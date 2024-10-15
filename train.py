import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import SegmentationDataset, LovaszSoftmax
import lovasz_losses as L
import numpy as np

# Data Augmentation
train_transform = A.Compose([
                A.HorizontalFlip(always_apply=None, p=0.5), # Face image already aligned, no need for vertical flip or rotation
                # Randomly change HSV values of the image
                A.HueSaturationValue(hue_shift_limit=(-20, 20),
                                     sat_shift_limit=(-20, 20),
                                     val_shift_limit=(-20, 20)),
                # Randomly scaling brightness & contrast based on max value of uint8 (255)
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                           contrast_limit=(-0.2, 0.2),
                                           brightness_by_max=True,
                                           p=0.5),
                # Normalize image using mean & S.D. from dataset
                A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                            std=(0.2677, 0.2408, 0.2334),
                            max_pixel_value=255.0,
                            normalization="standard",
                            p=1.0),
                ToTensorV2(),
            ])

val_transform = A.Compose([
    A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                std=(0.2677, 0.2408, 0.2334),
                max_pixel_value=255.0,
                normalization="standard",
                p=1.0),
    ToTensorV2(),
])

# Create datasets
train_dataset = SegmentationDataset(
    img_dir='dataset/train/train_image',
    mask_dir='dataset/train/train_mask',
    transform=train_transform
)

val_dataset = SegmentationDataset(
    img_dir='dataset/val/val_image',
    mask_dir='dataset/val/val_mask',
    transform=val_transform
)


# Create model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleSegmentationNet().to(device)
# model = CustomDeepLabV3(num_classes=19).to(device)
# model = LightweightDeepLabV3(num_classes=19).to(device)

from model import EnhancedLightweightDeepLabV3, ConfigurableEnhancedLightweightDeepLabV3
# model = EnhancedLightweightDeepLabV3(num_classes=19).to(device)
model = ConfigurableEnhancedLightweightDeepLabV3(num_classes=19, base_rate=2, atrous_depth=4).to(device)
# Check parameter count
param_count = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {param_count}")

criterion = LovaszSoftmax()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=5,
                                                       verbose=True)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device).long() # Also convert from torch.uint8 --> torch.Long (64-bit integer)

        optimizer.zero_grad()
        outputs = model(images)

        out = F.softmax(outputs, dim=1)
        loss = L.lovasz_softmax(out, masks)
        # loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)

            out = F.softmax(outputs, dim=1)
            loss = L.lovasz_softmax(out, masks)
            # loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    if epoch % 10 == 0:
        # Save the trained model
        torch.save(model.state_dict(), f'V2_epoch{epoch}.pth')

torch.save(model.state_dict(), 'V2.pth')

# torch.save({
#             'epoch': EPOCH,
#             'model_state_dict': net.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)

print("Training completed!")