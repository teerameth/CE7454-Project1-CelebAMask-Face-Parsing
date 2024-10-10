import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import SegmentationDataset
from model import SimpleSegmentationNet

# Data Augmentation
train_transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    # A.VerticalFlip(),     # Face data already aligned, no need to flip up-side down
    # A.RandomRotate90(),   # Face data already aligned, no need to rotate 90 deg
    A.HueSaturationValue(),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(),
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
model = SimpleSegmentationNet().to(device)

# Check parameter count
param_count = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {param_count}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device).long() # Also convert from torch.uint8 --> torch.Long (64-bit integer)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
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
            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'segmentation_model.pth')

# torch.save({
#             'epoch': EPOCH,
#             'model_state_dict': net.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)

print("Training completed!")