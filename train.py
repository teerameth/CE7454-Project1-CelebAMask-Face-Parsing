import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import SegmentationDataset
import lovasz_losses as L
import copy
import numpy as np
from model import MobileNetV3ASPP

_batch_size = 8
criterion = L.lovasz_softmax
backbone_width_multiplier = 2.34375
config = {
    'epochs': 128,
    'base_lr': 0.04764788015727089,
    'patience': 20,
    'scheduler': 'Cosine',
    'dropout_rate': 0.2,
    'artrous_depth': 6,
    'normalization': 'standard',
    'contrast_limit': 0.020115089762840793,
    'hue_shift_limit': 28,
    'sat_shift_limit': 48,
    'val_shift_limit': 17,
    'base_atrous_rate': 5,
    'brightness_limit': 0.05638424549287169,
    'aspp_output_channels': 256,
    'backbone_removed_layers': 5
    }

# Data Augmentation
train_transform = A.Compose([
    A.HorizontalFlip(always_apply=None, p=0.5), # Face image already aligned, no need for vertical flip or rotation
    # Randomly change HSV values of the image
    A.HueSaturationValue(hue_shift_limit=(-config["hue_shift_limit"], config["hue_shift_limit"]),
                         sat_shift_limit=(-config["sat_shift_limit"], config["sat_shift_limit"]),
                         val_shift_limit=(-config["val_shift_limit"], config["val_shift_limit"])),
    # Randomly scaling brightness & contrast based on max value of uint8 (255)
    A.RandomBrightnessContrast(brightness_limit=(-config["brightness_limit"], config["brightness_limit"]),
                               contrast_limit=(-config["contrast_limit"], config["contrast_limit"]),
                               brightness_by_max=True,
                               p=0.5),
    # Normalize image using mean & S.D. from dataset
    A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                std=(0.2677, 0.2408, 0.2334),
                max_pixel_value=255.0,
                normalization=config["normalization"],
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

def train_model(model, train_loader, val_loader, criterion, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Learning Rate Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(config["epochs"]):
        model.train()
        for batch_idx, (image, mask) in tqdm(enumerate(train_loader)):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(image)

            out = F.softmax(output, dim=1)
            # loss = L.lovasz_softmax(out, mask)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0
        area_intersect_all = np.zeros(19)
        area_union_all = np.zeros(19)
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                output = model(image)

                out = F.softmax(output, dim=1)
                # loss = L.lovasz_softmax(out, mask)
                loss = criterion(out, mask)
                val_loss += loss.item()

                ## Calculate mIOU on validation dataset ##
                for j in range(_batch_size):
                    gt_mask = np.array(mask.cpu()[j], dtype=np.float32)
                    pred_mask = np.array(output.cpu()[j], dtype=np.float32)
                    pred_mask = np.argmax(pred_mask, axis=0)  # Convert from prediction containing probability for each class (num_class, height, width) to class map (height, width) containing class id
                    for cls_idx in range(19):
                        area_intersect = np.sum((pred_mask == gt_mask) * (pred_mask == cls_idx))
                        area_pred_label = np.sum(pred_mask == cls_idx)
                        area_gt_label = np.sum(gt_mask == cls_idx)
                        area_union = area_pred_label + area_gt_label - area_intersect
                        area_intersect_all[cls_idx] += area_intersect
                        area_union_all[cls_idx] += area_union

        val_loss /= len(val_loader)

        ## Calculate mIOU ##
        iou_all = area_intersect_all / area_union_all * 100.0
        miou = iou_all.mean()
        print(miou)

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= config["patience"]:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss

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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Create model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetV3ASPP(num_classes=19,
                        base_rate=config["base_atrous_rate"],
                        atrous_depth=config["artrous_depth"],
                        aspp_output_channels=config["aspp_output_channels"],
                        backbone_removed_layers=config["backbone_removed_layers"],
                        backbone_width_multiplier=backbone_width_multiplier,
                        dropout_rate=config["dropout_rate"]).to(device)
# Check parameter count
param_count = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {param_count}")

trained_model, best_val_loss = train_model(model, train_loader, val_loader, criterion, config)

torch.save(trained_model.state_dict(), f"best_model.pth")