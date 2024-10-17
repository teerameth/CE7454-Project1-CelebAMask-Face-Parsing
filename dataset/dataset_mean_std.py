from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def calculate_mean_std(loaders):
    mean = 0.
    std = 0.
    total_images = 0

    # for batch_idx, (image, mask) in enumerate(loader):
    #     image, mask = image, mask
    for loader in loaders:
        for images, _ in tqdm(loader):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += np.array(images).mean(2).sum(0)
            std += np.array(images).std(2).sum(0)
            total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

transform = A.Compose([
    ToTensorV2(),
])
# Create datasets
train_dataset = SegmentationDataset(
    img_dir='train/train_image',
    mask_dir='train/train_mask',
    transform=transform
)
val_dataset = SegmentationDataset(
    img_dir='val/val_image',
    mask_dir='val/val_mask',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

mean, std = calculate_mean_std([train_loader, val_loader])
print(f"Mean: {mean/255.0}")
print(f"Standard Deviation: {std/255.0}")