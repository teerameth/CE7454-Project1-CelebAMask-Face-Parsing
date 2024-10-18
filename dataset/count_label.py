from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

counter = np.zeros((19), dtype=np.uint64)
def count_label(loaders):
    for loader in loaders:
        for _, masks in tqdm(loader):
            ids, counts = np.unique(masks, return_counts=True)

            for id, count in zip(ids, counts):
                counter[id] += int(count)

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

count_label([train_loader, val_loader])
print(counter)