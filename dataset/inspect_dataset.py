import os
import torch
import numpy as np
from PIL.ImageOps import colorize
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import cv2

### From: https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/utils.py
cmap = np.array([(0, 0, 0), (204, 0, 0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                         (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def color_to_class_id(self, mask):
        mask_out = np.zeros(mask.shape[:2], dtype=np.int64)
        colorized_image = cmap[mask]
        cv2.imwrite("ColorizedMask.jpg", cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB))
        print(mask.shape)
        print(mask)
        print(np.unique(mask))
        exit(0)
        for color, class_id in self.color_to_class.items():
            # mask_out[np.where(mask == color)] = class_id
            mask_out[(mask == color)] = class_id
        return mask_out
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx]).replace('jpg', 'png')

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        # mask = self.color_to_class_id(mask) # Convert color to class id

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # Albumentations already give output as tensor
        else:   # Convert to tensor first
            mask = torch.from_numpy(mask)
        return image, mask
# Create datasets
train_dataset = SegmentationDataset(
    img_dir='train/train_image',
    mask_dir='train/train_mask'
)

val_dataset = SegmentationDataset(
    img_dir='val/val_image',
    mask_dir='val/val_mask'
)


from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

classes = []
from tqdm import tqdm
for batch_data, batch_mask in tqdm(val_loader):
    for mask in batch_mask:
        for class_id in np.unique(mask):
            if class_id not in classes:
                classes.append(class_id)
                print(classes)