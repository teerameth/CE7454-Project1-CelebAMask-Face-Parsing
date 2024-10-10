import os
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2

from utils import SegmentationDataset, cmap
from model import SimpleSegmentationNet

num_class = 19
h, w = 512, 512

criterion = nn.CrossEntropyLoss()

val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

pred_dir = 'dataset/val/val_pred'

val_dataset = SegmentationDataset(
    img_dir='dataset/val/val_image',
    mask_dir='dataset/val/val_mask',
    transform=val_transform
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Create model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleSegmentationNet().to(device)
checkpoint = torch.load('segmentation_model.pth', weights_only=True)
model.load_state_dict(checkpoint)

# model.eval()
total_loss = 0
with torch.no_grad():
    for i, (images, masks) in tqdm(enumerate(val_loader)):
        images = images.to(device)
        masks = masks.to(device).long()

        outputs = model(images)
        loss = criterion(outputs, masks)

        mask = np.array(masks.cpu()).reshape((h, w))
        output = np.array(outputs.cpu()).reshape((num_class, h, w))
        class_map = np.argmax(output, axis=0)   # Convert from prediction containing probability for each class (num_class, height, width) to class map (height, width) containing class id

        pred_viz = cv2.cvtColor(cmap[class_map], cv2.COLOR_RGB2BGR)
        gt_viz = cv2.cvtColor(cmap[mask], cv2.COLOR_RGB2BGR)
        cv2.imshow("Ground Truth & Predicted", np.hstack([gt_viz, pred_viz]))
        cv2.waitKey(1)
        total_loss += loss.item()

        cv2.imwrite(os.path.join(pred_dir, f"{i}.png"), class_map)  # Write prediction image mask directly from class_map

print(total_loss / len(val_loader))

