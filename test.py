import os
from inspect import classify_class_attrs

import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from glob import glob

from utils import SegmentationDataset, LovaszSoftmax, cmap
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

num_class = 19
h, w = 512, 512

criterion = LovaszSoftmax()

transform = A.Compose([
            A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                        std=(0.2677, 0.2408, 0.2334),
                        max_pixel_value=255.0,
                        normalization="standard",
                        p=1.0),
            ToTensorV2(),
])

pred_dir = 'dataset/test/test_pred'

img_patt='dataset/test/test_image/*.jpg'
img_list = glob(img_patt)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleSegmentationNet().to(device)
# model = deeplabv3_mobilenet_v3_large(num_classes=19).to(device)

from model import EnhancedLightweightDeepLabV3
model = EnhancedLightweightDeepLabV3(num_classes=19).to(device)

checkpoint = torch.load('V3_epoch30.pth', weights_only=True)
model.load_state_dict(checkpoint)
param_count = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {param_count}")
# model.eval()
total_loss = 0
with torch.no_grad():
    for i, image_path in tqdm(enumerate(img_list)):
        image_name = os.path.basename(image_path).split('.')[0]
        image = np.array(Image.open(image_path).convert("RGB"))
        augmented = transform(image=image)
        images = augmented['image'].reshape((1, 3, 512, 512))
        images = images.to(device)
        model.eval()
        outputs = model(images)
        output = np.array(outputs.cpu()).reshape((num_class, h, w))
        class_map = np.argmax(output, axis=0)   # Convert from prediction containing probability for each class (num_class, height, width) to class map (height, width) containing class id

        pred_viz = cv2.cvtColor(cmap[class_map], cv2.COLOR_RGB2BGR)
        save_path = os.path.join(pred_dir, f"{image_name}.png")
        img = Image.fromarray(np.array(class_map, dtype=np.uint8))
        img.save(save_path)


