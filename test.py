from model import MobileNetV3ASPP
from utils import count_parameters
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

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

# Specify number of prediction classes and input image dimension
num_class = 19
h, w = 512, 512

# Apply normalization (the same setting used during training)
transform = A.Compose([
            A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                        std=(0.2677, 0.2408, 0.2334),
                        max_pixel_value=255.0,
                        normalization="standard",
                        p=1.0),
            ToTensorV2(),
])

# Input image pattern
img_patt='dataset/test/test_image/*.jpg'
# Output directiry
pred_dir = 'dataset/test/test_pred'
img_list = glob(img_patt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetV3ASPP(num_classes=19,
                        base_rate=config["base_atrous_rate"],
                        atrous_depth=config["artrous_depth"],
                        aspp_output_channels=config["aspp_output_channels"],
                        backbone_removed_layers=config["backbone_removed_layers"],
                        backbone_width_multiplier=backbone_width_multiplier,
                        dropout_rate=config["dropout_rate"]).to(device)
param_count = count_parameters(model)
print(f"Param Count: {param_count}")

model.load_state_dict(torch.load('best_model_bzwql5yt.pth'))

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

        save_path = os.path.join(pred_dir, f"{image_name}.png")
        img = Image.fromarray(np.array(class_map, dtype=np.uint8))
        img.save(save_path)