from model import MobileNetV3ASPP
from utils import count_parameters
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

api = wandb.Api()
run = api.run("teerameth/CE7454-Project1-CelebAMask-Face-Parsing-V2/bzwql5yt")  # Best successful run (got highest mIOU)

# Get the config used for the run
config = run.config
print(f"Run Condig:\nconfig")
# Get the metrics logged during the run
history = run.history()
# Get the summary statistics
summary = run.summary
# Get any files associated with the run
files = run.files()

#################################################################################################
## Apply binary search for backbone_width_multiplier that result in <= 2M trainable parameters ##
#################################################################################################
multiplier = {'low': 0.0, 'high': 5.0}
search_iter = 0
_max_search_iter = 100  # Maximum tries
_max_param_count = 2e6

while True:
    backbone_width_multiplier = (multiplier['low'] + multiplier['high']) / 2
    model = MobileNetV3ASPP(num_classes=19,
                                                     base_rate=config["base_atrous_rate"],
                                                     atrous_depth=config["artrous_depth"],
                                                     aspp_output_channels=config["aspp_output_channels"],
                                                     backbone_removed_layers=config["backbone_removed_layers"],
                                                     backbone_width_multiplier=backbone_width_multiplier,
                                                     dropout_rate=config["dropout_rate"])
    param_count = count_parameters(model)
    print(f"param_count: {param_count}, {backbone_width_multiplier}")
    if param_count > _max_param_count:
        multiplier['high'] = backbone_width_multiplier
    else:
        multiplier['low'] = backbone_width_multiplier
    search_iter += 1
    if search_iter > _max_search_iter or 0 < (_max_param_count - param_count) < 10000 or multiplier['high'] - \
            multiplier['low'] < 0.01:
        backbone_width_multiplier = multiplier['low']
        break

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