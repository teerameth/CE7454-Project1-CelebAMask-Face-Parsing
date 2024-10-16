from model import ConfigurableEnhancedLightweightDeepLabV3
from utils import SegmentationDataset, count_parameters
import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
wandb.login(key="e89b5bd847e91325b77ad10f073be9fea692b536")

api = wandb.Api()
run = api.run("teerameth/CE7454-Project1-CelebAMask-Face-Parsing-V2/bzwql5yt")

# Get the config used for the run
config = run.config
print(config)
# Get the metrics logged during the run
history = run.history()
print(history)
# Get the summary statistics
summary = run.summary
print(summary)
print(summary["mIOU"])
# Get any files associated with the run
files = run.files()
print(files)

# wandb.init(project="CE7454-Project1-CelebAMask-Face-Parsing-V2", resume="bzwql5yt")
# config = wandb.config
#
# print(config)

# model_file = run.file("best_model_bzwql5yt.pth")
# model_file.download(replace=True)

# Apply binary search for backbone_width_multiplier that result in <= 2M trainable parameters
multiplier = {'low': 0.0, 'high': 5.0}
search_iter = 0
_max_search_iter = 100  # Maximum tries
_max_param_count = 2e6
# wandb.init()
while True:
    backbone_width_multiplier = (multiplier['low'] + multiplier['high']) / 2
    model = ConfigurableEnhancedLightweightDeepLabV3(num_classes=19,
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



val_transform = A.Compose([
    A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                std=(0.2677, 0.2408, 0.2334),
                max_pixel_value=255.0,
                normalization="standard",
                p=1.0),
    ToTensorV2(),
])
val_dataset = SegmentationDataset(
    img_dir='dataset/val/val_image',
    mask_dir='dataset/val/val_mask',
    transform=val_transform
)
_batch_size = 1
val_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConfigurableEnhancedLightweightDeepLabV3(num_classes=19,
                                                 base_rate=config["base_atrous_rate"],
                                                 atrous_depth=config["artrous_depth"],
                                                 aspp_output_channels=config["aspp_output_channels"],
                                                 backbone_removed_layers=config["backbone_removed_layers"],
                                                 backbone_width_multiplier=backbone_width_multiplier,
                                                 dropout_rate=config["dropout_rate"]).to(device)
param_count = count_parameters(model)
print(f"Param Count: {param_count}")

model.load_state_dict(torch.load('best_model_bzwql5yt.pth'))

area_intersect_all = np.zeros(19)
area_union_all = np.zeros(19)
with torch.no_grad():
    for i, (images, masks) in tqdm(enumerate(val_loader)):
        images = images.to(device)
        gt_masks = masks.to(device).long()

        model.eval()
        pred_masks = model(images)  # Inference
        for j in range(_batch_size):
            gt_mask = np.array(gt_masks.cpu()[j])
            pred_mask = np.array(pred_masks.cpu()[j])
            pred_mask = np.argmax(pred_mask, axis=0)  # Convert from prediction containing probability for each class (num_class, height, width) to class map (height, width) containing class id

            for cls_idx in range(19):
                area_intersect = np.sum(
                    (pred_mask == gt_mask) * (pred_mask == cls_idx))

                area_pred_label = np.sum(pred_mask == cls_idx)
                area_gt_label = np.sum(gt_mask == cls_idx)
                area_union = area_pred_label + area_gt_label - area_intersect

                area_intersect_all[cls_idx] += area_intersect
                area_union_all[cls_idx] += area_union

    iou_all = area_intersect_all / area_union_all * 100.0
    miou = iou_all.mean()
    print(miou)