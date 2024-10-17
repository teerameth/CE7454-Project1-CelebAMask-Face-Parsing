from model import MobileNetV3ASPP
from utils import SegmentationDataset, count_parameters
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image as Image

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


val_transform = A.Compose([
    A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                std=(0.2677, 0.2408, 0.2334),
                max_pixel_value=255.0,
                normalization="standard",
                p=1.0),
    ToTensorV2(),
])

pred_dir = 'dataset/val/val_pred'

val_dataset = SegmentationDataset(
    img_dir='dataset/val/val_image',
    mask_dir='dataset/val/val_mask',
    transform=val_transform
)
_batch_size = 1
val_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)

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


def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1 score for semantic segmentation.

    Args:
    y_true: Ground truth segmentation mask
    y_pred: Predicted segmentation mask

    Returns:
    f1_score: The calculated F1 score
    """
    # Ensure the inputs are boolean masks
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    # Calculate True Positives, False Positives, and False Negatives
    TP = np.sum(np.logical_and(y_pred, y_true))
    FP = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
    FN = np.sum(np.logical_and(np.logical_not(y_pred), y_true))

    # Calculate Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def calculate_metrics(model, val_loader, device, num_classes=19, batch_size=1):
    area_intersect_all = np.zeros(num_classes)
    area_union_all = np.zeros(num_classes)
    f1_scores = []

    with torch.no_grad():
        for i, (images, masks) in tqdm(enumerate(val_loader)):
            images = images.to(device)
            gt_masks = masks.to(device).long()

            model.eval()
            pred_masks = model(images)  # Inference

            for j in range(batch_size):
                gt_mask = gt_masks[j].cpu().numpy()
                pred_mask = pred_masks[j].cpu().numpy()
                pred_mask = np.argmax(pred_mask, axis=0)  # Convert to class map

                # Calculate F1 score
                f1 = calculate_f1_score(gt_mask, pred_mask)
                f1_scores.append(f1)

                # Calculate mIOU
                for cls_idx in range(num_classes):
                    area_intersect = np.sum((pred_mask == gt_mask) * (pred_mask == cls_idx))
                    area_pred_label = np.sum(pred_mask == cls_idx)
                    area_gt_label = np.sum(gt_mask == cls_idx)
                    area_union = area_pred_label + area_gt_label - area_intersect

                    area_intersect_all[cls_idx] += area_intersect
                    area_union_all[cls_idx] += area_union

            save_path = os.path.join(pred_dir, f"{i}.png")
            img = Image.fromarray(np.array(pred_mask, dtype=np.uint8))
            img.save(save_path)
    # Calculate final metrics
    iou_all = area_intersect_all / area_union_all * 100.0
    miou = iou_all.mean()
    mean_f1 = np.mean(f1_scores)

    return miou, mean_f1


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assume model and val_loader are defined elsewhere
miou, mean_f1 = calculate_metrics(model, val_loader, device)
print(f"mIOU: {miou:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")