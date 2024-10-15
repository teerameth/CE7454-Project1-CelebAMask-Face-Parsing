import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import SegmentationDataset, cmap
from model import EnhancedLightweightDeepLabV3

num_class = 19
h, w = 512, 512
_batch_size = 4

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

val_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)

# Create model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedLightweightDeepLabV3(num_classes=19).to(device)

checkpoint = torch.load('V3_epoch30.pth', weights_only=True)
model.load_state_dict(checkpoint)
param_count = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {param_count}")

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

        # pred_viz = cv2.cvtColor(cmap[pred_mask], cv2.COLOR_RGB2BGR)
        # gt_viz = cv2.cvtColor(cmap[mask], cv2.COLOR_RGB2BGR)
        # cv2.imshow("Ground Truth & Predicted", np.hstack([gt_viz, pred_viz]))
        # cv2.imshow("Class Map", np.array(class_map, np.uint8))
        # cv2.imshow("Ground Truth", np.array(masks.reshape((512, 512)).cpu(), np.uint8))
        # cv2.waitKey(0)
        # total_loss += loss.item()
        # save_path = os.path.join(pred_dir, f"{i}.png")
        # img = Image.fromarray(np.array(pred_mask, dtype=np.uint8))
        # img.save(save_path)
        # cv2.imwrite(save_path, class_map)  # Write prediction image mask directly from class_map

# print(total_loss / len(val_loader))

