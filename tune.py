import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy

from triton.language import dtype

import wandb
from tqdm import tqdm
import numpy as np

from utils import SegmentationDataset, LovaszSoftmax, count_parameters
import lovasz_losses as L
from model import ConfigurableEnhancedLightweightDeepLabV3

wandb.login(key="e89b5bd847e91325b77ad10f073be9fea692b536")

_batch_size = 8

# def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, scaled_lr):
#     def lr_lambda(epoch):
#         if epoch < warmup_epochs:
#             return (scaled_lr - base_lr) * epoch / warmup_epochs + base_lr
#         else:
#             return scaled_lr - (epoch - warmup_epochs) * (scaled_lr - base_lr) / (total_epochs - warmup_epochs)
#
#     return LambdaLR(optimizer, lr_lambda)
# class WarmupLinearScaledLR(LambdaLR):
#     def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         super(WarmupLinearScaledLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
#
#     def lr_lambda(self, step):
#         if step < self.warmup_steps:
#             return float(step) / float(max(1, self.warmup_steps))
#         return max(0.0, float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps)))
class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # criterion = LovaszSoftmax()

    # Learning Rate Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
    if config.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    elif config.scheduler == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif config.scheduler == 'Linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, config.base_lr, total_iters=config.epochs)
    else:
        print("Invalid Scheduler")
        exit(1)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        for batch_idx, (image, mask) in tqdm(enumerate(train_loader)):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(image)

            out = F.softmax(output, dim=1)
            loss = L.lovasz_softmax(out, mask)
            # loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0
        area_intersect_all = np.zeros(19)
        area_union_all = np.zeros(19)
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                output = model(image)

                out = F.softmax(output, dim=1)
                loss = L.lovasz_softmax(out, mask)
                # loss = criterion(output, mask).item()
                val_loss += loss.item()

                ## Calculate mIOU ##
                for j in range(_batch_size):
                    gt_mask = np.array(mask.cpu()[j], dtype=np.float32)
                    pred_mask = np.array(output.cpu()[j], dtype=np.float32)
                    pred_mask = np.argmax(pred_mask, axis=0)  # Convert from prediction containing probability for each class (num_class, height, width) to class map (height, width) containing class id
                    for cls_idx in range(19):
                        area_intersect = np.sum((pred_mask == gt_mask) * (pred_mask == cls_idx))
                        area_pred_label = np.sum(pred_mask == cls_idx)
                        area_gt_label = np.sum(gt_mask == cls_idx)
                        area_union = area_pred_label + area_gt_label - area_intersect
                        area_intersect_all[cls_idx] += area_intersect
                        area_union_all[cls_idx] += area_union

        val_loss /= len(val_loader)

        ## Calculate mIOU ##
        iou_all = area_intersect_all / area_union_all * 100.0
        miou = iou_all.mean()

        wandb.log({
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "mIOU": miou,
        })

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss


def grid_search():
    sweep_config = {
        'method': 'bayes',  # Bayesian Search
        'metric': {'name': 'mIOU', 'goal': 'maximize'},
        'parameters': {
            ##################
            ## Architecture ##
            ##################
            'base_atrous_rate': {'distribution': 'int_uniform',
                                 'min': 2,
                                 'max': 8},
            'artrous_depth': {'distribution': 'int_uniform',
                              'min': 2,
                              'max': 6},
            'aspp_output_channels': {'values': [128, 256, 512]},
            # Number of N last layers removed from backbone network (MobileNetV3_small has 12 layers)
            'backbone_removed_layers': {'distribution': 'int_uniform',
                                        'min': 0,
                                        'max': 6},
            'dropout_rate': {
                "values": [0.1, 0.2, 0.3]
            },
            ##############
            ## Learning ##
            ##############
            'base_lr': {
                "distribution": 'log_uniform_values',
                "min": 1e-5,
                "max": 1e-1
            },
            'scheduler': {'values': ['Cosine', 'Plateau', 'Linear']},
            'epochs': {"values": [128, 256, 512]},
            'patience': {'value': 20},
            ##################
            ## Augmentation ##
            ##################
            'hue_shift_limit': {
                "min": 0,
                "max": 45
            },
            'sat_shift_limit': {
                "min": 0,
                "max": 64
            },
            'val_shift_limit': {
                "min": 0,
                "max": 64
            },
            'brightness_limit': {
                "min": 0.0,
                "max": 0.25
            },
            'contrast_limit': {
                "min": 0.0,
                "max": 0.25
            },
            'normalization': {
                # 'values': ["standard", "image", "image_per_channel", "min_max", "min_max_per_channel"]
                'value': "standard"
            }
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project="CE7454-Project1-CelebAMask-Face-Parsing-V2")


    best_overall_model = None
    best_overall_loss = float('inf')

    def train():
        nonlocal best_overall_model, best_overall_loss

        with wandb.init() as run:
            config = wandb.config

            # Apply binary search for backbone_width_multiplier that result in <= 2M trainable parameters
            multiplier = {'low': 0.0, 'high': 5.0}
            search_iter = 0
            _max_search_iter = 100  # Maximum tries
            _max_param_count  = 2e6
            while True:
                backbone_width_multiplier = (multiplier['low'] + multiplier['high']) / 2
                model = ConfigurableEnhancedLightweightDeepLabV3(num_classes=19,
                                                                 base_rate=config.base_atrous_rate,
                                                                 atrous_depth=config.artrous_depth,
                                                                 aspp_output_channels=config.aspp_output_channels,
                                                                 backbone_removed_layers=config.backbone_removed_layers,
                                                                 backbone_width_multiplier=backbone_width_multiplier,
                                                                 dropout_rate=config.dropout_rate)
                param_count = count_parameters(model)
                print(f"param_count: {param_count}, {backbone_width_multiplier}")
                if param_count > _max_param_count:
                    multiplier['high'] = backbone_width_multiplier
                else:
                    multiplier['low'] = backbone_width_multiplier
                search_iter += 1
                if search_iter > _max_search_iter or 0 < (_max_param_count - param_count) < 10000 or multiplier['high'] - multiplier['low'] < 0.01:
                    backbone_width_multiplier = multiplier['low']
                    break
            model = ConfigurableEnhancedLightweightDeepLabV3(num_classes=19,
                                                             base_rate=config.base_atrous_rate,
                                                             atrous_depth=config.artrous_depth,
                                                             aspp_output_channels=config.aspp_output_channels,
                                                             backbone_removed_layers=config.backbone_removed_layers,
                                                             backbone_width_multiplier=backbone_width_multiplier,
                                                             dropout_rate=config.dropout_rate)
            param_count = count_parameters(model)
            print(f"Param Count: {param_count}")
            wandb.log({"param_count": param_count,
                       "backbone_width_multiplier": backbone_width_multiplier})
            if param_count > _max_param_count:  # Invalid model (too large), skip training
                print("Can't find valid model size")
                wandb.log({
                    "val_loss": torch.inf
                })
                return

            # Data Augmentation
            train_transform = A.Compose([
                A.HorizontalFlip(always_apply=None, p=0.5), # Face image already aligned, no need for vertical flip or rotation
                # Randomly change HSV values of the image
                A.HueSaturationValue(hue_shift_limit=(-config.hue_shift_limit, config.hue_shift_limit),
                                     sat_shift_limit=(-config.sat_shift_limit, config.sat_shift_limit),
                                     val_shift_limit=(-config.val_shift_limit, config.val_shift_limit)),
                # Randomly scaling brightness & contrast based on max value of uint8 (255)
                A.RandomBrightnessContrast(brightness_limit=(-config.brightness_limit, config.brightness_limit),
                                           contrast_limit=(-config.contrast_limit, config.contrast_limit),
                                           brightness_by_max=True,
                                           p=0.5),
                # Normalize image using mean & S.D. from dataset
                A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                            std=(0.2677, 0.2408, 0.2334),
                            max_pixel_value=255.0,
                            normalization=config.normalization,
                            p=1.0),
                ToTensorV2(),
            ])
            val_transform = A.Compose([
                A.Normalize(mean=(0.5193, 0.4179, 0.3638),
                            std=(0.2677, 0.2408, 0.2334),
                            max_pixel_value=255.0,
                            normalization=config.normalization,
                            p=1.0),
                ToTensorV2(),
            ])
            # Create datasets
            train_dataset = SegmentationDataset(
                img_dir='dataset/train/train_image',
                mask_dir='dataset/train/train_mask',
                transform=train_transform
            )
            val_dataset = SegmentationDataset(
                img_dir='dataset/val/val_image',
                mask_dir='dataset/val/val_mask',
                transform=val_transform
            )
            train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)

            trained_model, best_val_loss = train_model(model, train_loader, val_loader, config)
            wandb.log({"best_val_loss": best_val_loss})
            # Update best overall model if this run has the lowest validation loss
            if best_val_loss < best_overall_loss:
                best_overall_loss = best_val_loss
                best_overall_model = copy.deepcopy(trained_model)

                # Save the best model weights
                torch.save(best_overall_model.state_dict(), f"best_model_{run.id}.pth")
                wandb.save(f"best_model_{run.id}.pth")

                # Log the best model information
                wandb.run.summary["best_model"] = f"best_model_{run.id}.pth"
                wandb.run.summary["best_val_loss"] = best_overall_loss


    wandb.agent(sweep_id, train)

    # After all runs, save the best overall model
    if best_overall_model is not None:
        torch.save(best_overall_model.state_dict(), "best_overall_model.pth")
        wandb.save("best_overall_model.pth")



if __name__ == "__main__":
    grid_search()