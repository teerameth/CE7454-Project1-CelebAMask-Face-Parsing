import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_small

#######################################
## Simple Network for system testing ##
#######################################
class SimpleSegmentationNet(nn.Module):
    def __init__(self, num_classes=19):
        super(SimpleSegmentationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.conv5(x)
        return x
#####################################
## Custom Network for this project ##
#####################################
class ConfigurableASPP(nn.Module):
    """ Configurable Atrous Spatial Pyramid Pooling (ASPP) module
    Module designed for resampling a given feature layer with dilated convolution to adjust/control effective FoV, suitable for semantic segmentation task.
    Attributes:
        in_channels: number of input channels
        out_channels: number of output channels
        base_rate: base number for atrous rate
        depth: atrous depth (number of multiplier used for extending atrous rate)
        droupout_rate: probability to activate the dropouts
    Examples:
        With base_rate=4 and depth=3 --> atrous rates=[4, 8, 12]
             base_rate=3 and depth=4 --> atrous rates=[3, 6, 9, 12]
    """
    def __init__(self, in_channels, out_channels, base_rate=6, atrous_depth=3, dropout_rate=0.1):
        super(ConfigurableASPP, self).__init__()

        atrous_rates = [base_rate * (i + 1) for i in range(atrous_depth)]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.atrous_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=rate, dilation=rate, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            ) for rate in atrous_rates  # Apply each atrous (dilation) rates
        ])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * (atrous_depth + 3), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        res = [self.conv1(x), self.conv2(x)]
        res.extend([branch(x) for branch in self.atrous_branches])

        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(global_features, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_features)

        res = torch.cat(res, dim=1)
        return self.output_conv(res)

class MobileNetV3ASPP(nn.Module):
    """ Custom Network with MobileNetV3_small as backbone and configurable ASPP module

    """
    def __init__(self,
                 num_classes,
                 base_rate=6,
                 atrous_depth=3,
                 aspp_output_channels=256,
                 backbone_removed_layers=0,
                 backbone_width_multiplier=1.0,
                 dropout_rate=0.1):
        super(MobileNetV3ASPP, self).__init__()

        self.num_classes = num_classes

        # Use the smallest MobileNetV3 (without pretrained weight) as backbone template and specify the width multiplier
        mobilenet = mobilenet_v3_small(weights=None, width_mult=backbone_width_multiplier).features

        ## Initialize our custom network
        self.backbone = nn.Sequential()
        for i_layer in range(12-backbone_removed_layers):   # Pick only first 6-12 layers from MobileNetV3_small architecture.
            self.backbone.add_module(str(i_layer), mobilenet[i_layer])

        # Get the number of features from the last layer of the backbone
        backbone_out_features = self.backbone[-1].out_channels

        # Configurable ASPP module
        self.aspp = ConfigurableASPP(backbone_out_features,
                                     aspp_output_channels,
                                     base_rate,
                                     atrous_depth,
                                     dropout_rate)

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_output_channels, aspp_output_channels, 3, padding=1, groups=aspp_output_channels, bias=False),
            nn.BatchNorm2d(aspp_output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(aspp_output_channels, aspp_output_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(aspp_output_channels, num_classes, 1)
        )

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Extract features
        features = self.backbone(x)

        # Apply ASPP
        x = self.aspp(features)

        # Apply classifier
        x = self.classifier(x)

        # Upsample the output to match input resolution
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x  # Return raw predictions for training

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)