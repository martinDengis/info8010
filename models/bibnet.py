import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified YOLO-style architecture for bib number detection
# Main reference:
#   - [Ultralytics YOLOv5 Architecture](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/)


# will often be called with same in_ and out_ channels
# goal: refine features rather than changing dimensionality
class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and leaky ReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ConvBlock called with same channels dim to simplify residual connections
# otherwise, would need adding a projection layer
class ResBlock(nn.Module):
    """Residual block with two convolutions and a skip connection"""

    def __init__(self, channels):
        super().__init__()

        self.conv1 = ConvBlock(channels, channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class DownsampleBlock(nn.Module):
    """Downsample block that reduces spatial dimensions by 2x"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x):
        return self.conv(x)


class Neck(nn.Module):
    """
    Feature pyramid network (FPN) neck to process and merge features from different scales.
    It connects the backbone modules and the BoundingBoxHead.
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        # Lateral connections (1x1 convs to reduce channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Top-down pathway (upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # After merging features
        self.smooth_convs = nn.ModuleList([
            ConvBlock(out_channels, out_channels, kernel_size=3)
            for _ in range(len(in_channels_list) - 1)
        ])

    def forward(self, features):
        # Process lateral connections
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]

        # Top-down pathway and feature fusion
        results = [laterals[-1]]  # Start with the deepest feature
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample the deeper feature
            upsampled = self.upsample(results[0])

            # Add the lateral connection
            fused = laterals[i] + upsampled

            # Apply smoothing
            if i < len(self.smooth_convs):
                fused = self.smooth_convs[i](fused)

            # Prepend to results
            results.insert(0, fused)

        return results


# block used for predicting bboxes
class BoundingBoxHead(nn.Module):
    """Head network for bounding box prediction  (no anchors != YOLOv5)"""

    def __init__(self, in_channels, num_coords=4):
        super().__init__()

        self.num_coords = num_coords

        # Convolutional layers before final prediction
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel_size=3)

        # Final prediction layer (no activation, raw outputs)
        self.pred = nn.Conv2d(in_channels, num_coords, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pred(x)

        # Reshape to [batch, H, W, 4] and then to [batch, H*W, 4]
        batch_size, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, 4]
        x = x.reshape(batch_size, height * width, self.num_coords)  # [batch, height*width, 4]

        return x


class BibNet(nn.Module):
    """
    BibNet model for bib number detection, based on a YOLO-style architecture.
    Uses anchor-free direct coordinate prediction.
    """

    def __init__(self,
                 input_channels=3,
                 backbone_channels=[64, 128, 256, 512],
                 neck_channels=256,
                 num_res_blocks=[1, 2, 8, 8],
                 num_coords=4):
        """
        Initialize BibNet model.

        Args:
            input_channels: Number of input image channels (default: 3 for RGB)
            backbone_channels: List of channel dimensions for each backbone stage
            neck_channels: Number of channels in the FPN neck
            num_res_blocks: Number of residual blocks in each backbone stage
            num_coords: Number of coordinates per bounding box (default: 4 for [x, y, w, h])
        """
        super().__init__()

        # Input convolution
        self.input_conv = ConvBlock(input_channels, backbone_channels[0], kernel_size=3)

        # Backbone
        self.backbone_stages = nn.ModuleList()

        in_channels = backbone_channels[0]
        for i, (out_channels, res_blocks) in enumerate(zip(backbone_channels, num_res_blocks)):
            stage = nn.Sequential()

            # Downsample (except for the first stage which already has input_conv)
            if i > 0:
                stage.add_module(f"downsample_{i}", DownsampleBlock(in_channels, out_channels))

            # Add residual blocks
            for j in range(res_blocks):
                stage.add_module(f"res_{i}_{j}", ResBlock(out_channels))

            self.backbone_stages.append(stage)
            in_channels = out_channels

        # FPN Neck
        self.neck = Neck(backbone_channels, neck_channels)

        # Bounding box prediction heads (one for each FPN level)
        self.bbox_heads = nn.ModuleList([
            BoundingBoxHead(neck_channels, num_coords)
            for _ in range(len(backbone_channels))
        ])

    def forward(self, x):
        """
        Forward pass of BibNet.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Flattened tensor of bounding box predictions [batch_size, total_predictions, 4]
        """
        # Initial convolution
        x = self.input_conv(x)

        # Extract features from backbone stages
        features = []
        for i, stage in enumerate(self.backbone_stages):
            x = stage(x) if i > 0 else stage(x)  # First stage doesn't need downsampling
            features.append(x)

        # Apply FPN neck
        fpn_features = self.neck(features)

        # Apply bbox heads
        bbox_preds = [head(feature) for head, feature in zip(self.bbox_heads, fpn_features)]

        # Concatenate predictions from different levels
        bbox_preds = torch.cat(bbox_preds, dim=1)  # [batch, sum(H*W), 4]

        return bbox_preds


def build_bibnet(cfg):
    """
    Build BibNet model from configuration.

    Args:
        cfg: Configuration dictionary with model parameters

    Returns:
        Initialized BibNet model
    """
    # Extract model parameters from cfg
    model_cfg = cfg.get('model', {})

    input_channels = model_cfg.get('input_channels', 3)
    backbone_channels = model_cfg.get('backbone_channels', [64, 128, 256, 512])
    neck_channels = model_cfg.get('neck_channels', 256)
    num_res_blocks = model_cfg.get('num_res_blocks', [1, 2, 8, 8])
    num_coords = model_cfg.get('num_coords', 4)

    # Create model
    model = BibNet(
        input_channels=input_channels,
        backbone_channels=backbone_channels,
        neck_channels=neck_channels,
        num_res_blocks=num_res_blocks,
        num_coords=num_coords
    )

    return model

