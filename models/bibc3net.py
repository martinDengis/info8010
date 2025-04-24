import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3Block(nn.Module):
    """
    C3 Block: Cross-stage Convolutional Connection Block
    Combines standard convolution with cross connections for improved feature extraction
    """

    def __init__(self, in_channels, out_channels, bottleneck_ratio=0.5, num_blocks=3):
        super().__init__()

        bottleneck_channels = int(out_channels * bottleneck_ratio)

        # Main branch conv
        self.cv1 = ConvBlock(in_channels, bottleneck_channels, kernel_size=1)

        # Secondary branch with multiple ConvBlocks in a Sequential structure
        self.cv2 = nn.Sequential(*[
            ConvBlock(bottleneck_channels, bottleneck_channels)
            for _ in range(num_blocks)
        ])

        # Parallel branch for cross-connection
        self.cv3 = ConvBlock(in_channels, bottleneck_channels, kernel_size=1)

        # Output fusion
        self.cv4 = ConvBlock(bottleneck_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y1 = self.cv2(self.cv1(x))
        y2 = self.cv3(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class PoolingPyramid(nn.Module):
    """
    Pooling Pyramid Module for multi-scale feature extraction
    Similar to Spatial Pyramid Pooling but with fixed output size
    """

    def __init__(self, channels, levels=[1, 2, 3, 6]):
        super().__init__()

        self.levels = levels
        out_channels = channels // len(levels)

        # Per-level convolutions after pooling
        self.convs = nn.ModuleList([
            ConvBlock(channels, out_channels, kernel_size=1)
            for _ in levels
        ])

        # Final fusion
        self.fusion = ConvBlock(out_channels * len(levels), channels, kernel_size=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        features = []

        for i, level in enumerate(self.levels):
            # Fixed-size adaptive pooling
            pooled = F.adaptive_avg_pool2d(x, output_size=(level, level))
            # Conv to reduce channels
            processed = self.convs[i](pooled)
            # Upsample back to original size
            upsampled = F.interpolate(processed, size=(h, w), mode='bilinear', align_corners=False)
            features.append(upsampled)

        # Concatenate features from all levels
        out = torch.cat(features, dim=1)
        # Final fusion
        return self.fusion(out)


class BoundingBoxPredictor(nn.Module):
    """
    Bounding box prediction head for CNN-based detection approach
    Predicts a fixed number of bounding boxes with confidence scores
    """

    def __init__(self, in_channels, feature_size, max_detections=100, num_coords=4):
        super().__init__()

        self.feature_size = feature_size  # H, W of the feature map
        self.max_detections = max_detections
        self.num_coords = num_coords

        # Reduce spatial dimensions with strided convolutions
        self.pre_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=2),
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=2),
        )

        # Calculate resulting feature map size after strided convolutions
        reduced_h = feature_size[0] // 4
        reduced_w = feature_size[1] // 4
        flattened_size = reduced_h * reduced_w * in_channels

        # Fully connected layers for box prediction
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, max_detections * num_coords)
        )

    def forward(self, x):
        # Get initial feature dimensions
        batch_size, channels, height, width = x.shape

        # Adaptively resize to the expected feature size for this model
        if height != self.feature_size[0] or width != self.feature_size[1]:
            x = F.adaptive_avg_pool2d(x, self.feature_size)

        # Reduce spatial dimensions
        x = self.pre_conv(x)

        # Predict boxes
        boxes = self.fc_layers(x)
        # Reshape to [batch_size, max_detections, num_coords]
        boxes = boxes.view(-1, self.max_detections, self.num_coords)
        # Apply sigmoid to normalize coordinates to [0, 1]
        boxes = torch.sigmoid(boxes)

        return boxes


class BibC3Net(nn.Module):
    """
    BibC3Net: CNN-based model for bib number detection using C3 blocks
    Predicts a fixed number of bounding boxes for each image
    """

    def __init__(self,
                 input_channels=3,
                 channels_list=[64, 128, 256, 512],
                 num_c3_blocks=[1, 2, 3, 4],
                 bottleneck_ratio=0.5,
                 feature_size=(16, 16),
                 max_detections=100,
                 num_coords=4,
                 use_spp=True):
        """
        Initialize BibC3Net model.

        Args:
            input_channels: Number of input image channels (default: 3 for RGB)
            channels_list: List of channel dimensions for each stage
            num_c3_blocks: Number of C3 blocks in each stage
            bottleneck_ratio: Ratio for bottleneck channels in C3 blocks
            feature_size: Size of feature map before bbox prediction (H, W)
            max_detections: Maximum number of detections per image
            num_coords: Number of coordinates per bounding box (default: 4 for [x, y, w, h])
            use_spp: Whether to use Spatial Pyramid Pooling
        """
        super().__init__()

        self.max_detections = max_detections
        self.feature_size = feature_size

        # Input convolution
        self.input_conv = ConvBlock(input_channels, channels_list[0], kernel_size=3)

        # Build network stages
        self.stages = nn.ModuleList()
        in_channels = channels_list[0]

        for i, (out_channels, num_blocks) in enumerate(zip(channels_list, num_c3_blocks)):
            stage = nn.Sequential()

            # Downsample at the beginning of each stage (except first)
            if i > 0:
                stage.add_module(f"downsample_{i}",
                                ConvBlock(in_channels, out_channels, kernel_size=3, stride=2))
            else:
                # For first stage, just adapt channels if needed
                if in_channels != out_channels:
                    stage.add_module(f"adapt_{i}",
                                    ConvBlock(in_channels, out_channels, kernel_size=1))

            # Add C3 blocks
            for j in range(num_blocks):
                stage.add_module(f"c3_{i}_{j}",
                                C3Block(out_channels, out_channels, bottleneck_ratio))

            self.stages.append(stage)
            in_channels = out_channels

        # Optional Spatial Pyramid Pooling
        self.use_spp = use_spp
        if use_spp:
            self.spp = PoolingPyramid(channels_list[-1])

        # Bounding box predictor
        self.bbox_predictor = BoundingBoxPredictor(
            channels_list[-1],
            feature_size,
            max_detections,
            num_coords
        )

    def forward(self, x):
        """
        Forward pass of BibC3Net.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Tensor of bounding box predictions [batch_size, max_detections, 4]
        """
        # Initial convolution
        x = self.input_conv(x)

        # Process through stages
        for stage in self.stages:
            x = stage(x)

        # Apply SPP if enabled
        if self.use_spp:
            x = self.spp(x)

        # Predict bounding boxes
        bbox_preds = self.bbox_predictor(x)

        return bbox_preds


def build_bibc3net(cfg):
    """
    Build BibC3Net model from configuration.

    Args:
        cfg: Configuration dictionary with model parameters

    Returns:
        Initialized BibC3Net model
    """
    # Extract model parameters from cfg
    model_cfg = cfg.get('model', {})

    input_channels = model_cfg.get('input_channels', 3)
    channels_list = model_cfg.get('channels_list', [64, 128, 256, 512])
    num_c3_blocks = model_cfg.get('num_c3_blocks', [1, 2, 3, 4])
    bottleneck_ratio = model_cfg.get('bottleneck_ratio', 0.5)
    feature_size = model_cfg.get('feature_size', (16, 16))
    max_detections = model_cfg.get('max_detections', 100)
    num_coords = model_cfg.get('num_coords', 4)
    use_spp = model_cfg.get('use_spp', True)

    # Create model
    model = BibC3Net(
        input_channels=input_channels,
        channels_list=channels_list,
        num_c3_blocks=num_c3_blocks,
        bottleneck_ratio=bottleneck_ratio,
        feature_size=feature_size,
        max_detections=max_detections,
        num_coords=num_coords,
        use_spp=use_spp
    )

    return model