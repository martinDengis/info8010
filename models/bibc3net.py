import torch
import torch.nn as nn
import torch.nn.functional as F


class C3Block(nn.Module):
    """
    C3 Block: A custom convolutional block for feature extraction.
    It consists of two convolutional layers with batch normalization, ReLU activation,
    and a skip connection for better gradient flow.
    """

    def __init__(self, in_channels, out_channels):
        super(C3Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection handling (1x1 conv if input and output channels differ)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # Skip connection
        x = F.relu(x)     # ReLU after addition
        return x


class PBlock(nn.Module):
    """
    P Block: A block containing a sequence of C3Blocks followed by pooling.
    Part of the pattern: [[CONV → ReLU]*C → POOL]*P
    """

    def __init__(self, in_channels, out_channels, c_blocks=2):
        super(PBlock, self).__init__()

        # Create sequence of C C3Blocks: [CONV → ReLU]*C
        layers = []
        current_channels = in_channels

        for _ in range(c_blocks):
            layers.append(C3Block(current_channels, out_channels))
            current_channels = out_channels

        # Add pooling layer: [[CONV → ReLU]*C → POOL]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Create the block
        self.p_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.p_block(x)


class BboxPredictionHead(nn.Module):
    """
    Bbox Prediction Head: A fully connected head for bounding box prediction.
    It supports multiple FC→ReLU layers, followed by a final FC layer.
    Output tensor has shape [batch_size, N, 5] where:
    - N is the number of bounding boxes (max_detections)
    - 5 is for [x, y, w, h, confidence]
    """

    def __init__(self, in_features, max_detections, num_fc_layers=1, hidden_dim=512):
        super(BboxPredictionHead, self).__init__()
        self.max_detections = max_detections

        # Create variable number of FC→ReLU layers
        layers = []
        current_dim = in_features

        for _ in range(num_fc_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim

        # Add the final FC layer
        # 5 values per box (x, y, w, h, conf)
        layers.append(nn.Linear(current_dim, max_detections * 5))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)

        # Reshape to [batch_size, max_detections, 5]
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_detections, 5)

        # Apply sigmoid to ensure outputs are in [0, 1]
        # All values (x, y, w, h, conf) should be normalized to [0, 1]
        x = torch.sigmoid(x)

        return x


class BibC3Net(nn.Module):
    """
    BibC3Net: CNN-based model for bib number detection using C3 blocks
    Predicts a fixed number of bounding boxes for each image
    """
    # INPUT → [[CONV → ReLU]*C → POOL?]*P → Global Average Pooling → [FC → ReLU]*L → FC → [BBOXES]

    def __init__(self, input_channels=3, max_detections=10, p_blocks=3, c_blocks=2,
                 feature_channels=[64, 128, 256], num_fc_layers=1, hidden_dim=512):
        """
        Initialize BibC3Net model.

        Args:
            input_channels: Number of input image channels (default: 3 for RGB)
            max_detections: Maximum number of bounding boxes to predict (N)
            p_blocks: Number of pooling blocks (P)
            c_blocks: Number of C3 blocks per pooling block (C)
            feature_channels: List of channel dimensions for each P block
            num_fc_layers: Number of FC layers in the prediction head
            hidden_dim: Hidden dimension size for FC layers
        """
        super(BibC3Net, self).__init__()

        if len(feature_channels) != p_blocks:
            raise ValueError(
                f"feature_channels must have length p_blocks ({p_blocks})")

        self.max_detections = max_detections

        # Initial convolution
        self.initial_block = nn.Sequential(
            nn.Conv2d(input_channels,
                      feature_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(feature_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Create P blocks, each with C * C3Blocks followed by pooling
        # [[CONV → ReLU]*C → POOL]*P
        self.p_blocks = nn.ModuleList()
        for p in range(p_blocks):
            # Define input and output channels for this PBlock
            in_ch = feature_channels[p-1] if p > 0 else feature_channels[0]
            out_ch = feature_channels[p]

            # Create and add the PBlock
            self.p_blocks.append(PBlock(in_ch, out_ch, c_blocks))

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Prediction head with customizable number of FC layers
        self.head = BboxPredictionHead(
            in_features=feature_channels[-1],
            max_detections=max_detections,
            num_fc_layers=num_fc_layers,
            hidden_dim=hidden_dim
        )

    def forward(self, x):
        """
        Forward pass of BibC3Net.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Tensor of bounding box predictions [batch_size, N, 5] where:
            - N is the number of bounding boxes (max_detections)
            - 5 is for [x, y, w, h, confidence]
            - x, y, w, h are normalized to [0, 1] and confidence is a score in [0, 1]
        """
        # Initial convolution
        x = self.initial_block(x)

        # Process through P blocks
        for p_block in self.p_blocks:
            x = p_block(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.flatten(1)  # Flatten to [batch_size, channels]

        # Prediction head
        bbox_preds = self.head(x)
        # print(f"bbox_preds: {bbox_preds.shape = }, {bbox_preds = }")

        return bbox_preds


def build_bibc3net(cfg):
    """
    Build BibC3Net model from configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Initialized BibC3Net model
    """
    model_cfg = cfg.get("model", {})
    in_channels = model_cfg.get("input_channels", 3)
    max_detections = model_cfg.get("max_detections", 10)
    p_blocks = model_cfg.get("p_blocks", 3)
    c_blocks = model_cfg.get("c_blocks", 2)
    feature_channels = model_cfg.get("feature_channels", [64, 128, 256])
    num_fc_layers = model_cfg.get("num_fc_layers", 1)
    hidden_dim = model_cfg.get("hidden_dim", 512)

    model = BibC3Net(
        input_channels=in_channels,
        max_detections=max_detections,
        p_blocks=p_blocks,
        c_blocks=c_blocks,
        feature_channels=feature_channels,
        num_fc_layers=num_fc_layers,
        hidden_dim=hidden_dim
    )
    return model
