import torch
import torch.nn as nn
import torch.nn.functional as F


class CBlock(nn.Module):
    """
    C Block: A custom convolutional block for feature extraction.
    It consists of two convolutional layers with batch normalization, ReLU activation,
    and a skip connection for better gradient flow.
    """

    def __init__(self, in_channels, out_channels):
        super(CBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection handling (1x1 conv if input and output channels differ)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # Skip connection
        x = F.leaky_relu(x)     # LeakyReLU after addition
        return x


class PBlock(nn.Module):
    """
    P Block: A block containing a sequence of CBlocks followed by pooling.
    Part of the pattern: [[CONV → ReLU]*C → POOL]*P
    """

    def __init__(self, in_channels, out_channels, c_blocks=2):
        super(PBlock, self).__init__()

        # Create sequence of C CBlocks: [CONV → ReLU]*C
        layers = []
        current_channels = in_channels

        for _ in range(c_blocks):
            layers.append(CBlock(current_channels, out_channels))
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
    It supports multiple FC→LeakyReLU layers, followed by a final FC layer.
    Output shape is [batch_size, split_size*split_size*(num_classes+num_boxes*5)]
    """

    def __init__(self, in_features, split_size=7, num_boxes=2, num_classes=1, hidden_size=512, num_fc_layers=1):
        super(BboxPredictionHead, self).__init__()
        self.num_fc_layers = max(1, num_fc_layers)

        # FC Network Creation similar to YOLOv1's _create_fcs
        layers = []

        # First layer from flattened features to hidden size
        layers.extend([
            nn.Linear(in_features * split_size * split_size, hidden_size),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1)
        ])

        # Additional hidden layers if requested
        for _ in range(self.num_fc_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Dropout(0.0))
            layers.append(nn.LeakyReLU(0.1))

        # Final output layer
        output_dim = split_size * split_size * (num_classes + num_boxes * 5)
        layers.append(nn.Linear(hidden_size, output_dim))

        self.fc_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_net(x)


class BibNet(nn.Module):
    """
    BibNet: CNN-based model for bib number detection using C blocks
    """
    # INPUT → [[CONV → ReLU]*C → POOL?]*P → Global Average Pooling → [FC → ReLU]*L → FC → [BBOXES]

    def __init__(self, input_channels=3, p_blocks=3, c_blocks=2, feature_channels=[64, 128, 256], split_size=7, num_boxes=2, num_classes=1, hidden_size=512, num_fc_layers=1):
        """
        Initialize BibNet model.
        """
        super(BibNet, self).__init__()

        if len(feature_channels) != p_blocks:
            raise ValueError(
                f"feature_channels must have length p_blocks ({p_blocks})")

        # Initial convolution
        self.initial_block = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(feature_channels[0]),
            nn.LeakyReLU(inplace=True)
        )

        # Create P blocks, each with C * CBlocks followed by pooling
        # [[CONV → ReLU]*C → POOL]*P
        self.p_blocks = nn.ModuleList()
        for p in range(p_blocks):
            # Define input and output channels for this PBlock
            in_ch = feature_channels[p-1] if p > 0 else feature_channels[0]
            out_ch = feature_channels[p]

            # Create and add the PBlock
            self.p_blocks.append(PBlock(in_ch, out_ch, c_blocks))

        # Adaptive Average Pooling
        self.aap = nn.AdaptiveAvgPool2d((split_size, split_size))

        # Last feature channel size will be the input to the prediction head
        final_feature_size = feature_channels[-1]

        # Prediction head with customizable number of FC layers
        self.head = BboxPredictionHead(
            in_features=final_feature_size,
            split_size=split_size,
            num_boxes=num_boxes,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_fc_layers=num_fc_layers
        )

    def forward(self, x):
        """
        Forward pass of BibNet.
        """
        # Initial convolution
        x = self.initial_block(x)

        # Process through P blocks
        for p_block in self.p_blocks:
            x = p_block(x)

        # Adaptive Average Pooling
        x = self.aap(x)

        # Prediction head
        bbox_preds = self.head(torch.flatten(x, start_dim=1))

        return bbox_preds


def build_bibnet(cfg):
    """
    Build BibNet model from configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Initialized BibNet model
    """
    model_cfg = cfg.get("model", {})
    in_channels = model_cfg.get("input_channels", 3)
    p_blocks = model_cfg.get("p_blocks", 3)
    c_blocks = model_cfg.get("c_blocks", 2)
    feature_channels = model_cfg.get("feature_channels", [64, 128, 256])
    split_size = model_cfg.get("split_size", 7)
    num_boxes = model_cfg.get("num_boxes", 2)
    num_classes = model_cfg.get("num_classes", 1)
    hidden_size = model_cfg.get("hidden_size", 512)
    num_fc_layers = model_cfg.get("num_fc_layers", 1)

    model = BibNet(
        input_channels=in_channels,
        p_blocks=p_blocks,
        c_blocks=c_blocks,
        feature_channels=feature_channels,
        split_size=split_size,
        num_boxes=num_boxes,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_fc_layers=num_fc_layers
    )
    return model
