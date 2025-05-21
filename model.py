"""
Model architecture for single-class object detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and leaky ReLU activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block for the object detection model.

    Args:
        channels (int): Number of channels
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class DownsampleBlock(nn.Module):
    """
    Downsample block for the object detection model.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = ConvBlock(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class BibNet(nn.Module):
    """
    Object detection model for bib number detection (single-class).

    Args:
        img_size (int): Input image size
        backbone_features (list): Number of features for each layer of the backbone
    """
    def __init__(self, img_size=416, backbone_features=[64, 128, 256, 512, 1024]):
        super(BibNet, self).__init__()
        self.img_size = img_size
        # Will be updated dynamically in forward pass
        self.grid_sizes = [img_size // 32, img_size // 16, img_size // 8]

        # Initial conv layer
        self.initial_layers = nn.Sequential(
            ConvBlock(3, backbone_features[0], kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Backbone
        self.backbone = nn.ModuleList()

        # Downsample blocks
        for i in range(len(backbone_features) - 1):
            self.backbone.append(
                nn.Sequential(
                    DownsampleBlock(backbone_features[i], backbone_features[i + 1]),
                    ResidualBlock(backbone_features[i + 1]),
                    ResidualBlock(backbone_features[i + 1])
                )
            )

        # Feature Pyramid Network (FPN)
        self.fpn_top = nn.Sequential(
            ConvBlock(backbone_features[-1], 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1)
        )

        # Upsampling path
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fpn_middle = nn.Sequential(
            ConvBlock(512 + backbone_features[-2], 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1)
        )

        # Second upsampling path
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fpn_bottom = nn.Sequential(
            ConvBlock(512 + backbone_features[-3], 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1)
        )

        # Prediction heads for three scales
        # Each head outputs [batch_size, 5, grid_size, grid_size]
        # where 5 represents (objectness, x, y, w, h)
        self.pred_head1 = nn.Conv2d(512, 5, kernel_size=1)
        self.pred_head2 = nn.Conv2d(512, 5, kernel_size=1)
        self.pred_head3 = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x):
        """Forward pass of the model."""
        batch_size = x.shape[0]
        feature_maps = []

        # Initial layers
        x = self.initial_layers(x)

        # Backbone
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i >= len(self.backbone) - 3:  # Save last 3 feature maps for FPN
                feature_maps.append(x)

        # FPN
        x = self.fpn_top(feature_maps[-1])

        # First prediction head (for smallest objects)
        p1 = self.pred_head1(x)

        # Upsample and concatenate with feature map from backbone
        x_upsampled = self.upsample1(x)

        # Ensure feature map sizes match before concatenation
        if x_upsampled.size()[2:] != feature_maps[-2].size()[2:]:
            # Adjust feature map size if needed
            x_upsampled = F.interpolate(
                x_upsampled,
                size=feature_maps[-2].size()[2:],
                mode='nearest'
            )

        x = torch.cat([x_upsampled, feature_maps[-2]], dim=1)
        x = self.fpn_middle(x)

        # Second prediction head (for medium objects)
        p2 = self.pred_head2(x)

        # Upsample and concatenate with feature map from backbone
        x_upsampled = self.upsample2(x)

        # Ensure feature map sizes match before concatenation
        if x_upsampled.size()[2:] != feature_maps[-3].size()[2:]:
            # Adjust feature map size if needed
            x_upsampled = F.interpolate(
                x_upsampled,
                size=feature_maps[-3].size()[2:],
                mode='nearest'
            )

        x = torch.cat([x_upsampled, feature_maps[-3]], dim=1)
        x = self.fpn_bottom(x)

        # Third prediction head (for largest objects)
        p3 = self.pred_head3(x)

        # Update grid sizes based on the actual output dimensions
        self.grid_sizes = [
            p1.shape[2],  # Small objects (highest res)
            p2.shape[2],  # Medium objects
            p3.shape[2],  # Large objects (lowest res)
        ]

        # Return predictions at three scales
        return [p1, p2, p3]

    def _transform_predictions(self, predictions, grid_size, stride):
        """
        Transform raw predictions to bounding boxes.
        Bounding boxes are returned in YOLO format,
        i.e., [x_center, y_center, width, height] and normalized.
        The objectness score is also returned.

        Args:
            predictions (Tensor): Raw predictions from one head [batch_size, 5, grid_size, grid_size]
            grid_size (int): Grid size for this prediction
            stride (int): Stride for this prediction level

        Returns:
            tuple: (boxes, objectness)
                - objectness is the confidence score for our single-class model
        """
        batch_size = predictions.shape[0]

        # Reshape to [batch_size, grid_size, grid_size, 5]
        predictions = predictions.permute(0, 2, 3, 1).contiguous()

        # Split predictions
        objectness = torch.sigmoid(predictions[..., 0])  # [batch_size, grid_size, grid_size]

        # Box coordinates
        x = torch.sigmoid(predictions[..., 1])  # Center x relative to cell [0, 1]
        y = torch.sigmoid(predictions[..., 2])  # Center y relative to cell [0, 1]
        w = torch.exp(predictions[..., 3])  # Width is log-scaled so take exp
        h = torch.exp(predictions[..., 4])  # Height is log-scaled so take exp

        # Create grid offsets
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, grid_size, grid_size]).to(predictions.device)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, grid_size, grid_size]).to(predictions.device)

        # Add offsets to predictions to get absolute coordinates
        x = (x + grid_x) * stride
        y = (y + grid_y) * stride
        w *= stride
        h *= stride

        # Normalize to [0, 1] based on image size
        x /= self.img_size
        y /= self.img_size
        w /= self.img_size
        h /= self.img_size

        # Return in [x_center, y_center, width, height] and normalized format
        boxes = torch.zeros_like(predictions[..., :4])
        boxes[..., 0] = x  # x_center
        boxes[..., 1] = y  # y_center
        boxes[..., 2] = w  # width
        boxes[..., 3] = h  # height

        return boxes, objectness

    def predict(self, x, conf_threshold=0.5, nms_threshold=0.5):
        """
        Make predictions with the model and apply NMS.

        Args:
            x (Tensor): Input image tensor [batch_size, C, H, W]
            conf_threshold (float): Confidence threshold
            nms_threshold (float): NMS threshold

        Returns:
            list: List of detection results per image, where each result is a dict
                  containing "boxes" and "scores".
                  - "boxes" in yolo format [x_center, y_center, width, height] and normalized.
                  - "scores" are the associated confidence score for each box.
        """
        # Forward pass
        predictions = self.forward(x)
        batch_size = x.shape[0]
        img_size = x.shape[2]

        # Get strides for each detection scale
        strides = [img_size // g for g in self.grid_sizes]

        all_results = []

        # Process each img in the batch
        for batch_idx in range(batch_size):
            boxes, scores = [], []

            # Process each prediction scale
            for pred_idx, pred in enumerate(predictions):
                grid_size = self.grid_sizes[pred_idx]
                stride = strides[pred_idx]

                # Transform predictions to boxes
                # boxes is in [x_center, y_center, width, height] format (normalized)
                boxes_pred, obj_pred = self._transform_predictions(
                    pred[batch_idx:batch_idx+1], grid_size, stride
                )

                # Flatten predictions
                boxes_pred = boxes_pred.view(-1, 4)
                obj_pred = obj_pred.view(-1)

                # For single-class detection, confidence is just the objectness score
                conf_pred = obj_pred

                # Filter by confidence threshold
                mask = conf_pred > conf_threshold
                boxes_filtered = boxes_pred[mask]
                scores_filtered = conf_pred[mask]

                # Add to overall predictions for this image
                boxes.append(boxes_filtered)
                scores.append(scores_filtered)

            # Combine predictions from all scales
            if boxes and len(boxes[0]) > 0:
                boxes = torch.cat(boxes, dim=0)
                scores = torch.cat(scores, dim=0)

                # Apply NMS
                keep = self.nms(boxes, scores, nms_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
            else:
                boxes = torch.zeros((0, 4), device=x.device)
                scores = torch.zeros(0, device=x.device)

            # Add results for this image
            all_results.append({
                "boxes": boxes,
                "scores": scores,
            })

        # Sort boxes by score for each image
        all_results = [
            {k: v[torch.argsort(r["scores"], descending=True)] if k in ["boxes", "scores"] else v
             for k, v in r.items()} for r in all_results
        ]

        return all_results

    def nms(self, boxes, scores, threshold):
        """
        Simple implementation of NMS.

        Args:
            boxes (Tensor): Boxes in YOLO format [x_center, y_center, width, height] [N, 4]
            scores (Tensor): Scores [N]
            threshold (float): IoU threshold

        Returns:
            Tensor: Indices of boxes to keep
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Convert YOLO format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        areas = width * height

        _, order = scores.sort(0, descending=True)
        keep = []

        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            # Calculate IoU of the current box with the rest
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than threshold
            inds = torch.where(iou < threshold)[0]
            order = order[inds + 1]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
