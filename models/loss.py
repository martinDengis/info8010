import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_utils import match_predictions_to_targets

class ConfidenceLoss(nn.Module):
    """Base class for confidence loss functions"""
    def __init__(self, reduction='sum'):
        super(ConfidenceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        raise NotImplementedError("Subclasses must implement forward method")


# Standard Binary Cross-Entropy (BCE) loss for confidence scores
class BCEConfidenceLoss(ConfidenceLoss):
    """Standard BCE confidence loss"""
    def __init__(self, reduction='sum'):
        super(BCEConfidenceLoss, self).__init__(reduction)
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, inputs, targets):
        return self.bce(inputs, targets)


# Focal Loss for confidence scores to address class imbalance
# Reference: https://arxiv.org/abs/1708.02002
class FocalConfidenceLoss(ConfidenceLoss):
    """Focal Loss for confidence scores"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super(FocalConfidenceLoss, self).__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Focal loss computation
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        modulating_factor = (1.0 - pt) ** self.gamma

        # Apply modulating and alpha factors
        loss = alpha_factor * modulating_factor * bce_loss

        # Apply reduction
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # 'none'
            return loss


class BboxLoss(nn.Module):
    def __init__(self, lambda_bbox=1.0, lambda_conf=1.0, iou_threshold=0.1,
                 conf_loss_type='bce', focal_alpha=0.25, focal_gamma=2.0):
        """
        Loss function for bounding box prediction.

        Args:
            lambda_bbox: Weight for bounding box regression loss
            lambda_conf: Weight for confidence loss
            iou_threshold: Minimum IoU to consider a match between prediction and ground truth
            conf_loss_type: Type of confidence loss ('bce' or 'focal')
            focal_alpha: Alpha parameter for focal loss (only used if conf_loss_type='focal')
            focal_gamma: Gamma parameter for focal loss (only used if conf_loss_type='focal')
        """
        super(BboxLoss, self).__init__()
        self.lambda_bbox = lambda_bbox
        self.lambda_conf = lambda_conf
        self.iou_threshold = iou_threshold
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

        # Initialize confidence loss based on type
        if conf_loss_type.lower() == 'bce':
            self.conf_loss = BCEConfidenceLoss(reduction='sum')
        elif conf_loss_type.lower() == 'focal':
            self.conf_loss = FocalConfidenceLoss(
                alpha=focal_alpha, gamma=focal_gamma, reduction='sum'
            )
        else:
            raise ValueError(f"Unsupported confidence loss type: {conf_loss_type}")

    def forward(self, batch, preds):
        """
        Compute loss between predictions and ground truth.

        Args:
            batch: Dictionary containing ground truth annotations:
                - 'images': Tensor of input images
                - 'bboxes': List of tensors, each containing ground truth boxes for an image
                - 'labels': List of tensors, each containing class labels for the boxes
            preds: Tensor of shape [batch_size, N, 5] where:
                - N is max_detections
                - 5 is for [x, y, w, h, confidence]

        Returns:
            Total loss, and a dictionary with individual loss components
        """
        batch_size = preds.size(0)
        num_preds = preds.size(1)  # N (max_detections)

        total_bbox_loss = 0.0
        total_conf_loss = 0.0
        total_matches = 0

        # Process each sample in batch separately
        for i in range(batch_size):
            # Extract predictions and ground truth for this sample
            pred_boxes = preds[i, :, :4]  # [N, 4] - (x, y, w, h)
            pred_conf = preds[i, :, 4]    # [N] - confidence scores

            height, width = batch['images'][i].shape[-2:]
            gt_boxes = batch['bboxes'][i]  # [M_i, 4] - (x, y, w, h)

            # Create a normalized copy of the ground truth boxes
            # - for direct comparison with predictions that are also normalized
            gt_boxes_normed = gt_boxes.clone()
            gt_boxes_normed[:, 0] /= width   # x
            gt_boxes_normed[:, 1] /= height  # y
            gt_boxes_normed[:, 2] /= width   # width
            gt_boxes_normed[:, 3] /= height  # height

            # Match predictions to ground truth
            matches, unmatched_preds = match_predictions_to_targets(
                pred_boxes, gt_boxes_normed, self.iou_threshold
            )

            # Initialize confidence targets with zeros (no match)
            conf_targets = torch.zeros_like(pred_conf)

            # Compute bbox regression loss for matched predictions
            if len(matches) > 0:
                matched_pred_indices = [m[0] for m in matches]
                matched_target_indices = [m[1] for m in matches]

                # Extract matched predictions and targets
                matched_preds = pred_boxes[matched_pred_indices]
                matched_targets = gt_boxes_normed[matched_target_indices]

                # Compute box regression loss (SmoothL1)
                bbox_loss = self.smooth_l1(matched_preds, matched_targets)

                # Set confidence targets for matches to 1.0
                conf_targets[matched_pred_indices] = 1.0

                # Add to total
                total_bbox_loss += bbox_loss
                total_matches += len(matches)

            # Use the selected confidence loss
            conf_loss = self.conf_loss(pred_conf, conf_targets)
            total_conf_loss += conf_loss

        # Normalize losses
        if total_matches > 0:
            total_bbox_loss /= total_matches
        else:
            total_bbox_loss = torch.tensor(0.0, device=preds.device)

        total_conf_loss /= (batch_size * num_preds)

        # Combine losses
        total_loss = self.lambda_bbox * total_bbox_loss + self.lambda_conf * total_conf_loss

        loss_dict = {
            'loss': total_loss,
            'bbox_loss': total_bbox_loss,
            'conf_loss': total_conf_loss
        }

        print(f"total_loss: {total_loss.item() = }, {loss_dict = }")
        return total_loss, loss_dict