from utils.loss_utils import intersection_over_union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_utils import match_predictions_to_targets, box_iou


# CIoU Loss for bounding box regression
# Reference: https://arxiv.org/abs/2005.03572
class CIoULoss(nn.Module):
    """Complete IoU Loss for bounding box regression"""

    def __init__(self, eps=1e-6):
        super(CIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes, reduction='mean'):
        """
        Calculate CIoU loss between prediction and target boxes

        Args:
            pred_boxes: [N, 4] - predicted boxes (x, y, w, h)
            target_boxes: [N, 4] - target boxes (x, y, w, h)
            reduction: Reduction method ('none', 'mean', 'sum')

        Returns:
            CIoU loss (1 - CIoU), where
            CIoU = IoU - (d² / c² + α * v)
        """
        # Convert to xyxy format for calculations
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Calculate IoU
        intersect_x1 = torch.maximum(pred_x1, target_x1)
        intersect_y1 = torch.maximum(pred_y1, target_y1)
        intersect_x2 = torch.minimum(pred_x2, target_x2)
        intersect_y2 = torch.minimum(pred_y2, target_y2)

        intersect_w = (intersect_x2 - intersect_x1).clamp(min=0)
        intersect_h = (intersect_y2 - intersect_y1).clamp(min=0)

        intersect_area = intersect_w * intersect_h
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - intersect_area

        iou = intersect_area / (union_area + self.eps)

        # Calculate distance component
        # Find the smallest enclosing box
        enclosing_x1 = torch.minimum(pred_x1, target_x1)
        enclosing_y1 = torch.minimum(pred_y1, target_y1)
        enclosing_x2 = torch.maximum(pred_x2, target_x2)
        enclosing_y2 = torch.maximum(pred_y2, target_y2)

        # Calculate diagonal distance of enclosing box
        c_square = ((enclosing_x2 - enclosing_x1)**2 +
                    (enclosing_y2 - enclosing_y1)**2)

        # Calculate center distance
        pred_cx = pred_boxes[:, 0]
        pred_cy = pred_boxes[:, 1]
        target_cx = target_boxes[:, 0]
        target_cy = target_boxes[:, 1]

        center_distance_square = (
            pred_cx - target_cx)**2 + (pred_cy - target_cy)**2

        # Calculate aspect ratio consistency term
        # v = (4 / (math.pi**2)) * (arctan(w_gt / h_gt) - arctan(w_pred / h_pred))^2
        # where w and h are the width and height of the boxes
        v = (4 / (math.pi**2)) * torch.pow(
            torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + self.eps)) -
            torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + self.eps)), 2
        )

        # Trade-off parameter, as in the paper
        alpha = v / (1 - iou + v + self.eps)

        # Final CIoU
        ciou = iou - (center_distance_square /
                      (c_square + self.eps) + alpha * v)

        # Return loss (1 - CIoU)
        loss = 1 - ciou

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ConfidenceLoss(nn.Module):
    """Base class for confidence loss functions"""

    def __init__(self, reduction='mean'):
        super(ConfidenceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        raise NotImplementedError("Subclasses must implement forward method")


# Standard Binary Cross-Entropy (BCE) loss for confidence scores
class BCEConfidenceLoss(ConfidenceLoss):
    """Standard BCE confidence loss"""

    def __init__(self, reduction='mean'):
        super(BCEConfidenceLoss, self).__init__(reduction)
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, inputs, targets):
        return self.bce(inputs, targets)


# Focal Loss for confidence scores to address class imbalance
# Reference: https://arxiv.org/abs/1708.02002
class FocalConfidenceLoss(ConfidenceLoss):
    """Focal Loss for confidence scores"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
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
    def __init__(self, lambda_bbox=1.0, lambda_conf=1.0, iou_threshold=0.5,
                 conf_loss_type='focal', focal_alpha=0.25, focal_gamma=2.0,
                 adaptive_threshold=True, adaptive_threshold_min=0.01, adaptive_threshold_epochs=0.2):
        """
        Improved loss function for bounding box prediction using CIoU loss.

        Args:
            lambda_bbox: Weight for bounding box regression loss
            lambda_conf: Weight for confidence loss
            iou_threshold: Minimum IoU to consider a match between prediction and ground truth
            conf_loss_type: Type of confidence loss ('bce' or 'focal')
            focal_alpha: Alpha parameter for focal loss (only used if conf_loss_type='focal')
            focal_gamma: Gamma parameter for focal loss (only used if conf_loss_type='focal')
            adaptive_threshold: Whether to use adaptive threshold for early training
            adaptive_threshold_min: Minimum threshold value to use at the start of training
            adaptive_threshold_epochs: Fraction of training used for threshold adaptation
        """
        super(BboxLoss, self).__init__()
        self.lambda_bbox = lambda_bbox
        self.lambda_conf = lambda_conf
        self.iou_threshold = iou_threshold
        self.current_epoch = 0
        self.max_epochs = 100
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_threshold_min = adaptive_threshold_min
        self.adaptive_threshold_epochs = adaptive_threshold_epochs

        # CIoU loss for bounding box regression
        self.ciou_loss = CIoULoss()

        # Initialize confidence loss based on type
        if conf_loss_type.lower() == 'bce':
            self.conf_loss = BCEConfidenceLoss(reduction='mean')
        elif conf_loss_type.lower() == 'focal':
            self.conf_loss = FocalConfidenceLoss(
                alpha=focal_alpha, gamma=focal_gamma, reduction='mean'
            )
        else:
            raise ValueError(
                f"Unsupported confidence loss type: {conf_loss_type}")

    def set_epoch_info(self, current_epoch, max_epochs):
        """Set current epoch and max epochs for adaptive threshold"""
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs

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
        match_rates = []

        # Process each sample in batch separately
        for i in range(batch_size):
            # Extract predictions and ground truth for this sample
            pred_boxes = preds[i, :, :4]  # [N, 4] - (x, y, w, h)
            pred_conf = preds[i, :, 4]    # [N] - confidence scores

            height, width = batch['images'][i].shape[-2:]
            gt_boxes = batch['bboxes'][i]  # [M_i, 4] - (x, y, w, h)

            # Create a normalized copy of the ground truth boxes
            gt_boxes_normed = gt_boxes.clone()
            gt_boxes_normed[:, 0] /= width   # x
            gt_boxes_normed[:, 1] /= height  # y
            gt_boxes_normed[:, 2] /= width   # width
            gt_boxes_normed[:, 3] /= height  # height

            # Initialize confidence targets with zeros (no match)
            conf_targets = torch.zeros_like(pred_conf)

            # Match predictions to ground truth
            matches, unmatched_preds = match_predictions_to_targets(
                pred_boxes, gt_boxes_normed, self.iou_threshold,
                epoch=self.current_epoch, max_epochs=self.max_epochs,
                use_adaptive_threshold=self.adaptive_threshold,
                min_threshold=self.adaptive_threshold_min,
                adaptive_epochs_fraction=self.adaptive_threshold_epochs
            )

            sample_bbox_loss = torch.tensor(0.0, device=preds.device)

            # Compute bbox regression loss for matched predictions
            if len(matches) > 0:
                matched_pred_indices = [m[0] for m in matches]
                matched_target_indices = [m[1] for m in matches]

                # Extract matched predictions and targets
                matched_preds = pred_boxes[matched_pred_indices]
                matched_targets = gt_boxes_normed[matched_target_indices]

                # Compute box regression loss using CIoU
                sample_bbox_loss = self.ciou_loss(
                    matched_preds, matched_targets)

                # Set confidence targets for matches to 1.0
                conf_targets[matched_pred_indices] = 1.0

            # Calculate match rate for this sample
            match_rate = len(matches) / len(gt_boxes) if len(gt_boxes) > 0 else 1.0

            match_rates.append(match_rate)

            # Use the selected confidence loss
            sample_conf_loss = self.conf_loss(pred_conf, conf_targets)

            # Add to batch totals
            total_bbox_loss += sample_bbox_loss
            total_conf_loss += sample_conf_loss

        # Calculate average match rate across batch
        avg_match_rate = sum(match_rates) / batch_size

        # Apply normalization by batch size
        total_bbox_loss /= batch_size
        total_conf_loss /= batch_size

        total_loss = (self.lambda_bbox * total_bbox_loss +
                      self.lambda_conf * total_conf_loss)

        # Create loss dictionary for monitoring
        loss_dict = {
            'loss': total_loss,
            'bbox_loss': total_bbox_loss,
            'conf_loss': total_conf_loss,
            'match_rate': torch.tensor(avg_match_rate, device=preds.device)
        }

        # Debug
        # print(f'total loss: {total_loss.item():.4f}, bbox loss: {total_bbox_loss.item():.4f}, '
        #       f'conf loss: {total_conf_loss.item():.4f} ({avg_match_rate:.4f})')
        return total_loss, loss_dict


# ==================================================
# YOLOv1 Loss
# ==================================================


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1,
                                          self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(
            predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(
            predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] +
            (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        # no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) *
                          predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) *
                          predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
