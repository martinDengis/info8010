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
    def __init__(self, lambda_bbox=5.0, lambda_conf=1.0, iou_threshold=0.1,
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
        nboxes = 0

        # Process each sample in batch separately
        for i in range(batch_size):
            # Extract predictions and ground truth for this sample
            pred_boxes = preds[i, :, :4]  # [N, 4] - (x, y, w, h)
            pred_conf = preds[i, :, 4]    # [N] - confidence scores

            height, width = batch['images'][i].shape[-2:]
            gt_boxes = batch['bboxes'][i]  # [M_i, 4] - (x, y, w, h)
            nboxes += len(gt_boxes)

            # Create a normalized copy of the ground truth boxes
            # - for direct compare with preds that are also normalized
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
        total_conf_loss /= (batch_size * num_preds)

        # Calculate match gap penalty
        match_gap_penalty = 0.0
        if nboxes > 0:
            # Calculate percentage of missed matches
            match_rate = total_matches / nboxes
            missed_rate = 1.0 - match_rate
            # Create a penalty that increases as missed_rate grows
            match_gap_penalty = missed_rate * 2.0  # Adjust multiplier as needed

            match_gap_tensor = torch.tensor(match_gap_penalty, device=preds.device)
        else:
            match_gap_tensor = torch.tensor(0.0, device=preds.device)

        # Handle bbox loss normalization
        if total_matches > 0:
            total_bbox_loss /= total_matches
        else:
            # When no matches, zero out bbox_loss
            total_bbox_loss = torch.tensor(0.0, device=preds.device)
            match_gap_penalty = 3.0  # Higher penalty for zero matches

        # Combine all loss components
        total_loss = self.lambda_bbox * total_bbox_loss + self.lambda_conf * total_conf_loss + match_gap_penalty

        loss_dict = {
            'loss': total_loss,
            'bbox_loss': total_bbox_loss,
            'conf_loss': total_conf_loss,
            'match_gap_penalty': match_gap_tensor,
            'match_rate': torch.tensor(total_matches / max(nboxes, 1), device=preds.device)
        }

        print(f"total_loss: {total_loss.item() = }, {loss_dict = }")
        return total_loss, loss_dict

# ==================================================
# YOLOv1 Loss
# ==================================================
from utils.loss_utils import intersection_over_union


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
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
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
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
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