import torch
import torch.nn as nn
from utils.loss_utils import ciou_loss, distribution_focal_loss

class BboxLoss(nn.Module):
    def __init__(self, ciou_weight=1.0, l1_weight=1.0, num_classes=1, reduction='mean'):
        super().__init__()
        self.ciou_weight = ciou_weight
        self.l1_weight = l1_weight
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, batch, preds):
        """
        Compute combined loss with CIoU and L1, handling variable target counts per image

        Args:
            batch: Dictionary containing ground truth data
            preds: Model predictions for bounding boxes [batch_size, total_predictions, 4]
        """
        # Get device from input tensors
        device = preds.device
        batch_size = preds.shape[0]

        # Extract ground truth data (boxes are lists of tensors)
        gt_bboxes_list = [boxes.to(device) for boxes in batch['bboxes']]
        gt_labels_list = [labels.to(device) for labels in batch['labels']]

        # Initialize loss accumulators
        total_ciou_loss = 0.0
        total_l1_loss = 0.0
        total_boxes = 0

        # each img processed independently because of different number of gt boxes
        for i in range(batch_size):
            if gt_bboxes_list[i].shape[0] == 0:
                continue    # no targets for this image

            # TODO:
            # Get predictions for this image's ground truth boxes
            # This would typically involve finding the closest predicted boxes to the ground truth
            # For simplicity, we're just using the ground truth boxes directly
            # Ideally, we'd need to match predictions to ground truth!!

            num_gt = gt_bboxes_list[i].shape[0]
            img_preds = preds[i][:num_gt]

            # Calculate CIoU loss for this image
            img_ciou_loss = ciou_loss(
                img_preds,
                gt_bboxes_list[i],
                reduction='none'  # handle reduction later
            )

            # Since BibNet model outputs direct box coordinates (not distributions),
            # we use L1 loss as the second regression component instead of DFL (!= YOLOv5)
            img_l1_loss = torch.nn.functional.l1_loss(
                img_preds,
                gt_bboxes_list[i],
                reduction='none'  # handle reduction later
            ).sum(dim=-1)

            # Accumulate losses
            total_ciou_loss += img_ciou_loss.sum()
            total_l1_loss += img_l1_loss.sum()
            total_boxes += num_gt

        # reduction
        if self.reduction == 'mean' and total_boxes > 0:
            total_ciou_loss = total_ciou_loss / total_boxes
            total_l1_loss = total_l1_loss / total_boxes

        # Combine with weights
        total_loss = (self.ciou_weight * total_ciou_loss) + (self.l1_weight * total_l1_loss)

        return total_loss