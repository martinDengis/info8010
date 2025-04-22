import torch
import torch.nn as nn
from utils.loss_utils import ciou_loss, distribution_focal_loss

class BboxLoss(nn.Module):
    def __init__(self, ciou_weight=1.0, dfl_weight=1.0, num_classes=1, num_bins=16, reduction='mean'):
        super().__init__()
        self.ciou_weight = ciou_weight
        self.dfl_weight = dfl_weight
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.reduction = reduction # 'mean', 'sum', or 'none' -> to obtain a scalar

    def forward(self, batch, preds):
        """
        Compute combined loss with CIoU and DFL components

        Args:
            batch: Dictionary containing ground truth data
            preds: Model predictions for bounding boxes
        """
        # Get device from input tensors
        device = preds.device

        # Extract ground truth data
        gt_bboxes = batch['bboxes'].to(device)
        gt_labels = batch['labels'].to(device)

        # Compute CIoU + DFL losses with specified reduction method
        ciou_loss_value = ciou_loss(preds, gt_bboxes, reduction=self.reduction)
        dfl_loss_value = distribution_focal_loss(preds, gt_bboxes, num_bins=self.num_bins, reduction=self.reduction)

        # Combine them with weights
        total_loss = (self.ciou_weight * ciou_loss_value) + (self.dfl_weight * dfl_loss_value)

        return total_loss