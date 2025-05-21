"""
Loss functions for object detection model.
"""
import torch
import torch.nn as nn
import numpy as np


class BibNetLoss(nn.Module):
    """
    Loss function for the bib number detection model.

    Args:
        img_size (int): Input image size
        grid_sizes (list): Grid sizes for each prediction scale
        lambda_obj (float): Weight for objectness loss
        lambda_noobj (float): Weight for no-objectness loss
        lambda_coord (float): Weight for coordinate loss
    """

    def __init__(
        self,
        img_size=416,
        grid_sizes=[13, 26, 52],
        lambda_obj=2.0,
        lambda_noobj=1.0,
        lambda_coord=5.0,
    ):
        super(BibNetLoss, self).__init__()
        self.img_size = img_size
        self.grid_sizes = grid_sizes
        self.strides = [img_size // g for g in grid_sizes]

        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord

        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, targets):
        """
        Compute the loss.

        Args:
            predictions (list): List of predictions from the model at different scales
            targets (list): List of target dictionaries with 'boxes', 'labels', etc.

        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        batch_size = len(targets)
        total_loss = 0
        loss_components = {
            "obj_loss": 0,
            "noobj_loss": 0,
            "bbox_loss": 0,
        }

        # Process each scale
        for scale_idx, (pred, grid_size, stride) in enumerate(zip(predictions, self.grid_sizes, self.strides)):
            # Reshape predictions: [batch_size, 5, grid_size, grid_size] -> [batch_size, grid_size, grid_size, 5]
            pred = pred.permute(0, 2, 3, 1).contiguous()

            # Extract predictions
            pred_obj = pred[..., 0]  # Objectness
            pred_x = pred[..., 1]    # Center x relative to cell
            pred_y = pred[..., 2]    # Center y relative to cell
            pred_w = pred[..., 3]    # Width
            pred_h = pred[..., 4]    # Height

            # Initialize target tensors
            target_obj = torch.zeros_like(pred_obj)
            target_x = torch.zeros_like(pred_x)
            target_y = torch.zeros_like(pred_y)
            target_w = torch.zeros_like(pred_w)
            target_h = torch.zeros_like(pred_h)

            for batch_idx in range(batch_size):
                # Get target boxes and labels
                # [num_boxes, 4] in YOLO format [x_center, y_center, width, height] (normalized)
                target_boxes = targets[batch_idx]["boxes"]

                if len(target_boxes) == 0:  # rare case where no bbox
                    continue

                # Extract coordinates directly
                target_cx = target_boxes[:, 0]  # x_center (normalized)
                target_cy = target_boxes[:, 1]  # y_center (normalized)
                target_w_norm = target_boxes[:, 2]  # width (normalized)
                target_h_norm = target_boxes[:, 3]  # height (normalized)

                # Scale UP the normalized coordinates to absolute values
                # i.e., convert to absolute pixel values (image is square)
                target_cx_abs = target_cx * self.img_size
                target_cy_abs = target_cy * self.img_size
                target_w_abs = target_w_norm * self.img_size
                target_h_abs = target_h_norm * self.img_size

                # Scale DOWN to the current grid cell
                # i.e., the result is a float like 5.3,
                # where 5 = grid cell idx
                # and 0.3 = relative coord within cell
                # i.e., stride is img_size / feature map size at this scale
                target_cx_grid = target_cx_abs / stride
                target_cy_grid = target_cy_abs / stride

                # Get grid cell indices
                # .long() to convert floating-point coord to int
                grid_i = target_cx_grid.long()
                grid_j = target_cy_grid.long()

                # Keep only valid grid indices
                valid_mask = (
                    (grid_i >= 0) & (grid_i < grid_size) &
                    (grid_j >= 0) & (grid_j < grid_size)
                )

                if not valid_mask.any():
                    continue

                # Filter targets
                grid_i = grid_i[valid_mask]
                grid_j = grid_j[valid_mask]

                target_cx_grid = target_cx_grid[valid_mask]
                target_cy_grid = target_cy_grid[valid_mask]
                target_w_abs = target_w_abs[valid_mask]
                target_h_abs = target_h_abs[valid_mask]

                # Compute relative coord _within_ cell
                # cfr. comment above
                target_cx_rel = target_cx_grid - grid_i.float()
                target_cy_rel = target_cy_grid - grid_j.float()

                # Compute log scale for width and height
                target_w_rel = torch.log(target_w_abs / stride + 1e-16)
                target_h_rel = torch.log(target_h_abs / stride + 1e-16)

                # Create target tensors in the correct shape to match predictions
                for i in range(len(grid_i)):
                    gi, gj = grid_i[i], grid_j[i]
                    # Ensure indices are within bounds
                    if gi < target_obj.shape[2] and gj < target_obj.shape[1]:
                        target_obj[batch_idx, gj, gi] = 1.0
                        target_x[batch_idx, gj, gi] = target_cx_rel[i]
                        target_y[batch_idx, gj, gi] = target_cy_rel[i]
                        target_w[batch_idx, gj, gi] = target_w_rel[i]
                        target_h[batch_idx, gj, gi] = target_h_rel[i]

            # ================
            # Compute losses
            # ================

            # Get objectness mask (i.e., where there is an object)
            obj_mask = target_obj > 0.5
            noobj_mask = ~obj_mask

            # 1. Objectness loss
            obj_loss = self.mse_loss(
                pred_obj[obj_mask], target_obj[obj_mask]
            ).sum()
            # obj_loss = self.bce_loss(
            #     pred_obj[obj_mask], target_obj[obj_mask]
            # ).sum()

            # 2. No-objectness loss
            noobj_loss = self.mse_loss(
                pred_obj[noobj_mask], target_obj[noobj_mask]
            ).sum()
            # noobj_loss = self.bce_loss(
            #     pred_obj[noobj_mask], target_obj[noobj_mask]
            # ).sum()

            # 3. Coordinate losses
            if obj_mask.sum() > 0:
                # Convert predictions to proper format
                # i.e., sigmoid for x and y to get coordinate in cell; [0, 1]
                # w and h are already in log scale => no sigmoid
                pred_x_sigmoid = torch.sigmoid(pred_x[obj_mask])
                pred_y_sigmoid = torch.sigmoid(pred_y[obj_mask])

                x_loss = self.mse_loss(
                    pred_x_sigmoid, target_x[obj_mask]
                ).sum()
                y_loss = self.mse_loss(
                    pred_y_sigmoid, target_y[obj_mask]
                ).sum()
                w_loss = self.mse_loss(
                    pred_w[obj_mask], target_w[obj_mask]
                ).sum()
                h_loss = self.mse_loss(
                    pred_h[obj_mask], target_h[obj_mask]
                ).sum()

                bbox_loss = x_loss + y_loss + w_loss + h_loss
            else:
                bbox_loss = torch.tensor(0.0, device=pred.device)

            # Apply weights
            obj_loss = self.lambda_obj * obj_loss
            noobj_loss = self.lambda_noobj * noobj_loss
            bbox_loss = self.lambda_coord * bbox_loss

            # Accumulate losses
            loss_components["obj_loss"] += obj_loss
            loss_components["noobj_loss"] += noobj_loss
            loss_components["bbox_loss"] += bbox_loss

            # Sum losses for this scale
            scale_loss = obj_loss + noobj_loss + bbox_loss
            total_loss += scale_loss

        # Normalize losses by batch size
        total_loss /= batch_size
        for k in loss_components:
            loss_components[k] = loss_components[k] / batch_size

        return total_loss, loss_components


class Evaluator:
    """
    Evaluation class for object detection.
    This class implements the COCO evaluation metric (AP at [.5:.05:.95]) using global AP calculation.
    Global AP means calculating a single precision-recall curve across all images in the dataset,
    rather than calculating AP per image and then averaging.

    It provides a metric of mAP (mean Average Precision) defined as the mean of APs across IoU thresholds.

    Args:
        iou_thresholds (list, optional): IoU thresholds for AP calculation
    """

    def __init__(self, iou_thresholds=None):
        if iou_thresholds is None:
            self.iou_thresholds = torch.tensor([0.5, 0.75, 0.95])
        else:
            self.iou_thresholds = torch.tensor(iou_thresholds)

    def calculate_global_ap(self, all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold):
        """
        Calculate global Average Precision (AP) at a specific IoU threshold.
        AP is calculated as the AUC of the precision-recall curve across all images.

        Args:
            all_pred_boxes (list): List of prediction boxes from all images, each with shape [N, 4]
            all_pred_scores (list): List of prediction scores from all images, each with shape [N]
            all_gt_boxes (list): List of ground truth boxes from all images, each with shape [M, 4]
            iou_threshold (float): IoU threshold for matching

        Returns:
            float: AP value
        """
        # Count total number of ground truth boxes across all images
        total_gt_boxes = sum(len(gt_boxes) for gt_boxes in all_gt_boxes)

        if total_gt_boxes == 0:
            if sum(len(pred) for pred in all_pred_boxes) == 0:
                # No predictions and no ground truth is perfect
                return 1.0
            else:
                # Has predictions but no ground truth
                return 0.0

        if sum(len(pred) for pred in all_pred_boxes) == 0:
            # No predictions but has ground truth
            return 0.0

        # Combine and prepare predictions
        combined_data = []
        for img_idx, (pred_boxes, pred_scores) in enumerate(zip(all_pred_boxes, all_pred_scores)):
            if len(pred_boxes) > 0:
                # Store (box, score, image_idx) for each prediction
                for box, score in zip(pred_boxes, pred_scores):
                    combined_data.append((box, score, img_idx))

        # Sort by confidence score in descending order
        combined_data.sort(key=lambda x: x[1], reverse=True)

        # Extract sorted components
        device = all_pred_boxes[0].device if all_pred_boxes and len(all_pred_boxes[0]) > 0 else torch.device('cpu')
        tp = torch.zeros(len(combined_data), dtype=torch.float32, device=device)
        fp = torch.zeros(len(combined_data), dtype=torch.float32, device=device)

        # Keep track of matched GT boxes per image
        gt_matched = [torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
                     for gt_boxes in all_gt_boxes]

        # Process predictions by image to avoid redundant IoU calculations
        img_to_preds = {}
        for i, (box, _, img_idx) in enumerate(combined_data):
            if img_idx not in img_to_preds:
                img_to_preds[img_idx] = []
            img_to_preds[img_idx].append((i, box))

        # Process each image separately
        for img_idx, gt_boxes in enumerate(all_gt_boxes):
            if img_idx not in img_to_preds or len(gt_boxes) == 0:
                # Mark all predictions for this image as false positives
                if img_idx in img_to_preds:
                    for pred_idx, _ in img_to_preds[img_idx]:
                        fp[pred_idx] = 1
                continue

            # Get all predictions for this image
            img_preds = img_to_preds[img_idx]
            pred_indices = [p[0] for p in img_preds]
            pred_boxes = torch.stack([p[1] for p in img_preds])

            # Calculate IoU matrix for all predictions in this image at once
            ious = calculate_iou_matrix(pred_boxes, gt_boxes, box_format='yolo')

            # For each prediction (in original sorted order)
            for i, pred_idx in enumerate(pred_indices):
                # Find best matching GT box
                pred_ious = ious[i]
                max_iou, max_gt_idx = torch.max(pred_ious, dim=0)

                # If IoU exceeds threshold and GT box not already matched
                if max_iou >= iou_threshold and not gt_matched[img_idx][max_gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[img_idx][max_gt_idx] = True
                else:
                    fp[pred_idx] = 1

        # Compute cumulative precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recalls = tp_cumsum / total_gt_boxes
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)

        # Add sentinel values to precision and recall
        precisions = torch.cat([torch.tensor([1.0], device=device), precisions])
        recalls = torch.cat([torch.tensor([0.0], device=device), recalls])

        # Make precision monotonically decreasing
        # i.e., for each recall level, take the maximum precision
        # this is done by looping backwards on the precision array
        # and the max precision is thus propagated backwards
        # motivation: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/htmldoc/index.html#sec:ap
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i-1] = torch.max(precisions[i-1], precisions[i])

        # Find indices where recall changes
        idx = torch.where(recalls[1:] != recalls[:-1])[0]

        # Calculate the AP as the sum of rectangular areas under the curve
        ap = torch.sum((recalls[idx+1] - recalls[idx]) * precisions[idx+1])

        return ap.item()

    def evaluate(self, predictions, targets):
        """
        Evaluate predictions against targets.

        Args:
            predictions (list): List of prediction dictionaries with 'boxes', 'scores'.
                Each dictionary has format:
                {
                    "boxes": tensor([[x_center, y_center, width, height], ...]),  # YOLO format (normalized)
                    "scores": tensor([0.5, ...])                                  # confidence scores
                }

            targets (list): List of target dictionaries with 'boxes', 'labels'.
                Each dictionary has format:
                {
                    "boxes": tensor([[x_center, y_center, width, height], ...]),  # YOLO format (normalized)
                    "labels": tensor([0, ...])                                    # class labels
                }

        Returns:
            dict: Evaluation results with global AP metrics and PR curves
        """
        results = {}
        ap_sum = 0.0
        pr_curves = {}

        # Extract all prediction boxes and ground truth boxes
        all_pred_boxes = [pred["boxes"] for pred in predictions]
        all_pred_scores = [pred["scores"] for pred in predictions]
        all_gt_boxes = [target["boxes"] for target in targets]

        # Process each IoU threshold
        for iou_threshold in self.iou_thresholds:
            # Calculate global AP for this IoU threshold
            ap = self.calculate_global_ap(
                all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold.item()
            )

            ap_sum += ap
            results[f"AP_{iou_threshold:.2f}"] = ap

        # Calculate mAP (mean over IoU thresholds)
        results["mAP"] = ap_sum / len(self.iou_thresholds)

        return results


def calculate_iou_matrix(boxes1, boxes2, box_format='yolo'):
    """
    Calculate IoU matrix between two sets of boxes.

    Args:
        boxes1 (Tensor): First set of boxes [N, 4]
        boxes2 (Tensor): Second set of boxes [M, 4]
        box_format (str): Format of input boxes, either 'yolo' [x_center, y_center, width, height] or 'corner' [x1, y1, x2, y2]

    Returns:
        Tensor: IoU matrix of shape [N, M]
    """
    # Convert from YOLO format to corner format if needed
    if box_format == 'yolo':
        # Convert boxes1 from YOLO to corner format
        x1_1 = boxes1[:, 0] - boxes1[:, 2] / 2  # x_center - width/2
        y1_1 = boxes1[:, 1] - boxes1[:, 3] / 2  # y_center - height/2
        x2_1 = boxes1[:, 0] + boxes1[:, 2] / 2  # x_center + width/2
        y2_1 = boxes1[:, 1] + boxes1[:, 3] / 2  # y_center + height/2

        # Convert boxes2 from YOLO to corner format
        x1_2 = boxes2[:, 0] - boxes2[:, 2] / 2  # x_center - width/2
        y1_2 = boxes2[:, 1] - boxes2[:, 3] / 2  # y_center - height/2
        x2_2 = boxes2[:, 0] + boxes2[:, 2] / 2  # x_center + width/2
        y2_2 = boxes2[:, 1] + boxes2[:, 3] / 2  # y_center + height/2
    else:  # corner format
        # Extract coordinates for the first set of boxes
        x1_1 = boxes1[:, 0]
        y1_1 = boxes1[:, 1]
        x2_1 = boxes1[:, 2]
        y2_1 = boxes1[:, 3]

        # Extract coordinates for the second set of boxes
        x1_2 = boxes2[:, 0]
        y1_2 = boxes2[:, 1]
        x2_2 = boxes2[:, 2]
        y2_2 = boxes2[:, 3]

    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate intersection
    inter_x1 = torch.max(x1_1[:, None], x1_2[None, :])
    inter_y1 = torch.max(y1_1[:, None], y1_2[None, :])
    inter_x2 = torch.min(x2_1[:, None], x2_2[None, :])
    inter_y2 = torch.min(y2_1[:, None], y2_2[None, :])

    width = torch.clamp(inter_x2 - inter_x1, min=0)
    height = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = width * height

    # Calculate union
    union = area1[:, None] + area2[None, :] - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-16)

    return iou
