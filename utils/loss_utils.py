import torch
import torch.nn.functional as F
import math

# Utils for loss functions

def box_iou(boxes1, boxes2):
    """
    Compute IoU between bounding boxes.

    Args:
        boxes1: Tensor of shape [N, 4] (x, y, w, h) where x, y are center coordinates
        boxes2: Tensor of shape [M, 4] (x, y, w, h) where x, y are center coordinates

    Returns:
        Tensor of shape [N, M] containing pairwise IoUs
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2) format
    boxes1_x1y1 = boxes1[:, :2] - boxes1[:, 2:] / 2
    boxes1_x2y2 = boxes1[:, :2] + boxes1[:, 2:] / 2
    boxes2_x1y1 = boxes2[:, :2] - boxes2[:, 2:] / 2
    boxes2_x2y2 = boxes2[:, :2] + boxes2[:, 2:] / 2

    # Calculate area of each box
    area1 = torch.prod(boxes1[:, 2:], dim=1)  # w * h
    area2 = torch.prod(boxes2[:, 2:], dim=1)  # w * h

    # Calculate intersection
    # Left-top and right-bottom corners of intersection
    lt = torch.max(boxes1_x1y1[:, None, :], boxes2_x1y1[None, :, :])
    rb = torch.min(boxes1_x2y2[:, None, :], boxes2_x2y2[None, :, :])

    # Check if there's an intersection
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    # Calculate union
    union = area1[:, None] + area2[None, :] - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero

    return iou

def match_predictions_to_targets(predictions, targets, iou_threshold=0.1):
    """
    Match predicted bounding boxes to ground truth using greedy matching.

    Args:
        predictions: Tensor of shape [N, 4] (x, y, w, h) - predicted boxes
        targets: Tensor of shape [M, 4] (x, y, w, h) - ground truth boxes
        iou_threshold: Minimum IoU to consider a match

    Returns:
        matches: List of (pred_idx, target_idx) pairs
        unmatched_preds: List of indices of unmatched predictions
    """
    if len(targets) == 0:
        # No targets, all predictions are unmatched
        return [], list(range(len(predictions)))

    if len(predictions) == 0:
        # No predictions, nothing to match
        return [], []

    # Calculate IoU matrix
    iou_matrix = box_iou(predictions, targets)

    matches = []
    matched_pred_indices = set()
    matched_target_indices = set()

    # Greedy matching - for each ground truth, find the best prediction
    while True:
        # Find highest remaining IoU
        if len(matched_pred_indices) == len(predictions) or len(matched_target_indices) == len(targets):
            break

        # Create a mask for already matched pairs
        pred_mask = torch.ones(len(predictions), dtype=torch.bool, device=predictions.device)
        pred_mask[list(matched_pred_indices)] = False

        target_mask = torch.ones(len(targets), dtype=torch.bool, device=targets.device)
        target_mask[list(matched_target_indices)] = False

        # Apply mask to IoU matrix
        masked_iou = iou_matrix.clone()
        masked_iou[~pred_mask, :] = -1
        masked_iou[:, ~target_mask] = -1

        # Find max IoU
        max_val, max_indices = masked_iou.view(-1).max(dim=0)

        # If best IoU is below threshold, we're done
        if max_val < iou_threshold:
            break

        # Convert flat index to 2D indices
        pred_idx = max_indices.item() // len(targets)
        target_idx = max_indices.item() % len(targets)

        # Add to matches
        matches.append((pred_idx, target_idx))
        matched_pred_indices.add(pred_idx)
        matched_target_indices.add(target_idx)

    # Identify unmatched predictions
    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_pred_indices]

    return matches, unmatched_preds


def rescale_bounding_boxes(normalized_boxes, image_width, image_height):
    """
    Convert normalized [0,1] bounding boxes to absolute pixel coordinates

    Args:
        normalized_boxes: Tensor of shape [N, 5] with [x, y, w, h, conf] in [0,1] range
        image_width: Width of the original image in pixels
        image_height: Height of the original image in pixels

    Returns:
        Tensor of shape [N, 5] with [x, y, w, h, conf] in pixel coordinates
    """
    scaled_boxes = normalized_boxes.clone()

    # Scale x and width by image width
    scaled_boxes[:, 0] *= image_width  # x coordinate
    scaled_boxes[:, 2] *= image_width  # width

    # Scale y and height by image height
    scaled_boxes[:, 1] *= image_height  # y coordinate
    scaled_boxes[:, 3] *= image_height  # height

    # Confidence score remains unchanged (already in [0,1])

    return scaled_boxes