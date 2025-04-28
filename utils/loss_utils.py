import torch
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment

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

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union (YOLOv1 loss function).

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def match_predictions_to_targets(predictions, targets, iou_threshold=0.1):
    """
    Match predicted bounding boxes to ground truth using the Hungarian algorithm.

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

    # Convert IoU to cost matrix (Hungarian algorithm minimizes cost)
    # We negate the IoU values to convert from a maximization to a minimization problem
    cost_matrix = -iou_matrix.detach().cpu().numpy()

    # Use the scipy implementation of the Hungarian algorithm
    pred_indices, target_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by IoU threshold
    matches = []
    for pred_idx, target_idx in zip(pred_indices, target_indices):
        if iou_matrix[pred_idx, target_idx] >= iou_threshold:
            matches.append((pred_idx, target_idx))

    # Identify unmatched predictions
    matched_pred_indices = set(pred_idx for pred_idx, _ in matches)
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