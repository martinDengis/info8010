import torch
import torch.nn.functional as F
import math


def bbox_to_xyxy(box, xywh=True):
    """Convert box from [x, y, w, h] to [x1, y1, x2, y2] format if needed"""
    if xywh:
        x, y, w, h = box.chunk(4, -1)
        box_xyxy = torch.cat((x - w / 2, y - h / 2, x + w / 2, y + h / 2), -1)
        return box_xyxy
    return box


def calculate_iou(box1, box2, xywh=True, eps=1e-7):
    """Calculate IoU between two bounding boxes"""
    # Convert to xyxy format
    box1_xyxy = bbox_to_xyxy(box1, xywh)
    box2_xyxy = bbox_to_xyxy(box2, xywh)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1_xyxy.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2_xyxy.chunk(4, -1)

    # Intersect
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # Output metrics
    iou = inter / union
    box1_coords = (b1_x1, b1_y1, b1_x2, b1_y2)
    box2_coords = (b2_x1, b2_y1, b2_x2, b2_y2)
    box1_dims = (w1, h1)
    box2_dims = (w2, h2)
    return iou, box1_coords, box2_coords, box1_dims, box2_dims


def calculate_center_distance(box1_coords, box2_coords, enclosing_box_diag, eps=1e-7):
    """Calculate the normalized center distance term for CIoU"""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1_coords
    b2_x1, b2_y1, b2_x2, b2_y2 = box2_coords

    # Enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps  # Convex diagonal squared

    # Center distance
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # Center dist ** 2

    return rho2 / c2


def calculate_aspect_ratio(box1_dims, box2_dims, iou, eps=1e-7):
    """Calculate the aspect ratio term for CIoU"""
    w1, h1 = box1_dims
    w2, h2 = box2_dims

    # Aspect ratio term
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return v * alpha


def ciou_loss(box_preds, box_targets, xywh=True, eps=1e-7, reduction='mean'):
    """
    Calculate CIoU loss between predicted and target boxes

    Args:
        box_preds (torch.Tensor): Predicted box coordinates [batch_size, n_boxes, 4]
        box_targets (torch.Tensor): Target box coordinates [batch_size, n_boxes, 4]
        xywh (bool): True if boxes in [x, y, w, h] format, False if [x1, y1, x2, y2]
        eps (float): Small constant to prevent division by zero
        reduction (str): Reduction method: 'none', 'mean', 'sum'

    Returns:
        torch.Tensor: CIoU loss (1 - CIoU)
    """
    orig_shape = box_preds.shape

    # Reshape if batch dimension
    if len(box_preds.shape) > 2:
        # Reshape to [total_boxes, 4]
        box_preds = box_preds.reshape(-1, 4)
        box_targets = box_targets.reshape(-1, 4)

    # IoU, center dist, and aspect ratio terms
    iou, box1_coords, box2_coords, box1_dims, box2_dims = calculate_iou(box_preds, box_targets, xywh, eps)
    center_distance = calculate_center_distance(box1_coords, box2_coords, None, eps)
    aspect_ratio = calculate_aspect_ratio(box1_dims, box2_dims, iou, eps)

    # Combine
    ciou = iou - (center_distance + aspect_ratio)
    loss = 1 - ciou  # Loss is 1 - CIoU

    # Apply reduction
    if reduction == 'none':
        # Reshape back to original dimensions
        if len(orig_shape) > 2:
            return loss.reshape(orig_shape[:-1])
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:  # default is mean
        return loss.mean()


def distribution_focal_loss(box_preds, box_targets, num_bins=16, reduction='mean'):
    """
    Distribution Focal Loss (DFL) for bounding box coordinate regression

    Args:
        box_preds (torch.Tensor): Predicted distributions for coordinates
            [batch_size, n_boxes, num_bins] or [batch_size, n_boxes, 4, num_bins]
        box_targets (torch.Tensor): Target coordinates (normalized between 0 and 1)
            [batch_size, n_boxes, 4]
        num_bins (int): Number of bins for coordinate discretization
        reduction (str): Reduction method: 'none', 'mean', 'sum'

    Returns:
        torch.Tensor: Distribution Focal Loss value
    """
    orig_shape = box_preds.shape

    # If we have one distribution per coordinate
    if len(box_preds.shape) == 4:  # [batch, boxes, 4, bins]
        box_preds = box_preds.reshape(-1, num_bins) # to [batch*boxes*4, bins]
        box_targets = box_targets.reshape(-1) # to [batch*boxes*4]

    # Scale target to range [0, num_bins-1]
    box_targets = box_targets * (num_bins - 1)

    # Get left and right bin indices
    target_left = box_targets.floor().long().clamp(0, num_bins - 2)
    target_right = target_left + 1

    # weights for left and right bins, and get their losses
    weight_left = target_right.float() - box_targets
    weight_right = 1 - weight_left

    loss_left = F.cross_entropy(
        box_preds, target_left, reduction='none') * weight_left
    loss_right = F.cross_entropy(
        box_preds, target_right, reduction='none') * weight_right

    # Combine and reduce
    loss = loss_left + loss_right
    if reduction == 'none':
        if len(orig_shape) == 4:  # [batch, boxes, 4, bins]
            return loss.reshape(orig_shape[:-1])
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:  # default is mean
        return loss.mean()
