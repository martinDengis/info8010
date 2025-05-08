# Utils for loss functions
import torch
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners/coco, if boxes (x,y,w,h) or (x1,y1,x2,y2) or (x_min,y_min,w,h)

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

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    elif box_format == "coco":
        # COCO format: (x_min, y_min, width, height)
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" or "coco" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    # Filter boxes below confidence threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort by confidence score (highest first)
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        # Take the box with highest confidence
        chosen_box = bboxes.pop(0)

        # Keep boxes that are either from different classes or don't overlap significantly
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # Different class
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold  # IoU below threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"
):
    """
    Calculates mean average precision for a single class

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        float: AP value for the single class
    """

    # used for numerical stability later on
    epsilon = 1e-6

    detections = []
    ground_truths = []

    # Only use class 0 (since we're working with a single class)
    for detection in pred_boxes:
        if detection[1] == 0:
            detections.append(detection)

    for true_box in true_boxes:
        if true_box[1] == 0:
            ground_truths.append(true_box)

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)

    # If none exists for this class then we can safely return 0
    if total_true_bboxes == 0:
        return 0.0

    for detection_idx, detection in enumerate(detections):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in ground_truths if bbox[0] == detection[0]
        ]

        num_gts = len(ground_truth_img)
        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[3:]),
                torch.tensor(gt[3:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    # torch.trapz for numerical integration
    return torch.trapz(precisions, recalls).item()


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    # Reshape to get the grid structure (batch_size, S, S, num_attributes)
    predictions = predictions.reshape(batch_size, 7, 7, 11)

    # Extract the two bounding boxes from each cell
    bboxes1 = predictions[..., 2:6]
    bboxes2 = predictions[..., 7:11]

    # Get confidence scores for both bounding boxes
    scores = torch.cat(
        (predictions[..., 1].unsqueeze(0), predictions[..., 6].unsqueeze(0)), dim=0
    )

    # Select the box with the highest confidence score
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    # Create grid cell indices for coordinate calculation
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    # Convert from cell coordinates to image coordinates
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    # Get the predicted class and best confidence score
    predicted_class = predictions[..., :1].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(
        predictions[..., 1], predictions[..., 6]).unsqueeze(-1)

    # Combine class, confidence, and bbox coordinates into final predictions
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item()
                          for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def get_precision_recall_curve(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"
):
    """
    Calculates precision-recall curve data for a single class

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        tuple: (AP value, precision values, recall values) for plotting
    """

    # used for numerical stability later on
    epsilon = 1e-6

    detections = []
    ground_truths = []

    # Only use class 0 (since we're working with a single class)
    for detection in pred_boxes:
        if detection[1] == 0:
            detections.append(detection)

    for true_box in true_boxes:
        if true_box[1] == 0:
            ground_truths.append(true_box)

    # find the amount of bboxes for each training example
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    # Convert to tensors of zeros for tracking
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)

    # If none exists for this class then we can safely return 0
    if total_true_bboxes == 0:
        # Return 2 points for default precision-recall curve
        return 0.0, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])

    # If no detections, we also need to return default values with 2 points
    if len(detections) == 0:
        return 0.0, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])


    for detection_idx, detection in enumerate(detections):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in ground_truths if bbox[0] == detection[0]
        ]

        best_iou = 0
        best_gt_idx = -1

        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[3:]),
                torch.tensor(gt[3:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

    # Add starting points for complete curve
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))

    # Calculate AP using trapezoidal rule
    ap = torch.trapz(precisions, recalls).item()

    return ap, precisions.detach().cpu(), recalls.detach().cpu()

def compute_auc(precisions, recalls):
    """
    Computes the Area Under the Curve from precision-recall values

    Parameters:
        precisions (torch.Tensor): Precision values
        recalls (torch.Tensor): Recall values

    Returns:
        float: AUC score
    """
    # Ensure the values are sorted by recall
    sorted_indices = torch.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]

    # Compute AUC using trapezoidal rule
    auc_value = torch.trapz(sorted_precisions, sorted_recalls).item()
    return auc_value

def evaluate_detection_metrics(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"
):
    """
    Computes detection metrics for wandb logging

    Parameters:
        pred_boxes (list): list of lists containing all predicted bboxes
        true_boxes (list): list of lists containing all ground truth bboxes
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        dict: Dictionary containing metrics and data for plotting
    """
    ap, precisions, recalls = get_precision_recall_curve(
        pred_boxes, true_boxes, iou_threshold, box_format
    )

    auc = compute_auc(precisions, recalls)

    # Convert to numpy for wandb logging
    precisions_np = precisions.numpy()
    recalls_np = recalls.numpy()

    metrics = {
        "AP": ap,
        "AUC": auc,
        "precision_recall_curve": {
            "precision": precisions_np,
            "recall": recalls_np
        }
    }

    return metrics