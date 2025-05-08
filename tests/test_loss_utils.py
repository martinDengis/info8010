import sys
import os
import numpy as np
import torch

# Add the parent directory to sys.path to be able to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions to test
from utils.loss_utils import average_precision


def test_average_precision_basic():
    """Test the average_precision function with a simple case."""
    # Create some sample predictions and ground truths
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],  # [img_idx, class, conf, x, y, w, h]
        [0, 0, 0.8, 0.2, 0.2, 0.6, 0.6],
        [0, 0, 0.7, 0.3, 0.3, 0.7, 0.7],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],  # [img_idx, class, conf, x, y, w, h]
        [0, 0, 1.0, 0.3, 0.3, 0.7, 0.7],
    ]

    ap = average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")

    # Since we have 2 true boxes and 2 of our predictions match perfectly
    # and 1 is a false positive, we should get a high AP value
    assert 0.75 <= ap <= 1.0, f"Expected AP around 0.75-1.0, got {ap}"


def test_average_precision_multiple_thresholds():
    """Test the average_precision function at different IoU thresholds."""
    # Create sample predictions and ground truths
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 0.8, 0.2, 0.2, 0.6, 0.6],
        [0, 0, 0.7, 0.3, 0.3, 0.7, 0.7],
        [1, 0, 0.9, 0.4, 0.4, 0.8, 0.8],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 1.0, 0.3, 0.3, 0.7, 0.7],
        [1, 0, 1.0, 0.4, 0.4, 0.8, 0.8],
    ]

    # Test at different IoU thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    ap_values = {}

    for threshold in thresholds:
        ap = average_precision(pred_boxes, true_boxes, iou_threshold=threshold, box_format="midpoint")
        ap_values[threshold] = ap

    # At lower thresholds, we expect higher AP values
    # At higher thresholds, we expect lower AP values
    for i in range(len(thresholds) - 1):
        assert ap_values[thresholds[i]] >= ap_values[thresholds[i+1]], \
            f"AP at threshold {thresholds[i]} should be >= AP at threshold {thresholds[i+1]}"

    # Check specific threshold values
    assert ap_values[0.5] > 0.5, f"Expected AP > 0.5 at IoU threshold 0.5, got {ap_values[0.5]}"


def test_average_precision_edge_cases():
    """Test the average_precision function with edge cases."""
    # Empty predictions
    pred_boxes = []
    true_boxes = [[0, 0, 1.0, 0.1, 0.1, 0.5, 0.5]]

    ap = average_precision(pred_boxes, true_boxes, iou_threshold=0.5)
    assert ap == 0.0, f"Expected AP = 0.0 for empty predictions, got {ap}"

    # Empty ground truth
    pred_boxes = [[0, 0, 0.9, 0.1, 0.1, 0.5, 0.5]]
    true_boxes = []

    ap = average_precision(pred_boxes, true_boxes, iou_threshold=0.5)
    assert ap == 0.0, f"Expected AP = 0.0 for empty ground truth, got {ap}"

    # Perfect predictions
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],
        [1, 0, 0.9, 0.2, 0.2, 0.6, 0.6],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],
        [1, 0, 1.0, 0.2, 0.2, 0.6, 0.6],
    ]

    ap = average_precision(pred_boxes, true_boxes, iou_threshold=0.5)
    assert abs(ap - 1.0) < 1e-5, f"Expected AP close to 1.0 for perfect predictions, got {ap}"


def test_average_precision_different_formats():
    """Test the average_precision function with different box formats."""
    # Midpoint format: [x, y, w, h]
    pred_boxes_midpoint = [
        [0, 0, 0.9, 0.5, 0.5, 0.4, 0.4],  # center at (0.5, 0.5) with width=0.4, height=0.4
    ]
    true_boxes_midpoint = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],  # same box
    ]

    ap_midpoint = average_precision(
        pred_boxes_midpoint, true_boxes_midpoint,
        iou_threshold=0.5, box_format="midpoint"
    )

    # Corners format: [x1, y1, x2, y2]
    # Convert the midpoint format to corners format
    pred_boxes_corners = [
        [0, 0, 0.9, 0.3, 0.3, 0.7, 0.7],  # (x1, y1, x2, y2)
    ]
    true_boxes_corners = [
        [0, 0, 1.0, 0.3, 0.3, 0.7, 0.7],  # same box
    ]

    ap_corners = average_precision(
        pred_boxes_corners, true_boxes_corners,
        iou_threshold=0.5, box_format="corners"
    )

    # The AP values should be the same since the boxes are exactly matching
    assert abs(ap_midpoint - ap_corners) < 1e-5, \
        f"AP values should be the same regardless of box format. Got {ap_midpoint} vs {ap_corners}"


def test_average_precision_with_mean():
    """Test computing mean average precision over multiple IoU thresholds."""
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 0.8, 0.2, 0.2, 0.6, 0.6],
        [0, 0, 0.7, 0.3, 0.3, 0.7, 0.7],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 1.0, 0.3, 0.3, 0.7, 0.7],
    ]

    # Compute AP at different thresholds
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    ap_values = []

    for threshold in iou_thresholds:
        ap = average_precision(pred_boxes, true_boxes, iou_threshold=float(threshold))
        ap_values.append(ap)

    # Compute mean AP (mAP)
    mean_ap = np.mean(ap_values)

    # Ensure the mean AP is a reasonable value
    assert 0 <= mean_ap <= 1, f"Mean AP should be between 0 and 1, got {mean_ap}"

    # The AP at higher thresholds should generally be lower
    assert ap_values[0] >= ap_values[-1], "AP at lower threshold should be higher than at higher threshold"


# Import the functions to test
from utils.loss_utils import (
    get_precision_recall_curve,
    compute_auc,
    evaluate_detection_metrics
)


def test_get_precision_recall_curve_basic():
    """Test the get_precision_recall_curve function with a simple case."""
    # Create some sample predictions and ground truths
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],  # [img_idx, class, conf, x, y, w, h]
        [0, 0, 0.8, 0.2, 0.2, 0.6, 0.6],
        [0, 0, 0.7, 0.3, 0.3, 0.7, 0.7],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],  # [img_idx, class, conf, x, y, w, h]
        [0, 0, 1.0, 0.3, 0.3, 0.7, 0.7],
    ]

    ap, precisions, recalls = get_precision_recall_curve(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"
    )

    # Check return types
    assert isinstance(ap, float), f"Expected AP to be float, got {type(ap)}"
    assert isinstance(precisions, torch.Tensor), f"Expected precisions to be torch.Tensor, got {type(precisions)}"
    assert isinstance(recalls, torch.Tensor), f"Expected recalls to be torch.Tensor, got {type(recalls)}"

    # Check shapes
    assert precisions.shape[0] == recalls.shape[0], "Precisions and recalls should have the same length"
    assert precisions.shape[0] > 1, "Should have multiple precision/recall points"

    # Check values
    assert 0 <= ap <= 1, f"AP should be between 0 and 1, got {ap}"
    assert (precisions >= 0).all() and (precisions <= 1).all(), "Precision values should be between 0 and 1"
    assert (recalls >= 0).all() and (recalls <= 1).all(), "Recall values should be between 0 and 1"

    # Check that precision starts at 1 and recall starts at 0
    assert abs(precisions[0].item() - 1.0) < 1e-5, f"First precision value should be 1.0, got {precisions[0].item()}"
    assert abs(recalls[0].item() - 0.0) < 1e-5, f"First recall value should be 0.0, got {recalls[0].item()}"


def test_get_precision_recall_curve_edge_cases():
    """Test the get_precision_recall_curve function with edge cases."""
    # Empty predictions
    pred_boxes = []
    true_boxes = [[0, 0, 1.0, 0.1, 0.1, 0.5, 0.5]]

    ap, precisions, recalls = get_precision_recall_curve(pred_boxes, true_boxes)
    assert ap == 0.0, f"Expected AP = 0.0 for empty predictions, got {ap}"
    assert precisions.shape[0] == 2 and recalls.shape[0] == 2, "Should have default precision/recall points"

    # Empty ground truth
    pred_boxes = [[0, 0, 0.9, 0.1, 0.1, 0.5, 0.5]]
    true_boxes = []

    ap, precisions, recalls = get_precision_recall_curve(pred_boxes, true_boxes)
    assert ap == 0.0, f"Expected AP = 0.0 for empty ground truth, got {ap}"
    assert precisions.shape[0] == 2 and recalls.shape[0] == 2, "Should have default precision/recall points"

    # Perfect predictions
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],
        [1, 0, 0.9, 0.2, 0.2, 0.6, 0.6],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],
        [1, 0, 1.0, 0.2, 0.2, 0.6, 0.6],
    ]

    ap, precisions, recalls = get_precision_recall_curve(pred_boxes, true_boxes)
    assert abs(ap - 1.0) < 1e-5, f"Expected AP close to 1.0 for perfect predictions, got {ap}"


def test_compute_auc_basic():
    """Test the compute_auc function with simple cases."""
    # Test with perfect classifier
    precisions = torch.tensor([1.0, 1.0, 1.0])
    recalls = torch.tensor([0.0, 0.5, 1.0])
    auc = compute_auc(precisions, recalls)
    assert abs(auc - 1.0) < 1e-5, f"Expected AUC = 1.0 for perfect classifier, got {auc}"

    # Test with random classifier
    precisions = torch.tensor([1.0, 0.5, 0.33])
    recalls = torch.tensor([0.0, 0.5, 1.0])
    auc = compute_auc(precisions, recalls)
    assert 0.4 <= auc <= 0.6, f"Expected AUC around 0.5 for random classifier, got {auc}"

    # Test with custom values
    precisions = torch.tensor([1.0, 0.8, 0.7, 0.6, 0.5])
    recalls = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    auc = compute_auc(precisions, recalls)
    assert 0.6 <= auc <= 0.8, f"Expected AUC around 0.7, got {auc}"


def test_compute_auc_unsorted():
    """Test the compute_auc function with unsorted inputs."""
    # Unsorted recalls
    precisions = torch.tensor([1.0, 0.7, 0.8, 0.6, 0.5])
    recalls = torch.tensor([0.0, 0.5, 0.25, 0.75, 1.0])
    auc = compute_auc(precisions, recalls)

    # Create sorted version for comparison
    sorted_indices = torch.argsort(recalls)
    sorted_precisions = precisions[sorted_indices]
    sorted_recalls = recalls[sorted_indices]
    expected_auc = torch.trapz(sorted_precisions, sorted_recalls).item()

    assert abs(auc - expected_auc) < 1e-5, f"AUC calculation should handle unsorted inputs, got {auc} vs expected {expected_auc}"


def test_evaluate_detection_metrics():
    """Test the evaluate_detection_metrics function."""
    # Create some sample predictions and ground truths
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 0.8, 0.2, 0.2, 0.6, 0.6],
        [0, 0, 0.7, 0.3, 0.3, 0.7, 0.7],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 1.0, 0.3, 0.3, 0.7, 0.7],
    ]

    metrics = evaluate_detection_metrics(pred_boxes, true_boxes)

    # Check the returned structure
    assert isinstance(metrics, dict), "Should return a dictionary"
    assert "AP" in metrics, "Result should contain AP key"
    assert "AUC" in metrics, "Result should contain AUC key"
    assert "precision_recall_curve" in metrics, "Result should contain precision_recall_curve key"
    assert "precision" in metrics["precision_recall_curve"], "Curve data should contain precision"
    assert "recall" in metrics["precision_recall_curve"], "Curve data should contain recall"

    # Check values
    assert 0 <= metrics["AP"] <= 1, f"AP should be between 0 and 1, got {metrics['AP']}"
    assert 0 <= metrics["AUC"] <= 1, f"AUC should be between 0 and 1, got {metrics['AUC']}"

    # Precision values should be float32 numpy arrays suitable for wandb logging
    assert isinstance(metrics["precision_recall_curve"]["precision"], np.ndarray), "Precision should be numpy array"
    assert isinstance(metrics["precision_recall_curve"]["recall"], np.ndarray), "Recall should be numpy array"

    # Test with a different IoU threshold
    metrics_strict = evaluate_detection_metrics(pred_boxes, true_boxes, iou_threshold=0.8)
    assert metrics_strict["AP"] <= metrics["AP"], "AP should decrease with stricter IoU threshold"


def test_metric_consistency():
    """Test that the metrics are consistent with different methods."""
    # Create some sample predictions and ground truths
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 0.8, 0.2, 0.2, 0.6, 0.6],
        [0, 0, 0.7, 0.3, 0.3, 0.7, 0.7],
    ]

    true_boxes = [
        [0, 0, 1.0, 0.1, 0.1, 0.5, 0.5],
        [0, 0, 1.0, 0.3, 0.3, 0.7, 0.7],
    ]

    # Get metrics using our combined function
    metrics = evaluate_detection_metrics(pred_boxes, true_boxes)

    # Get metrics using individual functions
    ap, precisions, recalls = get_precision_recall_curve(pred_boxes, true_boxes)
    auc = compute_auc(precisions, recalls)

    # Values should be identical
    assert abs(metrics["AP"] - ap) < 1e-5, "AP values should be identical"
    assert abs(metrics["AUC"] - auc) < 1e-5, "AUC values should be identical"


if __name__ == "__main__":
    test_average_precision_basic()
    test_average_precision_multiple_thresholds()
    test_average_precision_edge_cases()
    test_average_precision_different_formats()
    test_average_precision_with_mean()
    print("All AP tests passed!")

    test_get_precision_recall_curve_basic()
    test_get_precision_recall_curve_edge_cases()
    test_compute_auc_basic()
    test_compute_auc_unsorted()
    test_evaluate_detection_metrics()
    test_metric_consistency()
    print("All detection metrics tests passed!")
