import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# # Add the project root to the path so we can import from utils
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.loss_utils import match_predictions_to_targets, box_iou

def test_match_predictions_to_targets_normal_case():
    """Test the normal case where we have predictions and targets with good matches."""
    # Create test data
    predictions = torch.tensor([
        [0.1, 0.1, 0.2, 0.2],  # x, y, w, h for box 1
        [0.5, 0.5, 0.2, 0.2],  # x, y, w, h for box 2
        [0.8, 0.8, 0.2, 0.2],  # x, y, w, h for box 3
    ])

    targets = torch.tensor([
        [0.11, 0.11, 0.21, 0.21],  # x, y, w, h for ground truth 1 - should match prediction 0
        [0.51, 0.51, 0.19, 0.19],  # x, y, w, h for ground truth 2 - should match prediction 1
        [0.9, 0.9, 0.1, 0.1],      # x, y, w, h for ground truth 3 - low IoU with prediction 2
    ])

    # Run the matching function
    matches, unmatched_preds = match_predictions_to_targets(predictions, targets, iou_threshold=0.3)

    # Calculate IoU for verification
    iou_matrix = box_iou(predictions, targets)
    print("IoU Matrix:")
    print(iou_matrix)

    # Convert numpy indices to Python integers for more reliable assertions
    matches = [(int(pred_idx), int(target_idx)) for pred_idx, target_idx in matches]

    # Expected matches: (0,0) and (1,1) should match, but (2,2) might not meet threshold
    print(f"Matches: {matches}")
    print(f"Unmatched predictions: {unmatched_preds}")

    # Visualize the matches
    visualize_matches(predictions, targets, matches, "Normal Case")

    # Verify the matches
    assert len(matches) >= 2, "Expected at least 2 matches"
    # Check specific matches (the indices depend on the Hungarian algorithm's result)
    matched_pred_indices = [pred_idx for pred_idx, _ in matches]
    assert 0 in matched_pred_indices, "Prediction 0 should be matched"
    assert 1 in matched_pred_indices, "Prediction 1 should be matched"

def test_match_predictions_to_targets_no_targets():
    """Test the case where there are no targets."""
    predictions = torch.tensor([
        [0.1, 0.1, 0.2, 0.2],  # x, y, w, h for box 1
        [0.5, 0.5, 0.2, 0.2],  # x, y, w, h for box 2
    ])

    targets = torch.tensor([])  # Empty tensor

    # Run the matching function
    matches, unmatched_preds = match_predictions_to_targets(predictions, targets)

    print(f"Matches: {matches}")
    print(f"Unmatched predictions: {unmatched_preds}")

    # All predictions should be unmatched
    assert len(matches) == 0, "Expected no matches"
    assert len(unmatched_preds) == 2, "Expected all predictions to be unmatched"
    assert set(unmatched_preds) == {0, 1}, "Expected predictions 0 and 1 to be unmatched"

def test_match_predictions_to_targets_no_predictions():
    """Test the case where there are no predictions."""
    predictions = torch.tensor([])  # Empty tensor

    targets = torch.tensor([
        [0.1, 0.1, 0.2, 0.2],  # x, y, w, h for ground truth 1
        [0.5, 0.5, 0.2, 0.2],  # x, y, w, h for ground truth 2
    ])

    # Run the matching function
    matches, unmatched_preds = match_predictions_to_targets(predictions, targets)

    print(f"Matches: {matches}")
    print(f"Unmatched predictions: {unmatched_preds}")

    # No matches and no unmatched predictions
    assert len(matches) == 0, "Expected no matches"
    assert len(unmatched_preds) == 0, "Expected no unmatched predictions"

def test_match_predictions_to_targets_below_threshold():
    """Test the case where IoU is below threshold."""
    predictions = torch.tensor([
        [0.1, 0.1, 0.2, 0.2],  # x, y, w, h for box 1
    ])

    targets = torch.tensor([
        [0.3, 0.3, 0.2, 0.2],  # x, y, w, h for ground truth 1 - low IoU with prediction
    ])

    # Run the matching function with high threshold
    matches, unmatched_preds = match_predictions_to_targets(predictions, targets, iou_threshold=0.5)

    # Calculate IoU for verification
    iou_matrix = box_iou(predictions, targets)
    print("IoU Matrix:")
    print(iou_matrix)

    print(f"Matches: {matches}")
    print(f"Unmatched predictions: {unmatched_preds}")

    # Visualize the matches
    visualize_matches(predictions, targets, matches, "Below Threshold Case")

    # With high threshold, prediction should be unmatched
    assert len(matches) == 0, "Expected no matches due to high threshold"
    assert unmatched_preds == [0], "Expected prediction 0 to be unmatched"

def visualize_matches(predictions, targets, matches, title, save_path=None):
    """Visualize the bounding boxes and their matches."""
    # Skip visualization if either predictions or targets is empty
    if len(predictions) == 0 or len(targets) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)

    # Plot predictions (red)
    for i, (x, y, w, h) in enumerate(predictions):
        # Convert from center coordinates to top-left for Rectangle
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x-w/2, y-h/2, f'Pred {i}', color='r')

    # Plot targets (green)
    for i, (x, y, w, h) in enumerate(targets):
        # Convert from center coordinates to top-left for Rectangle
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x-w/2, y-h/2, f'GT {i}', color='g')

    # Plot matches with lines
    for pred_idx, target_idx in matches:
        pred_x, pred_y = predictions[pred_idx][0], predictions[pred_idx][1]
        target_x, target_y = targets[target_idx][0], targets[target_idx][1]
        ax.plot([pred_x, target_x], [pred_y, target_y], 'b-', linewidth=1)

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    print("Testing match_predictions_to_targets function...")

    print("\nTest Case 1: Normal case with good matches")
    test_match_predictions_to_targets_normal_case()

    print("\nTest Case 2: No targets")
    test_match_predictions_to_targets_no_targets()

    print("\nTest Case 3: No predictions")
    test_match_predictions_to_targets_no_predictions()

    print("\nTest Case 4: IoU below threshold")
    test_match_predictions_to_targets_below_threshold()

    print("\nAll tests completed!")
