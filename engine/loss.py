import torch
import torch.nn as nn
from utils.loss_utils import intersection_over_union

# ==================================================
# Bounding Box Loss
# Only slightly adapted from
#   https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py
# ==================================================


class BboxLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, split_size=7, num_boxes=2, num_classes=1):
        super(BboxLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes,
        """
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(
            -1,
            self.S,
            self.S,
            self.C + self.B*5
        )

        # Deconstruct
        po1 = predictions[..., 1:2]
        box_pred1 = predictions[..., 2:6]

        po2 = predictions[..., 6:7]
        box_pred2 = predictions[..., 7:11]

        ptarget = target[..., 1]
        ptarget_aux = target[..., 1:2]
        box_target = target[..., 2:6]

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(box_pred1, box_target)
        iou_b2 = intersection_over_union(box_pred2, box_target)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = ptarget.unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * \
            (bestbox * box_pred2 + (1 - bestbox) * box_pred1)

        box_targets = exists_box * box_target

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(
            box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (bestbox * po2 + (1 - bestbox) * po1)

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * ptarget_aux),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * po1, start_dim=1),
            torch.flatten((1 - exists_box) * ptarget_aux, start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * po2, start_dim=1),
            torch.flatten((1 - exists_box) * ptarget_aux, start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :1], end_dim=-2,),
            torch.flatten(exists_box * target[..., :1], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
