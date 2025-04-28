"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn

"""
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding)
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.cnnblock(x)


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.convnet = self._create_conv_layers()
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.convnet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        yolo_arch = [
            (7, 64, 2, 3),
            "M",
            (3, 192, 1, 1),
            "M",
            (1, 128, 1, 0),
            (3, 256, 1, 1),
            (1, 256, 1, 0),
            (3, 512, 1, 1),
            "M",
            [(1, 256, 1, 0), (3, 512, 1, 1), 4],
            (1, 512, 1, 0),
            (3, 1024, 1, 1),
            "M",
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
            (3, 1024, 1, 1),
            (3, 1024, 2, 1),
            (3, 1024, 1, 1),
            (3, 1024, 1, 1),
        ]

        for layer in yolo_arch:
            if type(layer) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        out_channels=layer[1],
                        kernel_size=layer[0],
                        stride=layer[2],
                        padding=layer[3],
                    )
                ]
                in_channels = layer[1]

            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(layer) == list:
                conv1 = layer[0]
                conv2 = layer[1]
                num_repeats = layer[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes, hidden_size=512):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, hidden_size),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, split_size * split_size * (num_classes + num_boxes * 5)),
        )


def build_yolov1_model(split_size=7, num_boxes=2, num_classes=1):
    """
    Build the YOLOv1 model with the specified parameters.

    Args:
        split_size (int): Number of grid cells along one dimension.
        num_boxes (int): Number of bounding boxes per grid cell.
        num_classes (int): Number of object classes.

    Returns:
        YOLOv1: The constructed YOLOv1 model.
    """
    return YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)