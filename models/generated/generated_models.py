from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torchvision.models.densenet import _densenet


class GeneratedModel1(nn.Module):
    def __init__(self):
        super(GeneratedModel1, self).__init__()
        self.seq0 = nn.Sequential(
            nn.BatchNorm2d(num_features=3, eps=1e-05),
        )
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                      dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList()
        for i in range(1, 62):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                          dilation=(1, 1), groups=1),
                nn.BatchNorm2d(num_features=64, eps=1e-05),
                nn.ReLU(),
            )
            self.layers.add_module('layer%d' % (i + 1), layer)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(802816, 100)
        )

    def forward(self, x):
        x_0 = self.seq0(x)
        x_1 = self.seq1(x_0)
        for layer in self.layers:
            x_1 = layer(x_0) + x_1
        x_1 = self.fc(x_1)
        return x_1


def GeneratedDensenet(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _densenet('GeneratedDensenet', 32, (6, 32, 61, 48), 64, pretrained, progress,
                     **kwargs)