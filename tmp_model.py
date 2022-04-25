import torch
from torch import nn


class Tmp(nn.Module):

    def __init__(self, in_shape=3):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=8751, kernel_size=(5, 4), stride=(1, 1), padding=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(num_features=8751),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(in_features=7018, out_features=7932),
            nn.Flatten(),
            nn.Linear(in_features=7932, out_features=4424),
            nn.Flatten(),
            nn.Linear(in_features=4424, out_features=4992),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.BatchNorm2d(num_features=0),
            nn.BatchNorm2d(num_features=2865),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(in_features=3092, out_features=1192),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=0),
            nn.Sigmoid(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
