import torch
from torch import nn


class Tmp(nn.Module):

    def __init__(self, in_shape=3):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq2 = nn.Sequential(
            nn.LogSoftmax(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_2 = x_1.permute([0, 3, 2, 1])
        x_2 = self.seq2(x_2)
        x_3 = x_2.permute([0, 3, 2, 1])
        return x_3
