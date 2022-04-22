import torch
from torch import nn


class Tmp(nn.Module):

    def __init__(self, in_shape=3):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq3 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq5 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq7 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq9 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq11 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq13 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq15 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq17 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq19 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq20 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq21 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq22 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq23 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq24 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq25 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq26 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq27 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq28 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq29 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq30 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq31 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq32 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq33 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq34 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq35 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq36 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq37 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq38 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq39 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq40 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq41 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq42 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq43 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq44 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq45 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq46 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq47 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq48 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq49 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq50 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq51 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq52 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq53 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq54 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq55 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq56 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq57 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq58 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq59 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq60 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq61 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq62 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq63 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq64 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq65 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq66 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq67 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq68 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq69 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq70 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq71 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq72 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq73 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq74 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq75 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq76 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq77 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq78 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq79 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq80 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq81 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq82 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq83 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq84 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq85 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq86 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq87 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq88 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq89 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq90 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq91 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq92 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq93 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq94 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq95 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq96 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq97 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq98 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq99 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq100 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq101 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq102 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq103 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq104 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq105 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq106 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq107 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq108 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq109 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq110 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq111 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq112 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq113 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq114 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq115 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq116 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq117 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq118 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq119 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq120 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq121 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq122 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq123 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq124 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq125 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq126 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq127 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq128 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq129 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq130 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq131 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq132 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq133 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq134 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq135 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq136 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq137 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq138 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq139 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq140 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq141 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq142 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq143 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq144 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq145 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq146 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq147 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq148 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq149 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq150 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq151 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq152 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq153 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq154 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq155 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq156 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq157 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq158 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq159 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq160 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq161 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq162 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq163 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq164 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq165 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq166 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq167 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq168 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq169 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq170 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq171 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq172 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq173 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq174 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq175 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq176 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq177 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq178 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq179 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq180 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq181 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq182 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq183 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq184 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq185 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq186 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq187 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq188 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq189 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq190 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq191 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq192 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq193 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq194 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq195 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq196 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq197 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq198 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq199 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq200 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq201 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq202 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq203 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq204 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq205 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq206 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq207 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq208 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq209 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq210 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq211 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq212 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq213 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq214 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq215 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq216 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq217 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq218 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq219 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq220 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq221 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq222 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq223 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq224 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq225 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq226 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq227 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq228 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq229 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq230 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq231 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq232 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq233 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[7, 7], stride=[7, 7], padding=[0, 0]),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1000),
        )
        self.seq234 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq235 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq236 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq237 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_237 = self.seq237(x_1)
        x_2 = self.seq2(x_1)
        x_3 = x_2 + x_237
        x_3 = self.seq3(x_3)
        x_4 = self.seq4(x_3)
        x_5 = x_4 + x_3
        x_5 = self.seq5(x_5)
        x_6 = self.seq6(x_5)
        x_7 = x_6 + x_5
        x_7 = self.seq7(x_7)
        x_8 = self.seq8(x_7)
        x_9 = x_8 + x_7
        x_9 = self.seq9(x_9)
        x_236 = self.seq236(x_9)
        x_10 = self.seq10(x_9)
        x_11 = x_10 + x_236
        x_11 = self.seq11(x_11)
        x_12 = self.seq12(x_11)
        x_13 = x_12 + x_11
        x_13 = self.seq13(x_13)
        x_14 = self.seq14(x_13)
        x_15 = x_14 + x_13
        x_15 = self.seq15(x_15)
        x_16 = self.seq16(x_15)
        x_17 = x_16 + x_15
        x_17 = self.seq17(x_17)
        x_18 = self.seq18(x_17)
        x_19 = x_18 + x_17
        x_19 = self.seq19(x_19)
        x_20 = self.seq20(x_19)
        x_21 = x_20 + x_19
        x_21 = self.seq21(x_21)
        x_22 = self.seq22(x_21)
        x_23 = x_22 + x_21
        x_23 = self.seq23(x_23)
        x_24 = self.seq24(x_23)
        x_25 = x_24 + x_23
        x_25 = self.seq25(x_25)
        x_26 = self.seq26(x_25)
        x_27 = x_26 + x_25
        x_27 = self.seq27(x_27)
        x_28 = self.seq28(x_27)
        x_29 = x_28 + x_27
        x_29 = self.seq29(x_29)
        x_30 = self.seq30(x_29)
        x_31 = x_30 + x_29
        x_31 = self.seq31(x_31)
        x_32 = self.seq32(x_31)
        x_33 = x_32 + x_31
        x_33 = self.seq33(x_33)
        x_34 = self.seq34(x_33)
        x_35 = x_34 + x_33
        x_35 = self.seq35(x_35)
        x_36 = self.seq36(x_35)
        x_37 = x_36 + x_35
        x_37 = self.seq37(x_37)
        x_38 = self.seq38(x_37)
        x_39 = x_38 + x_37
        x_39 = self.seq39(x_39)
        x_40 = self.seq40(x_39)
        x_41 = x_40 + x_39
        x_41 = self.seq41(x_41)
        x_235 = self.seq235(x_41)
        x_42 = self.seq42(x_41)
        x_43 = x_42 + x_235
        x_43 = self.seq43(x_43)
        x_44 = self.seq44(x_43)
        x_45 = x_44 + x_43
        x_45 = self.seq45(x_45)
        x_46 = self.seq46(x_45)
        x_47 = x_46 + x_45
        x_47 = self.seq47(x_47)
        x_48 = self.seq48(x_47)
        x_49 = x_48 + x_47
        x_49 = self.seq49(x_49)
        x_50 = self.seq50(x_49)
        x_51 = x_50 + x_49
        x_51 = self.seq51(x_51)
        x_52 = self.seq52(x_51)
        x_53 = x_52 + x_51
        x_53 = self.seq53(x_53)
        x_54 = self.seq54(x_53)
        x_55 = x_54 + x_53
        x_55 = self.seq55(x_55)
        x_56 = self.seq56(x_55)
        x_57 = x_56 + x_55
        x_57 = self.seq57(x_57)
        x_58 = self.seq58(x_57)
        x_59 = x_58 + x_57
        x_59 = self.seq59(x_59)
        x_60 = self.seq60(x_59)
        x_61 = x_60 + x_59
        x_61 = self.seq61(x_61)
        x_62 = self.seq62(x_61)
        x_63 = x_62 + x_61
        x_63 = self.seq63(x_63)
        x_64 = self.seq64(x_63)
        x_65 = x_64 + x_63
        x_65 = self.seq65(x_65)
        x_66 = self.seq66(x_65)
        x_67 = x_66 + x_65
        x_67 = self.seq67(x_67)
        x_68 = self.seq68(x_67)
        x_69 = x_68 + x_67
        x_69 = self.seq69(x_69)
        x_70 = self.seq70(x_69)
        x_71 = x_70 + x_69
        x_71 = self.seq71(x_71)
        x_72 = self.seq72(x_71)
        x_73 = x_72 + x_71
        x_73 = self.seq73(x_73)
        x_74 = self.seq74(x_73)
        x_75 = x_74 + x_73
        x_75 = self.seq75(x_75)
        x_76 = self.seq76(x_75)
        x_77 = x_76 + x_75
        x_77 = self.seq77(x_77)
        x_78 = self.seq78(x_77)
        x_79 = x_78 + x_77
        x_79 = self.seq79(x_79)
        x_80 = self.seq80(x_79)
        x_81 = x_80 + x_79
        x_81 = self.seq81(x_81)
        x_82 = self.seq82(x_81)
        x_83 = x_82 + x_81
        x_83 = self.seq83(x_83)
        x_84 = self.seq84(x_83)
        x_85 = x_84 + x_83
        x_85 = self.seq85(x_85)
        x_86 = self.seq86(x_85)
        x_87 = x_86 + x_85
        x_87 = self.seq87(x_87)
        x_88 = self.seq88(x_87)
        x_89 = x_88 + x_87
        x_89 = self.seq89(x_89)
        x_90 = self.seq90(x_89)
        x_91 = x_90 + x_89
        x_91 = self.seq91(x_91)
        x_92 = self.seq92(x_91)
        x_93 = x_92 + x_91
        x_93 = self.seq93(x_93)
        x_94 = self.seq94(x_93)
        x_95 = x_94 + x_93
        x_95 = self.seq95(x_95)
        x_96 = self.seq96(x_95)
        x_97 = x_96 + x_95
        x_97 = self.seq97(x_97)
        x_98 = self.seq98(x_97)
        x_99 = x_98 + x_97
        x_99 = self.seq99(x_99)
        x_100 = self.seq100(x_99)
        x_101 = x_100 + x_99
        x_101 = self.seq101(x_101)
        x_102 = self.seq102(x_101)
        x_103 = x_102 + x_101
        x_103 = self.seq103(x_103)
        x_104 = self.seq104(x_103)
        x_105 = x_104 + x_103
        x_105 = self.seq105(x_105)
        x_106 = self.seq106(x_105)
        x_107 = x_106 + x_105
        x_107 = self.seq107(x_107)
        x_108 = self.seq108(x_107)
        x_109 = x_108 + x_107
        x_109 = self.seq109(x_109)
        x_110 = self.seq110(x_109)
        x_111 = x_110 + x_109
        x_111 = self.seq111(x_111)
        x_112 = self.seq112(x_111)
        x_113 = x_112 + x_111
        x_113 = self.seq113(x_113)
        x_114 = self.seq114(x_113)
        x_115 = x_114 + x_113
        x_115 = self.seq115(x_115)
        x_116 = self.seq116(x_115)
        x_117 = x_116 + x_115
        x_117 = self.seq117(x_117)
        x_118 = self.seq118(x_117)
        x_119 = x_118 + x_117
        x_119 = self.seq119(x_119)
        x_120 = self.seq120(x_119)
        x_121 = x_120 + x_119
        x_121 = self.seq121(x_121)
        x_122 = self.seq122(x_121)
        x_123 = x_122 + x_121
        x_123 = self.seq123(x_123)
        x_124 = self.seq124(x_123)
        x_125 = x_124 + x_123
        x_125 = self.seq125(x_125)
        x_126 = self.seq126(x_125)
        x_127 = x_126 + x_125
        x_127 = self.seq127(x_127)
        x_128 = self.seq128(x_127)
        x_129 = x_128 + x_127
        x_129 = self.seq129(x_129)
        x_130 = self.seq130(x_129)
        x_131 = x_130 + x_129
        x_131 = self.seq131(x_131)
        x_132 = self.seq132(x_131)
        x_133 = x_132 + x_131
        x_133 = self.seq133(x_133)
        x_134 = self.seq134(x_133)
        x_135 = x_134 + x_133
        x_135 = self.seq135(x_135)
        x_136 = self.seq136(x_135)
        x_137 = x_136 + x_135
        x_137 = self.seq137(x_137)
        x_138 = self.seq138(x_137)
        x_139 = x_138 + x_137
        x_139 = self.seq139(x_139)
        x_140 = self.seq140(x_139)
        x_141 = x_140 + x_139
        x_141 = self.seq141(x_141)
        x_142 = self.seq142(x_141)
        x_143 = x_142 + x_141
        x_143 = self.seq143(x_143)
        x_144 = self.seq144(x_143)
        x_145 = x_144 + x_143
        x_145 = self.seq145(x_145)
        x_146 = self.seq146(x_145)
        x_147 = x_146 + x_145
        x_147 = self.seq147(x_147)
        x_148 = self.seq148(x_147)
        x_149 = x_148 + x_147
        x_149 = self.seq149(x_149)
        x_150 = self.seq150(x_149)
        x_151 = x_150 + x_149
        x_151 = self.seq151(x_151)
        x_152 = self.seq152(x_151)
        x_153 = x_152 + x_151
        x_153 = self.seq153(x_153)
        x_154 = self.seq154(x_153)
        x_155 = x_154 + x_153
        x_155 = self.seq155(x_155)
        x_156 = self.seq156(x_155)
        x_157 = x_156 + x_155
        x_157 = self.seq157(x_157)
        x_158 = self.seq158(x_157)
        x_159 = x_158 + x_157
        x_159 = self.seq159(x_159)
        x_160 = self.seq160(x_159)
        x_161 = x_160 + x_159
        x_161 = self.seq161(x_161)
        x_162 = self.seq162(x_161)
        x_163 = x_162 + x_161
        x_163 = self.seq163(x_163)
        x_164 = self.seq164(x_163)
        x_165 = x_164 + x_163
        x_165 = self.seq165(x_165)
        x_166 = self.seq166(x_165)
        x_167 = x_166 + x_165
        x_167 = self.seq167(x_167)
        x_168 = self.seq168(x_167)
        x_169 = x_168 + x_167
        x_169 = self.seq169(x_169)
        x_234 = self.seq234(x_169)
        x_170 = self.seq170(x_169)
        x_171 = x_170 + x_234
        x_171 = self.seq171(x_171)
        x_172 = self.seq172(x_171)
        x_173 = x_172 + x_171
        x_173 = self.seq173(x_173)
        x_174 = self.seq174(x_173)
        x_175 = x_174 + x_173
        x_175 = self.seq175(x_175)
        x_176 = self.seq176(x_175)
        x_177 = x_176 + x_175
        x_177 = self.seq177(x_177)
        x_178 = self.seq178(x_177)
        x_179 = x_178 + x_177
        x_179 = self.seq179(x_179)
        x_180 = self.seq180(x_179)
        x_181 = x_180 + x_179
        x_181 = self.seq181(x_181)
        x_182 = self.seq182(x_181)
        x_183 = x_182 + x_181
        x_183 = self.seq183(x_183)
        x_184 = self.seq184(x_183)
        x_185 = x_184 + x_183
        x_185 = self.seq185(x_185)
        x_186 = self.seq186(x_185)
        x_187 = x_186 + x_185
        x_187 = self.seq187(x_187)
        x_188 = self.seq188(x_187)
        x_189 = x_188 + x_187
        x_189 = self.seq189(x_189)
        x_190 = self.seq190(x_189)
        x_191 = x_190 + x_189
        x_191 = self.seq191(x_191)
        x_192 = self.seq192(x_191)
        x_193 = x_192 + x_191
        x_193 = self.seq193(x_193)
        x_194 = self.seq194(x_193)
        x_195 = x_194 + x_193
        x_195 = self.seq195(x_195)
        x_196 = self.seq196(x_195)
        x_197 = x_196 + x_195
        x_197 = self.seq197(x_197)
        x_198 = self.seq198(x_197)
        x_199 = x_198 + x_197
        x_199 = self.seq199(x_199)
        x_200 = self.seq200(x_199)
        x_201 = x_200 + x_199
        x_201 = self.seq201(x_201)
        x_202 = self.seq202(x_201)
        x_203 = x_202 + x_201
        x_203 = self.seq203(x_203)
        x_204 = self.seq204(x_203)
        x_205 = x_204 + x_203
        x_205 = self.seq205(x_205)
        x_206 = self.seq206(x_205)
        x_207 = x_206 + x_205
        x_207 = self.seq207(x_207)
        x_208 = self.seq208(x_207)
        x_209 = x_208 + x_207
        x_209 = self.seq209(x_209)
        x_210 = self.seq210(x_209)
        x_211 = x_210 + x_209
        x_211 = self.seq211(x_211)
        x_212 = self.seq212(x_211)
        x_213 = x_212 + x_211
        x_213 = self.seq213(x_213)
        x_214 = self.seq214(x_213)
        x_215 = x_214 + x_213
        x_215 = self.seq215(x_215)
        x_216 = self.seq216(x_215)
        x_217 = x_216 + x_215
        x_217 = self.seq217(x_217)
        x_218 = self.seq218(x_217)
        x_219 = x_218 + x_217
        x_219 = self.seq219(x_219)
        x_220 = self.seq220(x_219)
        x_221 = x_220 + x_219
        x_221 = self.seq221(x_221)
        x_222 = self.seq222(x_221)
        x_223 = x_222 + x_221
        x_223 = self.seq223(x_223)
        x_224 = self.seq224(x_223)
        x_225 = x_224 + x_223
        x_225 = self.seq225(x_225)
        x_226 = self.seq226(x_225)
        x_227 = x_226 + x_225
        x_227 = self.seq227(x_227)
        x_228 = self.seq228(x_227)
        x_229 = x_228 + x_227
        x_229 = self.seq229(x_229)
        x_230 = self.seq230(x_229)
        x_231 = x_230 + x_229
        x_231 = self.seq231(x_231)
        x_232 = self.seq232(x_231)
        x_233 = x_232 + x_231
        x_233 = self.seq233(x_233)
        return x_233
