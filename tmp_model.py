import torch
from torch import nn


class Tmp(nn.Module):

    def __init__(self, in_shape=3):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
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
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq9 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq13 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq15 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq17 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq19 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq41 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq42 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq43 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq45 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq46 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq47 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq49 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq50 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq51 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq53 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq54 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq55 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq57 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq58 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq59 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq61 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq62 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq63 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq65 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq66 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq67 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq69 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq70 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq71 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq73 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq74 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq75 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq77 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq78 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq79 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq81 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq82 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq83 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq85 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq86 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq87 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq89 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq90 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq91 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq93 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq94 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq95 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq97 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq98 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq99 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq101 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq102 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq103 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq105 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq106 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq107 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq109 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq110 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq111 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq113 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq114 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq115 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq117 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq118 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq119 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq121 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq122 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq123 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq125 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq126 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq127 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq129 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq130 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq131 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq133 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq134 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq135 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
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
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq169 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq170 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq171 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq173 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq174 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq175 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq177 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq178 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq179 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq181 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq182 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq183 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq185 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq186 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq187 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq189 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq190 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq191 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq193 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq194 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq195 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq197 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq198 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq199 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq201 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq202 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq203 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq205 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq206 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq207 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq209 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq210 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq211 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq213 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq214 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq215 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq217 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq218 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq219 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq221 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq222 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq223 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq225 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq226 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq227 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq229 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq230 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq231 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq233 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq234 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq235 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq237 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq238 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq239 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq241 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq242 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq243 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq245 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq246 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq247 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq249 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq250 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq251 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq253 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq254 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq255 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq257 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq258 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq259 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq261 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq262 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq263 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq265 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq266 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq267 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq269 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq270 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq271 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq273 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq274 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq275 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq277 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq278 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq279 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq281 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq282 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq283 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq285 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq286 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq287 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq289 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq290 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq291 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq293 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq294 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq295 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq297 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq298 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq299 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq301 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq302 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq303 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq305 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq306 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq307 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq309 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq310 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq311 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq313 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq314 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq315 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq317 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq318 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq319 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq321 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq322 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq323 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq325 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq326 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq327 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq329 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq330 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq331 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq333 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq334 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq335 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq337 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq338 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq339 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq341 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq342 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq343 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq345 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq346 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq347 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq349 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq350 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq351 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq353 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq354 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq355 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq357 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq358 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq359 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Sigmoid(),
        )
        self.seq361 = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1000),
        )
        self.seq362 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq363 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq364 = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq365 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_365 = self.seq365(x_1)
        x_2 = self.seq2(x_1)
        x_3 = x_2.mean([2, 3], keepdim=True)
        x_3 = self.seq3(x_3)
        x_4 = torch.mul(x_3, x_2)
        x_5 = x_4 + x_365
        x_5 = self.seq5(x_5)
        x_6 = self.seq6(x_5)
        x_7 = x_6.mean([2, 3], keepdim=True)
        x_7 = self.seq7(x_7)
        x_8 = torch.mul(x_7, x_6)
        x_9 = x_8 + x_5
        x_9 = self.seq9(x_9)
        x_10 = self.seq10(x_9)
        x_11 = x_10.mean([2, 3], keepdim=True)
        x_11 = self.seq11(x_11)
        x_12 = torch.mul(x_11, x_10)
        x_13 = x_12 + x_9
        x_13 = self.seq13(x_13)
        x_14 = self.seq14(x_13)
        x_15 = x_14.mean([2, 3], keepdim=True)
        x_15 = self.seq15(x_15)
        x_16 = torch.mul(x_15, x_14)
        x_17 = x_16 + x_13
        x_17 = self.seq17(x_17)
        x_364 = self.seq364(x_17)
        x_18 = self.seq18(x_17)
        x_19 = x_18.mean([2, 3], keepdim=True)
        x_19 = self.seq19(x_19)
        x_20 = torch.mul(x_19, x_18)
        x_21 = x_20 + x_364
        x_21 = self.seq21(x_21)
        x_22 = self.seq22(x_21)
        x_23 = x_22.mean([2, 3], keepdim=True)
        x_23 = self.seq23(x_23)
        x_24 = torch.mul(x_23, x_22)
        x_25 = x_24 + x_21
        x_25 = self.seq25(x_25)
        x_26 = self.seq26(x_25)
        x_27 = x_26.mean([2, 3], keepdim=True)
        x_27 = self.seq27(x_27)
        x_28 = torch.mul(x_27, x_26)
        x_29 = x_28 + x_25
        x_29 = self.seq29(x_29)
        x_30 = self.seq30(x_29)
        x_31 = x_30.mean([2, 3], keepdim=True)
        x_31 = self.seq31(x_31)
        x_32 = torch.mul(x_31, x_30)
        x_33 = x_32 + x_29
        x_33 = self.seq33(x_33)
        x_34 = self.seq34(x_33)
        x_35 = x_34.mean([2, 3], keepdim=True)
        x_35 = self.seq35(x_35)
        x_36 = torch.mul(x_35, x_34)
        x_37 = x_36 + x_33
        x_37 = self.seq37(x_37)
        x_38 = self.seq38(x_37)
        x_39 = x_38.mean([2, 3], keepdim=True)
        x_39 = self.seq39(x_39)
        x_40 = torch.mul(x_39, x_38)
        x_41 = x_40 + x_37
        x_41 = self.seq41(x_41)
        x_42 = self.seq42(x_41)
        x_43 = x_42.mean([2, 3], keepdim=True)
        x_43 = self.seq43(x_43)
        x_44 = torch.mul(x_43, x_42)
        x_45 = x_44 + x_41
        x_45 = self.seq45(x_45)
        x_46 = self.seq46(x_45)
        x_47 = x_46.mean([2, 3], keepdim=True)
        x_47 = self.seq47(x_47)
        x_48 = torch.mul(x_47, x_46)
        x_49 = x_48 + x_45
        x_49 = self.seq49(x_49)
        x_50 = self.seq50(x_49)
        x_51 = x_50.mean([2, 3], keepdim=True)
        x_51 = self.seq51(x_51)
        x_52 = torch.mul(x_51, x_50)
        x_53 = x_52 + x_49
        x_53 = self.seq53(x_53)
        x_54 = self.seq54(x_53)
        x_55 = x_54.mean([2, 3], keepdim=True)
        x_55 = self.seq55(x_55)
        x_56 = torch.mul(x_55, x_54)
        x_57 = x_56 + x_53
        x_57 = self.seq57(x_57)
        x_58 = self.seq58(x_57)
        x_59 = x_58.mean([2, 3], keepdim=True)
        x_59 = self.seq59(x_59)
        x_60 = torch.mul(x_59, x_58)
        x_61 = x_60 + x_57
        x_61 = self.seq61(x_61)
        x_62 = self.seq62(x_61)
        x_63 = x_62.mean([2, 3], keepdim=True)
        x_63 = self.seq63(x_63)
        x_64 = torch.mul(x_63, x_62)
        x_65 = x_64 + x_61
        x_65 = self.seq65(x_65)
        x_66 = self.seq66(x_65)
        x_67 = x_66.mean([2, 3], keepdim=True)
        x_67 = self.seq67(x_67)
        x_68 = torch.mul(x_67, x_66)
        x_69 = x_68 + x_65
        x_69 = self.seq69(x_69)
        x_70 = self.seq70(x_69)
        x_71 = x_70.mean([2, 3], keepdim=True)
        x_71 = self.seq71(x_71)
        x_72 = torch.mul(x_71, x_70)
        x_73 = x_72 + x_69
        x_73 = self.seq73(x_73)
        x_74 = self.seq74(x_73)
        x_75 = x_74.mean([2, 3], keepdim=True)
        x_75 = self.seq75(x_75)
        x_76 = torch.mul(x_75, x_74)
        x_77 = x_76 + x_73
        x_77 = self.seq77(x_77)
        x_78 = self.seq78(x_77)
        x_79 = x_78.mean([2, 3], keepdim=True)
        x_79 = self.seq79(x_79)
        x_80 = torch.mul(x_79, x_78)
        x_81 = x_80 + x_77
        x_81 = self.seq81(x_81)
        x_82 = self.seq82(x_81)
        x_83 = x_82.mean([2, 3], keepdim=True)
        x_83 = self.seq83(x_83)
        x_84 = torch.mul(x_83, x_82)
        x_85 = x_84 + x_81
        x_85 = self.seq85(x_85)
        x_86 = self.seq86(x_85)
        x_87 = x_86.mean([2, 3], keepdim=True)
        x_87 = self.seq87(x_87)
        x_88 = torch.mul(x_87, x_86)
        x_89 = x_88 + x_85
        x_89 = self.seq89(x_89)
        x_90 = self.seq90(x_89)
        x_91 = x_90.mean([2, 3], keepdim=True)
        x_91 = self.seq91(x_91)
        x_92 = torch.mul(x_91, x_90)
        x_93 = x_92 + x_89
        x_93 = self.seq93(x_93)
        x_94 = self.seq94(x_93)
        x_95 = x_94.mean([2, 3], keepdim=True)
        x_95 = self.seq95(x_95)
        x_96 = torch.mul(x_95, x_94)
        x_97 = x_96 + x_93
        x_97 = self.seq97(x_97)
        x_98 = self.seq98(x_97)
        x_99 = x_98.mean([2, 3], keepdim=True)
        x_99 = self.seq99(x_99)
        x_100 = torch.mul(x_99, x_98)
        x_101 = x_100 + x_97
        x_101 = self.seq101(x_101)
        x_102 = self.seq102(x_101)
        x_103 = x_102.mean([2, 3], keepdim=True)
        x_103 = self.seq103(x_103)
        x_104 = torch.mul(x_103, x_102)
        x_105 = x_104 + x_101
        x_105 = self.seq105(x_105)
        x_106 = self.seq106(x_105)
        x_107 = x_106.mean([2, 3], keepdim=True)
        x_107 = self.seq107(x_107)
        x_108 = torch.mul(x_107, x_106)
        x_109 = x_108 + x_105
        x_109 = self.seq109(x_109)
        x_110 = self.seq110(x_109)
        x_111 = x_110.mean([2, 3], keepdim=True)
        x_111 = self.seq111(x_111)
        x_112 = torch.mul(x_111, x_110)
        x_113 = x_112 + x_109
        x_113 = self.seq113(x_113)
        x_114 = self.seq114(x_113)
        x_115 = x_114.mean([2, 3], keepdim=True)
        x_115 = self.seq115(x_115)
        x_116 = torch.mul(x_115, x_114)
        x_117 = x_116 + x_113
        x_117 = self.seq117(x_117)
        x_118 = self.seq118(x_117)
        x_119 = x_118.mean([2, 3], keepdim=True)
        x_119 = self.seq119(x_119)
        x_120 = torch.mul(x_119, x_118)
        x_121 = x_120 + x_117
        x_121 = self.seq121(x_121)
        x_122 = self.seq122(x_121)
        x_123 = x_122.mean([2, 3], keepdim=True)
        x_123 = self.seq123(x_123)
        x_124 = torch.mul(x_123, x_122)
        x_125 = x_124 + x_121
        x_125 = self.seq125(x_125)
        x_126 = self.seq126(x_125)
        x_127 = x_126.mean([2, 3], keepdim=True)
        x_127 = self.seq127(x_127)
        x_128 = torch.mul(x_127, x_126)
        x_129 = x_128 + x_125
        x_129 = self.seq129(x_129)
        x_130 = self.seq130(x_129)
        x_131 = x_130.mean([2, 3], keepdim=True)
        x_131 = self.seq131(x_131)
        x_132 = torch.mul(x_131, x_130)
        x_133 = x_132 + x_129
        x_133 = self.seq133(x_133)
        x_363 = self.seq363(x_133)
        x_134 = self.seq134(x_133)
        x_135 = x_134.mean([2, 3], keepdim=True)
        x_135 = self.seq135(x_135)
        x_136 = torch.mul(x_135, x_134)
        x_137 = x_136 + x_363
        x_137 = self.seq137(x_137)
        x_138 = self.seq138(x_137)
        x_139 = x_138.mean([2, 3], keepdim=True)
        x_139 = self.seq139(x_139)
        x_140 = torch.mul(x_139, x_138)
        x_141 = x_140 + x_137
        x_141 = self.seq141(x_141)
        x_142 = self.seq142(x_141)
        x_143 = x_142.mean([2, 3], keepdim=True)
        x_143 = self.seq143(x_143)
        x_144 = torch.mul(x_143, x_142)
        x_145 = x_144 + x_141
        x_145 = self.seq145(x_145)
        x_146 = self.seq146(x_145)
        x_147 = x_146.mean([2, 3], keepdim=True)
        x_147 = self.seq147(x_147)
        x_148 = torch.mul(x_147, x_146)
        x_149 = x_148 + x_145
        x_149 = self.seq149(x_149)
        x_150 = self.seq150(x_149)
        x_151 = x_150.mean([2, 3], keepdim=True)
        x_151 = self.seq151(x_151)
        x_152 = torch.mul(x_151, x_150)
        x_153 = x_152 + x_149
        x_153 = self.seq153(x_153)
        x_154 = self.seq154(x_153)
        x_155 = x_154.mean([2, 3], keepdim=True)
        x_155 = self.seq155(x_155)
        x_156 = torch.mul(x_155, x_154)
        x_157 = x_156 + x_153
        x_157 = self.seq157(x_157)
        x_158 = self.seq158(x_157)
        x_159 = x_158.mean([2, 3], keepdim=True)
        x_159 = self.seq159(x_159)
        x_160 = torch.mul(x_159, x_158)
        x_161 = x_160 + x_157
        x_161 = self.seq161(x_161)
        x_162 = self.seq162(x_161)
        x_163 = x_162.mean([2, 3], keepdim=True)
        x_163 = self.seq163(x_163)
        x_164 = torch.mul(x_163, x_162)
        x_165 = x_164 + x_161
        x_165 = self.seq165(x_165)
        x_166 = self.seq166(x_165)
        x_167 = x_166.mean([2, 3], keepdim=True)
        x_167 = self.seq167(x_167)
        x_168 = torch.mul(x_167, x_166)
        x_169 = x_168 + x_165
        x_169 = self.seq169(x_169)
        x_170 = self.seq170(x_169)
        x_171 = x_170.mean([2, 3], keepdim=True)
        x_171 = self.seq171(x_171)
        x_172 = torch.mul(x_171, x_170)
        x_173 = x_172 + x_169
        x_173 = self.seq173(x_173)
        x_174 = self.seq174(x_173)
        x_175 = x_174.mean([2, 3], keepdim=True)
        x_175 = self.seq175(x_175)
        x_176 = torch.mul(x_175, x_174)
        x_177 = x_176 + x_173
        x_177 = self.seq177(x_177)
        x_178 = self.seq178(x_177)
        x_179 = x_178.mean([2, 3], keepdim=True)
        x_179 = self.seq179(x_179)
        x_180 = torch.mul(x_179, x_178)
        x_181 = x_180 + x_177
        x_181 = self.seq181(x_181)
        x_182 = self.seq182(x_181)
        x_183 = x_182.mean([2, 3], keepdim=True)
        x_183 = self.seq183(x_183)
        x_184 = torch.mul(x_183, x_182)
        x_185 = x_184 + x_181
        x_185 = self.seq185(x_185)
        x_186 = self.seq186(x_185)
        x_187 = x_186.mean([2, 3], keepdim=True)
        x_187 = self.seq187(x_187)
        x_188 = torch.mul(x_187, x_186)
        x_189 = x_188 + x_185
        x_189 = self.seq189(x_189)
        x_190 = self.seq190(x_189)
        x_191 = x_190.mean([2, 3], keepdim=True)
        x_191 = self.seq191(x_191)
        x_192 = torch.mul(x_191, x_190)
        x_193 = x_192 + x_189
        x_193 = self.seq193(x_193)
        x_194 = self.seq194(x_193)
        x_195 = x_194.mean([2, 3], keepdim=True)
        x_195 = self.seq195(x_195)
        x_196 = torch.mul(x_195, x_194)
        x_197 = x_196 + x_193
        x_197 = self.seq197(x_197)
        x_198 = self.seq198(x_197)
        x_199 = x_198.mean([2, 3], keepdim=True)
        x_199 = self.seq199(x_199)
        x_200 = torch.mul(x_199, x_198)
        x_201 = x_200 + x_197
        x_201 = self.seq201(x_201)
        x_202 = self.seq202(x_201)
        x_203 = x_202.mean([2, 3], keepdim=True)
        x_203 = self.seq203(x_203)
        x_204 = torch.mul(x_203, x_202)
        x_205 = x_204 + x_201
        x_205 = self.seq205(x_205)
        x_206 = self.seq206(x_205)
        x_207 = x_206.mean([2, 3], keepdim=True)
        x_207 = self.seq207(x_207)
        x_208 = torch.mul(x_207, x_206)
        x_209 = x_208 + x_205
        x_209 = self.seq209(x_209)
        x_210 = self.seq210(x_209)
        x_211 = x_210.mean([2, 3], keepdim=True)
        x_211 = self.seq211(x_211)
        x_212 = torch.mul(x_211, x_210)
        x_213 = x_212 + x_209
        x_213 = self.seq213(x_213)
        x_214 = self.seq214(x_213)
        x_215 = x_214.mean([2, 3], keepdim=True)
        x_215 = self.seq215(x_215)
        x_216 = torch.mul(x_215, x_214)
        x_217 = x_216 + x_213
        x_217 = self.seq217(x_217)
        x_218 = self.seq218(x_217)
        x_219 = x_218.mean([2, 3], keepdim=True)
        x_219 = self.seq219(x_219)
        x_220 = torch.mul(x_219, x_218)
        x_221 = x_220 + x_217
        x_221 = self.seq221(x_221)
        x_222 = self.seq222(x_221)
        x_223 = x_222.mean([2, 3], keepdim=True)
        x_223 = self.seq223(x_223)
        x_224 = torch.mul(x_223, x_222)
        x_225 = x_224 + x_221
        x_225 = self.seq225(x_225)
        x_226 = self.seq226(x_225)
        x_227 = x_226.mean([2, 3], keepdim=True)
        x_227 = self.seq227(x_227)
        x_228 = torch.mul(x_227, x_226)
        x_229 = x_228 + x_225
        x_229 = self.seq229(x_229)
        x_230 = self.seq230(x_229)
        x_231 = x_230.mean([2, 3], keepdim=True)
        x_231 = self.seq231(x_231)
        x_232 = torch.mul(x_231, x_230)
        x_233 = x_232 + x_229
        x_233 = self.seq233(x_233)
        x_234 = self.seq234(x_233)
        x_235 = x_234.mean([2, 3], keepdim=True)
        x_235 = self.seq235(x_235)
        x_236 = torch.mul(x_235, x_234)
        x_237 = x_236 + x_233
        x_237 = self.seq237(x_237)
        x_238 = self.seq238(x_237)
        x_239 = x_238.mean([2, 3], keepdim=True)
        x_239 = self.seq239(x_239)
        x_240 = torch.mul(x_239, x_238)
        x_241 = x_240 + x_237
        x_241 = self.seq241(x_241)
        x_242 = self.seq242(x_241)
        x_243 = x_242.mean([2, 3], keepdim=True)
        x_243 = self.seq243(x_243)
        x_244 = torch.mul(x_243, x_242)
        x_245 = x_244 + x_241
        x_245 = self.seq245(x_245)
        x_246 = self.seq246(x_245)
        x_247 = x_246.mean([2, 3], keepdim=True)
        x_247 = self.seq247(x_247)
        x_248 = torch.mul(x_247, x_246)
        x_249 = x_248 + x_245
        x_249 = self.seq249(x_249)
        x_250 = self.seq250(x_249)
        x_251 = x_250.mean([2, 3], keepdim=True)
        x_251 = self.seq251(x_251)
        x_252 = torch.mul(x_251, x_250)
        x_253 = x_252 + x_249
        x_253 = self.seq253(x_253)
        x_254 = self.seq254(x_253)
        x_255 = x_254.mean([2, 3], keepdim=True)
        x_255 = self.seq255(x_255)
        x_256 = torch.mul(x_255, x_254)
        x_257 = x_256 + x_253
        x_257 = self.seq257(x_257)
        x_258 = self.seq258(x_257)
        x_259 = x_258.mean([2, 3], keepdim=True)
        x_259 = self.seq259(x_259)
        x_260 = torch.mul(x_259, x_258)
        x_261 = x_260 + x_257
        x_261 = self.seq261(x_261)
        x_262 = self.seq262(x_261)
        x_263 = x_262.mean([2, 3], keepdim=True)
        x_263 = self.seq263(x_263)
        x_264 = torch.mul(x_263, x_262)
        x_265 = x_264 + x_261
        x_265 = self.seq265(x_265)
        x_266 = self.seq266(x_265)
        x_267 = x_266.mean([2, 3], keepdim=True)
        x_267 = self.seq267(x_267)
        x_268 = torch.mul(x_267, x_266)
        x_269 = x_268 + x_265
        x_269 = self.seq269(x_269)
        x_270 = self.seq270(x_269)
        x_271 = x_270.mean([2, 3], keepdim=True)
        x_271 = self.seq271(x_271)
        x_272 = torch.mul(x_271, x_270)
        x_273 = x_272 + x_269
        x_273 = self.seq273(x_273)
        x_274 = self.seq274(x_273)
        x_275 = x_274.mean([2, 3], keepdim=True)
        x_275 = self.seq275(x_275)
        x_276 = torch.mul(x_275, x_274)
        x_277 = x_276 + x_273
        x_277 = self.seq277(x_277)
        x_278 = self.seq278(x_277)
        x_279 = x_278.mean([2, 3], keepdim=True)
        x_279 = self.seq279(x_279)
        x_280 = torch.mul(x_279, x_278)
        x_281 = x_280 + x_277
        x_281 = self.seq281(x_281)
        x_282 = self.seq282(x_281)
        x_283 = x_282.mean([2, 3], keepdim=True)
        x_283 = self.seq283(x_283)
        x_284 = torch.mul(x_283, x_282)
        x_285 = x_284 + x_281
        x_285 = self.seq285(x_285)
        x_286 = self.seq286(x_285)
        x_287 = x_286.mean([2, 3], keepdim=True)
        x_287 = self.seq287(x_287)
        x_288 = torch.mul(x_287, x_286)
        x_289 = x_288 + x_285
        x_289 = self.seq289(x_289)
        x_290 = self.seq290(x_289)
        x_291 = x_290.mean([2, 3], keepdim=True)
        x_291 = self.seq291(x_291)
        x_292 = torch.mul(x_291, x_290)
        x_293 = x_292 + x_289
        x_293 = self.seq293(x_293)
        x_294 = self.seq294(x_293)
        x_295 = x_294.mean([2, 3], keepdim=True)
        x_295 = self.seq295(x_295)
        x_296 = torch.mul(x_295, x_294)
        x_297 = x_296 + x_293
        x_297 = self.seq297(x_297)
        x_298 = self.seq298(x_297)
        x_299 = x_298.mean([2, 3], keepdim=True)
        x_299 = self.seq299(x_299)
        x_300 = torch.mul(x_299, x_298)
        x_301 = x_300 + x_297
        x_301 = self.seq301(x_301)
        x_302 = self.seq302(x_301)
        x_303 = x_302.mean([2, 3], keepdim=True)
        x_303 = self.seq303(x_303)
        x_304 = torch.mul(x_303, x_302)
        x_305 = x_304 + x_301
        x_305 = self.seq305(x_305)
        x_306 = self.seq306(x_305)
        x_307 = x_306.mean([2, 3], keepdim=True)
        x_307 = self.seq307(x_307)
        x_308 = torch.mul(x_307, x_306)
        x_309 = x_308 + x_305
        x_309 = self.seq309(x_309)
        x_310 = self.seq310(x_309)
        x_311 = x_310.mean([2, 3], keepdim=True)
        x_311 = self.seq311(x_311)
        x_312 = torch.mul(x_311, x_310)
        x_313 = x_312 + x_309
        x_313 = self.seq313(x_313)
        x_314 = self.seq314(x_313)
        x_315 = x_314.mean([2, 3], keepdim=True)
        x_315 = self.seq315(x_315)
        x_316 = torch.mul(x_315, x_314)
        x_317 = x_316 + x_313
        x_317 = self.seq317(x_317)
        x_318 = self.seq318(x_317)
        x_319 = x_318.mean([2, 3], keepdim=True)
        x_319 = self.seq319(x_319)
        x_320 = torch.mul(x_319, x_318)
        x_321 = x_320 + x_317
        x_321 = self.seq321(x_321)
        x_322 = self.seq322(x_321)
        x_323 = x_322.mean([2, 3], keepdim=True)
        x_323 = self.seq323(x_323)
        x_324 = torch.mul(x_323, x_322)
        x_325 = x_324 + x_321
        x_325 = self.seq325(x_325)
        x_326 = self.seq326(x_325)
        x_327 = x_326.mean([2, 3], keepdim=True)
        x_327 = self.seq327(x_327)
        x_328 = torch.mul(x_327, x_326)
        x_329 = x_328 + x_325
        x_329 = self.seq329(x_329)
        x_330 = self.seq330(x_329)
        x_331 = x_330.mean([2, 3], keepdim=True)
        x_331 = self.seq331(x_331)
        x_332 = torch.mul(x_331, x_330)
        x_333 = x_332 + x_329
        x_333 = self.seq333(x_333)
        x_334 = self.seq334(x_333)
        x_335 = x_334.mean([2, 3], keepdim=True)
        x_335 = self.seq335(x_335)
        x_336 = torch.mul(x_335, x_334)
        x_337 = x_336 + x_333
        x_337 = self.seq337(x_337)
        x_338 = self.seq338(x_337)
        x_339 = x_338.mean([2, 3], keepdim=True)
        x_339 = self.seq339(x_339)
        x_340 = torch.mul(x_339, x_338)
        x_341 = x_340 + x_337
        x_341 = self.seq341(x_341)
        x_342 = self.seq342(x_341)
        x_343 = x_342.mean([2, 3], keepdim=True)
        x_343 = self.seq343(x_343)
        x_344 = torch.mul(x_343, x_342)
        x_345 = x_344 + x_341
        x_345 = self.seq345(x_345)
        x_362 = self.seq362(x_345)
        x_346 = self.seq346(x_345)
        x_347 = x_346.mean([2, 3], keepdim=True)
        x_347 = self.seq347(x_347)
        x_348 = torch.mul(x_347, x_346)
        x_349 = x_348 + x_362
        x_349 = self.seq349(x_349)
        x_350 = self.seq350(x_349)
        x_351 = x_350.mean([2, 3], keepdim=True)
        x_351 = self.seq351(x_351)
        x_352 = torch.mul(x_351, x_350)
        x_353 = x_352 + x_349
        x_353 = self.seq353(x_353)
        x_354 = self.seq354(x_353)
        x_355 = x_354.mean([2, 3], keepdim=True)
        x_355 = self.seq355(x_355)
        x_356 = torch.mul(x_355, x_354)
        x_357 = x_356 + x_353
        x_357 = self.seq357(x_357)
        x_358 = self.seq358(x_357)
        x_359 = x_358.mean([2, 3], keepdim=True)
        x_359 = self.seq359(x_359)
        x_360 = torch.mul(x_359, x_358)
        x_361 = x_360 + x_357
        x_361 = self.seq361(x_361)
        return x_361
