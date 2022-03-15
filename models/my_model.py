from torch import nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq3 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq5 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq7 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq9 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq11 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq13 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq15 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq17 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq19 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq20 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq21 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq22 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq23 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq24 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq25 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq26 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq27 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq28 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq29 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq30 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq31 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq32 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq33 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq34 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq35 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq36 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq37 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq38 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq39 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq40 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq41 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq42 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq43 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq44 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq45 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq46 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq47 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq48 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq49 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq50 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq51 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq52 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq53 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq54 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq55 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq56 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq57 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq58 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq59 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq60 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq61 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq62 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq63 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq64 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq65 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
        )
        self.seq66 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq67 = nn.Sequential(
            #  Unsupportable layer type: {node['op']},
            nn.ReLU(),
            #  Unsupportable layer type: {node['op']},
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1000),
        )
        self.seq68 = nn.Sequential(
            nn.Conv2d(in_channels=1000, out_channels=2048, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq69 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq70 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
        self.seq71 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1),
            #  Unsupportable layer type: {node['op']},
        )
