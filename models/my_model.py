from torch import nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[11, 11], stride=[4, 4], padding=[2, 2], dilation=[1, 1], groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
        )
