import random

import torch
from torch import nn
import torch.nn.functional as F
from graph import MAX_NODE


class AE(nn.Module):
    def __init__(self, **kwargs):
        # params: input_shape == output_shape
        # input_shape % hidden_shape == 0
        super(AE, self).__init__()
        self.enc_linear1 = nn.Linear(in_features=kwargs["shapes"][0], out_features=kwargs["shapes"][1])
        self.enc_linear2 = nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][2])
        self.enc_linear3 = nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][2])
        self.enc_N = torch.distributions.Normal(0, 1)
        self.enc_kl = 0


        self.decoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][1]),
            nn.Hardtanh(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        x = F.relu(self.enc_linear1(x))
        mu = self.enc_linear2(x)
        sigma = torch.exp(self.enc_linear3(x))
        z = mu + sigma * self.enc_N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z

    def decode(self, x):
        return self.decoder(x)

