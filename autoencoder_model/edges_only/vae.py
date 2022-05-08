import numpy as np
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from graph import ATTRIBUTES_POS_COUNT, node_to_ops, attribute_parameters


class VAE(nn.Module):
    def __init__(self, shapes, negative_slope=0.01):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(shapes[0], shapes[1])),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(shapes[1], shapes[2])),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
        ]))
        self.fc_mu = nn.Linear(shapes[2], shapes[3])
        self.fc_var = nn.Linear(shapes[2], shapes[3])
        self.decoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(shapes[3], shapes[2])),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(shapes[2], shapes[1])),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer3', nn.Linear(shapes[1], shapes[0])),
            ('sigmoid', nn.Sigmoid()),
        ]))
        self._init_weights()

    def forward(self, x, training=True):
        if training:
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_var(h)
            z = self._reparameterize(mu, logvar)
            y = self.decoder(z)
            return y, mu, logvar
        else:
            z = self.represent(x)
            y = self.decoder(z)
            return y

    def represent(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self._reparameterize(mu, logvar)
        return z

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).type_as(mu)
        z = mu + std * esp
        return z

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
