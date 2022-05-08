import random

import torch
from torch import nn
import torch.nn.functional as F
from graph import MAX_NODE, ATTRIBUTES_POS_COUNT, attribute_parameters, node_to_ops


class VAE(nn.Module):
    def __init__(self, **kwargs):
        # params: input_shape == output_shape
        # input_shape % hidden_shape == 0
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][0], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][2]),
            nn.LeakyReLU()
        )
        self.hidden2mu = nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][2])
        self.hidden2log_var = nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][2])
        self.alpha = 1
        self.decoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][0]),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(1, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return mu, log_var, self.decoder(hidden)

    def reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        return self.decoder(x)

    def training_step(self, inputs):
        mu, log_var, x_out = self.forward(inputs)
        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss(reduction='sum')
        recon_loss = recon_loss_criterion(inputs.view(-1), x_out.view(-1))
        loss = recon_loss * self.alpha + kl_loss
        return loss, x_out.view(-1)

    def one_encode(self, inputs):
        x = inputs.view(1, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return hidden