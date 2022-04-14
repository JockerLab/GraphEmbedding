import random

import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: (batch_size, embedding_dim, node_dim)
        # target: (batch_size, embedding_dim, node_dim)

        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_output_size = self.decoder.output_size
        outputs = torch.zeros(target_len, batch_size, target_output_size)

        hidden, cell = self.encoder(source)
        context_hidden, context_cell = hidden, cell

        input = target[:, 0, :]  # SOS_token
        # input: (batch_size, node_dim)

        for i in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)
            # cell: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = target[:, i, :] if random.random() < teacher_forcing_ratio else output

        return outputs, context_hidden, context_cell
