import random

import torch
from torch import nn
from graph import MAX_NODE


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

        hidden = self.encoder(source)
        context_hidden = hidden

        input = target[:, 0, :]  # SOS_token
        # input: (batch_size, node_dim)

        for i in range(1, target_len):
            output, hidden = self.decoder(input, hidden, context_hidden)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = target[:, i, :] if random.random() < teacher_forcing_ratio else output

        return outputs, context_hidden

    def encode(self, source):
        # source: (batch_size, embedding_dim, node_dim)
        hidden = torch.cat([torch.tensor(source.size(1)).view(1, 1, -1), self.encoder(source).view(1, 1, -1)], 2)
        return hidden

    def decode(self, embedding, node_embedding_size, SOS_token, embedding_len, batch_size=1):
        hidden = embedding
        context = embedding
        input = SOS_token.view(1, -1)
        outputs = torch.zeros(embedding_len, batch_size, node_embedding_size)
        for i in range(embedding_len):
            output, hidden = self.decoder(input, hidden, context)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = output

        return outputs

