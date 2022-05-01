import random

import torch
from torch import nn
from graph import MAX_NODE


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, data_len, teacher_forcing_ratio=0.5):
        # source: (batch_size, embedding_dim, node_dim)
        # target: (batch_size, embedding_dim, node_dim)

        target_output_size = self.decoder.output_size
        outputs = torch.zeros(data_len + 1, target_output_size)

        hidden = self.encoder(source[:(data_len + 1)])

        input = target[0]  # SOS_token
        # input: (batch_size, node_dim)

        for i in range(1, data_len + 1):
            output, hidden = self.decoder(input, hidden)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = target[i] if random.random() < teacher_forcing_ratio else output.argmax(1)[0]

        return outputs

    #TODO:  data_len, concat
    def encode(self, source, data_len):
        # source: (batch_size, embedding_dim, node_dim)
        return data_len, self.encoder(source[:(data_len + 1)])

    def decode(self, embedding, node_embedding_size, SOS_token, data_len, batch_size=1):
        hidden = embedding
        input = SOS_token[0]
        target_output_size = self.decoder.output_size
        outputs = torch.zeros(data_len + 1, target_output_size)
        for i in range(1, data_len + 1):
            output, hidden = self.decoder(input, hidden)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = output.argmax(1)[0]

        return outputs