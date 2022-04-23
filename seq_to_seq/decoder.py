import torch
from torch import nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        # output_size = node_dim
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(output_size + hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(output_size + hidden_size * 2, output_size)

    def forward(self, input, hidden, context):
        # input: (batch_size, node_dim)
        # context: (1, batch_size, node_dim)

        input = self.dropout(input.unsqueeze(0))
        # input: (1, batch_size, node_dim)

        emb_con = torch.cat((input, context), dim=2)
        # emb_con: (1, batch_size, node_dim + hidden_dim)

        output, hidden = self.gru(emb_con, hidden)
        # output: (1, batch_size, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)

        output = torch.cat((input.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)

        # output: (batch_size, node_dim + hidden_dim * 2)

        output = torch.tanh(output)
        output = self.fc(output)
        # output: (batch_size, output_size)

        return output, hidden
