import torch
from torch import nn
import torch.nn.functional as F


class DecoderRNNEdges(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout, max_node):
        # output_size = node_dim
        super(DecoderRNNEdges, self).__init__()
        self.output_size = max_node
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_node, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, max_node)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input: (batch_size, node_dim)
        # context: (1, batch_size, node_dim)

        input = input.unsqueeze(0)
        input = self.dropout(self.embedding(input).unsqueeze(0))
        # input: (1, batch_size, node_dim)

        output, hidden = self.gru(input, hidden)
        # output: (1, batch_size, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)

        output = self.softmax(self.fc(output.squeeze(0)))
        # output: (batch_size, output_size)

        return output, hidden