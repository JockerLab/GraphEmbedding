import json
import math
import os
import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset
from seq_to_seq.encoder import EncoderRNN
from seq_to_seq.decoder import DecoderRNN
from seq_to_seq.seq2seq import Seq2Seq


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
teacher_forcing_ratio = 0.5
SOS_token = torch.tensor([[[-1.] * NODE_EMBEDDING_DIMENSION]])
EOS_token = torch.tensor([[[1.] * NODE_EMBEDDING_DIMENSION]])


def train(loader, model, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(loader):
        # data: (batch_size, embedding_dim, node_dim)
        source = target = data
        # source: (batch_size, embedding_dim, node_dim)
        # target: (batch_size, embedding_dim, node_dim)

        optimizer.zero_grad()

        source = torch.cat([SOS_token, source, EOS_token], 1)
        target = torch.cat([SOS_token, target, EOS_token], 1)

        output, hidden, cell = model(source, target)
        # output: (embedding_dim, batch_size, node_dim)

        final_embedding = torch.cat([hidden.view(-1), cell.view(-1)])

        output = output[1:].view(-1, output.shape[2])
        target = target[:, 1:, :].view(-1, target.shape[2])
        # output = (batch_size * (embedding_dim - 1), node_dim)
        # target: (batch_size * (embedding_dim - 1), node_dim)

        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(loader, model, criterion):
    model.eval()
    epoch_loss = 0
    for i, data in enumerate(loader):
        # data: (batch_size, embedding_dim, node_dim)
        source = target = data
        # source: (batch_size, embedding_dim, node_dim)
        # target: (batch_size, embedding_dim, node_dim)

        source = torch.cat([SOS_token, source, EOS_token], 1)
        target = torch.cat([SOS_token, target, EOS_token], 1)

        output, hidden, cell = model(source, target, 0)
        # output: (embedding_dim, batch_size, node_dim)

        output = output[1:].view(-1, output.shape[2])
        target = target[:, 1:, :].view(-1, target.shape[2])
        # output = (batch_size * (embedding_dim - 1), node_dim)
        # target: (batch_size * (embedding_dim - 1), node_dim)

        loss = criterion(output, target)
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    hidden_size = 256
    num_layers = 1  # 2
    dropout = 0  # 0.5
    N_EPOCHS = 20
    best_valid_loss = float('inf')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EmbeddingDataset(
        root='../data/embeddings/',
        transform=torch.tensor,
        normalize=True
    )
    loader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=False)

    encoder = EncoderRNN(NODE_EMBEDDING_DIMENSION, hidden_size, num_layers, dropout).to(device)
    decoder = DecoderRNN(NODE_EMBEDDING_DIMENSION, hidden_size, num_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss(reduction='sum')

    test_input = []
    test_loss_last = 0
    with open(f'../data/embeddings/test.json', 'r') as f:
        test_input = json.load(f)
    vals = []
    if os.path.isfile('../data/embeddings/min_max.json'):
        with open(f'../data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]
        for i in range(len(test_input)):
            for j in range(len(test_input[i])):
                if max_vals[j] == min_vals[j]:
                    test_input[i][j] = max_vals[j]
                else:
                    test_input[i][j] = 2 * (test_input[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]) - 1
    test_len = len(test_input)
    test_input = torch.tensor(test_input).view(1, test_len, -1)
    test_input = torch.cat([SOS_token, test_input, EOS_token], 1)

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = 0
        eval_loss = 0
        train_loss += train(loader, model, optimizer, criterion, clip=1)
        eval_loss += evaluate(loader, model, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        test_embedding = model.encode(test_input)
        test_output = model.decode(test_embedding, NODE_EMBEDDING_DIMENSION, SOS_token, EOS_token.view(1, -1), num_layers=num_layers, batch_size=1)
        if test_output.size(dim=0) > test_len + 1:
            print('Incorrect test output size!')
        test_loss = criterion(test_input[:, 1:, :].view(test_len + 1, -1), test_output[:(test_len + 1), :])
        test_loss_change = test_loss - test_loss_last
        test_loss_last = test_loss

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            torch.save(model.state_dict(), 'seq2seq_model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')
