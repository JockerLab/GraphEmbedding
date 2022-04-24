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
ATTRIBUTES_POS_COUNT = 50
teacher_forcing_ratio = 0.5  # 1
SOS_token = torch.tensor([[[-1.] * NODE_EMBEDDING_DIMENSION]])
# EOS_token = torch.tensor([[[1.] * NODE_EMBEDDING_DIMENSION]])
# cur_dfs = []
#
#
# def get_dfs(v, data):
#     cur_dfs.append(v)
#     get_dfs(data[0][v][ATTRIBUTES_POS_COUNT], data)


def train(loader, model, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(loader):
        # data: (batch_size, embedding_dim, node_dim)
        # global cur_dfs
        # cur_dfs = []
        # get_dfs(0, data)
        source = target = data  # [:ATTRIBUTES_POS_COUNT]
        # source: (batch_size, embedding_dim, node_dim)
        # target: (batch_size, embedding_dim, node_dim)

        optimizer.zero_grad()

        source = torch.cat([SOS_token, source], 1)
        target = torch.cat([SOS_token, target], 1)

        output, hidden, cell = model(source, target)
        # output: (embedding_dim, batch_size, node_dim)

        # final_embedding = hidden

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
    with torch.no_grad():
        for i, data in enumerate(loader):
            # data: (batch_size, embedding_dim, node_dim)
            source = target = data
            # source: (batch_size, embedding_dim, node_dim)
            # target: (batch_size, embedding_dim, node_dim)

            source = torch.cat([SOS_token, source], 1)
            target = torch.cat([SOS_token, target], 1)

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
    num_layers = 1  # 2
    N_EPOCHS = 10
    best_valid_loss_total_mean = float('inf')
    best_valid_loss_total_sum = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EmbeddingDataset(
        root='../data/embeddings/',
        transform=torch.tensor,
        normalize=True
    )
    loader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=False)

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
    test_input = torch.cat([SOS_token, test_input], 1)

    dropouts = [0, 0.5]
    hidden_sizes = [128, 256]  # [128, 256, 512, 1024]
    optimizers = [optim.Adamax]  # [optim.Adam, optim.AdamW, optim.Adamax]
    learning_rates = [1e-3]  # [1e-2, 1e-3]
    reductions = ['mean', 'sum']  # ['mean', 'sum']
    iter_num = 0

    for hidden_size in hidden_sizes:
        for dropout in dropouts:
            for optim_func in optimizers:
                for lr in learning_rates:
                    for reduction in reductions:
                        iter_num += 1
                        print(f'Iter {iter_num} is processing')
                        print(f'{hidden_size}, {dropout}, {optim_func}, {lr}, {reduction}')
                        encoder = EncoderRNN(NODE_EMBEDDING_DIMENSION, hidden_size, num_layers, dropout).to(device)
                        decoder = DecoderRNN(NODE_EMBEDDING_DIMENSION, hidden_size, num_layers, dropout).to(device)
                        model = Seq2Seq(encoder, decoder).to(device)
                        optimizer = optim_func(model.parameters(), lr=lr)
                        criterion = nn.MSELoss(reduction=reduction)

                        best_valid_loss = float('inf')
                        train_loss = 0
                        eval_loss = 0
                        test_loss_change = 0
                        test_loss_last = 0
                        for epoch in range(N_EPOCHS):
                            start_time = time.time()
                            train_loss = 0
                            eval_loss = 0
                            train_loss += train(loader, model, optimizer, criterion, clip=1)
                            eval_loss += evaluate(loader, model, criterion)
                            end_time = time.time()

                            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                            test_embedding = model.encode(test_input)
                            test_embedding_len = int(test_embedding[0])
                            test_output = model.decode(test_embedding, NODE_EMBEDDING_DIMENSION, SOS_token, test_embedding_len)
                            test_loss = criterion(test_input[0, 1:, :], test_output[1:, :])
                            test_loss_change = test_loss - test_loss_last
                            test_loss_last = test_loss

                            if eval_loss < best_valid_loss:
                                best_valid_loss = eval_loss
                                torch.save(model.state_dict(), f'seq2seq_model_{iter_num}.pt')

                            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                            print(f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}')
                        if best_valid_loss < best_valid_loss_total_sum and reduction == 'sum':
                            best_valid_loss_total_sum = best_valid_loss
                            torch.save(model.state_dict(),
                                       f'seq2seq_best_sum_{iter_num}.pt')
                        if best_valid_loss < best_valid_loss_total_mean and reduction == 'mean':
                            best_valid_loss_total_mean = best_valid_loss
                            torch.save(model.state_dict(),
                                       f'seq2seq_best_mean_{iter_num}.pt')
                        print('\n\n-------------------------------------\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')
