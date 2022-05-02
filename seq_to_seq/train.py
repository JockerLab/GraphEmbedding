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
MAX_N = 3000
teacher_forcing_ratio = 0.5  # 1
SOS_token = torch.tensor([0])
# EOS_token = torch.tensor([[[1.] * (NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT)]])


def fill_matrix(matrix):
    for i in range(MAX_N - matrix.size(1)):
        row = torch.tensor([[[-1.] * ATTRIBUTES_POS_COUNT]])
        matrix = torch.cat([matrix, row], dim=1)
    return matrix


def train(loader, model, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(loader):
        # data: (batch_size, embedding_dim, node_dim)

        for row in data[0]:
            data_len = int(row[ATTRIBUTES_POS_COUNT])
            if data_len <= 0:
                continue
            source = target = row[(ATTRIBUTES_POS_COUNT + 1):]
            # source: (batch_size, embedding_dim, node_dim)
            # target: (batch_size, embedding_dim, node_dim)
            source = torch.cat([SOS_token, source])
            target = torch.cat([SOS_token, target])
            output = model(source, target, data_len)
            # output: (embedding_dim, batch_size, node_dim)
            output = output[1:, :]
            target = target[1:(data_len + 1)]
            # output = (batch_size * (embedding_dim - 1), node_dim)
            # target: (batch_size * (embedding_dim - 1), node_dim)
            loss = criterion(output, target)
            optimizer.zero_grad()
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

            for row in data[0]:
                data_len = int(row[ATTRIBUTES_POS_COUNT])
                if data_len <= 0:
                    continue
                source = target = row[(ATTRIBUTES_POS_COUNT + 1):]
                # source: (batch_size, embedding_dim, node_dim)
                # target: (batch_size, embedding_dim, node_dim)
                source = torch.cat([SOS_token, source])
                target = torch.cat([SOS_token, target])
                output = model(source, target, data_len)
                # output: (embedding_dim, batch_size, node_dim)
                output = output[1:, :]
                target = target[1:(data_len + 1)]
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
    N_EPOCHS = 5
    best_valid_loss_total_mean = float('inf')
    best_valid_loss_total_sum = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EmbeddingDataset(
        root='../data/embeddings/',
        transform=torch.LongTensor,
        normalize=False
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
            for j in range(ATTRIBUTES_POS_COUNT):
                if max_vals[j] == min_vals[j]:
                    test_input[i][j] = max_vals[j]
                else:
                    test_input[i][j] = 2 * (test_input[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]) - 1
    test_len = len(test_input)
    test_input = torch.LongTensor(test_input).unsqueeze(0)
    # test_input = torch.cat([SOS_token, test_input[0, 0, (ATTRIBUTES_POS_COUNT + 1):]], 1)

    dropouts = [0]
    hidden_sizes = [20]  # [128, 256, 512, 1024]
    optimizers = [optim.Adamax]  # [optim.Adam, optim.AdamW, optim.Adamax]
    learning_rates = [1e-3]  # [1e-2, 1e-3]
    reductions = ['sum']  # ['mean', 'sum']
    iter_num = 0
    train_losses = []
    eval_losses = []
    test_losses = []

    for hidden_size in hidden_sizes:
        for dropout in dropouts:
            for optim_func in optimizers:
                for lr in learning_rates:
                    for reduction in reductions:
                        iter_num += 1
                        print(f'Iter {iter_num} is processing')
                        print(f'{hidden_size}, {dropout}, {optim_func}, {lr}, {reduction}')
                        encoder = EncoderRNN(NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT - 1, hidden_size, num_layers, dropout, MAX_N).to(device)
                        decoder = DecoderRNN(NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT - 1, hidden_size, num_layers, dropout, MAX_N).to(device)
                        model = Seq2Seq(encoder, decoder).to(device)
                        optimizer = optim_func(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()  # NLLLoss

                        best_valid_loss = float('inf')
                        train_loss = 0
                        test_loss = 0
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

                            for row in test_input[0]:
                                data_len = int(row[ATTRIBUTES_POS_COUNT])
                                if data_len <= 0:
                                    continue
                                test_in = row[(ATTRIBUTES_POS_COUNT + 1):]
                                test_in = torch.cat([SOS_token, test_in])
                                _, test_embedding = model.encode(test_in, data_len)
                                test_output = model.decode(test_embedding, NODE_EMBEDDING_DIMENSION, SOS_token, data_len)
                                test_loss += criterion(test_output[1:, :], test_in[1:(data_len + 1)])
                                test_loss_change = test_loss - test_loss_last
                                test_loss_last = test_loss

                            train_losses.append(float(train_loss))
                            eval_losses.append(float(eval_loss))
                            test_losses.append(float(test_loss))
                            with open(f'losses.json', 'w') as f:
                                f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

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