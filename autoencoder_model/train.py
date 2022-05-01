import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset
from autoencoder_model.autoencoder import AE

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_NODE = 1_500
teacher_forcing_ratio = 0.5  # 1
SOS_token = torch.tensor([[[-1.] * NODE_EMBEDDING_DIMENSION]])
EOS_token = torch.tensor([[[1.] * NODE_EMBEDDING_DIMENSION]])


def train(loader, model, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(loader):
        # data: (batch_size, embedding_dim, node_dim)

        optimizer.zero_grad()

        for j in range(len(data[0])):
            output = model(data[0][j][:ATTRIBUTES_POS_COUNT])
            loss = ((data[0][j][:ATTRIBUTES_POS_COUNT] - output) ** 2).sum() + model.enc_kl
            # loss = criterion(output, data[0][j][:ATTRIBUTES_POS_COUNT])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        for j in range(MAX_NODE - len(data[0])):
            test_data = torch.tensor([-1.] * ATTRIBUTES_POS_COUNT)
            output = model(test_data)
            loss = ((test_data - output) ** 2).sum() + model.enc_kl
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(loader, model, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            # data: (batch_size, embedding_dim, node_dim)

            for j in range(len(data[0])):
                output = model(data[0][j][:ATTRIBUTES_POS_COUNT])
                loss = ((data[0][j][:ATTRIBUTES_POS_COUNT] - output) ** 2).sum() + model.enc_kl
                # loss = criterion(output, data[0][j][:ATTRIBUTES_POS_COUNT])
                epoch_loss += loss.item()
            for j in range(MAX_NODE - len(data[0])):
                test_data = torch.tensor([-1.] * ATTRIBUTES_POS_COUNT)
                output = model(test_data)
                loss = ((test_data - output) ** 2).sum() + model.enc_kl
                epoch_loss += loss.item()

    return epoch_loss / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    num_layers = 1  # 2
    N_EPOCHS = 20
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
                    test_input[i][j] = float(max_vals[j])
                elif test_input[i][j] == -1:
                    test_input[i][j] = -1.
                else:
                    test_input[i][j] = (test_input[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j])
    test_len = len(test_input)
    test_input = torch.tensor(test_input).view(1, test_len, -1)
    test_input = torch.cat([SOS_token, test_input[:, :, :NODE_EMBEDDING_DIMENSION]], 1)

    dropouts = [0]
    hidden_sizes = [2048]  # [128, 256, 512, 1024]
    optimizers = [optim.Adam]  # [optim.Adam, optim.AdamW, optim.Adamax]
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
                        # print(f'Iter {iter_num} is processing')
                        with open(f'logs.txt', 'w') as f:
                            f.write(f'{hidden_size}, {dropout}, {optim_func}, {lr}, {reduction}\n')
                        model = AE(shapes=[ATTRIBUTES_POS_COUNT, 40, 30]).to(device)
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
                            train_loss += train(loader, model, optimizer, criterion)
                            eval_loss += evaluate(loader, model, criterion)
                            end_time = time.time()

                            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                            test_loss = 0
                            for j in range(len(test_input[0])):
                                test_embedding = model.encode(test_input[0][j][:ATTRIBUTES_POS_COUNT])
                                test_output = model.decode(test_embedding)
                                test_loss += ((test_input[0][j][:ATTRIBUTES_POS_COUNT] - test_output) ** 2).sum() + model.enc_kl
                                # test_loss += criterion(test_input[0][j][:ATTRIBUTES_POS_COUNT], test_output)
                            for j in range(MAX_NODE - len(test_input[0])):
                                test_data = torch.tensor([-1.] * ATTRIBUTES_POS_COUNT)
                                test_embedding = model.encode(test_data)
                                test_output = model.decode(test_embedding)
                                test_loss += ((test_data - test_output) ** 2).sum() + model.enc_kl

                            test_loss_change = test_loss - test_loss_last
                            test_loss_last = test_loss

                            train_losses.append(float(train_loss))
                            eval_losses.append(float(eval_loss))
                            test_losses.append(float(test_loss))
                            with open(f'losses.json', 'w') as f:
                                f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

                            if eval_loss < best_valid_loss:
                                best_valid_loss = eval_loss
                                torch.save(model.state_dict(), f'autoencoder_model_{iter_num}.pt')
                            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                            print(f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}')
                            with open(f'logs.txt', 'a') as f:
                                f.write(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
                                f.write(
                                    f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}\n')
                        if best_valid_loss < best_valid_loss_total_sum and reduction == 'sum':
                            best_valid_loss_total_sum = best_valid_loss
                            torch.save(model.state_dict(),
                                       f'autoencoder_best_sum_{iter_num}.pt')
                        if best_valid_loss < best_valid_loss_total_mean and reduction == 'mean':
                            best_valid_loss_total_mean = best_valid_loss
                            torch.save(model.state_dict(),
                                       f'autoencoder_best_mean_{iter_num}.pt')
                        # print('\n\n-------------------------------------\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')
