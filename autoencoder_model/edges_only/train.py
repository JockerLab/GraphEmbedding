import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import optim, float32
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import EmbeddingDataset
from autoencoder_model.edges_only.vae import VAE
from graph import attribute_parameters, node_to_ops


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_NODE = 3_000
train_losses = []
eval_losses = []
test_losses = []
min_vals = []
max_vals = []


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(loader, model, optimizer):
    model.train()
    epoch_loss = 0
    cnt = 0
    # for i, data in enumerate(loader):
    #     # data: (batch_size, embedding_dim, node_dim)
    #
    #     for row in data:
    #         data_len = int(row[ATTRIBUTES_POS_COUNT])
    #         if data_len <= 0:
    #             continue
    #         source = torch.tensor(row[(ATTRIBUTES_POS_COUNT + 1):], dtype=float32).view(1, -1)
    #         for j in range(len(source[0])):
    #             if source[0][j] == -1.:
    #                 source[0][j] = 0.
    #             else:
    #                 source[0][j] = float(source[0][j]) / 3000.
    #         cnt += 1
    #         optimizer.zero_grad()
    #         outputs, mu, logvar = model(source)
    #         loss = loss_fn(outputs, source, mu, logvar)
    #         loss.backward()
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    #         optimizer.step()
    #         epoch_loss += loss.item()

    for step in range(5000):
        data_len = random.randint(1, 62)
        row = [0.] * 62
        for j in range(data_len):
            row[j] = float(random.randint(1, 2999)) / 3000.
        row = torch.tensor(row, dtype=float32).view(1, -1)
        source = row
        cnt += 1
        optimizer.zero_grad()
        outputs, mu, logvar = model(source)
        loss = loss_fn(outputs, source, mu, logvar)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / cnt


def evaluate(loader, model):
    model.eval()
    epoch_loss = 0
    cnt = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            # data: (batch_size, embedding_dim, node_dim)

            for row in data:
                data_len = int(row[ATTRIBUTES_POS_COUNT])
                if data_len <= 0:
                    continue
                source = torch.tensor(row[(ATTRIBUTES_POS_COUNT + 1):], dtype=float32).view(1, -1)
                for j in range(len(source[0])):
                    if source[0][j] == -1.:
                        source[0][j] = 0.
                    else:
                        source[0][j] = float(source[0][j]) / 3000.
                cnt += 1
                outputs, mu, logvar = model(source)
                loss = loss_fn(outputs, source, mu, logvar)
                epoch_loss += loss.item()

    return epoch_loss / cnt


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # https://github__com.teameo.ca/thuyngch/Variational-Autoencoder-PyTorch/blob/master/models/vae.py
    num_layers = 1
    N_EPOCHS = 30
    dropout = 0
    rnn_hidden_size = 30
    hidden_size = 64
    lr = 1e-3

    # Local
    # paths = ['../../', '']
    # CTlab
    paths = ['/nfs/home/vshaldin/embeddings/', '/nfs/home/vshaldin/embeddings/autoencoder_model/all_attributes_2/']

    best_valid_loss_total_mean = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EmbeddingDataset(
        root=f'{paths[0]}data/embeddings/',
        train=True,
        normalize=True
    )
    test_dataset = EmbeddingDataset(
        root=f'{paths[0]}data/embeddings/',
        train=False,
        normalize=True
    )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    model = VAE(shapes=[62, 45, 30, 20]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    with open(f'{paths[0]}data/embeddings/test.json', 'r') as f:
        test_input = json.load(f)
    vals = []
    if os.path.isfile(f'{paths[0]}data/embeddings/min_max.json'):
        with open(f'{paths[0]}data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        for i in range(len(test_input)):
            for j in range(NODE_EMBEDDING_DIMENSION):
                if j > ATTRIBUTES_POS_COUNT:
                    if test_input[i][j] == -1:
                        test_input[i][j] = 0.
                    else:
                        test_input[i][j] /= 3000.
    test_len = len(test_input)
    test_row = torch.tensor(test_input[0][(ATTRIBUTES_POS_COUNT + 1):])

    best_valid_loss = float('inf')
    train_loss = 0
    eval_loss = 0
    test_loss_change = 0
    test_loss_last = 0

    print(f'Testing:\n{test_row.tolist()}\n')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = 0
        eval_loss = 0
        test_loss = 0
        train_loss += train(train_loader, model, optimizer)
        eval_loss += evaluate(test_loader, model)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        test_row = test_row.view(1, -1)
        test_outputs, mu, logvar = model(test_row)
        test_loss = loss_fn(test_outputs, test_row, mu, logvar)
        test_loss_change = test_loss - test_loss_last
        test_loss_last = test_loss
        print(f'After epoch {epoch + 1}:\n{(test_outputs[0] - test_row[0]).tolist()}\n')

        train_losses.append(float(train_loss))
        eval_losses.append(float(eval_loss))
        test_losses.append(float(test_loss))

        with open(f'{paths[1]}losses.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            torch.save(model.state_dict(), f'{paths[1]}vae_model.pt')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}\n')
