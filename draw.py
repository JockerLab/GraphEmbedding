import json

import matplotlib.pyplot as plt
from graph import attribute_parameters


# TODO: resnets
plt.title('Обучение моделей ResNet18')
plt.xlabel('Номер итерации')
plt.ylabel('Потери при обучении')

iters = [i + 1 for i in range(20)]
convert_train = []
convert_test = []
with open('Experiments/resnets/convert_losses.json', 'r') as f:
    vals = json.load(f)
    convert_train = vals['train']
    convert_test = vals['test']
naive_train = []
naive_test = []
with open('Experiments/resnets/naive_losses.json', 'r') as f:
    vals = json.load(f)
    naive_train = vals['train']
    naive_test = vals['test']
origin_train = []
origin_test = []
with open('Experiments/resnets/origin/origin_losses.json', 'r') as f:
    vals = json.load(f)
    origin_train = vals['train']
    origin_test = vals['test']

plt.plot(iters, convert_train, 'r')
plt.plot(iters, convert_test, 'r--')
plt.plot(iters, naive_train, 'g')
plt.plot(iters, naive_test, 'g--')
plt.plot(iters, origin_train, 'b')
plt.plot(iters, origin_test, 'b--')

plt.legend(['Train Convert', 'Test Convert', 'Train Naive', 'Test Naive', 'Train Original', 'Test Original'], loc=1)
# plt.show()
plt.savefig('Experiments/resnets/1.png')


# TODO: DVAES
# plt.xlabel('Номер итерации')
# plt.ylabel('Потери при обучении')
#
# iters = [i + 1 for i in range(10)]
# dvae_recon = []
# dvae_kl = []
# with open('Experiments/DVAE/train_loss.txt', 'r') as f:
#     for n, line in enumerate(f, 1):
#         line = list(map(float, line.rstrip('\n').split(' ')))
#         dvae_recon.append(line[1])
#         dvae_kl.append(line[2])
# dvae_emb_recon = []
# dvae_emb_kl = []
# with open('Experiments/DVAE_EMB/train_loss.txt', 'r') as f:
#     for n, line in enumerate(f, 1):
#         line = list(map(float, line.rstrip('\n').split(' ')))
#         dvae_emb_recon.append(line[1])
#         dvae_emb_kl.append(line[2])
# with open('Experiments/MY/attributes_losses.json', 'r') as f:
#     vals = json.load(f)
#     my_recon = vals['recon']
#     my_kl = vals['kl']
# with open('Experiments/MY/edge_losses.json', 'r') as f:
#     vals = json.load(f)
#     for i in range(10):
#         my_recon[i] += vals['recon'][i]
#         my_kl[i] += vals['kl'][i]
#
# plt.plot(iters, dvae_recon, 'r')
# plt.plot(iters, dvae_kl, 'r--')
# plt.plot(iters, dvae_emb_recon, 'g')
# plt.plot(iters, dvae_emb_kl, 'g--')
# plt.plot(iters, my_recon, 'b')
# plt.plot(iters, my_kl, 'b--')
#
# plt.legend(['Reconstruct DVAE', 'KL DVAE', 'Reconstruct DVAE-EMB', 'KL DVAE-EMB', 'Reconstruct Model', 'KL Model'], loc=1)
# plt.savefig('Experiments/DVAES.png')

# TODO
#
# plt.title('Обучение модели VAE')
# plt.xlabel('Номер итерации')
# plt.ylabel('Потери при обучении')
#
# iters = [i for i in range(50)]
#
# with open('seq_to_seq/attributes_only/losses.json', 'r') as f:
#     vals = json.load(f)
#     attrs_train = vals['train']
#     attrs_test = vals['eval']
# with open('seq_to_seq/attributes_only/losses.json', 'r') as f:
#     vals = json.load(f)
#     edges_train = vals['train']
#     edges_test = vals['eval']
#
# plt.plot(iters, attrs_train, 'r')
# plt.plot(iters, attrs_test, 'r--')
# plt.plot(iters, edges_train, 'g')
# plt.plot(iters, edges_test, 'g--')
#
# plt.legend(['Train attributes', 'Test attributes', 'Train edges', 'Test edges'], loc=1)
# plt.savefig('Experiments/AE.png')