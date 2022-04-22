import json
import os

from graph import reversed_attributes, ATTRIBUTES_POS_COUNT

NODE_EMBEDDING_DIMENSION = 113


def normalize_dataset(dataset):
    # R -> [-1; 1]
    min_vals = [float('inf')] * NODE_EMBEDDING_DIMENSION
    max_vals = [float('-inf')] * NODE_EMBEDDING_DIMENSION
    vals = []
    if os.path.isfile('../data/embeddings/min_max.json'):
        with open(f'../data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]
    else:
        for emb in range(len(dataset)):
            for i in range(len(dataset[emb])):
                for pos, attr in reversed_attributes.items():
                    if 'len' not in attr:
                        n = NODE_EMBEDDING_DIMENSION - pos
                    else:
                        n = attr['len']
                    for j in range(n):
                        min_vals[pos + j] = min(min_vals[pos + j], dataset[emb][i][pos + j])
                        if 'len' not in attr:
                            max_vals[pos + j] = attr['range'][1]
                            continue
                        if attr['range'][1] < float('inf'):
                            max_vals[pos + j] = attr['range'][1]
                        else:
                            max_vals[pos + j] = max(max_vals[pos + j], dataset[emb][i][pos + j])
        with open(f'../data/embeddings/min_max.json', 'w') as f:
            f.write(json.dumps([min_vals, max_vals]))
    for emb in range(len(dataset)):
        for i in range(len(dataset[emb])):
            for j in range(len(dataset[emb][i])):
                if max_vals[j] == min_vals[j]:
                    dataset[emb][i][j] = max_vals[j]
                else:
                    dataset[emb][i][j] = 2 * (dataset[emb][i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]) - 1
    return dataset


def denormalize_vector(x):
    # TODO: [-1; 1] -> R
    return x
