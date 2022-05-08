import json
import os

from graph import reversed_attributes, attribute_parameters, ATTRIBUTES_POS_COUNT

NODE_EMBEDDING_DIMENSION = 113


def normalize_dataset(dataset):
    min_vals = [-1.] * NODE_EMBEDDING_DIMENSION
    max_vals = [float('-inf')] * NODE_EMBEDDING_DIMENSION
    vals = []
    if os.path.isfile('../../data/embeddings/min_max.json'):
        with open(f'../../data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]
    for emb in range(len(dataset)):
        for i in range(len(dataset[emb])):
            for j in range(NODE_EMBEDDING_DIMENSION):
                if j >= ATTRIBUTES_POS_COUNT:
                    continue
                if dataset[emb][i][j] == -1 or max_vals[j] == -1:
                    dataset[emb][i][j] = 0.
                else:
                    dataset[emb][i][j] = (dataset[emb][i][j] + 1.) / (max_vals[j] + 1.)
    return dataset

