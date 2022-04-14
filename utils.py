import json

NODE_EMBEDDING_DIMENSION = 100


def normalize_dataset(dataset):
    # R -> [-1; 1]
    min_vals = [float('inf')] * NODE_EMBEDDING_DIMENSION
    max_vals = [float('-inf')] * NODE_EMBEDDING_DIMENSION
    for emb in range(len(dataset)):
        for i in range(len(dataset[emb])):
            for j in range(len(dataset[emb][i])):
                min_vals[j] = min(min_vals[j], dataset[emb][i][j])
                max_vals[j] = max(max_vals[j], dataset[emb][i][j])
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
