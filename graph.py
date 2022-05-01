import sys

import networkx as nx
from torchvision.models import resnet101, densenet201, vgg19_bn, mnasnet1_3, squeezenet1_1, resnet50, \
    inception_v3

from models.original.original_alexnet import AlexNet
from models.original.original_unet import UNet
import hiddenlayer as hl
import torch
import pandas
import json
from functools import reduce

ATTRIBUTES_POS_COUNT = 50
NODE_EMBEDDING_DIMENSION = 113
NONE_REPLACEMENT = -1
MAX_NODE = 1_000  # for 200 layers in network

node_to_ops = {
    "Conv": 0,
    "LeakyRelu": 1,
    "MaxPool": 2,
    "Flatten": 3,
    "Linear": 4,
    "Sigmoid": 5,
    "BatchNorm": 6,
    "Relu": 7,
    "Add": 8,
    "GlobalAveragePool": 9,
    "AveragePool": 10,
    "Concat": 11,
    "Pad": 12,
    "ReduceMean": 13,
    "Tanh": 14,
    "ConvTranspose": 15,
    "Slice": 16,
    "Elu": 17,
    "Constant": 18,
    "Reshape": 19,
    "Mul": 20,
    "Transpose": 21,
    "LogSoftmax": 22,
}

pads_to_mods = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}

attribute_to_pos = {
    "alpha": 0,
    "axes": [1, 2, 3, 4],
    "axis": 5,
    "dilations": [6, 7],
    "ends": [8, 9, 10, 11],
    "epsilon": 12,
    "group": 13,
    "keepdims": 14,
    "kernel_shape": [15, 16],
    "mode": 17,
    "momentum": 18,
    "op": 19,
    "output_shape": [20, 21, 22, 23],
    "pads": [24, 25, 26, 27, 28, 29, 30, 31],
    "starts": [32, 33, 34, 35],
    "steps": [36, 37, 38, 39],
    "strides": [40, 41],
    "value": [42, 43, 44, 45],
    "perm": [46, 47, 48, 49]
    # "skip_connections": [50, ...]
}

reversed_attributes = {
    0: {'op': 'alpha', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    1: {'op': 'axes', 'len': 4, 'type': 'int', 'range': [0, float('inf')]},
    5: {'op': 'axis', 'len': 1, 'type': 'int', 'range': [0, float('inf')]},
    6: {'op': 'dilations', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    8: {'op': 'ends', 'len': 4, 'type': 'int', 'range': [0, sys.maxsize]},
    12: {'op': 'epsilon', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    13: {'op': 'group', 'len': 1, 'type': 'int', 'range': [0, float('inf')]},
    14: {'op': 'keepdims', 'len': 1, 'type': 'int', 'range': [0, float('inf')]},
    15: {'op': 'kernel_shape', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    17: {'op': 'mode', 'len': 1, 'type': 'int', 'range': [0, len(pads_to_mods)]},
    18: {'op': 'momentum', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    19: {'op': 'op', 'len': 1, 'type': 'int', 'range': [0, len(node_to_ops)]},
    20: {'op': 'output_shape', 'len': 4, 'type': 'int', 'range': [0, float('inf')]},
    24: {'op': 'pads', 'len': 8, 'type': 'int', 'range': [0, float('inf')]},
    32: {'op': 'starts', 'len': 4, 'type': 'int', 'range': [0, sys.maxsize]},
    36: {'op': 'steps', 'len': 4, 'type': 'int', 'range': [0, sys.maxsize]},
    40: {'op': 'strides', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    42: {'op': 'value', 'len': 4, 'type': 'int', 'range': [0, float('inf')]},
    46: {'op': 'perm', 'len': 4, 'type': 'int', 'range': [0, 4]},
    50: {'len': 1, 'type': 'int', 'range': [0, NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT]},
    51: {'type': 'int', 'range': [0, MAX_NODE]},
}

# autoencoder_model = Autoencoder()
# autoencoder_model.load_state_dict(torch.load("models/autoencoder_model.pth"))


class NeuralNetworkGraph(nx.DiGraph):
    """Parse graph from network"""

    def __init__(self, model, test_batch):
        """Initialize structure with embedding for each node from `model` and graph from `HiddenLayer`"""
        super().__init__()
        hl_graph = hl.build_graph(model, test_batch, transforms=None)
        self.__colors = {}
        self.__input_shapes = {}
        self.__id_to_node = {}
        self.embedding = []
        self.__parse_graph(hl_graph)

    @staticmethod
    def denormalize_vector(embedding):
        with open(f'./data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]
        for i in range(len(embedding)):
            for j in range(len(embedding[i])):
                if max_vals[j] == min_vals[j]:
                    embedding[i][j] = max_vals[j]
                else:
                    embedding[i][j] = ((max_vals[j] - min_vals[j]) / 2) * (embedding[i][j] + 1) + min_vals[j]
        return embedding

    # TODO: remove autoencoder_model
    @classmethod
    def get_graph(cls, embedding, autoencoder, is_naive=False, is_normalize_needed=False):
        """Create graph from embedding and return it. Get embedding type of list"""
        graph = cls.__new__(cls)
        super(NeuralNetworkGraph, graph).__init__()
        SOS_token = torch.tensor([[[-1.] * NODE_EMBEDDING_DIMENSION]])
        decoded = embedding if is_naive else autoencoder.decode(embedding, NODE_EMBEDDING_DIMENSION, SOS_token).tolist()
        denormalized = decoded if not is_normalize_needed else cls.denormalize_vector(decoded)
        valid_naive = NeuralNetworkGraph.replace_none_in_embedding(denormalized, is_need_replace=False)
        graph.embedding = cls.__fix_attributes(valid_naive)
        graph.__create_graph()
        return graph

    @staticmethod
    def __fix_attributes(embedding):
        for e in range(len(embedding)):
            for pos, attr in reversed_attributes.items():
                if 'len' not in attr:
                    n = NODE_EMBEDDING_DIMENSION - pos
                else:
                    n = attr['len']
                for i in range(n):
                    if embedding[e][pos + i] is None:
                        continue
                    if attr['type'] == 'int':
                        embedding[e][pos + i] = int(round(embedding[e][pos + i]))
                    if attr['type'] == 'float':
                        embedding[e][pos + i] = float(embedding[e][pos + i])
                    if embedding[e][pos + i] < attr['range'][0]:
                        embedding[e][pos + i] = attr['range'][0]
                    if attr['range'][1] <= embedding[e][pos + i]:
                        embedding[e][pos + i] = attr['range'][1]
        return embedding

    def get_naive_embedding(self):
        """Return naive embedding"""
        return self.__fix_attributes(self.embedding)

    @staticmethod
    def replace_none_in_embedding(embedding, is_need_replace=True):
        for i in range(len(embedding)):
            for j in range(len(embedding[i])):
                if is_need_replace and embedding[i][j] is None:
                    embedding[i][j] = NONE_REPLACEMENT
                if not is_need_replace and round(embedding[i][j]) == NONE_REPLACEMENT:
                    embedding[i][j] = None
        return embedding

    def get_embedding(self, autoencoder):
        """Return embedding"""
        input = self.__fix_attributes(self.embedding)
        input = self.replace_none_in_embedding(input)
        input_len = len(input)
        input = torch.tensor(input).view(1, input_len, -1)
        SOS_token = torch.tensor([[[-1.] * NODE_EMBEDDING_DIMENSION]])
        input = torch.cat([SOS_token, input], 1)
        return autoencoder.encode(torch.tensor(input)).tolist()

    def __create_graph(self):
        """Create `networkx.DiGraph` graph from embedding"""
        counter = 0
        for embedding in self.embedding:
            """Add node with attributes to graph"""
            params = {}
            for pos, attr_info in reversed_attributes.items():
                if 'op' not in attr_info:
                    continue
                is_set = True
                if attr_info['len'] > 1:
                    attr = []
                    for i in range(attr_info['len']):
                        if embedding[pos + i] is not None:
                            attr.append(embedding[pos + i])
                        else:
                            break
                    if len(attr) == 0:
                        is_set = False
                else:
                    if embedding[pos] is None:
                        is_set = False
                    else:
                        if pos == attribute_to_pos['op']:
                            attr = str(list(filter(lambda x: node_to_ops[x] == embedding[pos], node_to_ops))[0])
                        elif pos == attribute_to_pos['mode']:
                            attr = str(list(filter(lambda x: pads_to_mods[x] == embedding[pos], pads_to_mods))[0])
                        else:
                            attr = embedding[pos]
                if is_set:
                    params[attr_info['op']] = attr
            self.add_node(counter, **params)

            """Add edge to graph"""
            # for i in range(embedding[ATTRIBUTES_POS_COUNT]):
            #     self.add_edge(counter, embedding[ATTRIBUTES_POS_COUNT + i + 1])
            counter += 1

        for i in range(1, len(self.nodes)):
            self.add_edge(i - 1, i)

    def __add_edges(self, graph):
        """Add edges with changed node's names"""
        for edge in graph.edges:
            v = self.__id_to_node.get(edge[0])
            u = self.__id_to_node.get(edge[1])
            self.__input_shapes[u] = edge[2]
            if v == u:
                continue
            self.add_edge(v, u)

    def __is_supported(self, v):
        """Check if graph is supported"""
        self.__colors[v] = 1
        result = True
        for u in self.adj[v]:
            if self.__colors.get(u, 0) == 0:
                result &= self.__is_supported(u)
            elif self.__colors.get(u, 0) == 1:
                result = False
        self.__colors[v] = 2
        return result

    def __calculate_embedding(self):
        """Calculate embedding for each node"""
        for id in self.nodes:
            node = self.nodes[id]
            embedding = [None] * NODE_EMBEDDING_DIMENSION

            """
            Take output_shape and check it. output_shape might be None or
            size 2 (for linear), size 4 (for convolutional).
            """
            if not node['output_shape'] or node['output_shape'] == []:
                output_shape = self.__input_shapes.get(id)
                if output_shape:
                    node['output_shape'] = output_shape
                    self.nodes[id]['output_shape'] = output_shape
                else:
                    del node['output_shape']

            """
            Set node's parameters to embedding vector in order described in attribute_to_pos dictionary 
            and map string parameters to its' numeric representation.
            """
            for param in node:
                op_name = param
                if isinstance(node[param], list):
                    current_poses = attribute_to_pos[op_name]
                    for i in range(len(node[param])):
                        embedding[current_poses[i]] = node[param][i]
                else:
                    value = node[param]
                    if param == 'op':
                        value = node_to_ops[value]
                    if param == 'mode' and node['op'] == 'Pad':
                        value = pads_to_mods[value]
                    if op_name in attribute_to_pos:
                        cur_pos = attribute_to_pos[op_name][0] if isinstance(attribute_to_pos[op_name], list) else attribute_to_pos[op_name]
                        embedding[cur_pos] = value

            edge_list = list(self.adj[id])
            # embedding.extend(edge_list)
            if len(edge_list) + ATTRIBUTES_POS_COUNT + 1 <= NODE_EMBEDDING_DIMENSION:
                embedding[ATTRIBUTES_POS_COUNT] = len(edge_list)
                for i in range(0, len(edge_list)):
                    embedding[ATTRIBUTES_POS_COUNT + i + 1] = edge_list[i]
            else:
                print('This graph is not supported!')
            self.embedding.append(embedding)

    def __parse_graph(self, graph):
        """Parse `HiddenLayer` graph and create `networkx.DiGraph` with same node attributes"""
        try:
            counter = 0

            """Renumber nodes and add it to graph"""
            values = {}
            for id in graph.nodes:
                graph.nodes[id].params['output_shape'] = graph.nodes[id].output_shape
                graph.nodes[id].params['op'] = graph.nodes[id].op
                if graph.nodes[id].params['op'] == 'Constant':
                    to = list(filter(lambda x: x[0] == id, graph.edges))[0][1]
                    if torch.is_tensor(graph.nodes[id].params['value']):
                        values[to] = {'value': graph.nodes[id].params['value'].tolist(), 'from': id}
                    else:
                        values[to] = {'value': graph.nodes[id].params['value'], 'from': id}
                    continue
                self.__id_to_node[id] = counter
                counter += 1

            for id in graph.nodes:
                if graph.nodes[id].params['op'] == 'Constant':
                    continue
                if id in values:
                    graph.nodes[id].params['value'] = values[id]['value']
                    self.__id_to_node[values[id]['from']] = self.__id_to_node[id]
                self.add_node(self.__id_to_node[id], **graph.nodes[id].params)

            self.__add_edges(graph)
            is_supported = self.__is_supported(0)

            if is_supported:
                self.__calculate_embedding()
            else:
                print('Graph is not supported. This network is not supported.')
        except KeyError as e:
            print(f'Operation or layer is not supported: {e}.')
            raise KeyError(f'Operation or layer is not supported: {e}.')

    @staticmethod
    def check_equality(graph1, graph2):
        """Check two graphs on equality. Return if they are equal and message"""
        if graph1.edges != graph2.edges:
            return False, 'Edges are not equal'
        if sorted(list(graph1.nodes)) != sorted(list(graph2.nodes)):
            return False, 'Nodes are not equal'
        for node in graph1.nodes:
            if graph1.nodes[node] != graph2.nodes[node]:
                return False, 'Node params are not equal'
        return True, 'Graphs are equal'


if __name__ == '__main__':
    # model = NeuralNetwork()
    # model = resnet101()
    # model = AlexNet()
    # model = densenet201()
    # model = mnasnet1_3()
    # model = squeezenet1_1()
    # model = resnet50()
    # model = vgg19_bn()
    # model = inception_v3(aux_logits=False)
    model = UNet(3, 10)

    xs = torch.zeros([1, 3, 224, 224])  # for other models from torchvision.models
    # xs = torch.zeros([64, 3, 28, 28])  # for MnasNet and NeuralNetwork
    # xs = torch.zeros([64, 3, 299, 299])  # for inception

    # g1 = NeuralNetworkGraph(model=model, test_batch=xs)
    # g2 = NeuralNetworkGraph.get_graph(g1.get_embedding())
    # is_equal, message = NeuralNetworkGraph.check_equality(g1, g2)
    # print(message)

    models = {
        "alexnet": AlexNet(),
        "resnet50": resnet50(),
        "resnet101": resnet101(),
        "unet": UNet(3, 10),
        "vgg": vgg19_bn(),
        "densenet": densenet201(),
        "inception": inception_v3(aux_logits=False),
        "mnasnet": mnasnet1_3(),
        "squeezenet": squeezenet1_1(),
    }

    # cnt = 0
    # for name, model in models.items():
    #     cnt += 1
    #     xs = torch.zeros([1, 3, 224, 224])
    #     if name == 'mnasnet':
    #         xs = torch.zeros([64, 3, 28, 28])
    #     if name == 'inception':
    #         xs = torch.zeros([64, 3, 299, 299])
    #     g = NeuralNetworkGraph(model=model, test_batch=xs)
    #     embedding = g.get_naive_embedding()
    #     for e in embedding:
    #         for i in range(len(e)):
    #             if e[i] == None:
    #                 e[i] = NONE_REPLACEMENT
    #     with open(f'./data/embeddings/{cnt}.json', 'w') as f:
    #         f.write(json.dumps(embedding))

    # with open('embeddings/naive/embeddings_dims.txt', 'w') as f:
    #     for name, model in models.items():
    #         xs = torch.zeros([1, 3, 224, 224])
    #         if name == 'mnasnet':
    #             xs = torch.zeros([64, 3, 28, 28])
    #         if name == 'inception':
    #             xs = torch.zeros([64, 3, 299, 299])
    #         g = NeuralNetworkGraph(model=model, test_batch=xs)
    #         dim = NODE_EMBEDDING_DIMENSION
    #         f.write(f'{name}:\nlen = {len(g.embedding)}\nnode_dim = {dim}\n\n')
    #
    #         with open(f'embeddings/naive/naive_{name}.txt', 'w') as f1:
    #             f1.write(json.dumps(g.embedding))
