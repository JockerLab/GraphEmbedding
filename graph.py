import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from KD_Lib.models import ResNet18, LeNet
from torch import nn
from copy import deepcopy
# from karateclub import DeepWalk, Diff2Vec
from torchvision.models import resnet101, densenet201, alexnet, vgg19_bn, mnasnet1_3, squeezenet1_1
from network import NeuralNetwork
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import datasets, models
from ast import literal_eval
import hiddenlayer as hl
from functools import reduce
import torch
import json

node_ops = {
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
}

pads_mods = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}

ops_with_different_dims = ["output_shape", "pads"]

ATTRIBUTES_POS_COUNT = 37
attribute_pos = {
    "dilations": [0, 1],
    "group": 2,
    "kernel_shape": [3, 4],
    "pads_4": [5, 6, 7, 8],
    "strides": [9, 10],
    "output_shape_4": [11, 12, 13, 14],
    "alpha": 15,
    "axis": 16,
    "output_shape_2": [17, 18],
    "beta": 19,
    "transB": 20,
    "epsilon": 21,
    "momentum": 22,
    "mode": 23,
    "pads_8": [24, 25, 26, 27, 28, 29, 30, 31],
    "value": 32,
    "axes": [33, 34],
    "keepdims": 35,
    "op": 36,
    # "skip_connections": [37, ...]
}

reversed_attribute_pos = {
    0: ['dilations', 2],
    2: ['group', 1],
    3: ['kernel_shape', 2],
    5: ['pads', 4],
    9: ['strides', 2],
    11: ['output_shape', 4],
    15: ['alpha', 1],
    16: ['axis', 1],
    17: ['output_shape', 2],
    19: ['beta', 1],
    20: ['transB', 1],
    21: ['epsilon', 1],
    22: ['momentum', 1],
    23: ['mode', 1],
    24: ['pads', 8],
    32: ['value', 1],
    33: ['axes', 2],
    35: ['keepdims', 1],
    36: ['op', 1]
}


class NeuralNetworkGraph(nx.DiGraph):
    # Parse graph from network
    def __init__(self, model, test_batch):
        super().__init__()
        hl_graph = hl.build_graph(model, test_batch, transforms=None)
        self.__edge_list = {}
        self.__colors = {}
        self.__ancestor_edge = {}
        self.__id_node = {}
        self.embedding = []
        self.__parse_graph(hl_graph)

    # Create graph from embedding
    @classmethod
    def get_graph(cls, embedding):
        graph = cls.__new__(cls)
        super(NeuralNetworkGraph, graph).__init__()
        graph.embedding = embedding
        graph.__create_graph()
        return graph

    def __create_graph(self):
        counter = 0
        for embedding in self.embedding:
            # Add node with attributes to graph
            params = {}
            for pos in reversed_attribute_pos:
                isSet = True
                if reversed_attribute_pos[pos][1] > 1:
                    attr = []
                    for i in range(reversed_attribute_pos[pos][1]):
                        if embedding[pos + i] is None:
                            isSet = False
                            break
                        attr.append(embedding[pos + i])
                else:
                    if embedding[pos] is None:
                        isSet = False
                    else:
                        if pos == 36:  # Attribute 'op'
                            attr = str(list(filter(lambda x: node_ops[x] == embedding[pos], node_ops))[0])
                        elif pos == 23:  # Attribute 'mode'
                            attr = str(list(filter(lambda x: pads_mods[x] == embedding[pos], pads_mods))[0])
                        else:
                            attr = embedding[pos]
                if isSet:
                    params[reversed_attribute_pos[pos][0]] = attr
            self.add_node(counter, **params)

            # Add edge to graph
            for i in range(embedding[ATTRIBUTES_POS_COUNT]):
                self.add_edge(counter, embedding[ATTRIBUTES_POS_COUNT + i + 1])
            counter += 1

    def __make_edge_list(self, graph):
        for edge in graph.edges:
            v = self.__id_node[edge[0]]
            u = self.__id_node[edge[1]]
            cur_list = self.__edge_list.get(v, [])
            cur_list.append(u)
            self.__edge_list[v] = cur_list
            self.__ancestor_edge[u] = edge
            self.add_edge(v, u)

    def __is_top_sorted(self, v):
        self.__colors[v] = 1
        result = True
        for u in self.__edge_list.get(v, []):
            if self.__colors.get(u, 0) == 0:
                result &= self.__is_top_sorted(u)
            elif self.__colors.get(u, 0) == 1:
                result = False
        self.__colors[v] = 2
        return result

    def __get_embedding(self, graph):
        counter = 0
        for id in graph.nodes:
            node = graph.nodes[id]
            embedding = [None] * ATTRIBUTES_POS_COUNT

            # Take output_shape and check it. output_shape might be None or
            # size 2 (for linear), size 4 (for convolutional).
            if not node.output_shape or node.output_shape == []:
                output_shape = self.__ancestor_edge[self.__id_node[id]][2]
                node.params['output_shape'] = output_shape
                self.nodes[counter]['output_shape'] = output_shape

            for param in node.params:
                op_name = param
                if isinstance(node.params[param], list):
                    if param in ops_with_different_dims:
                        op_name += '_' + str(len(node.params[param]))
                    current_poses = attribute_pos[op_name]
                    for i in range(len(node.params[param])):
                        embedding[current_poses[i]] = node.params[param][i]
                else:
                    value = node.params[param]
                    if param == 'op':
                        value = node_ops[value]
                    if param == 'mode' and node.op == 'Pad':
                        value = pads_mods[value]
                    embedding[attribute_pos[op_name]] = value

            node_id = self.__id_node[id]
            if node_id in self.__edge_list:
                edge_list = self.__edge_list[node_id]
            else:
                edge_list = []
            embedding.extend([len(edge_list), *edge_list])
            self.embedding.append(embedding)
            counter += 1

    def __parse_graph(self, graph):
        try:
            counter = 0

            # Renumber nodes and add it to graph
            for id in graph.nodes:
                self.__id_node[id] = counter
                graph.nodes[id].params['output_shape'] = graph.nodes[id].output_shape
                graph.nodes[id].params['op'] = graph.nodes[id].op
                self.add_node(counter, **graph.nodes[id].params)
                counter += 1

            # Add edges to graph
            self.__make_edge_list(graph)

            # Check if graph is top sorted
            is_top_sorted = self.__is_top_sorted(0)
            if not is_top_sorted:
                print('Graph is not top sorted. This network is not supported.')
            else:
                print('Graph is top sorted.')

            # Get embeddings for each node
            self.__get_embedding(graph)
        except KeyError as e:
            print(f"Operation or layer is not supported: {e}.")


def check_graphs(g1, g2):
    if g1.edges != g2.edges:
        print('Edges are not equal')
    if sorted(list(g1.nodes)) != sorted(list(g2.nodes)):
        print('Nodes are not equal')
    for node in g1.nodes:
        if g1.nodes[node] != g2.nodes[node]:
            print('Node params are not equal')

if __name__ == '__main__':
    # model = NeuralNetwork()
    # model = alexnet()
    # model = densenet201()
    # model = mnasnet1_3()
    # model = squeezenet1_1()
    # model = vgg19_bn()
    # model = LeNet(in_channels=1, img_size=28)
    model = resnet101()

    # xs = torch.zeros([1, 1, 28, 28])  # for NeuralNetwork()
    xs = torch.zeros([1, 3, 224, 224])  # for other models from torchvision.models

    g1 = NeuralNetworkGraph(model=model, test_batch=xs)
    g2 = NeuralNetworkGraph.get_graph(g1.embedding)
    check_graphs(g1, g2)

    # with open('embeddings/naive_densenet201_embedding.txt', 'r') as f:
    #     embedding = json.load(f)
    #     g2 = NeuralNetworkGraph.get_graph(embedding)
    #     check_graphs(g1, g2)

    # Graph visualization

    # plt.figure(figsize=(40, 30))
    # nx.draw(g)
    # plt.show()

    # Graph embedding in 2-dim

    # model = Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=2)
    # model = DeepWalk(dimensions=2)
    # model.fit(g)
    # embedding = model.get_embedding()
    # print(embedding)
    # plt.figure(figsize=(40, 30))
    # plt.scatter(embedding[:, 0], embedding[:, 1], s=1000)
    # plt.show()

    # node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=1)
    # model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # model.wv.save_word2vec_format('./embeddings/node2vec.txt')
