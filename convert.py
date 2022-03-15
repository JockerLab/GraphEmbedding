from torchvision.models import resnet101, densenet201, alexnet, vgg19_bn, mnasnet1_3, squeezenet1_1
from graph import NeuralNetworkGraph
from torch import nn
from network import NeuralNetwork
import torch
import os
import types


# def do_forward(graph, node_to_layer, node_to_operation):
#     def forward(self, x):
#         current_node = 0
#         edges = graph.adj.get(current_node)
#         while True:
#             if current_node in node_to_layer:
#                 current_operation_id = node_to_layer[current_node]
#                 current_operation = self.layers[current_operation_id]
#             if current_node in node_to_operation:
#                 current_operation_id = node_to_operation[current_node]
#                 current_operation = self.operations[current_operation_id]
#             x = current_operation(x)
#             if len(graph.adj.get(current_node)) == 0:
#                 break
#             current_node = next(iter(edges))
#             edges = graph.adj.get(current_node)
#
#         # TODO:
#         if len(edges) > 1:
#             pass
#
#         return x
#
#     return forward


class Converter:
    def __init__(self, graph, filepath='models/my_model.py', model_name="Model"):
        self.graph = graph
        self.operations = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.current_dim = 3
        self.node_to_layer = {}
        self.node_to_operation = {}
        self.tabulation = '    '
        self.sequences = {1: [self.__choose_type(self.graph.nodes[0])]}
        self.node_to_sequence = {0: 1}
        self.__create_layers(0)

        with open(filepath, 'w') as file:
            self.__write_model(file, model_name)
            self.__write_layers(file)
            #self.__create_forward(file)

    def __select_layer(self, node):
        # TODO: не все параметры есть. Некоторых может не быть
        # TODO: padding размера 4, а не 2
        old_dim = self.current_dim
        self.current_dim = node['output_shape'][1]
        if node['op'] == "Linear":
            return f"nn.Linear(in_features={old_dim}, out_features={self.current_dim})"
        if node['op'] == "Conv":
            return f"nn.Conv2d(" \
                f"in_channels={old_dim}, " \
                f"out_channels={self.current_dim}, " \
                f"kernel_size={node['kernel_shape']}, " \
                f"stride={node['strides']}, " \
                f"padding={node['pads'][:2]}, " \
                f"dilation={node['dilations']}, " \
                f"groups={node['group']})"

    def __select_operation(self, node):
        # TODO: не все параметры есть. Некоторых может не быть
        # TODO: padding размера 4, а не 2
        # TODO: может не быть node['output_shape'][1]
        self.current_dim = node['output_shape'][1]
        if node['op'] == "Relu":
            return "nn.ReLU()"
        if node['op'] == "MaxPool":
            return f"nn.MaxPool2d(" \
                   f"kernel_size={node['kernel_shape']}, " \
                   f"stride={node['strides']}, " \
                   f"padding={node['pads'][:2]})"
        if node['op'] == "AveragePool":
            return f"nn.AdaptiveAvgPool2d({(node['output_shape'][2], node['output_shape'][3])})"
        if node['op'] == "Flatten":
            return "nn.Flatten()"

    def __choose_type(self, node):
        if node['op'] in ["Linear", "Conv"]:
            return self.__select_layer(node)

        if node['op'] in [
            "Relu", "MaxPool", "AveragePool", "Flatten",
        ]:
            return self.__select_operation(node)

        #print(f"Unsupportable layer type: {node['op']}.")
        return "#  Unsupportable layer type: {node['op']}"

    def __create_layers(self, cur_node):
        edges = self.graph.adj.get(cur_node)
        current_sequence = len(self.sequences)
        for v in edges:
            if v in self.node_to_sequence:
                continue
            node = self.graph.nodes[v]
            layer = self.__choose_type(node)
            if len(edges) > 1 or len(self.graph.degree._pred[v]) > 1:
                current_sequence = len(self.sequences) + 1
            array = self.sequences.get(current_sequence, [])
            array.append(layer)
            self.sequences[current_sequence] = array
            self.node_to_sequence[v] = current_sequence
            self.__create_layers(v)

    def __write_layers(self, file):
        for key, value in self.sequences.items():
            file.write(self.tabulation * 2 + f'self.seq{key} = nn.Sequential(\n')
            for elem in value:
                file.write(self.tabulation * 3 + elem + ',\n')
            file.write(self.tabulation * 2 + ')\n')

    def __write_model(self, file, model_name):
        # Write imports
        file.write('from torch import nn\n\n\n')
        # Write model initialization
        file.write(f'class {model_name}(nn.Module):\n\n')
        file.write(self.tabulation + 'def __init__(self):\n')
        file.write(self.tabulation * 2 + f'super({model_name}, self).__init__()\n')


def check_graphs(g1, g2):
    if g1.edges != g2.edges:
        print('Edges are not equal')
    if sorted(list(g1.nodes)) != sorted(list(g2.nodes)):
        print('Nodes are not equal')
    for node in g1.nodes:
        if g1.nodes[node] != g2.nodes[node]:
            print('Node params are not equal')


if __name__ == '__main__':
    # TODO: supress messages
    model = resnet101()
    xs = torch.zeros([1, 3, 224, 224])
    g1 = NeuralNetworkGraph(model=model, test_batch=xs)

    graph = NeuralNetworkGraph.get_graph(g1.embedding)
    network = Converter(graph)

    # g2 = NeuralNetworkGraph(model=network, test_batch=xs)

    # if check_graphs(g1, g2):
    #     print("Graphs are equal!")
