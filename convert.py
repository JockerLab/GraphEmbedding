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
            self.__write_model_init(file, model_name)
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

    @staticmethod
    def __write_line(file, line, tab=''):
        file.write(tab + line + '\n')

    def __write_layers(self, file):
        for key, value in self.sequences.items():
            Converter.__write_line(file, f'self.seq{key} = nn.Sequential(', self.tabulation * 2)
            for elem in value:
                Converter.__write_line(file, elem + ',', self.tabulation * 3)
            Converter.__write_line(file, ')', self.tabulation * 2)

    def __write_model_init(self, file, model_name):
        Converter.__write_line(file, 'from torch import nn\n\n')
        Converter.__write_line(file, f'class {model_name}(nn.Module):\n')
        Converter.__write_line(file, 'def __init__(self):', self.tabulation)
        Converter.__write_line(file, f'super({model_name}, self).__init__()', self.tabulation * 2)


if __name__ == '__main__':
    model = alexnet()
    xs = torch.zeros([1, 3, 224, 224])
    g1 = NeuralNetworkGraph(model=model, test_batch=xs)

    graph = NeuralNetworkGraph.get_graph(g1.embedding)
    network = Converter(graph)

    # g2 = NeuralNetworkGraph(model=network, test_batch=xs)
    # is_equal, message = NeuralNetworkGraph.check_equality(g1, g2)
    # if not is_equal:
    #     print(message)
