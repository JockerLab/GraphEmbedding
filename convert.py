from torchvision.models import resnet101, densenet201, alexnet, vgg19_bn, mnasnet1_3, squeezenet1_1
from graph import NeuralNetworkGraph
from torch import nn
import networkx as nx
from mapping import NetworkMapping
from models.converted_alexnet import ConvertedAlexNet
from models.converted_mnasnet import ConvertedMnasNet
from models.converted_resnet import ConvertedResNet
from models.converted_densenet import ConvertedDenseNet
from models.converted_squeezenet import ConvertedSqueezeNet
from models.converted_vgg import ConvertedVGG
from network import NeuralNetwork
from models.original_alexnet import AlexNet
import torch
import os
import types


class Converter:
    def __init__(self, graph, filepath='models/my_model.py', model_name="Model"):
        self.graph = graph
        self.operations = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.node_to_layer = {}
        self.node_to_operation = {}
        self.tabulation = '    '
        self.out_dim = {0: self.graph.nodes[0]['output_shape'][1]}
        self.sequences = {1: [NetworkMapping.map_node(self.graph.nodes[0], 3, self.out_dim[0])]}
        self._graph_seq = nx.DiGraph()
        self._graph_seq.add_node(1, **{'op': self.graph.nodes[0]['op']})
        self.node_to_sequence = {0: 1}
        self.__create_layers(0)

        with open(filepath, 'w') as file:
            self.__write_model_init(file, model_name)
            self.__write_layers(file)
            self.__write_forward(file)

    def __create_layers(self, cur_node):
        edges = self.graph.adj.get(cur_node)
        current_sequence = len(self.sequences)

        if cur_node == 23:
            kek = 0

        for v in edges:
            if v in self.node_to_sequence:
                continue
            node = self.graph.nodes[v]
            # Skip concat and add
            layer = None
            old_dim = self.out_dim[cur_node]
            self.out_dim[v] = node['output_shape'][1] if node['output_shape'] else old_dim
            if node['op'] not in ['Concat', 'Add']:
                layer = NetworkMapping.map_node(node, old_dim, self.out_dim[v])
            if len(edges) > 1 \
                    or len(self.graph.pred[v]) > 1 \
                    or (node['op'] in ['Concat', 'Pad'] and len(self.graph.pred[v]) <= 1):
                current_sequence = len(self.sequences) + 1
                if node['op'] == 'Pad':
                    if current_sequence not in self._graph_seq.nodes:
                        self._graph_seq.add_node(current_sequence, **{'op': node['op'], 'pads': node['pads']})
                else:
                    if current_sequence not in self._graph_seq.nodes:
                        self._graph_seq.add_node(current_sequence, **{'op': node['op']})
            array = self.sequences.get(current_sequence, [])
            if layer:
                if node['op'] == 'Linear' and 'nn.Flatten()' not in array:
                    array.append(NetworkMapping.map_node({'op': 'Flatten'}))
                array.append(layer)
            self.sequences[current_sequence] = array
            self.node_to_sequence[v] = current_sequence
            self.__create_layers(v)

        current_sequence = self.node_to_sequence[cur_node]
        for v in edges:
            new_sequence = self.node_to_sequence[v]
            if new_sequence == current_sequence:
                continue
            self._graph_seq.add_edge(current_sequence, new_sequence)

    @staticmethod
    def __write_line(file, line, tab=''):
        file.write(tab + line + '\n')

    def __write_layers(self, file):
        for key, value in self.sequences.items():
            if len(value) == 0:
                continue
            Converter.__write_line(file, f'self.seq{key} = nn.Sequential(', self.tabulation * 2)
            for elem in value:
                Converter.__write_line(file, elem + ',', self.tabulation * 3)
            Converter.__write_line(file, ')', self.tabulation * 2)
        Converter.__write_line(file, '')

    def __write_forward(self, file):
        Converter.__write_line(file, 'def forward(self, x_0):', self.tabulation)
        q = [1]
        used = {1}
        last = 1
        while len(q) > 0:
            v = q[0]
            last = v
            q.pop(0)
            for u in self._graph_seq.adj[v]:
                if u not in used:
                    used.add(u)
                    q.append(u)

            if len(self._graph_seq.pred[v]) > 1:
                if self._graph_seq.nodes[v]['op'] == 'Concat':
                    inputs = []
                    for u in self._graph_seq.pred[v]:
                        inputs.append(f'x_{u}')
                    Converter.__write_line(file, f'x_{v} = torch.cat([{", ".join(map(str, inputs))}], 1)',
                                           self.tabulation * 2)
                    Converter.__write_line(file,
                                           f'x_{v} = self.seq{v}(x_{v})',
                                           self.tabulation * 2)
                if self._graph_seq.nodes[v]['op'] == 'Add':
                    inputs = []
                    for u in self._graph_seq.pred[v]:
                        inputs.append(f'x_{u}')
                    Converter.__write_line(file, f'x_{v} = {" + ".join(map(str, inputs))}',
                                           self.tabulation * 2)
                    if len(self.sequences[v]) > 0:
                        Converter.__write_line(file,
                                               f'x_{v} = self.seq{v}(x_{v})',
                                               self.tabulation * 2)
            else:
                if self._graph_seq.nodes[v]['op'] == 'Concat':
                    Converter.__write_line(file,
                                           f'x_{v} = torch.cat([x_{next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))}], 1)',
                                           self.tabulation * 2)
                    Converter.__write_line(file,
                                           f'x_{v} = self.seq{v}(x_{v})',
                                           self.tabulation * 2)
                elif self._graph_seq.nodes[v]['op'] == 'Pad':
                    # TODO: pads
                    Converter.__write_line(file,
                                           f'x_{v} = torch.nn.functional.pad(x_{next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))}, {self._graph_seq.nodes[v]["pads"]})',
                                           self.tabulation * 2)
                    Converter.__write_line(file,
                                           f'x_{v} = self.seq{v}(x_{v})',
                                           self.tabulation * 2)
                else:
                    Converter.__write_line(file,
                                           f'x_{v} = self.seq{v}(x_{next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))})',
                                           self.tabulation * 2)
        Converter.__write_line(file, f'return x_{last}', self.tabulation * 2)

    def __write_model_init(self, file, model_name):
        Converter.__write_line(file, 'import torch')
        Converter.__write_line(file, 'from torch import nn\n\n')
        Converter.__write_line(file, f'class {model_name}(nn.Module):\n')
        Converter.__write_line(file, 'def __init__(self):', self.tabulation)
        Converter.__write_line(file, f'super({model_name}, self).__init__()', self.tabulation * 2)


if __name__ == '__main__':
    # model = resnet101()
    # model = NeuralNetwork()
    # model = alexnet()
    # model = densenet201()
    # model = mnasnet1_3()
    model = squeezenet1_1()
    # model = vgg19_bn()
    # model = resnet101()
    xs = torch.zeros([1, 3, 224, 224])
    # xs = torch.zeros([64, 3, 28, 28])
    g1 = NeuralNetworkGraph(model=model, test_batch=xs)
    # network = Converter(g1, filepath='models/converted_squeezenet.py', model_name='ConvertedSqueezeNet')
    g2 = NeuralNetworkGraph(model=ConvertedSqueezeNet(), test_batch=xs)
    is_equal, message = NeuralNetworkGraph.check_equality(g1, g2)
    print(message)
