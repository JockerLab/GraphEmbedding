# Print all attributes for models

from KD_Lib.models import ResNet18, LeNet, Shallow, ModLeNet, LSTMNet, NetworkInNetwork

from torchvision.models import alexnet, resnet101, densenet201, googlenet, inception_v3, mnasnet1_3, mobilenet_v3_large, squeezenet1_1, vgg19_bn
from graph import NeuralNetworkGraph
from network import NeuralNetwork
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch

# training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=False,
#     transform=ToTensor(),
# )
xs = torch.zeros([64, 3, 224, 224])
models = {
    'My Network': NeuralNetwork(),
    # 'googlenet': googlenet(),
    # 'inception_v3': inception_v3(),
    # 'mobilenet_v3_large': mobilenet_v3_large(),

    'ResNet101': resnet101(),
    'Alexnet': alexnet(),
    'densenet201': densenet201(),
    'mnasnet1_3': mnasnet1_3(),
    'squeezenet1_1': squeezenet1_1(),
    'vgg19_bn': vgg19_bn()
}
dic_attrs = dict()

with open('layers.txt', 'w') as f:
    for model in models:
        print(model)
        if model == 'My Network':
            xs = torch.zeros([64, 1, 28, 28])
        else:
            xs = torch.zeros([64, 3, 224, 224])
        g = NeuralNetworkGraph(model=models[model], test_batch=xs)
        f.write(model + '\n')
        for node in g.nodes:
            attrs = g.nodes[node]
            op = attrs['op']
            del attrs['op']
            f.write(op + ': ' + str(attrs) + '\n')
            if not op in dic_attrs:
                dic_attrs[op] = dict()
            for attr in attrs:
                if not attr in dic_attrs[op]:
                    dic_attrs[op][attr] = attrs[attr]
        f.write('\n')
    f.write('\n----------------------------------------------------------------------\n')
    for dic_attr in dic_attrs:
        f.write(dic_attr + ': ' + str(dic_attrs[dic_attr]) + '\n')
    f.write('\n----------------------------------------------------------------------\n')
    map_counter = 0
    map_attrs = dict()
    for dic_attr in dic_attrs:
        for attr in dic_attrs[dic_attr]:
            new_attr = attr
            if attr == 'output_shape' and dic_attrs[dic_attr][attr]:
                new_attr += "_" + str(len(dic_attrs[dic_attr][attr]))
            if attr == 'pads' and dic_attrs[dic_attr][attr]:
                new_attr += "_" + str(len(dic_attrs[dic_attr][attr]))
            if not new_attr in map_attrs:
                if isinstance(dic_attrs[dic_attr][attr], list):
                    map_pos = []
                    for i in range(0, len(dic_attrs[dic_attr][attr])):
                        map_pos.append(map_counter)
                        map_counter += 1
                    f.write(new_attr + ': ' + str(map_pos) + '\n')
                    map_attrs[new_attr] = map_pos
                else:
                    f.write(attr + ': ' + str(map_counter) + '\n')
                    map_attrs[new_attr] = map_counter
                    map_counter += 1
            elif isinstance(dic_attrs[dic_attr][attr], list) and len(dic_attrs[dic_attr][attr]) != len(map_attrs[new_attr]):
                print('Error -- ' + new_attr + ': ' + str(dic_attrs[dic_attr][attr]) + ', ' + str(map_attrs[new_attr]))
