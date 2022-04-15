# Print all attributes for models
import importlib
import json
import os
import zipfile

from KD_Lib.models import ResNet18, LeNet, Shallow, ModLeNet, LSTMNet, NetworkInNetwork

from torchvision.models import alexnet, resnet101, densenet201, googlenet, inception_v3, mnasnet1_3, mobilenet_v3_large, squeezenet1_1, vgg19_bn

from convert import Converter
from graph import NeuralNetworkGraph
from models.generated.generated_models import *
from models.original.original_unet import UNet
from network import NeuralNetwork
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms
from torchvision import datasets
import torch
from torchvision import datasets, transforms as T
import torchvision.models as models


models = {
'resnet18': models.resnet18(),
'resnet34': models.resnet34(),
'resnet50': models.resnet50(),
'resnet101': models.resnet101(),
'resnet152': models.resnet152(),
'resnext50_32x4d': models.resnext50_32x4d(),
'resnext101_32x8d': models.resnext101_32x8d(),
'wide_resnet50_2': models.wide_resnet50_2(),
'wide_resnet101_2': models.wide_resnet101_2(),
'unet': UNet(3, 10),
'alexnet':models.alexnet(),
'vgg16': models.vgg16(),
'vgg11': models.vgg11(),
'vgg11_bn': models.vgg11_bn(),
'vgg13': models.vgg13(),
'vgg13_bn': models.vgg13_bn(),
'vgg16_bn': models.vgg16_bn(),
'vgg19_bn': models.vgg19_bn(),
'vgg19': models.vgg19(),
'squeezenet1_0': models.squeezenet1_0(),
'squeezenet1_1': models.squeezenet1_1(),
'densenet161' : models.densenet161(),
'densenet121': models.densenet121(),
'densenet169': models.densenet169(),
'densenet201': models.densenet201(),
'inception': models.inception_v3(aux_logits=False),
'googlenet': models.googlenet(aux_logits=False),
'mnasnet1_0': models.mnasnet1_0(),
'mnasnet0_5': models.mnasnet0_5(),
'mnasnet0_75': models.mnasnet0_75(),
'mnasnet1_3': models.mnasnet1_3(),
'gen1':    GeneratedModel1(),
'GeneratedDensenet': GeneratedDensenet(),
'classification': NaturalSceneClassification(),
}


# Test models convertation
with open('./tmp_model.py', 'w') as f:
    f.write('')
import tmp_model
for name, model in models.items():
    print(f'{name} model is processing')
    xs = torch.zeros([4, 3, 224, 224])
    if name == 'inception':
        xs = torch.zeros([4, 3, 299, 299])
    if name == 'classification':
        xs = torch.zeros([128, 3, 150, 150])
    g = NeuralNetworkGraph(model=model, test_batch=xs)
    Converter(g, filepath='./tmp_model.py', model_name='Tmp')
    importlib.reload(tmp_model)
    print('    ->' + str(len(list(tmp_model.Tmp().modules()))))
    tmp_model.Tmp()(xs)
os.remove('./tmp_model.py')

# # Add embedding model to archive dataset
# cnt = 34
# with zipfile.ZipFile('./data/embeddings/embeddings-zip.zip', 'a') as archive:
#     for name, model in models.items():
#         print(f'{name} model is processing')
#         cnt += 1
#         xs = torch.zeros([4, 3, 224, 224])
#         if name == 'inception':
#             xs = torch.zeros([4, 3, 299, 299])
#         if name == 'classification':
#             xs = torch.zeros([128, 3, 150, 150])
#         g = NeuralNetworkGraph(model=model, test_batch=xs)
#         embedding = g.get_naive_embedding()
#         for e in embedding:
#             for i in range(len(e)):
#                 if e[i] == None:
#                     e[i] = -1
#         archive.writestr(f'{cnt}.json', json.dumps(embedding))



# training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=False,
#     transform=ToTensor(),
# )
# xs = torch.zeros([64, 3, 224, 224])
# models = {
#     'My Network': NeuralNetwork(),
#     # 'googlenet': googlenet(),
#     # 'inception_v3': inception_v3(),
#     # 'mobilenet_v3_large': mobilenet_v3_large(),
#
#     'ResNet101': resnet101(),
#     'Alexnet': alexnet(),
#     'densenet201': densenet201(),
#     'mnasnet1_3': mnasnet1_3(),
#     'squeezenet1_1': squeezenet1_1(),
#     'vgg19_bn': vgg19_bn()
# }
# dic_attrs = dict()
#
# with open('layers.txt', 'w') as f:
#     for model in models:
#         print(model)
#         if model == 'My Network':
#             xs = torch.zeros([64, 1, 28, 28])
#         else:
#             xs = torch.zeros([64, 3, 224, 224])
#         g = NeuralNetworkGraph(model=models[model], test_batch=xs)
#         f.write(model + '\n')
#         for node in g.nodes:
#             attrs = g.nodes[node]
#             op = attrs['op']
#             del attrs['op']
#             f.write(op + ': ' + str(attrs) + '\n')
#             if not op in dic_attrs:
#                 dic_attrs[op] = dict()
#             for attr in attrs:
#                 if not attr in dic_attrs[op]:
#                     dic_attrs[op][attr] = attrs[attr]
#         f.write('\n')
#     f.write('\n----------------------------------------------------------------------\n')
#     for dic_attr in dic_attrs:
#         f.write(dic_attr + ': ' + str(dic_attrs[dic_attr]) + '\n')
#     f.write('\n----------------------------------------------------------------------\n')
#     map_counter = 0
#     map_attrs = dict()
#     for dic_attr in dic_attrs:
#         for attr in dic_attrs[dic_attr]:
#             new_attr = attr
#             if attr == 'output_shape' and dic_attrs[dic_attr][attr]:
#                 new_attr += "_" + str(len(dic_attrs[dic_attr][attr]))
#             if attr == 'pads' and dic_attrs[dic_attr][attr]:
#                 new_attr += "_" + str(len(dic_attrs[dic_attr][attr]))
#             if not new_attr in map_attrs:
#                 if isinstance(dic_attrs[dic_attr][attr], list):
#                     map_pos = []
#                     for i in range(0, len(dic_attrs[dic_attr][attr])):
#                         map_pos.append(map_counter)
#                         map_counter += 1
#                     f.write(new_attr + ': ' + str(map_pos) + '\n')
#                     map_attrs[new_attr] = map_pos
#                 else:
#                     f.write(attr + ': ' + str(map_counter) + '\n')
#                     map_attrs[new_attr] = map_counter
#                     map_counter += 1
#             elif isinstance(dic_attrs[dic_attr][attr], list) and len(dic_attrs[dic_attr][attr]) != len(map_attrs[new_attr]):
#                 print('Error -- ' + new_attr + ': ' + str(dic_attrs[dic_attr][attr]) + ', ' + str(map_attrs[new_attr]))
