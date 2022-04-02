import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models import resnet101, densenet201, alexnet

from models.converted_alexnet import ConvertedAlexNet
from models.converted_densenet import ConvertedDenseNet
from models.converted_resnet import ConvertedResNet
from models.converted_mnasnet import ConvertedMnasNet
from models.original_alexnet import AlexNet


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(dataset):
    training_data = dataset(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X, y
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    args = sys.argv[1:]

    # train_dataloader, test_dataloader = load_data(datasets.MNIST)
    train_dataloader, test_dataloader = load_data(datasets.CIFAR10)
    # model = NeuralNetwork()
    model = ConvertedResNet()

    # model.load_state_dict(torch.load("models/model.pth"))
    # model.eval()
    # summary(model, input_size=(1, 28, 28))
    # print(model)

    for i in range(len(args)):
        if args[i] == '--train':
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            epochs = 5
            for t in range(epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                train(train_dataloader, model, loss_fn, optimizer)
                test(test_dataloader, model, loss_fn)
            print("Done!")

            torch.save(model.state_dict(), "model.pth")
            print("Saved PyTorch Model State to model.pth")

        if args[i] == '--distill':
            pass
