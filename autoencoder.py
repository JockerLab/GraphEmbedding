from torch import nn, optim
import torch
from torchvision import transforms, datasets

from graph import NODE_EMBEDDING_DIMENSION


class Autoencoder(nn.Module):
    def __init__(self, classes=NODE_EMBEDDING_DIMENSION):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=classes, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=10, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train():
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data",
                             train=True,
                             download=False,
                             transform=tensor_transform)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=32,
                                         shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(784).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    epochs = 20
    for epoch in range(epochs):
        sum_loss = 0
        for (image, _) in loader:
            image = image.reshape(-1, 28 * 28)
            output = model(image)
            optimizer.zero_grad()
            loss = loss_function(output, image)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        print(f'Epoch {epoch}/{epochs}\n    Avg Loss: {sum_loss / len(loader)}')


if __name__ == '__main__':
    train()
