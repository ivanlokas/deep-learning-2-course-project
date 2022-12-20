import torch.nn as nn


class PrototypeDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class PrototypeGenerator(nn.Module):
    def __init__(self, noise_dimension):
        super().__init__()

        self.noise_dimension = noise_dimension

        self.layers = nn.Sequential(
            nn.Linear(noise_dimension, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
