import torch.nn as nn


class ComplexDiscriminator(nn.Module):
    def __init__(self, image_size, n_channels):
        super().__init__()

        self.image_size = image_size
        self.n_channels = n_channels

        self.layers = nn.Sequential(
            nn.Linear(self.image_size ** 2, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, self.n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class ComplexGenerator(nn.Module):
    def __init__(self, image_size, noise_dimension):
        super().__init__()

        self.image_size = image_size
        self.noise_dimension = noise_dimension

        self.layers = nn.Sequential(
            nn.Linear(noise_dimension, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, self.image_size ** 2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
