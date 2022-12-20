import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.prototype import PrototypeDiscriminator, PrototypeGenerator
from util import train

if __name__ == "__main__":
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    batch_size = 32
    learning_rate = 3e-4
    n_epochs = 50
    noise_dimension = 64

    # Datasets
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root="datasets/", transform=transforms, download=True)
    validate_dataset = datasets.MNIST(root="datasets/", transform=transforms, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Models
    discriminator = PrototypeDiscriminator().to(device)
    generator = PrototypeGenerator(noise_dimension).to(device)

    # Model arguments
    criterion = nn.BCELoss()

    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate)
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=learning_rate)

    # Train model
    train(
        discriminator=discriminator,
        generator=generator,
        train_dataloader=train_dataloader,
        device=device,
        criterion=criterion,
        discriminator_optimizer=discriminator_optimizer,
        generator_optimizer=generator_optimizer,
        batch_size=batch_size,
        n_epochs=n_epochs,
        noise_dimension=noise_dimension,
    )
