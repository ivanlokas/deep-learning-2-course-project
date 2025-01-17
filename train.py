import torch
from torch import nn
from torchvision import transforms

from datasets.load import get_loaders
from models.complex import ComplexDiscriminator, ComplexGenerator
from models.convolutional import ConvolutionalDiscriminator, ConvolutionalGenerator
from util import train

if __name__ == "__main__":
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    batch_size = 64
    learning_rate = 5e-4
    betas = (0.5, 0.999)
    n_epochs = 100
    noise_dimension = 128
    image_size = 256
    n_channels = 3
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # Dataset folder name
    dataset_name = 'cars'

    # Dataloaders
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataloader, validation_dataloader, test_dataloader = get_loaders(
        dataset_name,
        batch_size=batch_size,
        transform=transform,
    )

    # Models

    discriminator = ConvolutionalDiscriminator(image_size, n_channels).to(device)
    generator = ConvolutionalGenerator(image_size, noise_dimension).to(device)

    # Model arguments
    criterion = nn.BCELoss()

    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=learning_rate,
        betas=betas
    )

    generator_optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=learning_rate,
        betas=betas
    )

    save_state = True
    save_state_dir = f'convolutional2_bs_{batch_size}_ne_{n_epochs}_lr_{learning_rate}_sz_{image_size}'

    # Train model
    train(
        discriminator=discriminator,
        generator=generator,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=device,
        criterion=criterion,
        discriminator_optimizer=discriminator_optimizer,
        generator_optimizer=generator_optimizer,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_channels=n_channels,
        image_size=image_size,
        noise_dimension=noise_dimension,
        save_state=save_state,
        save_state_dir=save_state_dir,
    )
