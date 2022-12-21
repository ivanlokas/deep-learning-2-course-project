import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def train(
        discriminator,
        generator,
        train_dataloader,
        device,
        criterion,
        discriminator_optimizer,
        generator_optimizer,
        batch_size,
        n_epochs,
        n_channels,
        image_size,
        noise_dimension,
) -> None:
    """
    Generic GAN training function

    Args:
        discriminator: Discriminator that will be trained
        generator: Generator that will be trained
        train_dataloader: Training dataloader
        device: Device that will be used
        criterion: Criterion function
        discriminator_optimizer: Discriminator optimizer
        generator_optimizer: Generator optimizer
        batch_size: Batch size
        n_epochs: Number of epochs
        n_channels: Number of channels
        image_size: Image size
        noise_dimension: Noise dimension
    """

    writer = SummaryWriter()
    fixed_noise = torch.randn((batch_size * n_channels, noise_dimension)).to(device)

    for epoch in range(n_epochs):
        for index, (features, labels) in enumerate(train_dataloader):
            features_real = features.view(-1, image_size ** 2).to(device)
            noise = torch.randn(batch_size * n_channels, noise_dimension).to(device)

            # Train dataset
            features_real = features_real.float()
            features_generated = generator(noise).float()

            # Train discriminator
            discriminator_real = discriminator(features_real).view(-1)
            loss_discriminator_real = criterion(discriminator_real, torch.ones_like(discriminator_real))

            discriminator_generated = discriminator(features_generated).view(-1)
            loss_discriminator_generated = criterion(discriminator_generated, torch.zeros_like(discriminator_generated))

            loss_discriminator = (loss_discriminator_real + loss_discriminator_generated) / 2

            discriminator.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            discriminator_optimizer.step()

            # Train generator
            output = discriminator(features_generated).view(-1)
            loss_generator = criterion(output, torch.ones_like(output))

            generator.zero_grad()
            loss_generator.backward()
            generator_optimizer.step()

            # Update writer
            writer_index = epoch * len(train_dataloader) + index

            writer.add_scalar("Discriminator Loss", loss_discriminator, writer_index)
            writer.add_scalar("Generator Loss", loss_generator, writer_index)

            if index == 0:
                with torch.no_grad():
                    features_real = features_real.view(batch_size, n_channels, image_size, image_size)
                    features_generated = generator(fixed_noise).view(batch_size, n_channels, image_size, image_size)

                    grid_real = torchvision.utils.make_grid(features_real, normalize=True)
                    grid_generated = torchvision.utils.make_grid(features_generated, normalize=True)

                    writer.add_image("Real", grid_real, epoch)
                    writer.add_image("Generated", grid_generated, epoch)
