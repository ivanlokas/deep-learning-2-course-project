import os
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def evaluate_discriminator(discriminator, generator, device, criterion, features_real, batch_size, noise_dimension):
    noise = torch.randn(batch_size, noise_dimension, 1, 1).to(device)
    features_generated = generator(noise).float()

    discriminator_real = discriminator(features_real)
    loss_discriminator_real = criterion(discriminator_real, torch.ones_like(discriminator_real))

    discriminator_generated = discriminator(features_generated)
    loss_discriminator_generated = criterion(discriminator_generated, torch.zeros_like(discriminator_generated))

    loss_discriminator = (loss_discriminator_real + loss_discriminator_generated)

    return loss_discriminator


def train_discriminator(discriminator, generator, device, criterion, discriminator_optimizer, features_real, batch_size, noise_dimension):
    discriminator_optimizer.zero_grad()

    loss_discriminator = evaluate_discriminator(discriminator, generator, device, criterion, features_real, batch_size, noise_dimension)

    loss_discriminator.backward()
    discriminator_optimizer.step()

    return loss_discriminator


def evaluate_generator(discriminator, generator, device, criterion, batch_size, noise_dimension):
    noise = torch.randn(batch_size, noise_dimension, 1, 1).to(device)
    features_generated = generator(noise).float()

    output = discriminator(features_generated)
    loss_generator = criterion(output, torch.ones_like(output))

    return loss_generator


def train_generator(discriminator, generator, device, criterion, generator_optimizer, batch_size, noise_dimension):
    generator_optimizer.zero_grad()

    loss_generator = evaluate_generator(discriminator, generator, device, criterion, batch_size, noise_dimension)

    loss_generator.backward()
    generator_optimizer.step()

    return loss_generator


def train(
        discriminator,
        generator,
        train_dataloader,
        validation_dataloader,
        device,
        criterion,
        discriminator_optimizer,
        generator_optimizer,
        batch_size,
        n_epochs,
        n_channels,
        image_size,
        noise_dimension,
        save_state=False,
        save_state_dir=None,
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
        save_state: True if model should be saved, False otherwise
        save_state_dir: Directory where model will be saved
    """

    discriminator_start_state_dict = discriminator.state_dict()
    generator_start_state_dict = generator.state_dict()

    writer = SummaryWriter()
    fixed_noise = torch.randn((batch_size, noise_dimension, 1, 1)).to(device)

    # Specify save state directory
    path = Path(__file__).parent.parent / 'states' / save_state_dir

    # Save model state
    if save_state:
        # Create save state directory, if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save starting state
        torch.save(discriminator_start_state_dict, path / f'discriminator_epoch_{0}')
        torch.save(generator_start_state_dict, path / f'generator_epoch_{0}')

    print(len(train_dataloader), len(validation_dataloader))

    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        for index, (features, labels) in enumerate(train_dataloader):
            features_real = features.view(-1, n_channels, image_size, image_size).to(device)

            # Train dataset
            features_real = features_real.float()

            # Train discriminator
            loss_discriminator = train_discriminator(discriminator, generator, device, criterion, discriminator_optimizer,
                                                     features_real, batch_size, noise_dimension)

            # Train generator
            loss_generator = train_generator(discriminator, generator, device, criterion, generator_optimizer,
                                             batch_size, noise_dimension)

            # Update writer
            writer_index = epoch * len(train_dataloader) + index

            writer.add_scalar("Discriminator Loss", loss_discriminator, writer_index)
            writer.add_scalar("Generator Loss", loss_generator, writer_index)

        with torch.no_grad():
            features_real = features_real.view(batch_size, n_channels, image_size, image_size)
            features_generated = generator(fixed_noise).view(batch_size, n_channels, image_size, image_size)

            grid_real = torchvision.utils.make_grid(features_real, normalize=True)
            grid_generated = torchvision.utils.make_grid(features_generated, normalize=True)

            writer.add_image("Real", grid_real, epoch)
            writer.add_image("Generated", grid_generated, epoch)

            # Save model state
            if save_state:
                # Save state at end of epoch
                torch.save(discriminator.state_dict(), path / f'discriminator_epoch_{epoch + 1}')
                torch.save(generator.state_dict(), path / f'generator_epoch_{epoch + 1}')

        generator.eval()
        discriminator.eval()
        losses_discriminator = []
        losses_generator = []
        with torch.no_grad():
            for index, (features, labels) in enumerate(validation_dataloader):
                features_real = features.view(-1, n_channels, image_size, image_size).to(device)

                # Train dataset
                features_real = features_real.float()

                # Train discriminator
                loss_discriminator = evaluate_discriminator(discriminator, generator, device, criterion,
                                                            features_real, batch_size, noise_dimension)
                losses_discriminator.append(loss_discriminator.item())

                # Train generator
                loss_generator = evaluate_generator(discriminator, generator, device, criterion,
                                                    batch_size, noise_dimension)
                losses_generator.append(loss_generator.item())

        loss_discriminator_mean = sum(losses_discriminator) / len(losses_discriminator)
        loss_generator_mean = sum(losses_generator) / len(losses_generator)

        print(f"Epoch {epoch}: discriminator {loss_discriminator_mean}, generator {loss_generator_mean}")

    writer.close()
