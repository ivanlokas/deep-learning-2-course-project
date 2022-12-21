from matplotlib import pyplot as plt

from pathlib import Path

import torch
from torchvision import transforms

from models.complex import ComplexGenerator

if __name__ == "__main__":
    # Image transform
    transform = transforms.ToPILImage()

    # Hyper parameters
    batch_size = 64
    learning_rate = 1e-5
    betas = (0.5, 0.999)
    n_epochs = 50
    noise_dimension = 128
    image_size = 32
    n_channels = 3

    # Model
    generator = ComplexGenerator(image_size=image_size, noise_dimension=noise_dimension)

    # Load state dict
    path = Path(__file__).parent / 'states' \
           / \
           f'complex_bs_{batch_size}_ne_{n_epochs}_lr_{learning_rate}' \
           / f'generator_epoch_{0}'
    generator.load_state_dict(torch.load(path))

    # Generate image
    noise = torch.randn(n_channels, noise_dimension)
    model_output = generator(noise).float()
    model_output = torch.reshape(model_output, (3, image_size, image_size))
    image = transform(model_output)

    # Display image
    plt.imshow(image)
    plt.show()
