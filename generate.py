from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torchvision import transforms, utils

from models.convolutional import ConvolutionalGenerator

if __name__ == "__main__":
    mean, std = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5)
    # Image transform
    transform = transforms.Compose([
        transforms.Normalize(mean, std),
        # transforms.ToPILImage(),
    ])

    # Hyper parameters
    batch_size = 64
    learning_rate = 1e-3
    betas = (0.5, 0.999)
    n_epochs = 100
    noise_dimension = 128
    image_size = 256
    n_channels = 3
    epoch = 44

    # Model
    generator = ConvolutionalGenerator(image_size=image_size, noise_dimension=noise_dimension)

    # Load state dict
    path = Path(__file__).parent / 'states' \
           / f'convolutional_small_bs_{batch_size}_ne_{n_epochs}_lr_{learning_rate}_sz_{image_size}' \
           / f'generator_epoch_{epoch}'
    generator.load_state_dict(torch.load(path))

    # Generate image
    # torch.manual_seed(1)
    noise = torch.randn(64, noise_dimension, 1, 1)
    model_output = generator(noise).float().detach()
    model_output = torch.reshape(model_output, (-1, 3, image_size, image_size))
    print(model_output[0, 0, :10, -10:])
    image = transform(model_output)
    print(image[0, 0, :10, -10:])

    # Display image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(utils.make_grid(image, nrow=8).permute(1, 2, 0))
    fig.show()
