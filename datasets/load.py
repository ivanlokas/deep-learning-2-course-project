from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_loaders(dataset_name: str, batch_size: int = 32, transform=transforms.ToTensor()) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retrieves batch loaders with given batch size

    Args:
        dataset_name (str): Relative dataset path
        batch_size (int): Batch size
        transform: Transforms

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloaders
    """

    DATA_DIR = Path(__file__).parent / dataset_name

    data = ImageFolder(DATA_DIR, transform=transform)

    c = len(data) // 4
    b = len(data) // 2 - c
    a = len(data) - b - c

    train_data, validation_data, test_data = random_split(data, [a, b, c])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader
