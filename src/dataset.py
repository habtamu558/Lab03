
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(data_dir="data", batch_size=128):
    """
    Download Fashion-MNIST and prepare training,
    validation, and testing DataLoaders.
    """

    # Convert images to tensors and normalize them.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Download the official training dataset.
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download the official test dataset.
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split 60,000 images into 55,000 training
    # images and 5,000 validation images.
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
