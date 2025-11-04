"""
CIFAR-10 dataset loading and federated partitioning.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple


def get_cifar10_loaders(data_dir='./data', batch_size=32, num_workers=2):
    """
    Get CIFAR-10 train and test data loaders.

    Args:
        data_dir: Directory to store/load CIFAR-10 data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, test_loader
    """
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download and load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                    download=True, transform=transform_train)

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                   download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def partition_cifar10_noniid(train_dataset, n_clients: int = 5, alpha: float = 0.5,
                             seed: int = 42) -> List[Subset]:
    """
    Partition CIFAR-10 dataset among clients in a non-IID manner using Dirichlet distribution.

    Args:
        train_dataset: CIFAR-10 training dataset
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed

    Returns:
        List of Subset datasets for each client
    """
    np.random.seed(seed)

    # Get labels
    if hasattr(train_dataset, 'targets'):
        labels = np.array(train_dataset.targets)
    else:
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    n_classes = 10  # CIFAR-10 has 10 classes
    n_samples = len(labels)

    # Initialize client data holders
    client_indices = [[] for _ in range(n_clients)]

    # For each class, use Dirichlet distribution to split among clients
    for class_idx in range(n_classes):
        # Get indices for this class
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * n_clients)

        # Split the class data according to proportions
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()  # Ensure all samples assigned

        # Assign to clients
        idx = 0
        for client_idx, split_size in enumerate(splits):
            if split_size > 0:
                client_indices[client_idx].extend(class_indices[idx:idx + split_size])
                idx += split_size

    # Create Subset datasets for each client
    client_datasets = []
    for indices in client_indices:
        np.random.shuffle(indices)
        client_datasets.append(Subset(train_dataset, indices))

    return client_datasets


def get_federated_cifar10(n_clients: int = 5, batch_size: int = 32,
                          non_iid: bool = True, alpha: float = 0.5,
                          seed: int = 42, data_dir='./data'):
    """
    Get federated CIFAR-10 data loaders.

    Args:
        n_clients: Number of clients
        batch_size: Batch size for training
        non_iid: Whether to use non-IID partitioning
        alpha: Dirichlet concentration parameter (for non-IID)
        seed: Random seed
        data_dir: Directory to store/load data

    Returns:
        Tuple of (client_loaders, test_loader, full_train_loader)
    """
    # Get full datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                    download=True, transform=transform_train)

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                   download=True, transform=transform_test)

    # Partition training data
    if non_iid:
        client_datasets = partition_cifar10_noniid(train_dataset, n_clients, alpha, seed)
    else:
        # IID split
        indices = list(range(len(train_dataset)))
        np.random.seed(seed)
        np.random.shuffle(indices)

        split_size = len(train_dataset) // n_clients
        client_datasets = []
        for i in range(n_clients):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_clients - 1 else len(train_dataset)
            client_datasets.append(Subset(train_dataset, indices[start_idx:end_idx]))

    # Create data loaders
    client_loaders = [
        DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        for client_dataset in client_datasets
    ]

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    # Full training loader (for pre-training)
    full_train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)

    return client_loaders, test_loader, full_train_loader


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing CIFAR-10 federated data loading...")

    client_loaders, test_loader, full_train_loader = get_federated_cifar10(
        n_clients=5, batch_size=64, non_iid=True, alpha=0.5
    )

    print(f"\nNumber of clients: {len(client_loaders)}")
    for i, loader in enumerate(client_loaders):
        print(f"  Client {i}: {len(loader.dataset)} samples, {len(loader)} batches")

    print(f"\nTest set: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    print(f"Full train set: {len(full_train_loader.dataset)} samples")

    # Check data distribution per client
    print("\nClass distribution per client:")
    for i, loader in enumerate(client_loaders):
        dataset = loader.dataset
        if hasattr(dataset.dataset, 'targets'):
            labels = np.array([dataset.dataset.targets[idx] for idx in dataset.indices])
        else:
            labels = np.array([dataset.dataset[idx][1] for idx in dataset.indices])

        class_counts = np.bincount(labels, minlength=10)
        print(f"  Client {i}: {class_counts}")
