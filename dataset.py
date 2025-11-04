"""
Dataset generation and non-IID partitioning for federated learning demo.
"""

import numpy as np
from sklearn.datasets import make_moons
from typing import List, Tuple


def generate_dataset(dataset_name: str = "moons", n_samples: int = 2000, noise: float = 0.1, seed: int = 42):
    """
    Generate a simple 2D classification dataset.

    Args:
        dataset_name: Type of dataset ("moons" or future: "mnist")
        n_samples: Total number of samples
        noise: Noise level for make_moons
        seed: Random seed for reproducibility

    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Labels array of shape (n_samples,)
    """
    np.random.seed(seed)

    if dataset_name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X, y


def partition_data_noniid(X: np.ndarray, y: np.ndarray, n_clients: int = 5,
                          alpha: float = 0.5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data among clients in a non-IID manner using Dirichlet distribution.

    Args:
        X: Features array
        y: Labels array
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed

    Returns:
        List of (X_client, y_client) tuples for each client
    """
    np.random.seed(seed)
    n_samples = len(y)
    n_classes = len(np.unique(y))

    # Initialize client data holders
    client_data = [[] for _ in range(n_clients)]

    # For each class, use Dirichlet distribution to split among clients
    for class_idx in range(n_classes):
        # Get indices for this class
        class_indices = np.where(y == class_idx)[0]
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
                client_data[client_idx].extend(class_indices[idx:idx + split_size])
                idx += split_size

    # Convert to arrays and shuffle each client's data
    client_datasets = []
    for client_idx in range(n_clients):
        indices = np.array(client_data[client_idx])
        np.random.shuffle(indices)
        client_datasets.append((X[indices], y[indices]))

    return client_datasets


def partition_data_iid(X: np.ndarray, y: np.ndarray, n_clients: int = 5,
                       seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data among clients in an IID manner (for comparison).

    Args:
        X: Features array
        y: Labels array
        n_clients: Number of clients
        seed: Random seed

    Returns:
        List of (X_client, y_client) tuples for each client
    """
    np.random.seed(seed)
    n_samples = len(y)

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split evenly
    split_size = n_samples // n_clients
    client_datasets = []

    for i in range(n_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_clients - 1 else n_samples
        client_indices = indices[start_idx:end_idx]
        client_datasets.append((X[client_indices], y[client_indices]))

    return client_datasets


def get_federated_data(dataset_name: str = "moons", n_clients: int = 5,
                      n_samples: int = 2000, non_iid: bool = True,
                      alpha: float = 0.5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate and partition dataset for federated learning.

    Args:
        dataset_name: Type of dataset
        n_clients: Number of clients
        n_samples: Total number of samples
        non_iid: Whether to use non-IID partitioning
        alpha: Dirichlet concentration parameter (for non-IID)
        seed: Random seed

    Returns:
        List of (X_client, y_client) tuples for each client
    """
    # Generate dataset
    X, y = generate_dataset(dataset_name, n_samples=n_samples, seed=seed)

    # Partition data
    if non_iid:
        client_datasets = partition_data_noniid(X, y, n_clients=n_clients, alpha=alpha, seed=seed)
    else:
        client_datasets = partition_data_iid(X, y, n_clients=n_clients, seed=seed)

    return client_datasets


if __name__ == "__main__":
    # Test the dataset generation
    client_data = get_federated_data(n_clients=5, n_samples=2000, non_iid=True, alpha=0.5)

    print(f"Number of clients: {len(client_data)}")
    for i, (X_client, y_client) in enumerate(client_data):
        print(f"Client {i}: {len(X_client)} samples, class distribution: {np.bincount(y_client.astype(int))}")
