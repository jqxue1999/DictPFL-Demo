"""
Simple 2-layer MLP model for federated learning demo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np


class SimpleMLP(nn.Module):
    """
    Simple 2-layer Multi-Layer Perceptron for binary classification.

    Architecture:
        Input -> Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 2):
        """
        Initialize the MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
        """
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model_params(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    Extract model parameters as numpy arrays.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to numpy arrays
    """
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.cpu().numpy().copy()
    return params


def set_model_params(model: nn.Module, params: Dict[str, np.ndarray]):
    """
    Set model parameters from numpy arrays.

    Args:
        model: PyTorch model
        params: Dictionary mapping parameter names to numpy arrays
    """
    for name, param in model.named_parameters():
        if name in params:
            # Use .copy() to avoid sharing memory between numpy and torch
            param.data = torch.from_numpy(params[name].copy()).float()


def get_model_gradients(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    Extract model gradients as numpy arrays.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to gradient numpy arrays
    """
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.cpu().numpy().copy()
        else:
            grads[name] = np.zeros_like(param.data.cpu().numpy())
    return grads


def train_one_epoch(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   lr: float = 0.01, batch_size: int = 32) -> float:
    """
    Train model for one epoch on local data.

    Args:
        model: PyTorch model
        X: Features array
        y: Labels array
        lr: Learning rate
        batch_size: Batch size

    Returns:
        Average loss for the epoch
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Convert to tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    total_loss = 0.0
    n_batches = 0

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_X = X_tensor[batch_indices]
        batch_y = y_tensor[batch_indices]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate model accuracy on data.

    Args:
        model: PyTorch model
        X: Features array
        y: Labels array

    Returns:
        Accuracy (fraction correct)
    """
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean().item()

    return accuracy


if __name__ == "__main__":
    # Test the model
    model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
    print(f"Model: {model}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")

    # Test forward pass
    X_test = np.random.randn(10, 2)
    y_test = np.random.randint(0, 2, 10)

    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Random accuracy: {accuracy:.4f}")

    # Test training
    loss = train_one_epoch(model, X_test, y_test, lr=0.01, batch_size=5)
    print(f"Training loss: {loss:.4f}")
