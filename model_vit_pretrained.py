"""
Vision Transformer with ImageNet pre-trained weights for CIFAR-10 federated learning.

Uses torchvision's pre-trained ViT-B/16 model and adapts it for CIFAR-10:
- Resizes CIFAR-10 images from 32x32 to 224x224 (ImageNet size)
- Replaces classification head for 10 classes
- For DictPFL: Uses pre-trained weights
- For baselines: Trains from scratch (random initialization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Dict
import numpy as np


class ViTForCIFAR10(nn.Module):
    """
    Vision Transformer adapted for CIFAR-10 using ImageNet pre-trained weights.

    Architecture:
    - Base: ViT-B/16 (patch size 16x16)
    - Input: 224x224 (upscaled from CIFAR-10's 32x32)
    - Hidden dim: 768
    - Transformer layers: 12
    - Attention heads: 12
    - Total parameters: ~86M
    """

    def __init__(self, num_classes=10, use_pretrained=True):
        super().__init__()

        if use_pretrained:
            # Load pre-trained ViT-B/16 from ImageNet
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = vit_b_16(weights=weights)
            print("✓ Loaded ImageNet pre-trained ViT-B/16 weights")
        else:
            # Random initialization for baselines
            self.vit = vit_b_16(weights=None)
            print("✗ Using random initialization (training from scratch)")

        # Replace classification head for CIFAR-10 (10 classes)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

        # Image resizing transform (32x32 -> 224x224)
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # x: (B, 3, 32, 32) - CIFAR-10 images
        # Resize to ImageNet size
        x = self.resize(x)  # (B, 3, 224, 224)

        # ViT forward pass
        return self.vit(x)


def get_model_params_vit(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract model parameters as numpy arrays."""
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.cpu().numpy().copy()
    return params


def set_model_params_vit(model: nn.Module, params: Dict[str, np.ndarray]):
    """Set model parameters from numpy arrays."""
    for name, param in model.named_parameters():
        if name in params:
            param.data = torch.from_numpy(params[name].copy()).float()


def train_one_epoch_vit(model: nn.Module, dataloader, lr: float = 0.0001, device='cpu'):
    """
    Train ViT model for one epoch.

    Note: Using lower learning rate (0.0001) for fine-tuning pre-trained model.
    """
    model.train()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    n_batches = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_model_vit(model: nn.Module, dataloader, device='cpu') -> float:
    """Evaluate ViT model accuracy."""
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the pre-trained ViT model
    print("Testing ImageNet Pre-trained ViT for CIFAR-10...")

    # Test with pre-trained weights
    print("\n1. With ImageNet pre-trained weights:")
    model_pretrained = ViTForCIFAR10(num_classes=10, use_pretrained=True)
    total_params = count_parameters(model_pretrained)
    print(f"   Total parameters: {total_params:,}")

    # Test with random initialization
    print("\n2. With random initialization:")
    model_scratch = ViTForCIFAR10(num_classes=10, use_pretrained=False)

    # Test forward pass with CIFAR-10 sized input
    x = torch.randn(2, 3, 32, 32)  # CIFAR-10 size
    print(f"\n3. Forward pass test:")
    print(f"   Input shape: {x.shape}")

    output = model_pretrained(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Output logits (sample): {output[0].detach().numpy()}")

    # Show some parameter names and shapes
    print(f"\n4. Sample parameter breakdown:")
    for i, (name, param) in enumerate(model_pretrained.named_parameters()):
        if i < 10:  # Show first 10 parameters
            print(f"   {name}: {tuple(param.shape)} = {param.numel():,} params")
        elif i == 10:
            print(f"   ... ({total_params:,} total parameters)")
            break
