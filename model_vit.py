"""
Vision Transformer (ViT) model for federated learning demo.
DictPFL uses pre-trained weights, while baselines train from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np


class SimpleViT(nn.Module):
    """
    Simplified Vision Transformer for CIFAR-10.

    This is a lightweight ViT suitable for the demo:
    - Image size: 32x32 (CIFAR-10)
    - Patch size: 4x4
    - Embedding dim: 128
    - Num heads: 4
    - Num layers: 4
    """

    def __init__(self, image_size=32, patch_size=4, num_classes=10,
                 dim=128, depth=4, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # 3 channels

        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        B = x.shape[0]

        # Create patches: (B, num_patches, patch_dim)
        patches = self.extract_patches(x)

        # Patch embedding: (B, num_patches, dim)
        x = self.patch_to_embedding(patches)

        # Add CLS token: (B, num_patches+1, dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embedding[:, :(x.size(1))]

        # Transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)

        # Classification
        x = self.layer_norm(x)
        cls_token_final = x[:, 0]  # Take CLS token
        return self.mlp_head(cls_token_final)

    def extract_patches(self, x):
        """Extract patches from image."""
        B, C, H, W = x.shape
        ps = self.patch_size

        # Reshape to patches
        x = x.reshape(B, C, H // ps, ps, W // ps, ps)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//ps, W//ps, C, ps, ps)
        x = x.reshape(B, (H // ps) * (W // ps), C * ps * ps)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""

    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()

        self.attention = MultiHeadAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, heads, dropout):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        assert dim % heads == 0, "Dimension must be divisible by number of heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


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


def train_one_epoch_vit(model: nn.Module, dataloader, lr: float = 0.001, device='cpu'):
    """Train ViT model for one epoch."""
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


def load_pretrained_vit(model: nn.Module, pretrain_epochs: int = 5,
                       train_loader=None, device='cpu'):
    """
    Pre-train the ViT model on the dataset.
    This is used for DictPFL to have a good initialization.
    """
    print(f"Pre-training ViT for {pretrain_epochs} epochs...")
    model = model.to(device)

    for epoch in range(pretrain_epochs):
        loss = train_one_epoch_vit(model, train_loader, lr=0.001, device=device)
        print(f"  Pre-train epoch {epoch+1}/{pretrain_epochs}: loss={loss:.4f}")

    print("Pre-training complete!")
    return model


if __name__ == "__main__":
    # Test the ViT model
    print("Testing SimpleViT...")

    model = SimpleViT(image_size=32, patch_size=4, num_classes=10,
                     dim=128, depth=4, heads=4, mlp_dim=256)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Show parameter breakdown
    print("\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} = {param.numel():,} parameters")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output[0].detach().numpy()}")
