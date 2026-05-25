"""
autoencoder/encoder.py — Deterministic CNN encoder for image feature extraction.

NOT a VAE: no mu/log_var, no sampling, no reparameterization.
Simply compresses a (3, 80, 160) image into a 95-dim latent vector.
Input is normalized: x = x.float() / 255.0.
"""

import torch
import torch.nn as nn

import parameters as params


class CNNEncoder(nn.Module):
    """Deterministic CNN encoder: image → fixed-dim latent vector."""

    def __init__(self, latent_dim=None):
        super().__init__()
        self.latent_dim = latent_dim or params.LATENT_DIM  # 95

        # 4-layer CNN
        # Input: (B, 3, 80, 160)
        self.conv = nn.Sequential(
            # Layer 1: (3, 80, 160) → (32, 40, 80)
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: (32, 40, 80) → (64, 20, 40)
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: (64, 20, 40) → (128, 10, 20)
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: (128, 10, 20) → (256, 5, 10)
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size: 256 * 5 * 10 = 12800
        self._flat_dim = 256 * 5 * 10

        # FC to latent vector
        self.fc = nn.Sequential(
            nn.Linear(self._flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.latent_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 80, 160) uint8 or float tensor.
               If values > 1.0, assumed to be uint8 range → normalized.
        Returns:
            latent: (B, latent_dim) feature vector.
        """
        # Normalize input
        if x.dtype == torch.uint8 or x.max() > 1.0:
            x = x.float() / 255.0

        features = self.conv(x)
        features = features.reshape(features.size(0), -1)  # flatten
        latent = self.fc(features)
        return latent
