from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import labels_to_int


class BurstAutoencoder(nn.Module):
    """MLP autoencoder that compresses burst traces to a low-dimensional latent.

    The bottleneck is used as a learned, dense representation of each burst
    trace, so clustering runs on a compact vector instead of the raw sparse
    burst counts.
    """

    def __init__(
        self,
        burst_len: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
    ) -> None:
        super().__init__()
        enc_layers: list[nn.Module] = []
        in_dim = burst_len
        for hidden_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden_dim))
            enc_layers.append(nn.ReLU())
            in_dim = hidden_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden_dim))
            dec_layers.append(nn.ReLU())
            in_dim = hidden_dim
        dec_layers.append(nn.Linear(in_dim, burst_len))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def normalize_bursts(bursts: np.ndarray) -> np.ndarray:
    """Scale each burst trace to [-1, 1] before feeding the autoencoder."""
    data = np.asarray(bursts, dtype=np.float32)
    max_abs = np.abs(data).max(axis=1, keepdims=True)
    max_abs[max_abs < 1.0] = 1.0
    return data / max_abs


def train_autoencoder(
    bursts: np.ndarray,
    *,
    latent_dim: int,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dims: tuple[int, ...] = (1024, 512, 256),
    seed: int = 1,
    device: torch.device,
) -> BurstAutoencoder:
    model = BurstAutoencoder(bursts.shape[1], latent_dim, hidden_dims=hidden_dims).to(device)
    x_norm = normalize_bursts(bursts)
    x_tensor = torch.from_numpy(x_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = x_tensor.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    for _epoch in range(epochs):
        perm = torch.randperm(n, generator=generator, device=device)
        for start in range(0, n, batch_size):
            batch = x_tensor[perm[start:start + batch_size]]
            _z, recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def encode_website_profiles(
    model: BurstAutoencoder,
    bursts: np.ndarray,
    labels: np.ndarray,
    *,
    exclude_labels: set[int] | None = None,
    batch_size: int = 512,
    device: torch.device,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Encode every burst trace and average per website into profile vectors."""
    labels_int = labels_to_int(labels)
    exclude_labels = exclude_labels or set()
    x_norm = normalize_bursts(bursts)

    latent_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], batch_size):
            batch = torch.from_numpy(x_norm[start:start + batch_size]).to(device)
            latent_parts.append(model.encode(batch).cpu().numpy())
    latent = np.concatenate(latent_parts, axis=0)

    site_labels = sorted(set(int(v) for v in labels_int) - exclude_labels)
    profiles: dict[int, np.ndarray] = {}
    for site in site_labels:
        profiles[site] = latent[labels_int == site].mean(axis=0)
    return site_labels, profiles
