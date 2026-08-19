from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class GeneratorConfig:
    burst_len: int = 2000
    hidden_dims: tuple[int, ...] = (512, 512, 1024, 1024)
    dropout: float = 0.05

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratorConfig":
        copied = dict(data)
        copied.pop("noise_dim", None)
        copied["hidden_dims"] = tuple(copied.get("hidden_dims", (512, 512, 1024, 1024)))
        return cls(**copied)


class BurstGenerator(nn.Module):
    """MLP generator that maps a burst trace to non-negative burst padding amounts."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = config.burst_len
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, config.burst_len))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_generator_from_checkpoint(checkpoint: dict, map_location: str | torch.device = "cpu") -> BurstGenerator:
    config = GeneratorConfig.from_dict(checkpoint["config"])
    model = BurstGenerator(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(map_location)
    return model

