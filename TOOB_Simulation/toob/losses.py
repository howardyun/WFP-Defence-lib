from __future__ import annotations

import torch
import torch.nn.functional as F

from .burst import overhead_ratio


def untargeted_attack_loss(logits: torch.Tensor, labels: torch.Tensor, mode: str = "true_prob") -> torch.Tensor:
    """Loss minimized by the generator to reduce true-class confidence."""
    if mode == "true_prob":
        probs = torch.softmax(logits, dim=1)
        return probs.gather(1, labels.view(-1, 1)).mean()
    if mode == "true_logit":
        return logits.gather(1, labels.view(-1, 1)).mean()
    if mode == "negative_ce":
        return -F.cross_entropy(logits, labels)
    raise ValueError(f"Unknown attack loss mode: {mode}")


def overhead_hinge_loss(original: torch.Tensor, defended: torch.Tensor, threshold: float) -> torch.Tensor:
    overhead = overhead_ratio(original, defended)
    return torch.relu(overhead - threshold).mean()


def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    if delta.shape[1] <= 1:
        return delta.new_tensor(0.0)
    return torch.mean(torch.abs(delta[:, 1:] - delta[:, :-1]))

