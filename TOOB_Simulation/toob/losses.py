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


def overhead_budget_loss(
    original: torch.Tensor,
    defended: torch.Tensor,
    target: float,
    *,
    mode: str = "hinge",
    tolerance: float = 0.0,
) -> torch.Tensor:
    """Bandwidth loss for training-time overhead control.

    ``hinge`` preserves the old behavior: under-budget samples are not
    penalized. ``target_l1`` and ``target_l2`` penalize both underuse and
    overuse, which is useful when you want a point on the bandwidth-accuracy
    curve to land near a chosen target.
    """
    overhead = overhead_ratio(original, defended)
    if mode == "hinge":
        return torch.relu(overhead - target).mean()
    if mode == "target_l1":
        return torch.abs(overhead - target).mean()
    if mode == "target_l2":
        return torch.mean((overhead - target) ** 2)
    if mode == "band":
        lower = max(0.0, target - tolerance)
        upper = target + tolerance
        return (torch.relu(lower - overhead) + torch.relu(overhead - upper)).mean()
    raise ValueError(f"Unknown overhead loss mode: {mode}")


def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    if delta.shape[1] <= 1:
        return delta.new_tensor(0.0)
    return torch.mean(torch.abs(delta[:, 1:] - delta[:, :-1]))
