"""Template for connecting a trained PyTorch DF checkpoint.

Copy this file, replace `DFNet` with your actual DF model definition, and pass:

    --detector-builder C:/path/to/your_df_builder.py:build_model

The returned model must accept the tensor layout selected by
`--detector-input-layout`.
"""

from __future__ import annotations

import torch
from torch import nn


class DFNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Replace this template with your real DF architecture.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_model(num_classes: int) -> nn.Module:
    return DFNet(num_classes=num_classes)

