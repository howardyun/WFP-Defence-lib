from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from .data import labels_to_int


def load_website_to_set(path: str | Path) -> dict[int, list[int]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path, allow_pickle=True).item()
    elif suffix == ".json":
        text = path.read_text(encoding="utf-8")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = ast.literal_eval(text)
    else:
        raise ValueError(f"Unsupported mapping format: {path}")

    mapping: dict[int, list[int]] = {}
    for key, value in raw.items():
        site = int(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            mapping[site] = [int(v) for v in value]
        else:
            mapping[site] = [int(value)]
    return mapping


def labels_to_pseudo(
    labels: np.ndarray,
    website_to_set: dict[int, list[int]],
    *,
    strategy: str = "first",
    seed: int = 1,
    drop_unmapped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Map real website labels to cluster pseudo labels."""
    labels_int = labels_to_int(labels)
    rng = np.random.default_rng(seed)
    pseudo = []
    keep_indices = []

    for idx, label in enumerate(labels_int):
        sets = website_to_set.get(int(label))
        if not sets:
            if drop_unmapped:
                continue
            raise KeyError(f"No pseudo label mapping found for website label {int(label)}")
        if strategy == "first":
            pseudo_value = sets[0]
        elif strategy == "random":
            pseudo_value = rng.choice(sets)
        elif strategy == "round_robin":
            pseudo_value = sets[idx % len(sets)]
        else:
            raise ValueError(f"Unknown pseudo-label strategy: {strategy}")
        keep_indices.append(idx)
        pseudo.append(pseudo_value)
    return np.asarray(pseudo, dtype=np.int64), np.asarray(keep_indices, dtype=np.int64)
