from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from .data import labels_to_int


def load_website_to_pseudo_label(path: str | Path) -> dict[int, int]:
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

    if "website_to_pseudo_label" in raw:
        raw = raw["website_to_pseudo_label"]

    mapping: dict[int, int] = {}
    for key, value in raw.items():
        site = int(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 1:
                raise ValueError(
                    "Expected website_to_pseudo_label values to be scalar pseudo labels. "
                    "Use a mapping like {'0': 0, '1': 0, '2': 1}."
                )
            mapping[site] = int(value[0])
        else:
            mapping[site] = int(value)
    return mapping


def labels_to_pseudo(
    labels: np.ndarray,
    website_to_pseudo_label: dict[int, int],
    *,
    drop_unmapped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Map real website labels to cluster pseudo labels."""
    labels_int = labels_to_int(labels)
    pseudo = []
    keep_indices = []

    for idx, label in enumerate(labels_int):
        pseudo_value = website_to_pseudo_label.get(int(label))
        if pseudo_value is None:
            if drop_unmapped:
                continue
            raise KeyError(f"No pseudo label mapping found for website label {int(label)}")
        keep_indices.append(idx)
        pseudo.append(pseudo_value)
    return np.asarray(pseudo, dtype=np.int64), np.asarray(keep_indices, dtype=np.int64)


load_website_to_set = load_website_to_pseudo_label
