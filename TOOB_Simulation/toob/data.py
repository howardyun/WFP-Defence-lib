from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def labels_to_int(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return np.argmax(labels, axis=1).astype(np.int64)
    return labels.astype(np.int64)


def load_npz_dataset(path: str | Path, *, data_key: str = "data", labels_key: str = "labels") -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        if data_key not in npz:
            raise KeyError(f"{path} does not contain data key {data_key!r}")
        if labels_key not in npz:
            raise KeyError(f"{path} does not contain labels key {labels_key!r}")
        data = np.asarray(npz[data_key])
        labels = labels_to_int(np.asarray(npz[labels_key]))
    return data, labels


def save_npz_dataset(path: str | Path, **arrays: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


class BurstDataset(Dataset):
    def __init__(
        self,
        bursts: np.ndarray,
        labels: np.ndarray,
        pseudo_labels: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        labels = labels_to_int(labels)
        if pseudo_labels is not None:
            pseudo_labels = labels_to_int(pseudo_labels)

        if indices is None:
            indices = np.arange(len(labels))

        self.bursts = torch.from_numpy(np.asarray(bursts[indices], dtype=np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.pseudo_labels = (
            torch.from_numpy(pseudo_labels[indices].astype(np.int64))
            if pseudo_labels is not None
            else None
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        if self.pseudo_labels is None:
            return self.bursts[index], self.labels[index]
        return self.bursts[index], self.labels[index], self.pseudo_labels[index]


def select_by_pseudo(pseudo_labels: np.ndarray, pseudo_label: int) -> np.ndarray:
    pseudo = labels_to_int(pseudo_labels)
    return np.flatnonzero(pseudo == pseudo_label)

