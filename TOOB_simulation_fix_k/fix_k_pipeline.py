from __future__ import annotations

import csv
import json
import math
import os
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_optional(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    return value if value else None


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"", "0", "false", "no", "off"}


def env_words(name: str, default: str) -> list[str]:
    return [part for part in os.environ.get(name, default).split() if part]


def tag_value(value: Any) -> str:
    return str(value).replace(" ", "_").replace(".", "p").replace("-", "_")


def require_file(path: str | Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def log_command(argv: list[str]) -> None:
    print("+ " + shlex.join(argv), flush=True)


def labels_to_int(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return np.argmax(labels, axis=1).astype(np.int64)
    return labels.astype(np.int64)


def load_npz_dataset(path: str | Path, data_key: str = "data", labels_key: str = "labels") -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        if data_key not in npz:
            raise KeyError(f"{path} does not contain data key {data_key!r}; keys={list(npz.keys())}")
        if labels_key not in npz:
            raise KeyError(f"{path} does not contain labels key {labels_key!r}; keys={list(npz.keys())}")
        data = np.asarray(npz[data_key])
        labels = labels_to_int(np.asarray(npz[labels_key]))
    return data, labels


def save_npz(path: str | Path, **arrays: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def direction_to_burst(direction_data: np.ndarray, max_bursts: int) -> np.ndarray:
    directions = np.asarray(direction_data)
    if directions.ndim != 2:
        raise ValueError(f"direction_data must be 2-D, got {directions.shape}")

    bursts = np.zeros((directions.shape[0], max_bursts), dtype=np.float32)
    for row_idx, row in enumerate(directions):
        burst_idx = -1
        cur_sign = 0.0
        for value in row:
            sign = float(np.sign(value))
            if sign == 0.0:
                break
            if sign != cur_sign:
                burst_idx += 1
                cur_sign = sign
                if burst_idx >= max_bursts:
                    break
            bursts[row_idx, burst_idx] += sign
    return bursts


def direction_to_sign_sequence(direction_data: np.ndarray, trace_len: int) -> np.ndarray:
    directions = np.sign(np.asarray(direction_data, dtype=np.float32))
    if directions.ndim != 2:
        raise ValueError(f"direction_data must be 2-D, got {directions.shape}")
    if directions.shape[1] > trace_len:
        return directions[:, :trace_len].astype(np.float32, copy=False)
    if directions.shape[1] < trace_len:
        directions = np.pad(directions, ((0, 0), (0, trace_len - directions.shape[1])), mode="constant")
    return directions.astype(np.float32, copy=False)


def burst_to_direction(burst_data: np.ndarray, trace_len: int) -> np.ndarray:
    bursts = np.asarray(burst_data)
    if bursts.ndim != 2:
        raise ValueError(f"burst_data must be 2-D, got {bursts.shape}")
    directions = np.zeros((bursts.shape[0], trace_len), dtype=np.float32)
    for row_idx, row in enumerate(bursts):
        cursor = 0
        for burst in row:
            count = int(round(abs(float(burst))))
            if count <= 0:
                continue
            sign = 1.0 if burst > 0 else -1.0
            end = min(trace_len, cursor + count)
            directions[row_idx, cursor:end] = sign
            cursor = end
            if cursor >= trace_len:
                break
    return directions


def summarize_bursts(bursts: np.ndarray) -> str:
    return (
        f"BurstStats(traces={bursts.shape[0]}, max_bursts={bursts.shape[1]}, "
        f"nonzero_bursts={int(np.count_nonzero(bursts))}, "
        f"max_abs_burst={float(np.max(np.abs(bursts))) if bursts.size else 0.0})"
    )


def apply_burst_perturbation(burst_batch: torch.Tensor, delta_batch: torch.Tensor, round_output: bool = False) -> torch.Tensor:
    adv = burst_batch + delta_batch * torch.sign(burst_batch)
    return torch.round(adv) if round_output else adv


def overhead_ratio(original: torch.Tensor, defended: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    added = torch.sum(torch.abs(defended) - torch.abs(original), dim=1)
    base = torch.sum(torch.abs(original), dim=1).clamp_min(eps)
    return added / base


def soft_burst_to_direction(burst_batch: torch.Tensor, trace_len: int, tau: float, chunk_size: int) -> torch.Tensor:
    lengths = torch.abs(burst_batch).clamp_min(0)
    signs = torch.sign(burst_batch)
    ends = torch.cumsum(lengths, dim=1)
    starts = ends - lengths
    positions = torch.arange(trace_len, device=burst_batch.device, dtype=burst_batch.dtype).view(1, 1, trace_len) + 0.5
    projected = torch.zeros((burst_batch.shape[0], trace_len), device=burst_batch.device, dtype=burst_batch.dtype)
    for start_idx in range(0, burst_batch.shape[1], chunk_size):
        end_idx = min(start_idx + chunk_size, burst_batch.shape[1])
        chunk_starts = starts[:, start_idx:end_idx].unsqueeze(-1)
        chunk_ends = ends[:, start_idx:end_idx].unsqueeze(-1)
        chunk_signs = signs[:, start_idx:end_idx].unsqueeze(-1)
        occupancy = torch.sigmoid((positions - chunk_starts) / tau) - torch.sigmoid((positions - chunk_ends) / tau)
        projected = projected + torch.sum(chunk_signs * occupancy, dim=1)
    return projected.clamp(-1.0, 1.0)


def hard_burst_to_direction(burst_batch: torch.Tensor, trace_len: int, chunk_size: int) -> torch.Tensor:
    lengths = torch.round(torch.abs(burst_batch)).clamp_min(0)
    signs = torch.sign(burst_batch)
    ends = torch.cumsum(lengths, dim=1)
    starts = ends - lengths
    positions = torch.arange(trace_len, device=burst_batch.device, dtype=burst_batch.dtype).view(1, 1, trace_len) + 0.5
    projected = torch.zeros((burst_batch.shape[0], trace_len), device=burst_batch.device, dtype=burst_batch.dtype)
    for start_idx in range(0, burst_batch.shape[1], chunk_size):
        end_idx = min(start_idx + chunk_size, burst_batch.shape[1])
        chunk_starts = starts[:, start_idx:end_idx].unsqueeze(-1)
        chunk_ends = ends[:, start_idx:end_idx].unsqueeze(-1)
        chunk_signs = signs[:, start_idx:end_idx].unsqueeze(-1)
        occupancy = ((positions > chunk_starts) & (positions <= chunk_ends)).to(burst_batch.dtype)
        projected = projected + torch.sum(chunk_signs * occupancy, dim=1)
    return projected.clamp(-1.0, 1.0)


def ste_burst_to_direction(burst_batch: torch.Tensor, trace_len: int, tau: float, chunk_size: int) -> torch.Tensor:
    soft = soft_burst_to_direction(burst_batch, trace_len=trace_len, tau=tau, chunk_size=chunk_size)
    hard = hard_burst_to_direction(burst_batch, trace_len=trace_len, chunk_size=chunk_size)
    return soft + (hard - soft).detach()


def format_detector_input(x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "nl":
        return x
    if layout == "ncl":
        return x.unsqueeze(1)
    if layout == "nchw":
        return x.unsqueeze(1).unsqueeze(-1)
    raise ValueError(f"Unknown detector input layout: {layout}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, pool_size: int, pool_stride: int, dropout_p: float, activation: type[nn.Module]):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.MaxPool1d(pool_size, pool_stride, padding=0),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DF(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        filter_num = [32, 64, 128, 256]
        kernel_size = 8
        pool_size = 8
        pool_stride_size = 4
        length_after_extraction = 18
        self.feature_extraction = nn.Sequential(
            ConvBlock(1, filter_num[0], kernel_size, 1, pool_size, pool_stride_size, 0.1, nn.ELU),
            ConvBlock(filter_num[0], filter_num[1], kernel_size, 1, pool_size, pool_stride_size, 0.1, nn.ReLU),
            ConvBlock(filter_num[1], filter_num[2], kernel_size, 1, pool_size, pool_stride_size, 0.1, nn.ReLU),
            ConvBlock(filter_num[2], filter_num[3], kernel_size, 1, pool_size, pool_stride_size, 0.1, nn.ReLU),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filter_num[3] * length_after_extraction, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extraction(x))


def load_detector(checkpoint_path: str | Path, num_classes: int, device: torch.device) -> DF:
    model = DF(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if isinstance(checkpoint, dict):
        checkpoint = {str(key).removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@dataclass
class GeneratorConfig:
    noise_dim: int
    burst_len: int
    hidden_dims: tuple[int, ...] = (512, 512, 1024, 1024)
    dropout: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorConfig":
        copied = dict(data)
        copied["hidden_dims"] = tuple(copied.get("hidden_dims", (512, 512, 1024, 1024)))
        return cls(**copied)


class BurstGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = config.noise_dim
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_generator_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> BurstGenerator:
    model = BurstGenerator(GeneratorConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def untargeted_attack_loss(logits: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "true_prob":
        probs = torch.softmax(logits, dim=1)
        return probs.gather(1, labels.view(-1, 1)).mean()
    if mode == "true_logit":
        return logits.gather(1, labels.view(-1, 1)).mean()
    if mode == "negative_ce":
        return -F.cross_entropy(logits, labels)
    raise ValueError(f"Unknown attack loss: {mode}")


def overhead_budget_loss(original: torch.Tensor, defended: torch.Tensor, target: float, mode: str, tolerance: float) -> torch.Tensor:
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
    raise ValueError(f"Unknown overhead loss: {mode}")


def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    if delta.shape[1] <= 1:
        return delta.new_tensor(0.0)
    return torch.mean(torch.abs(delta[:, 1:] - delta[:, :-1]))


class BurstDataset(Dataset):
    def __init__(self, bursts: np.ndarray, labels: np.ndarray, pseudo_labels: np.ndarray, indices: np.ndarray):
        self.bursts = torch.from_numpy(np.asarray(bursts[indices], dtype=np.float32))
        self.labels = torch.from_numpy(labels_to_int(labels[indices]).astype(np.int64))
        self.pseudo_labels = torch.from_numpy(labels_to_int(pseudo_labels[indices]).astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.bursts[index], self.labels[index], self.pseudo_labels[index]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    noise_dim: int
    overhead_threshold: float
    lambda_overhead: float
    overhead_loss: str
    overhead_tolerance: float
    lambda_tv: float
    attack_loss: str
    trace_len: int
    detector_input_layout: str
    projection_mode: str
    soft_projection_tau: float
    projection_chunk_size: int
    seed: int


def detector_input_from_bursts(bursts: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    if config.projection_mode == "soft":
        features = soft_burst_to_direction(bursts, config.trace_len, config.soft_projection_tau, config.projection_chunk_size)
    elif config.projection_mode == "hard":
        features = hard_burst_to_direction(bursts, config.trace_len, config.projection_chunk_size)
    elif config.projection_mode == "ste":
        features = ste_burst_to_direction(bursts, config.trace_len, config.soft_projection_tau, config.projection_chunk_size)
    else:
        raise ValueError(f"Unknown projection mode: {config.projection_mode}")
    return format_detector_input(features, config.detector_input_layout)


def select_by_pseudo(pseudo_labels: np.ndarray, pseudo_label: int) -> np.ndarray:
    return np.flatnonzero(labels_to_int(pseudo_labels) == int(pseudo_label))


def train_one_generator(
    *,
    pseudo_label: int,
    bursts: np.ndarray,
    labels: np.ndarray,
    pseudo_labels: np.ndarray,
    detector: nn.Module,
    output_dir: Path,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    indices = select_by_pseudo(pseudo_labels, pseudo_label)
    if len(indices) == 0:
        raise ValueError(f"Pseudo label {pseudo_label} has no samples")

    dataset = BurstDataset(bursts, labels, pseudo_labels, indices)
    generator = torch.Generator().manual_seed(config.seed + int(pseudo_label))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)

    gen_config = GeneratorConfig(noise_dim=config.noise_dim, burst_len=bursts.shape[1])
    model = BurstGenerator(gen_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        rows = []
        progress = tqdm(loader, desc=f"pseudo={pseudo_label} epoch={epoch}", leave=False)
        for x, y, _ in progress:
            x = x.to(device)
            y = y.to(device)
            z = torch.randn(x.shape[0], config.noise_dim, device=device)
            delta = model(z)
            adv = apply_burst_perturbation(x, delta)
            logits = detector(detector_input_from_bursts(adv, config))

            attack = untargeted_attack_loss(logits, y, config.attack_loss)
            overhead_loss = overhead_budget_loss(x, adv, config.overhead_threshold, config.overhead_loss, config.overhead_tolerance)
            tv = total_variation_loss(delta)
            loss = attack + config.lambda_overhead * overhead_loss + config.lambda_tv * tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                true_prob = probs.gather(1, y.view(-1, 1)).mean()
                adv_pred = torch.argmax(logits, dim=1)
                adv_acc = torch.mean((adv_pred == y).float())
                overhead = overhead_ratio(x, adv).mean()
                clean_logits = detector(detector_input_from_bursts(x, config))
                clean_probs = torch.softmax(clean_logits, dim=1)
                clean_true_prob = clean_probs.gather(1, y.view(-1, 1)).mean()
                clean_acc = torch.mean((torch.argmax(clean_logits, dim=1) == y).float())

            row = {
                "loss": float(loss.detach().cpu()),
                "attack": float(attack.detach().cpu()),
                "overhead_loss": float(overhead_loss.detach().cpu()),
                "tv": float(tv.detach().cpu()),
                "overhead": float(overhead.detach().cpu()),
                "true_prob": float(true_prob.detach().cpu()),
                "adv_acc": float(adv_acc.detach().cpu()),
                "attack_success": float((1.0 - adv_acc).detach().cpu()),
                "clean_acc": float(clean_acc.detach().cpu()),
                "clean_true_prob": float(clean_true_prob.detach().cpu()),
            }
            rows.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4f}", overhead=f"{row['overhead']:.3f}", adv_acc=f"{row['adv_acc']:.3f}")

        summary = {"epoch": epoch, "samples": int(len(indices))}
        for key in rows[0]:
            summary[key] = float(np.mean([row[key] for row in rows]))
        history.append(summary)
        print(
            f"pseudo={pseudo_label} epoch={epoch}/{config.epochs} "
            f"loss={summary['loss']:.4f} attack={summary['attack']:.4f} "
            f"overhead_loss={summary['overhead_loss']:.4f} tv={summary['tv']:.4f} "
            f"overhead={summary['overhead']:.3f} clean_acc={summary['clean_acc']:.3f} "
            f"adv_acc={summary['adv_acc']:.3f} attack_success={summary['attack_success']:.3f} "
            f"clean_true_prob={summary['clean_true_prob']:.3f} true_prob={summary['true_prob']:.3f}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"generator_pseudo_{pseudo_label}.pt"
    torch.save(
        {
            "pseudo_label": int(pseudo_label),
            "config": gen_config.to_dict(),
            "train_config": asdict(config),
            "state_dict": model.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    (output_dir / f"generator_pseudo_{pseudo_label}.history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"pseudo_label": int(pseudo_label), "samples": int(len(indices)), "checkpoint": str(checkpoint_path), "history": history}


def compute_profile(bursts: np.ndarray, method: str) -> np.ndarray:
    if method == "super":
        return np.concatenate([np.mean(np.abs(bursts), axis=0).ravel(), np.mean(bursts, axis=0).ravel()])
    if method == "mean_abs":
        return np.mean(np.abs(bursts), axis=0).ravel()
    if method == "mean_signed":
        return np.mean(bursts, axis=0).ravel()
    raise ValueError(f"Unknown profile method: {method}")


def build_website_profiles(
    bursts: np.ndarray,
    labels: np.ndarray,
    exclude_labels: set[int],
    profile_method: str,
) -> tuple[list[int], np.ndarray, dict[int, int]]:
    labels_int = labels_to_int(labels)
    site_labels = sorted(set(int(value) for value in labels_int) - exclude_labels)
    profiles = []
    sample_counts = {}
    for label in site_labels:
        mask = labels_int == label
        sample_counts[label] = int(np.sum(mask))
        profiles.append(compute_profile(bursts[mask], profile_method))
    if not profiles:
        raise ValueError("No website labels left after exclusions")
    return site_labels, np.asarray(profiles, dtype=np.float32), sample_counts


def normalize_profiles(x: np.ndarray, method: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if method == "none":
        return x
    if method == "zscore":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        return (x - mean) / np.maximum(std, 1e-6)
    if method == "l2":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-6)
    raise ValueError(f"Unknown normalize: {method}")


def kmeans_once(x: np.ndarray, k: int, rng: np.random.Generator, max_iter: int) -> tuple[np.ndarray, np.ndarray, float]:
    n = x.shape[0]
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members) == 0:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                centers[cluster_id] = x[farthest]
            else:
                centers[cluster_id] = members.mean(axis=0)
    distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    inertia = float(np.sum(np.min(distances, axis=1)))
    return labels, centers, inertia


def best_kmeans(x: np.ndarray, k: int, seed: int, restarts: int, max_iter: int) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for offset in range(restarts):
        rng = np.random.default_rng(seed + offset * 1009)
        result = kmeans_once(x, k, rng, max_iter)
        if best is None or result[2] < best[2]:
            best = result
    assert best is not None
    return best


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def silhouette(distance_matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    values = []
    for i in range(distance_matrix.shape[0]):
        own = labels == labels[i]
        other = ~own
        if np.sum(own) <= 1 or not np.any(other):
            values.append(0.0)
            continue
        a = float(np.mean(distance_matrix[i, own & (np.arange(len(labels)) != i)]))
        b = min(float(np.mean(distance_matrix[i, labels == label])) for label in set(labels[other]))
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values)), float(np.min(values))


def evaluate_fixed_k(
    bursts: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    exclude_labels: set[int],
    profile_method: str,
    normalize: str,
    seed: int,
    restarts: int,
    max_iter: int,
    min_cluster_size: int,
) -> dict[str, Any]:
    site_labels, profiles, sample_counts = build_website_profiles(bursts, labels, exclude_labels, profile_method)
    x = normalize_profiles(profiles, normalize)
    cluster_labels, centers, inertia = best_kmeans(x, k, seed + k * 997, restarts, max_iter)
    distance_matrix = pairwise_distances(x)
    sil_mean, sil_min = silhouette(distance_matrix, cluster_labels)

    pseudo_label_to_websites: dict[str, list[int]] = {}
    site_counts = []
    sample_count_values = []
    for cluster_id in range(k):
        websites = [int(site_labels[index]) for index in np.flatnonzero(cluster_labels == cluster_id)]
        pseudo_label_to_websites[str(cluster_id)] = sorted(websites)
        site_counts.append(len(websites))
        sample_count_values.append(sum(sample_counts[website] for website in websites))

    min_sites = min(site_counts)
    max_sites = max(site_counts)
    balance = min_sites / max(max_sites, 1)
    small_penalty = max(0.0, (min_cluster_size - min_sites) / max(min_cluster_size, 1))
    selection_score = sil_mean - 0.10 * (1.0 - balance) - 0.25 * small_penalty
    return {
        "k": int(k),
        "derived_set_size": int(math.ceil(len(site_labels) / k)),
        "num_websites": int(len(site_labels)),
        "silhouette_mean": float(sil_mean),
        "silhouette_min": float(sil_min),
        "inertia": float(inertia),
        "cluster_site_count_min": int(min_sites),
        "cluster_site_count_max": int(max_sites),
        "cluster_site_count_std": float(np.std(site_counts)),
        "cluster_sample_count_min": int(min(sample_count_values)),
        "cluster_sample_count_max": int(max(sample_count_values)),
        "cluster_sample_count_std": float(np.std(sample_count_values)),
        "balance_ratio": float(balance),
        "selection_score": float(selection_score),
        "pseudo_label_to_websites": pseudo_label_to_websites,
    }


def website_to_pseudo_from_row(row: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for pseudo_label, websites in row["pseudo_label_to_websites"].items():
        for website in websites:
            mapping[int(website)] = int(pseudo_label)
    return dict(sorted(mapping.items()))


def write_fixed_k_mapping(row: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cluster_count_probe.csv"
    json_path = output_dir / "cluster_count_probe.json"
    recommendation_path = output_dir / "recommendation.json"
    mapping_path = output_dir / "mapping_recommended.json"

    flat = {key: value for key, value in row.items() if key != "pseudo_label_to_websites"}
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    json_path.write_text(json.dumps([row], indent=2), encoding="utf-8")
    recommendation_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    mapping = {
        "k": int(row["k"]),
        "derived_set_size": int(row["derived_set_size"]),
        "website_to_pseudo_label": {str(k): int(v) for k, v in website_to_pseudo_from_row(row).items()},
        "pseudo_label_to_websites": row["pseudo_label_to_websites"],
        "cluster_quality": {key: row[key] for key in row if key not in {"pseudo_label_to_websites"}},
    }
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(
        f"fixed-K mapping: K={row['k']}, derived_set_size~{row['derived_set_size']}, "
        f"silhouette={row['silhouette_mean']:.6f}, balance={row['balance_ratio']:.3f}"
    )
    print(f"saved mapping: {mapping_path}")
    return mapping_path


def load_mapping(path: str | Path) -> dict[int, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "website_to_pseudo_label" not in data:
        raise KeyError(f"{path} does not contain website_to_pseudo_label")
    return {int(k): int(v) for k, v in data["website_to_pseudo_label"].items()}


def labels_to_pseudo(labels: np.ndarray, mapping: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pseudo = []
    keep = []
    for index, label in enumerate(labels_to_int(labels)):
        pseudo_label = mapping.get(int(label))
        if pseudo_label is None:
            continue
        keep.append(index)
        pseudo.append(pseudo_label)
    return np.asarray(pseudo, dtype=np.int64), np.asarray(keep, dtype=np.int64)


def cluster_quality_for_mapping(bursts: np.ndarray, labels: np.ndarray, mapping: dict[int, int], profile_method: str, exclude_labels: set[int]) -> dict[str, Any]:
    site_labels, profiles, sample_counts = build_website_profiles(bursts, labels, exclude_labels, profile_method)
    mapped_labels = np.asarray([mapping[int(site)] for site in site_labels], dtype=np.int64)
    distances = pairwise_distances(profiles)
    sil_mean, sil_min = silhouette(distances, mapped_labels)
    cluster_site_counts = []
    cluster_sample_counts = []
    for pseudo_label in sorted(set(mapped_labels.tolist())):
        websites = [int(site) for site, mapped in zip(site_labels, mapped_labels) if mapped == pseudo_label]
        cluster_site_counts.append(len(websites))
        cluster_sample_counts.append(sum(sample_counts[website] for website in websites))
    return {
        "silhouette_mean": float(sil_mean),
        "silhouette_min": float(sil_min),
        "cluster_website_count": {"min": int(min(cluster_site_counts)), "max": int(max(cluster_site_counts))},
        "cluster_sample_count": {"min": int(min(cluster_sample_counts)), "max": int(max(cluster_sample_counts))},
    }


@dataclass(frozen=True)
class Config:
    mode: str
    out_root: Path
    cluster_counts: list[int]
    train_dataset: str
    train_data_key: str
    train_labels_key: str
    valid_dataset: str | None
    valid_data_key: str
    valid_labels_key: str
    burst_limit: int | None
    valid_limit: int | None
    max_bursts: int
    trace_len: int
    profile_method: str
    normalize: str
    exclude_labels: list[int]
    eval_exclude_labels: list[int]
    detector_checkpoint: str
    num_classes: int
    detector_input_layout: str
    overhead_targets: list[float]
    lambda_overheads: list[float]
    overhead_losses: list[str]
    overhead_tolerance: float
    attack_losses: list[str]
    lambda_tv: float
    projection_mode: str
    soft_projection_tau: float
    projection_chunk_size: int
    lr: float
    epochs: int
    batch_size: int
    noise_dim: int
    seed: int
    device: str | None
    reuse_intermediates: bool
    skip_existing: bool
    run_eval: bool
    budget_slack: float
    select_metric: str
    metrics: list[str]
    average: str
    kmeans_restarts: int
    kmeans_max_iter: int
    min_cluster_size: int


def build_config(mode: str) -> Config:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    smoke_limit = env_int("SMOKE_LIMIT", 200)
    burst_limit = int(env_optional("BURST_LIMIT", str(smoke_limit if mode == "smoke" else "")) or 0) or None
    valid_limit = int(env_optional("VALID_LIMIT", str(smoke_limit if mode == "smoke" else "")) or 0) or None
    return Config(
        mode=mode,
        out_root=Path(env_str("OUT_ROOT", "TOOB_simulation_fix_k/outputs")),
        cluster_counts=[int(value) for value in env_words("CLUSTER_COUNTS", "4")],
        train_dataset=env_str("TRAIN_DATASET", env_str("DATASET", "TOOB_simulation_fix_k/data/raw/train.npz")),
        train_data_key=env_str("TRAIN_DATA_KEY", env_str("DATA_KEY", "X")),
        train_labels_key=env_str("TRAIN_LABELS_KEY", env_str("LABELS_KEY", "y")),
        valid_dataset=env_optional("VALID_DATASET", "TOOB_simulation_fix_k/data/raw/valid.npz"),
        valid_data_key=env_str("VALID_DATA_KEY", env_str("DATA_KEY", "X")),
        valid_labels_key=env_str("VALID_LABELS_KEY", env_str("LABELS_KEY", "y")),
        burst_limit=burst_limit,
        valid_limit=valid_limit,
        max_bursts=env_int("MAX_BURSTS", 2000),
        trace_len=env_int("TRACE_LEN", 5000),
        profile_method=env_str("PROFILE_METHOD", "super"),
        normalize=env_str("NORMALIZE", "zscore"),
        exclude_labels=[int(value) for value in env_words("EXCLUDE_LABELS", "95")],
        eval_exclude_labels=[int(value) for value in env_words("EVAL_EXCLUDE_LABELS", env_str("EXCLUDE_LABELS", "95"))],
        detector_checkpoint=env_str("DF_CHECKPOINT", "TOOB_simulation_fix_k/checkpoints/df_cw/max_f1.pth"),
        num_classes=env_int("NUM_CLASSES", 95),
        detector_input_layout=env_str("DETECTOR_INPUT_LAYOUT", "ncl"),
        overhead_targets=[float(value) for value in env_words("OVERHEAD_TARGETS", "0.20 0.30 0.40")],
        lambda_overheads=[float(value) for value in env_words("LAMBDA_OVERHEADS", "1.0")],
        overhead_losses=env_words("OVERHEAD_LOSSES", "hinge"),
        overhead_tolerance=env_float("OVERHEAD_TOLERANCE", 0.02),
        attack_losses=env_words("ATTACK_LOSSES", "true_prob"),
        lambda_tv=env_float("LAMBDA_TV", 0.0),
        projection_mode=env_str("PROJECTION_MODE", "ste"),
        soft_projection_tau=env_float("SOFT_PROJECTION_TAU", 1.5),
        projection_chunk_size=env_int("PROJECTION_CHUNK_SIZE", 64),
        lr=env_float("LR", 1e-4),
        epochs=env_int("SMOKE_EPOCHS" if mode == "smoke" else "FULL_EPOCHS", 1 if mode == "smoke" else 30),
        batch_size=env_int("SMOKE_BATCH_SIZE" if mode == "smoke" else "FULL_BATCH_SIZE", 4 if mode == "smoke" else 64),
        noise_dim=env_int("SMOKE_NOISE_DIM" if mode == "smoke" else "FULL_NOISE_DIM", 64 if mode == "smoke" else 2000),
        seed=env_int("SEED", 1),
        device=env_optional("DEVICE"),
        reuse_intermediates=env_bool("REUSE_INTERMEDIATES", True),
        skip_existing=env_bool("SKIP_EXISTING", True),
        run_eval=env_bool("RUN_EVAL", True),
        budget_slack=env_float("BUDGET_SLACK", 0.02),
        select_metric=env_str("SELECT_METRIC", "accuracy"),
        metrics=env_words("EVAL_METRICS", "accuracy precision recall f1"),
        average=env_str("EVAL_AVERAGE", "macro"),
        kmeans_restarts=env_int("KMEANS_RESTARTS", 20),
        kmeans_max_iter=env_int("KMEANS_MAX_ITER", 100),
        min_cluster_size=env_int("MIN_CLUSTER_SIZE", 1),
    )


def serializable_config(config: Config) -> dict[str, Any]:
    data = asdict(config)
    data["out_root"] = str(config.out_root)
    return data


def maybe_prepare_bursts(config: Config, input_npz: str, data_key: str, labels_key: str, output_npz: Path, limit: int | None) -> None:
    if config.reuse_intermediates and output_npz.exists():
        print(f"reuse burst dataset: {output_npz}")
        return
    data, labels = load_npz_dataset(input_npz, data_key, labels_key)
    if limit is not None:
        data = data[:limit]
        labels = labels[:limit]
    bursts = direction_to_burst(data, config.max_bursts)
    save_npz(output_npz, data=bursts, labels=labels)
    print(f"saved burst dataset: {output_npz}")
    print(summarize_bursts(bursts))


def maybe_build_mapping(config: Config, burst_npz: Path, k: int, output_dir: Path) -> Path:
    mapping_path = output_dir / "mapping_recommended.json"
    if config.reuse_intermediates and mapping_path.exists():
        print(f"reuse fixed-K mapping: {mapping_path}")
        return mapping_path
    bursts, labels = load_npz_dataset(burst_npz)
    row = evaluate_fixed_k(
        bursts,
        labels,
        k,
        exclude_labels=set(config.exclude_labels),
        profile_method=config.profile_method,
        normalize=config.normalize,
        seed=config.seed,
        restarts=config.kmeans_restarts,
        max_iter=config.kmeans_max_iter,
        min_cluster_size=config.min_cluster_size,
    )
    return write_fixed_k_mapping(row, output_dir)


def maybe_apply_mapping(config: Config, burst_npz: Path, mapping_json: Path, output_npz: Path, output_json: Path) -> None:
    if config.reuse_intermediates and output_npz.exists() and output_json.exists():
        print(f"reuse pseudo labels: {output_npz}")
        print(f"reuse pseudo json: {output_json}")
        return
    bursts, labels = load_npz_dataset(burst_npz)
    mapping = load_mapping(mapping_json)
    pseudo, keep_indices = labels_to_pseudo(labels, mapping)
    kept_labels = labels[keep_indices]
    save_npz(output_npz, labels=kept_labels, pseudo_labels=pseudo, keep_indices=keep_indices)
    unique, counts = np.unique(pseudo, return_counts=True)
    grouped: dict[int, list[int]] = {}
    for website, pseudo_label in mapping.items():
        grouped.setdefault(pseudo_label, []).append(website)
    summary = {
        "source": "fixed_k_mapping",
        "labels_npz": str(burst_npz),
        "mapping": str(mapping_json),
        "total_samples": int(len(labels)),
        "kept_samples": int(len(keep_indices)),
        "dropped_samples": int(len(labels) - len(keep_indices)),
        "website_to_pseudo_label": {str(k): int(v) for k, v in sorted(mapping.items())},
        "pseudo_label_to_websites": {str(k): sorted(int(v) for v in values) for k, values in sorted(grouped.items())},
        "pseudo_label_counts": {str(int(label)): int(count) for label, count in zip(unique, counts)},
        "cluster_quality": cluster_quality_for_mapping(bursts, labels, mapping, config.profile_method, set(config.exclude_labels)),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved pseudo labels: {output_npz}")
    print(f"saved pseudo json: {output_json}")


def train_generators(
    config: Config,
    train_config: TrainConfig,
    burst_npz: Path,
    pseudo_npz: Path,
    generator_dir: Path,
    device: torch.device,
) -> None:
    manifest_path = generator_dir / "manifest.json"
    if config.reuse_intermediates and manifest_path.exists():
        print(f"reuse generators: {generator_dir}")
        return
    bursts, labels = load_npz_dataset(burst_npz)
    with np.load(pseudo_npz, allow_pickle=True) as npz:
        pseudo_labels = np.asarray(npz["pseudo_labels"]).astype(np.int64)
        keep_indices = np.asarray(npz["keep_indices"]).astype(np.int64)
    bursts = bursts[keep_indices]
    labels = labels[keep_indices]
    detector = load_detector(config.detector_checkpoint, config.num_classes, device)
    generator_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pseudo_label in [int(value) for value in np.unique(pseudo_labels)]:
        result = train_one_generator(
            pseudo_label=pseudo_label,
            bursts=bursts,
            labels=labels,
            pseudo_labels=pseudo_labels,
            detector=detector,
            output_dir=generator_dir,
            config=train_config,
            device=device,
        )
        results.append(result)
    manifest = {
        "burst_npz": str(burst_npz),
        "pseudo_npz": str(pseudo_npz),
        "detector_checkpoint": config.detector_checkpoint,
        "num_classes": config.num_classes,
        "train_config": asdict(train_config),
        "generators": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved generator manifest: {manifest_path}")


def export_defended_dataset(config: Config, burst_npz: Path, pseudo_npz: Path, generator_dir: Path, output_npz: Path, device: torch.device) -> None:
    bursts, labels = load_npz_dataset(burst_npz)
    with np.load(pseudo_npz, allow_pickle=True) as npz:
        pseudo_labels = np.asarray(npz["pseudo_labels"]).astype(np.int64)
        keep_indices = np.asarray(npz["keep_indices"]).astype(np.int64)
    bursts = bursts[keep_indices]
    labels = labels[keep_indices]

    generators = {}
    for path in generator_dir.glob("generator_pseudo_*.pt"):
        checkpoint = torch.load(path, map_location=device)
        model = build_generator_from_checkpoint(checkpoint, device)
        generators[int(checkpoint["pseudo_label"])] = model
    if not generators:
        raise FileNotFoundError(f"No generator_pseudo_*.pt files in {generator_dir}")

    defended = np.zeros_like(bursts, dtype=np.float32)
    overhead_values = np.zeros((bursts.shape[0],), dtype=np.float32)
    for pseudo_label in np.unique(pseudo_labels):
        pseudo_label = int(pseudo_label)
        model = generators[pseudo_label]
        indices = np.flatnonzero(pseudo_labels == pseudo_label)
        for start in tqdm(range(0, len(indices), 256), desc=f"export pseudo={pseudo_label}"):
            batch_indices = indices[start:start + 256]
            x = torch.from_numpy(bursts[batch_indices].astype(np.float32)).to(device)
            z = torch.randn(x.shape[0], model.config.noise_dim, device=device)
            with torch.no_grad():
                adv = apply_burst_perturbation(x, model(z), round_output=True)
                overhead_values[batch_indices] = overhead_ratio(x, adv).detach().cpu().numpy()
            defended[batch_indices] = adv.detach().cpu().numpy()
    directions = burst_to_direction(defended, config.trace_len)
    save_npz(output_npz, data=directions, direction_data=directions, labels=labels, pseudo_labels=pseudo_labels, overhead=overhead_values)
    print(f"saved defended dataset: {output_npz}")
    print(f"mean overhead: {float(np.mean(overhead_values)):.6f}")
    print(f"median overhead: {float(np.median(overhead_values)):.6f}")


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, metric_names: list[str], average: str, num_classes: int) -> dict[str, float]:
    labels = labels_to_int(labels)
    predictions = labels_to_int(predictions)
    classes = np.union1d(labels, predictions).astype(np.int64)
    tp_values = []
    fp_values = []
    fn_values = []
    support_values = []
    for cls in classes:
        true_mask = labels == cls
        pred_mask = predictions == cls
        tp_values.append(int(np.sum(true_mask & pred_mask)))
        fp_values.append(int(np.sum(~true_mask & pred_mask)))
        fn_values.append(int(np.sum(true_mask & ~pred_mask)))
        support_values.append(int(np.sum(true_mask)))

    def safe(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    precision_values = np.asarray([safe(tp, tp + fp) for tp, fp in zip(tp_values, fp_values)], dtype=np.float64)
    recall_values = np.asarray([safe(tp, tp + fn) for tp, fn in zip(tp_values, fn_values)], dtype=np.float64)
    f1_values = np.asarray([safe(2 * p * r, p + r) for p, r in zip(precision_values, recall_values)], dtype=np.float64)
    support = np.asarray(support_values, dtype=np.float64)

    def avg(values: np.ndarray) -> float:
        if average == "macro":
            return float(np.mean(values)) if len(values) else 0.0
        if average == "weighted":
            total = float(np.sum(support))
            return float(np.sum(values * support) / total) if total > 0 else 0.0
        if average == "micro":
            tp = float(np.sum(tp_values))
            fp = float(np.sum(fp_values))
            fn = float(np.sum(fn_values))
            if values is precision_values:
                return safe(tp, tp + fp)
            if values is recall_values:
                return safe(tp, tp + fn)
            p = safe(tp, tp + fp)
            r = safe(tp, tp + fn)
            return safe(2 * p * r, p + r)
        raise ValueError(f"Unknown average: {average}")

    result = {}
    if "accuracy" in metric_names:
        result["accuracy"] = float(np.mean(labels == predictions)) if len(labels) else 0.0
    if "precision" in metric_names:
        result[f"precision_{average}"] = avg(precision_values)
    if "recall" in metric_names:
        result[f"recall_{average}"] = avg(recall_values)
    if "f1" in metric_names:
        result[f"f1_score_{average}"] = avg(f1_values)
    return result


def evaluate_dataset(config: Config, input_npz: Path, output_json: Path, device: torch.device) -> None:
    data, labels = load_npz_dataset(input_npz, data_key="data", labels_key="labels")
    overhead_values = None
    with np.load(input_npz, allow_pickle=True) as npz:
        if "overhead" in npz:
            overhead_values = np.asarray(npz["overhead"])
    if config.eval_exclude_labels:
        keep = ~np.isin(labels, np.asarray(config.eval_exclude_labels, dtype=np.int64))
        data = data[keep]
        labels = labels[keep]
        if overhead_values is not None:
            overhead_values = overhead_values[keep]

    detector = load_detector(config.detector_checkpoint, config.num_classes, device)
    detector_data = direction_to_sign_sequence(data, config.trace_len)
    predictions = []
    for start in tqdm(range(0, len(labels), 256), desc="evaluate detector"):
        batch = torch.from_numpy(detector_data[start:start + 256].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = detector(format_detector_input(batch, config.detector_input_layout))
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        predictions.append(pred)
    predictions_np = np.concatenate(predictions).astype(np.int64) if predictions else np.asarray([], dtype=np.int64)
    metrics = compute_metrics(labels, predictions_np, config.metrics, config.average, config.num_classes)
    overhead_summary = None
    if overhead_values is not None:
        overhead_summary = {
            "mean": float(np.mean(overhead_values)) if overhead_values.size else 0.0,
            "median": float(np.median(overhead_values)) if overhead_values.size else 0.0,
            "max": float(np.max(overhead_values)) if overhead_values.size else 0.0,
        }
    summary = {
        "input_npz": str(input_npz),
        "exclude_labels": config.eval_exclude_labels,
        "detector": {"checkpoint": config.detector_checkpoint, "num_classes": config.num_classes, "input_layout": config.detector_input_layout},
        "metrics": metrics,
        "average": config.average,
        "num_samples": int(len(labels)),
        "num_predictions": int(len(predictions_np)),
        "overhead": overhead_summary,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    if overhead_summary:
        print(f"  overhead_mean: {overhead_summary['mean']:.6f}")
        print(f"  overhead_median: {overhead_summary['median']:.6f}")
    print(f"saved metrics: {output_json}")


def flatten_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    overhead = summary.get("overhead") or {}
    return {
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_score_macro": metrics.get("f1_score_macro"),
        "overhead_mean": overhead.get("mean"),
        "overhead_median": overhead.get("median"),
        "overhead_max": overhead.get("max"),
        "num_samples": summary.get("num_samples"),
        "eval_average": summary.get("average"),
    }


def write_run_config(path: Path, config: Config, row: dict[str, Any]) -> None:
    payload = {"pipeline": "TOOB_simulation_fix_k_standalone", "config": serializable_config(config)}
    payload.update(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_candidate(
    config: Config,
    *,
    k: int,
    mapping_json: Path,
    train_burst_npz: Path,
    train_pseudo_npz: Path,
    eval_burst_npz: Path,
    eval_pseudo_npz: Path,
    run_dir: Path,
    target: float,
    lambda_overhead: float,
    overhead_loss: str,
    attack_loss: str,
    device: torch.device,
) -> None:
    metrics_json = run_dir / "defense_eval_metrics.json"
    if config.skip_existing and metrics_json.exists():
        print(f"skip existing run: {run_dir.name}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    generator_dir = run_dir / "generators"
    adv_npz = run_dir / "toob_adv_direction.npz"
    train_config = TrainConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        noise_dim=config.noise_dim,
        overhead_threshold=target,
        lambda_overhead=lambda_overhead,
        overhead_loss=overhead_loss,
        overhead_tolerance=config.overhead_tolerance,
        lambda_tv=config.lambda_tv,
        attack_loss=attack_loss,
        trace_len=config.trace_len,
        detector_input_layout=config.detector_input_layout,
        projection_mode=config.projection_mode,
        soft_projection_tau=config.soft_projection_tau,
        projection_chunk_size=config.projection_chunk_size,
        seed=config.seed,
    )
    row = {
        "run_name": run_dir.name,
        "cluster_count": k,
        "mapping": str(mapping_json),
        "overhead_threshold": target,
        "lambda_overhead": lambda_overhead,
        "overhead_loss": overhead_loss,
        "overhead_tolerance": config.overhead_tolerance,
        "attack_loss": attack_loss,
        "lambda_tv": config.lambda_tv,
        "projection_mode": config.projection_mode,
        "soft_projection_tau": config.soft_projection_tau,
        "lr": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "noise_dim": config.noise_dim,
        "detector_checkpoint": config.detector_checkpoint,
        "num_classes": config.num_classes,
        "eval_exclude_labels": " ".join(str(value) for value in config.eval_exclude_labels),
        "output_npz": str(adv_npz),
        "metrics_json": str(metrics_json),
    }
    write_run_config(run_dir / "run_config.json", config, row)
    log_command(["standalone-train", f"k={k}", f"target={target}", f"lambda={lambda_overhead}", overhead_loss, attack_loss])
    train_generators(config, train_config, train_burst_npz, train_pseudo_npz, generator_dir, device)
    export_defended_dataset(config, eval_burst_npz, eval_pseudo_npz, generator_dir, adv_npz, device)
    if config.run_eval:
        evaluate_dataset(config, adv_npz, metrics_json, device)


def collect_rows(out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_root.glob("k*/run_*/defense_eval_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {key: value for key, value in config.items() if key not in {"config"}}
        row["metrics_json"] = str(metrics_path)
        row.update(flatten_metrics(summary))
        rows.append(row)
    return rows


def write_table(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any, default: float = 1e9) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def best_by_target(rows: list[dict[str, Any]], budget_slack: float, select_metric: str) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        target = row.get("overhead_threshold")
        if target is not None:
            grouped.setdefault(float(target), []).append(row)
    best_rows = []
    for target, candidates in sorted(grouped.items()):
        eligible = [
            row for row in candidates
            if row.get("overhead_mean") is not None and float(row["overhead_mean"]) <= target + budget_slack
        ]
        if eligible:
            selected = min(eligible, key=lambda row: number(row.get(select_metric)))
            reason = "within_budget"
        else:
            selected = min(candidates, key=lambda row: abs(number(row.get("overhead_mean")) - target))
            reason = "closest_overhead"
        copied = dict(selected)
        copied["target"] = target
        copied["selection_reason"] = reason
        copied["budget_slack"] = budget_slack
        copied["select_metric"] = select_metric
        best_rows.append(copied)
    return best_rows


def print_config(config: Config) -> None:
    print("Standalone fixed-K TOOB pipeline")
    print(f"  mode: {config.mode}")
    print(f"  out root: {config.out_root}")
    print(f"  cluster counts: {' '.join(str(value) for value in config.cluster_counts)}")
    print(f"  train dataset: {config.train_dataset}")
    print(f"  valid dataset: {config.valid_dataset or '<none>'}")
    print(f"  detector checkpoint: {config.detector_checkpoint}")
    print(f"  num classes: {config.num_classes}")
    print(f"  targets: {' '.join(str(value) for value in config.overhead_targets)}")
    print(f"  lambdas: {' '.join(str(value) for value in config.lambda_overheads)}")
    print(f"  losses: {' '.join(config.overhead_losses)} / {' '.join(config.attack_losses)}")
    print(f"  epochs/batch/noise: {config.epochs}/{config.batch_size}/{config.noise_dim}")
    print(f"  projection: {config.projection_mode}")


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "full"
    config = build_config(mode)
    print_config(config)
    require_file(config.train_dataset)
    if config.valid_dataset:
        require_file(config.valid_dataset)
    require_file(config.detector_checkpoint)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device: {device}")

    config.out_root.mkdir(parents=True, exist_ok=True)
    shared_burst = Path(env_str("BURST_NPZ", str(config.out_root / "shared_cache" / "burst_dataset.npz")))
    maybe_prepare_bursts(config, config.train_dataset, config.train_data_key, config.train_labels_key, shared_burst, config.burst_limit)

    for k in config.cluster_counts:
        if k < 2:
            raise ValueError(f"K must be >= 2, got {k}")
        k_dir = config.out_root / f"k{k}"
        mapping_json = maybe_build_mapping(config, shared_burst, k, k_dir / "mapping")
        train_pseudo_npz = k_dir / "cache" / "train_pseudo_labels.npz"
        train_pseudo_json = k_dir / "cache" / "train_pseudo_labels.json"
        maybe_apply_mapping(config, shared_burst, mapping_json, train_pseudo_npz, train_pseudo_json)

        if config.valid_dataset:
            eval_burst = k_dir / "cache" / "valid_burst_dataset.npz"
            eval_pseudo_npz = k_dir / "cache" / "valid_pseudo_labels.npz"
            eval_pseudo_json = k_dir / "cache" / "valid_pseudo_labels.json"
            maybe_prepare_bursts(config, config.valid_dataset, config.valid_data_key, config.valid_labels_key, eval_burst, config.valid_limit)
            maybe_apply_mapping(config, eval_burst, mapping_json, eval_pseudo_npz, eval_pseudo_json)
        else:
            eval_burst = shared_burst
            eval_pseudo_npz = train_pseudo_npz

        for target in config.overhead_targets:
            for lambda_overhead in config.lambda_overheads:
                for overhead_loss in config.overhead_losses:
                    for attack_loss in config.attack_losses:
                        run_name = f"run_t{tag_value(target)}_l{tag_value(lambda_overhead)}_{overhead_loss}_{attack_loss}"
                        run_candidate(
                            config,
                            k=k,
                            mapping_json=mapping_json,
                            train_burst_npz=shared_burst,
                            train_pseudo_npz=train_pseudo_npz,
                            eval_burst_npz=eval_burst,
                            eval_pseudo_npz=eval_pseudo_npz,
                            run_dir=k_dir / run_name,
                            target=target,
                            lambda_overhead=lambda_overhead,
                            overhead_loss=overhead_loss,
                            attack_loss=attack_loss,
                            device=device,
                        )

    rows = collect_rows(config.out_root)
    write_table(rows, config.out_root / "summary.csv", config.out_root / "summary.json")
    best_rows = best_by_target(rows, config.budget_slack, config.select_metric)
    write_table(best_rows, config.out_root / "best_by_target.csv", config.out_root / "best_by_target.json")
    print(f"saved summary: {config.out_root / 'summary.csv'}")
    print(f"saved best: {config.out_root / 'best_by_target.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
