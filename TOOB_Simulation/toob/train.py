from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .burst import apply_burst_perturbation, overhead_ratio, soft_burst_to_direction
from .data import BurstDataset, select_by_pseudo
from .detector import format_detector_input
from .generator import BurstGenerator, GeneratorConfig
from .losses import overhead_budget_loss, total_variation_loss, untargeted_attack_loss


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    noise_dim: int = 256
    overhead_threshold: float = 0.22
    lambda_overhead: float = 1.0
    overhead_loss: str = "hinge"
    overhead_tolerance: float = 0.0
    lambda_tv: float = 0.001
    attack_loss: str = "true_prob"
    detector_input_kind: str = "direction"
    detector_input_layout: str = "ncl"
    detector_input_length: int = 5000
    soft_projection_tau: float = 1.5
    projection_chunk_size: int = 128
    seed: int = 1


def train_one_generator(
    *,
    pseudo_label: int,
    bursts: np.ndarray,
    labels: np.ndarray,
    pseudo_labels: np.ndarray,
    detector: torch.nn.Module,
    output_dir: str | Path,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    indices = select_by_pseudo(pseudo_labels, pseudo_label)
    if len(indices) == 0:
        raise ValueError(f"Pseudo label {pseudo_label} has no samples")

    dataset = BurstDataset(bursts, labels, pseudo_labels, indices)
    generator = torch.Generator().manual_seed(config.seed + int(pseudo_label))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)

    gen_config = GeneratorConfig(noise_dim=config.noise_dim, burst_len=bursts.shape[1])
    model = BurstGenerator(gen_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_rows = []
        progress = tqdm(loader, desc=f"pseudo={pseudo_label} epoch={epoch}", leave=False)
        for batch in progress:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            z = torch.randn(x.shape[0], config.noise_dim, device=device)

            delta = model(z)
            adv = apply_burst_perturbation(x, delta)
            if config.detector_input_kind == "burst":
                detector_features = adv
            elif config.detector_input_kind == "direction":
                detector_features = soft_burst_to_direction(
                    adv,
                    trace_len=config.detector_input_length,
                    tau=config.soft_projection_tau,
                    chunk_size=config.projection_chunk_size,
                )
            else:
                raise ValueError(f"Unknown detector input kind: {config.detector_input_kind}")
            detector_x = format_detector_input(
                detector_features,
                kind=config.detector_input_kind,
                layout=config.detector_input_layout,
            )
            logits = detector(detector_x)

            loss_attack = untargeted_attack_loss(logits, y, mode=config.attack_loss)
            loss_overhead = overhead_budget_loss(
                x,
                adv,
                config.overhead_threshold,
                mode=config.overhead_loss,
                tolerance=config.overhead_tolerance,
            )
            loss_tv = total_variation_loss(delta)
            loss = loss_attack + config.lambda_overhead * loss_overhead + config.lambda_tv * loss_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                overhead = overhead_ratio(x, adv).mean()
                true_prob = torch.softmax(logits, dim=1).gather(1, y.view(-1, 1)).mean()
            row = {
                "loss": float(loss.detach().cpu()),
                "attack": float(loss_attack.detach().cpu()),
                "overhead_loss": float(loss_overhead.detach().cpu()),
                "tv": float(loss_tv.detach().cpu()),
                "overhead": float(overhead.detach().cpu()),
                "true_prob": float(true_prob.detach().cpu()),
            }
            epoch_rows.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4f}", overhead=f"{row['overhead']:.3f}")

        summary = {
            "epoch": epoch,
            "samples": int(len(indices)),
        }
        for key in ("loss", "attack", "overhead_loss", "tv", "overhead", "true_prob"):
            summary[key] = float(np.mean([row[key] for row in epoch_rows]))
        history.append(summary)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"generator_pseudo_{pseudo_label}.pt"
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

    history_path = out_dir / f"generator_pseudo_{pseudo_label}.history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "pseudo_label": int(pseudo_label),
        "samples": int(len(indices)),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
