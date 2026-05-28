from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toob.burst import apply_burst_perturbation, burst_to_direction, overhead_ratio
from toob.data import load_npz_dataset, save_npz_dataset
from toob.generator import build_generator_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate defended/adversarial dataset from trained generators.")
    parser.add_argument("--burst-npz", required=True)
    parser.add_argument("--pseudo-npz", required=True)
    parser.add_argument("--generator-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-kind", choices=("burst", "direction", "both"), default="burst")
    parser.add_argument("--max-trace-len", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--round", action="store_true", help="Round defended burst counts before saving/export.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_generators(generator_dir: str | Path, device: torch.device) -> dict[int, torch.nn.Module]:
    models = {}
    for path in Path(generator_dir).glob("generator_pseudo_*.pt"):
        checkpoint = torch.load(path, map_location=device)
        pseudo_label = int(checkpoint["pseudo_label"])
        model = build_generator_from_checkpoint(checkpoint, map_location=device)
        model.eval()
        models[pseudo_label] = model
    if not models:
        raise FileNotFoundError(f"No generator_pseudo_*.pt files found in {generator_dir}")
    return models


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    bursts, labels = load_npz_dataset(args.burst_npz)
    with np.load(args.pseudo_npz, allow_pickle=True) as npz:
        pseudo_labels = np.asarray(npz["pseudo_labels"]).astype(np.int64)
        if "keep_indices" in npz:
            keep_indices = np.asarray(npz["keep_indices"]).astype(np.int64)
            bursts = bursts[keep_indices]
            labels = labels[keep_indices]

    generators = load_generators(args.generator_dir, device)
    defended = np.zeros_like(bursts, dtype=np.float32)
    overhead_values = np.zeros((bursts.shape[0],), dtype=np.float32)

    for pseudo_label in np.unique(pseudo_labels):
        pseudo_label = int(pseudo_label)
        if pseudo_label not in generators:
            raise KeyError(f"Missing generator for pseudo label {pseudo_label}")
        model = generators[pseudo_label]
        indices = np.flatnonzero(pseudo_labels == pseudo_label)
        noise_dim = model.config.noise_dim
        for start in tqdm(range(0, len(indices), args.batch_size), desc=f"pseudo={pseudo_label}"):
            batch_indices = indices[start:start + args.batch_size]
            x = torch.from_numpy(bursts[batch_indices].astype(np.float32)).to(device)
            z = torch.randn(x.shape[0], noise_dim, device=device)
            with torch.no_grad():
                delta = model(z)
                adv = apply_burst_perturbation(x, delta, round_output=args.round)
                batch_overhead = overhead_ratio(x, adv).detach().cpu().numpy()
            defended[batch_indices] = adv.detach().cpu().numpy()
            overhead_values[batch_indices] = batch_overhead

    arrays = {
        "labels": labels,
        "pseudo_labels": pseudo_labels,
        "overhead": overhead_values,
    }

    if args.output_kind in ("burst", "both"):
        arrays["data"] = defended
        arrays["burst_data"] = defended
    if args.output_kind in ("direction", "both"):
        direction = burst_to_direction(defended, max_trace_len=args.max_trace_len)
        if args.output_kind == "direction":
            arrays["data"] = direction
        arrays["direction_data"] = direction

    save_npz_dataset(args.output, **arrays)
    print(f"saved: {args.output}")
    if len(overhead_values) > 0:
        print(f"mean overhead: {float(np.mean(overhead_values)):.6f}")
        print(f"median overhead: {float(np.median(overhead_values)):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
