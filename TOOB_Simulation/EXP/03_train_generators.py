from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ensure_project_root

ensure_project_root(required_modules=("data", "detector", "train"))
from toob.data import load_npz_dataset
from toob.detector import load_detector
from toob.train import TrainConfig, train_one_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one burst generator per pseudo label.")
    parser.add_argument("--burst-npz", required=True)
    parser.add_argument("--pseudo-npz", required=True)
    parser.add_argument("--detector-builder", required=True, help="module:function or file.py:function")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--detector-state-dict-key")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--detector-input-kind", choices=("burst", "direction"), default="direction")
    parser.add_argument("--detector-input-layout", choices=("nl", "ncl", "nchw"), default="ncl")
    parser.add_argument("--detector-input-length", type=int, default=5000)
    parser.add_argument("--projection-mode", choices=("soft", "ste"), default="ste")
    parser.add_argument("--soft-projection-tau", type=float, default=1.5)
    parser.add_argument("--projection-chunk-size", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pseudo-labels", nargs="*", type=int, help="Optional subset of pseudo labels to train.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise-dim", type=int, default=256, help="Deprecated: generator is conditional on burst input.")
    parser.add_argument("--overhead-threshold", type=float, default=0.22)
    parser.add_argument("--lambda-overhead", type=float, default=1.0)
    parser.add_argument("--overhead-loss", choices=("hinge", "target_l1", "target_l2", "band"), default="hinge")
    parser.add_argument("--overhead-tolerance", type=float, default=0.0)
    parser.add_argument("--lambda-tv", type=float, default=0.001)
    parser.add_argument("--attack-loss", choices=("true_prob", "true_logit", "negative_ce"), default="true_logit")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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

    detector = load_detector(
        builder_spec=args.detector_builder,
        checkpoint_path=args.detector_checkpoint,
        num_classes=args.num_classes,
        device=device,
        state_dict_key=args.detector_state_dict_key,
    )

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        overhead_threshold=args.overhead_threshold,
        lambda_overhead=args.lambda_overhead,
        overhead_loss=args.overhead_loss,
        overhead_tolerance=args.overhead_tolerance,
        lambda_tv=args.lambda_tv,
        attack_loss=args.attack_loss,
        detector_input_kind=args.detector_input_kind,
        detector_input_layout=args.detector_input_layout,
        detector_input_length=args.detector_input_length,
        projection_mode=args.projection_mode,
        soft_projection_tau=args.soft_projection_tau,
        projection_chunk_size=args.projection_chunk_size,
        seed=args.seed,
    )

    selected = args.pseudo_labels
    if selected is None:
        selected = [int(v) for v in np.unique(pseudo_labels)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "burst_npz": str(args.burst_npz),
        "pseudo_npz": str(args.pseudo_npz),
        "detector_builder": args.detector_builder,
        "detector_checkpoint": str(args.detector_checkpoint),
        "num_classes": args.num_classes,
        "train_config": asdict(train_config),
        "generators": [],
    }

    for pseudo_label in selected:
        result = train_one_generator(
            pseudo_label=pseudo_label,
            bursts=bursts,
            labels=labels,
            pseudo_labels=pseudo_labels,
            detector=detector,
            output_dir=output_dir,
            config=train_config,
            device=device,
        )
        manifest["generators"].append(result)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
