from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toob.data import load_npz_dataset, save_npz_dataset
from toob.pseudo import labels_to_pseudo, load_website_to_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pseudo labels from Palette website_to_set mapping.")
    parser.add_argument("--labels-npz", required=True, help=".npz file containing labels.")
    parser.add_argument("--mapping", required=True, help="Palette website_to_set .npy or .json.")
    parser.add_argument("--output", required=True, help="Output .npz with labels and pseudo_labels.")
    parser.add_argument("--json-output", help="Optional readable JSON summary path. Defaults to output path with .json suffix.")
    parser.add_argument("--strategy", choices=("first", "random", "round_robin"), default="first")
    parser.add_argument("--drop-unmapped", action="store_true", help="Drop labels missing from the Palette mapping.")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, labels = load_npz_dataset(args.labels_npz)
    mapping = load_website_to_set(args.mapping)
    pseudo, keep_indices = labels_to_pseudo(
        labels,
        mapping,
        strategy=args.strategy,
        seed=args.seed,
        drop_unmapped=args.drop_unmapped,
    )
    kept_labels = labels[keep_indices]
    save_npz_dataset(args.output, labels=kept_labels, pseudo_labels=pseudo, keep_indices=keep_indices)
    unique, counts = np.unique(pseudo, return_counts=True)

    json_output = Path(args.json_output) if args.json_output else Path(args.output).with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "labels_npz": str(args.labels_npz),
        "mapping": str(args.mapping),
        "strategy": args.strategy,
        "drop_unmapped": bool(args.drop_unmapped),
        "total_samples": int(len(labels)),
        "kept_samples": int(len(keep_indices)),
        "dropped_samples": int(len(labels) - len(keep_indices)),
        "website_to_set": {str(k): v for k, v in sorted(mapping.items())},
        "pseudo_label_counts": {str(int(label)): int(count) for label, count in zip(unique, counts)},
        "samples": [
            {
                "original_index": int(original_idx),
                "label": int(label),
                "pseudo_label": int(pseudo_label),
            }
            for original_idx, label, pseudo_label in zip(keep_indices, kept_labels, pseudo)
        ],
    }
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {args.output}")
    print(f"saved json: {json_output}")
    print(f"kept samples: {len(keep_indices)} / {len(labels)}")
    print("pseudo label counts:")
    for label, count in zip(unique, counts):
        print(f"  {int(label)}: {int(count)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
