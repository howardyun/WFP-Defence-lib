from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ensure_project_root

ensure_project_root(required_modules=("cluster", "data", "pseudo"))
from toob.cluster import cluster_websites
from toob.data import load_npz_dataset, save_npz_dataset
from toob.pseudo import labels_to_pseudo, load_website_to_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cluster pseudo labels from a burst dataset.")
    parser.add_argument("--labels-npz", required=True, help=".npz file containing burst data and labels.")
    parser.add_argument("--mapping", help="Optional precomputed website_to_set .npy or .json.")
    parser.add_argument("--output", required=True, help="Output .npz with labels and pseudo_labels.")
    parser.add_argument("--json-output", help="Optional readable JSON summary path. Defaults to output path with .json suffix.")
    parser.add_argument("--strategy", choices=("first", "random", "round_robin"), default="first")
    parser.add_argument("--drop-unmapped", action="store_true", help="Drop labels missing from the mapping.")
    parser.add_argument("--exclude-labels", nargs="*", type=int, default=[], help="Labels excluded before clustering, e.g. open-world label 95.")
    parser.add_argument("--set-size", type=int, default=30, help="Target anonymity set size.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of clustering rounds.")
    parser.add_argument("--profile-method", choices=("super", "mean_abs", "mean_signed"), default="super")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bursts, labels = load_npz_dataset(args.labels_npz)
    exclude_labels = set(args.exclude_labels)
    if args.mapping:
        mapping = load_website_to_set(args.mapping)
        cluster_info = None
        source = "mapping"
    else:
        cluster_result = cluster_websites(
            bursts,
            labels,
            set_size=args.set_size,
            rounds=args.rounds,
            seed=args.seed,
            exclude_labels=exclude_labels,
            profile_method=args.profile_method,
        )
        mapping = cluster_result.website_to_set
        cluster_info = {
            "set_size": int(args.set_size),
            "rounds": int(args.rounds),
            "profile_method": args.profile_method,
            "excluded_labels": sorted(int(v) for v in exclude_labels),
            "site_labels": [int(v) for v in cluster_result.labels],
            "total_sets": [[int(v) for v in anonymity_set] for anonymity_set in cluster_result.total_sets],
        }
        source = "toob_cluster"
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
        "source": source,
        "mapping": str(args.mapping) if args.mapping else None,
        "strategy": args.strategy,
        "drop_unmapped": bool(args.drop_unmapped),
        "total_samples": int(len(labels)),
        "kept_samples": int(len(keep_indices)),
        "dropped_samples": int(len(labels) - len(keep_indices)),
        "website_to_set": {str(k): v for k, v in sorted(mapping.items())},
        "pseudo_label_counts": {str(int(label)): int(count) for label, count in zip(unique, counts)},
        "cluster": cluster_info,
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
