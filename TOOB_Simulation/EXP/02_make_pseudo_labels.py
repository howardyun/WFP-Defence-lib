from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ensure_project_root

ensure_project_root(required_modules=("cluster", "data", "encoder", "pseudo"))
from toob.cluster import (
    build_website_profiles,
    cluster_websites_from_profiles,
    cluster_websites_kmeans,
    evaluate_cluster_quality,
)
from toob.data import load_npz_dataset, save_npz_dataset
from toob.encoder import encode_website_profiles, train_autoencoder
from toob.pseudo import labels_to_pseudo, load_website_to_pseudo_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cluster pseudo labels from a burst dataset.")
    parser.add_argument("--labels-npz", required=True, help=".npz file containing burst data and labels.")
    parser.add_argument("--mapping", help="Optional precomputed website_to_pseudo_label .npy or .json.")
    parser.add_argument("--output", required=True, help="Output .npz with labels and pseudo_labels.")
    parser.add_argument("--json-output", help="Optional readable JSON summary path. Defaults to output path with .json suffix.")
    parser.add_argument("--drop-unmapped", action="store_true", help="Drop labels missing from the mapping.")
    parser.add_argument("--exclude-labels", nargs="*", type=int, default=[], help="Labels excluded before clustering, e.g. open-world label 95.")
    parser.add_argument("--set-size", type=int, default=30, help="Target anonymity set size.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of clustering rounds.")
    parser.add_argument("--profile-method", choices=("super", "mean_abs", "mean_signed"), default="super")
    parser.add_argument("--cluster-method", choices=("greedy", "kmeans"), default="greedy")
    parser.add_argument("--num-clusters", type=int, default=None, help="K for kmeans clustering; defaults to ceil(num_sites/set_size).")
    parser.add_argument("--use-encoder", action="store_true", help="Use an MLP autoencoder latent representation for clustering.")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=40)
    parser.add_argument("--encoder-batch-size", type=int, default=256)
    parser.add_argument("--encoder-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bursts, labels = load_npz_dataset(args.labels_npz)
    exclude_labels = set(args.exclude_labels)
    profiles = None
    if args.mapping:
        website_to_pseudo_label = load_website_to_pseudo_label(args.mapping)
        grouped: dict[int, list[int]] = {}
        for website, pseudo_label in website_to_pseudo_label.items():
            grouped.setdefault(pseudo_label, []).append(website)
        pseudo_label_to_websites = {
            pseudo_label: sorted(websites)
            for pseudo_label, websites in sorted(grouped.items())
        }
        cluster_info = None
        source = "mapping"
    else:
        device = torch.device(args.device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if args.use_encoder:
            model = train_autoencoder(
                bursts,
                latent_dim=args.latent_dim,
                epochs=args.encoder_epochs,
                batch_size=args.encoder_batch_size,
                lr=args.encoder_lr,
                seed=args.seed,
                device=device,
            )
            site_labels, profiles = encode_website_profiles(
                model,
                bursts,
                labels,
                exclude_labels=exclude_labels,
                device=device,
            )
        else:
            site_labels, profiles = build_website_profiles(
                bursts,
                labels,
                exclude_labels=exclude_labels,
                profile_method=args.profile_method,
            )

        if args.cluster_method == "kmeans":
            num_clusters = args.num_clusters
            if num_clusters is None:
                num_clusters = max(2, math.ceil(len(site_labels) / args.set_size))
            cluster_result = cluster_websites_kmeans(
                profiles,
                num_clusters=num_clusters,
                seed=args.seed,
            )
        else:
            cluster_result = cluster_websites_from_profiles(
                profiles,
                set_size=args.set_size,
                seed=args.seed,
            )

        website_to_pseudo_label = cluster_result.website_to_pseudo_label
        pseudo_label_to_websites = cluster_result.pseudo_label_to_websites
        cluster_info = {
            "set_size": int(args.set_size),
            "rounds": int(args.rounds),
            "profile_method": args.profile_method,
            "cluster_method": args.cluster_method,
            "num_clusters": num_clusters if args.cluster_method == "kmeans" else None,
            "use_encoder": bool(args.use_encoder),
            "latent_dim": int(args.latent_dim) if args.use_encoder else None,
            "excluded_labels": sorted(int(v) for v in exclude_labels),
            "site_labels": [int(v) for v in cluster_result.labels],
            "num_pseudo_labels": len(cluster_result.pseudo_label_to_websites),
        }
        source = "toob_cluster_encoder" if args.use_encoder else "toob_cluster"
    pseudo, keep_indices = labels_to_pseudo(
        labels,
        website_to_pseudo_label,
        drop_unmapped=args.drop_unmapped,
    )
    kept_labels = labels[keep_indices]
    cluster_quality = evaluate_cluster_quality(
        bursts,
        labels,
        website_to_pseudo_label,
        profile_method=args.profile_method,
        exclude_labels=exclude_labels,
        profiles=profiles,
    )
    save_npz_dataset(args.output, labels=kept_labels, pseudo_labels=pseudo, keep_indices=keep_indices)
    unique, counts = np.unique(pseudo, return_counts=True)

    json_output = Path(args.json_output) if args.json_output else Path(args.output).with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "labels_npz": str(args.labels_npz),
        "source": source,
        "mapping": str(args.mapping) if args.mapping else None,
        "drop_unmapped": bool(args.drop_unmapped),
        "total_samples": int(len(labels)),
        "kept_samples": int(len(keep_indices)),
        "dropped_samples": int(len(labels) - len(keep_indices)),
        "website_to_pseudo_label": {
            str(k): int(v)
            for k, v in sorted(website_to_pseudo_label.items())
        },
        "pseudo_label_to_websites": {
            str(k): [int(v) for v in websites]
            for k, websites in sorted(pseudo_label_to_websites.items())
        },
        "pseudo_label_counts": {str(int(label)): int(count) for label, count in zip(unique, counts)},
        "cluster": cluster_info,
        "cluster_quality": cluster_quality,
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
    print("cluster quality:")
    print(f"  silhouette_mean: {cluster_quality['silhouette_mean']:.6f}")
    print(
        "  cluster website count min/max: "
        f"{cluster_quality['cluster_website_count']['min']:.0f}/"
        f"{cluster_quality['cluster_website_count']['max']:.0f}"
    )
    print(
        "  cluster sample count min/max: "
        f"{cluster_quality['cluster_sample_count']['min']:.0f}/"
        f"{cluster_quality['cluster_sample_count']['max']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
