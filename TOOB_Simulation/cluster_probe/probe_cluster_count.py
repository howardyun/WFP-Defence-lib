from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toob.data import labels_to_int, load_npz_dataset  # noqa: E402


DEFAULT_BURST_CANDIDATES = (
    PROJECT_ROOT / "outputs_train_tuning" / "cluster_cache_set30_super_exclude95" / "burst_dataset.npz",
    PROJECT_ROOT / "outputs" / "burst_dataset.npz",
    PROJECT_ROOT / "outputs_smoke" / "burst_dataset.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a suitable TOOB pseudo-label cluster count.")
    parser.add_argument("--burst-npz", help="Burst dataset .npz. If omitted, common main-flow paths are tried.")
    parser.add_argument("--data-key", default="data")
    parser.add_argument("--labels-key", default="labels")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "cluster_probe" / "results"))
    parser.add_argument("--k-values", nargs="*", type=int, help="Explicit K values to evaluate.")
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--exclude-labels", nargs="*", type=int, default=[95])
    parser.add_argument("--profile-method", choices=("super", "mean_abs", "mean_signed"), default="super")
    parser.add_argument("--normalize", choices=("zscore", "l2", "none"), default="zscore")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--balance-weight", type=float, default=0.10)
    parser.add_argument("--small-cluster-penalty", type=float, default=0.25)
    return parser.parse_args()


def resolve_burst_path(path: str | None) -> Path:
    if path:
        resolved = Path(path)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"burst dataset not found: {resolved}")

    for candidate in DEFAULT_BURST_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in DEFAULT_BURST_CANDIDATES)
    raise FileNotFoundError(f"No default burst dataset found. Tried:\n{searched}")


def compute_profile(bursts: np.ndarray, method: str) -> np.ndarray:
    if method == "super":
        abs_profile = np.mean(np.abs(bursts), axis=0)
        signed_profile = np.mean(bursts, axis=0)
        return np.concatenate([abs_profile.ravel(), signed_profile.ravel()])
    if method == "mean_abs":
        return np.mean(np.abs(bursts), axis=0).ravel()
    if method == "mean_signed":
        return np.mean(bursts, axis=0).ravel()
    raise ValueError(f"unknown profile method: {method}")


def build_website_profiles(
    bursts: np.ndarray,
    labels: np.ndarray,
    *,
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
        raise ValueError("No website labels remain after exclusions")
    return site_labels, np.stack(profiles, axis=0).astype(np.float64), sample_counts


def normalize_profiles(profiles: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return profiles
    if mode == "l2":
        denom = np.linalg.norm(profiles, axis=1, keepdims=True)
        return profiles / np.maximum(denom, 1e-12)
    if mode == "zscore":
        mean = np.mean(profiles, axis=0, keepdims=True)
        std = np.std(profiles, axis=0, keepdims=True)
        return (profiles - mean) / np.maximum(std, 1e-12)
    raise ValueError(f"unknown normalization mode: {mode}")


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=x.dtype)
    first = int(rng.integers(0, n))
    centers[0] = x[first]
    closest_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for center_idx in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 0:
            centers[center_idx] = x[int(rng.integers(0, n))]
        else:
            probs = closest_sq / total
            centers[center_idx] = x[int(rng.choice(n, p=probs))]
        new_sq = np.sum((x - centers[center_idx]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def repair_empty_clusters(
    x: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    distances_sq: np.ndarray,
) -> np.ndarray:
    k = centers.shape[0]
    counts = np.bincount(labels, minlength=k)
    if np.all(counts > 0):
        return labels

    labels = labels.copy()
    nearest_sq = distances_sq[np.arange(x.shape[0]), labels]
    for empty_cluster in np.flatnonzero(counts == 0):
        donor_order = np.argsort(nearest_sq)[::-1]
        for candidate in donor_order:
            old_cluster = int(labels[candidate])
            if counts[old_cluster] > 1:
                labels[candidate] = int(empty_cluster)
                counts[old_cluster] -= 1
                counts[empty_cluster] += 1
                break
    return labels


def run_kmeans(
    x: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    centers = kmeans_plus_plus_init(x, k, rng)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        distances_sq = np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances_sq, axis=1).astype(np.int64)
        new_labels = repair_empty_clusters(x, new_labels, centers, distances_sq)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members):
                centers[cluster_id] = np.mean(members, axis=0)
    inertia = float(np.sum((x - centers[labels]) ** 2))
    return labels, centers, inertia


def best_kmeans(
    x: np.ndarray,
    k: int,
    *,
    seed: int,
    restarts: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        labels, centers, inertia = run_kmeans(x, k, rng=rng, max_iter=max_iter)
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    assert best is not None
    return best


def silhouette_score(distance_matrix: np.ndarray, cluster_labels: np.ndarray) -> tuple[float, float]:
    values = []
    for idx, own_cluster in enumerate(cluster_labels):
        own_indices = np.flatnonzero(cluster_labels == own_cluster)
        own_indices = own_indices[own_indices != idx]
        a_value = float(np.mean(distance_matrix[idx, own_indices])) if len(own_indices) else 0.0
        other_means = []
        for cluster_id in sorted(set(int(value) for value in cluster_labels)):
            if cluster_id == int(own_cluster):
                continue
            other_indices = np.flatnonzero(cluster_labels == cluster_id)
            if len(other_indices):
                other_means.append(float(np.mean(distance_matrix[idx, other_indices])))
        b_value = min(other_means) if other_means else 0.0
        denominator = max(a_value, b_value)
        values.append(0.0 if denominator == 0 else (b_value - a_value) / denominator)
    return float(np.mean(values)), float(np.min(values))


def calinski_harabasz_score(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    n = x.shape[0]
    k = centers.shape[0]
    if k <= 1 or n <= k:
        return 0.0
    overall = np.mean(x, axis=0)
    between = 0.0
    within = 0.0
    for cluster_id in range(k):
        members = x[labels == cluster_id]
        if len(members) == 0:
            continue
        between += len(members) * float(np.sum((centers[cluster_id] - overall) ** 2))
        within += float(np.sum((members - centers[cluster_id]) ** 2))
    if within <= 0:
        return 0.0
    return float((between / (k - 1)) / (within / (n - k)))


def davies_bouldin_score(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    k = centers.shape[0]
    scatters = np.zeros(k, dtype=np.float64)
    for cluster_id in range(k):
        members = x[labels == cluster_id]
        if len(members):
            scatters[cluster_id] = float(np.mean(np.linalg.norm(members - centers[cluster_id], axis=1)))
    values = []
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j:
                continue
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            ratios.append((scatters[i] + scatters[j]) / max(distance, 1e-12))
        values.append(max(ratios) if ratios else 0.0)
    return float(np.mean(values))


def evaluate_k(
    x: np.ndarray,
    site_labels: list[int],
    sample_counts: dict[int, int],
    k: int,
    *,
    seed: int,
    restarts: int,
    max_iter: int,
    min_cluster_size: int,
    balance_weight: float,
    small_cluster_penalty: float,
) -> dict[str, Any]:
    cluster_labels, centers, inertia = best_kmeans(x, k, seed=seed + k * 997, restarts=restarts, max_iter=max_iter)
    distance_matrix = pairwise_distances(x)
    sil_mean, sil_min = silhouette_score(distance_matrix, cluster_labels)
    ch_score = calinski_harabasz_score(x, cluster_labels, centers)
    db_score = davies_bouldin_score(x, cluster_labels, centers)

    cluster_site_counts = []
    cluster_sample_counts = []
    pseudo_label_to_websites: dict[str, list[int]] = {}
    for cluster_id in range(k):
        member_indices = np.flatnonzero(cluster_labels == cluster_id)
        websites = [int(site_labels[index]) for index in member_indices]
        pseudo_label_to_websites[str(cluster_id)] = sorted(websites)
        cluster_site_counts.append(int(len(websites)))
        cluster_sample_counts.append(int(sum(sample_counts[website] for website in websites)))

    min_sites = min(cluster_site_counts)
    max_sites = max(cluster_site_counts)
    balance_ratio = min_sites / max(max_sites, 1)
    small_penalty = max(0.0, (min_cluster_size - min_sites) / max(min_cluster_size, 1))
    score = sil_mean - balance_weight * (1.0 - balance_ratio) - small_cluster_penalty * small_penalty

    num_websites = len(site_labels)
    return {
        "k": int(k),
        "suggested_set_size": int(math.ceil(num_websites / k)),
        "num_websites": int(num_websites),
        "silhouette_mean": sil_mean,
        "silhouette_min": sil_min,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score,
        "inertia": inertia,
        "cluster_site_count_min": int(min_sites),
        "cluster_site_count_max": int(max_sites),
        "cluster_site_count_std": float(np.std(cluster_site_counts)),
        "cluster_sample_count_min": int(min(cluster_sample_counts)),
        "cluster_sample_count_max": int(max(cluster_sample_counts)),
        "cluster_sample_count_std": float(np.std(cluster_sample_counts)),
        "balance_ratio": float(balance_ratio),
        "selection_score": float(score),
        "pseudo_label_to_websites": pseudo_label_to_websites,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "k",
        "suggested_set_size",
        "num_websites",
        "silhouette_mean",
        "silhouette_min",
        "calinski_harabasz",
        "davies_bouldin",
        "inertia",
        "cluster_site_count_min",
        "cluster_site_count_max",
        "cluster_site_count_std",
        "cluster_sample_count_min",
        "cluster_sample_count_max",
        "cluster_sample_count_std",
        "balance_ratio",
        "selection_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    burst_path = resolve_burst_path(args.burst_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bursts, labels = load_npz_dataset(burst_path, data_key=args.data_key, labels_key=args.labels_key)
    site_labels, profiles, sample_counts = build_website_profiles(
        bursts,
        labels,
        exclude_labels=set(args.exclude_labels),
        profile_method=args.profile_method,
    )
    x = normalize_profiles(profiles, args.normalize)

    if args.k_values:
        k_values = sorted(set(args.k_values))
    else:
        k_values = list(range(args.min_k, args.max_k + 1))
    k_values = [k for k in k_values if 1 < k <= len(site_labels)]
    if not k_values:
        raise ValueError("No valid K values to evaluate")

    rows = [
        evaluate_k(
            x,
            site_labels,
            sample_counts,
            k,
            seed=args.seed,
            restarts=args.restarts,
            max_iter=args.max_iter,
            min_cluster_size=args.min_cluster_size,
            balance_weight=args.balance_weight,
            small_cluster_penalty=args.small_cluster_penalty,
        )
        for k in k_values
    ]
    rows_for_selection = [
        row for row in rows
        if row["cluster_site_count_min"] >= args.min_cluster_size
    ] or rows
    recommendation = max(rows_for_selection, key=lambda row: row["selection_score"])

    report = {
        "input": {
            "burst_npz": str(burst_path),
            "data_key": args.data_key,
            "labels_key": args.labels_key,
            "profile_method": args.profile_method,
            "normalize": args.normalize,
            "exclude_labels": [int(value) for value in args.exclude_labels],
            "num_websites": len(site_labels),
            "site_labels": [int(value) for value in site_labels],
        },
        "selection": {
            "min_cluster_size": int(args.min_cluster_size),
            "balance_weight": float(args.balance_weight),
            "small_cluster_penalty": float(args.small_cluster_penalty),
        },
        "recommendation": recommendation,
        "results": rows,
    }

    csv_path = output_dir / "cluster_count_probe.csv"
    json_path = output_dir / "cluster_count_probe.json"
    recommendation_path = output_dir / "recommendation.json"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    recommendation_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    print(f"burst dataset: {burst_path}")
    print(f"evaluated K: {' '.join(str(value) for value in k_values)}")
    print(
        "recommendation: "
        f"K={recommendation['k']}, "
        f"SET_SIZE~{recommendation['suggested_set_size']}, "
        f"silhouette={recommendation['silhouette_mean']:.6f}, "
        f"balance={recommendation['balance_ratio']:.3f}"
    )
    print(f"saved csv: {csv_path}")
    print(f"saved json: {json_path}")
    print(f"saved recommendation: {recommendation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
