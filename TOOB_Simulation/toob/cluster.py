from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import labels_to_int


@dataclass
class ClusterResult:
    website_to_pseudo_label: dict[int, int]
    pseudo_label_to_websites: dict[int, list[int]]
    labels: list[int]
    total_sets: list[list[int]]


def _profile(bursts: np.ndarray, method: str = "super") -> np.ndarray:
    """Compute a per-website traffic profile from burst data.

    Parameters
    ----------
    bursts : np.ndarray, shape (N, C, L)
        Burst representations (direction or packet-count sequences).
    method : str
        ``super``  – concatenate mean-absolute and mean-signed profiles.
        ``mean_abs`` – mean absolute values across samples.
        ``mean_signed`` – mean signed values across samples.
    """
    if method == "super":
        abs_profile = np.mean(np.abs(bursts), axis=0)
        signed_profile = np.mean(bursts, axis=0)
        return np.concatenate([abs_profile.ravel(), signed_profile.ravel()])
    elif method == "mean_abs":
        return np.mean(np.abs(bursts), axis=0).ravel()
    elif method == "mean_signed":
        return np.mean(bursts, axis=0).ravel()
    raise ValueError(f"Unknown profile method: {method}")


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def evaluate_cluster_quality(
    bursts: np.ndarray,
    labels: np.ndarray,
    website_to_pseudo_label: dict[int, int],
    *,
    profile_method: str = "super",
    exclude_labels: set[int] | None = None,
    profiles: dict[int, np.ndarray] | None = None,
) -> dict:
    """Evaluate pseudo-label cluster quality on per-website burst profiles.

    When ``profiles`` is provided it is used directly instead of computing
    ``_profile`` from ``bursts``; this lets callers evaluate clustering done in
    a learned latent space.
    """
    labels_int = labels_to_int(labels)
    exclude_labels = exclude_labels or set()
    observed_labels = sorted(
        label for label in set(int(v) for v in labels_int)
        if label not in exclude_labels and label in website_to_pseudo_label
    )
    if not observed_labels:
        return {
            "profile_method": profile_method,
            "num_websites": 0,
            "num_pseudo_labels": 0,
            "silhouette_mean": 0.0,
            "silhouette_min": 0.0,
            "intra_pairwise_distance": _summarize([]),
            "inter_centroid_distance": _summarize([]),
            "cluster_website_count": _summarize([]),
            "cluster_sample_count": _summarize([]),
            "per_cluster": {},
        }

    profile_vectors: dict[int, np.ndarray] = {}
    sample_counts: dict[int, int] = {}
    for label in observed_labels:
        mask = labels_int == label
        if profiles is not None:
            profile_vectors[label] = np.asarray(profiles[label], dtype=np.float64)
        else:
            profile_vectors[label] = _profile(bursts[mask], method=profile_method)
        sample_counts[label] = int(np.sum(mask))

    grouped: dict[int, list[int]] = {}
    for label in observed_labels:
        grouped.setdefault(int(website_to_pseudo_label[label]), []).append(label)
    grouped = {key: sorted(value) for key, value in sorted(grouped.items())}

    label_to_index = {label: idx for idx, label in enumerate(observed_labels)}
    stacked_profiles = np.stack([profile_vectors[label] for label in observed_labels], axis=0)
    distance_matrix = np.linalg.norm(
        stacked_profiles[:, np.newaxis, :] - stacked_profiles[np.newaxis, :, :],
        axis=2,
    )

    per_cluster = {}
    intra_means = []
    centroid_profiles = {}
    for pseudo_label, websites in grouped.items():
        indices = [label_to_index[label] for label in websites]
        cluster_profiles = np.stack([profile_vectors[label] for label in websites], axis=0)
        centroid = np.mean(cluster_profiles, axis=0)
        centroid_profiles[pseudo_label] = centroid

        pairwise_values = []
        if len(indices) > 1:
            for left_pos, left_idx in enumerate(indices):
                for right_idx in indices[left_pos + 1:]:
                    pairwise_values.append(float(distance_matrix[left_idx, right_idx]))
        if pairwise_values:
            intra_means.append(float(np.mean(pairwise_values)))

        sample_count = int(sum(sample_counts[label] for label in websites))
        per_cluster[str(pseudo_label)] = {
            "websites": [int(label) for label in websites],
            "website_count": int(len(websites)),
            "sample_count": sample_count,
            "intra_pairwise_distance_mean": float(np.mean(pairwise_values)) if pairwise_values else 0.0,
            "intra_pairwise_distance_max": float(np.max(pairwise_values)) if pairwise_values else 0.0,
            "centroid_radius_mean": float(np.mean(np.linalg.norm(cluster_profiles - centroid, axis=1))),
        }

    inter_centroid_values = []
    pseudo_labels = sorted(centroid_profiles)
    for left_pos, left_label in enumerate(pseudo_labels):
        for right_label in pseudo_labels[left_pos + 1:]:
            inter_centroid_values.append(
                float(np.linalg.norm(centroid_profiles[left_label] - centroid_profiles[right_label]))
            )

    silhouette_values = []
    for label in observed_labels:
        idx = label_to_index[label]
        own_pseudo = int(website_to_pseudo_label[label])
        own_websites = grouped[own_pseudo]
        own_indices = [label_to_index[item] for item in own_websites if item != label]
        a_value = float(np.mean(distance_matrix[idx, own_indices])) if own_indices else 0.0

        other_means = []
        for pseudo_label, websites in grouped.items():
            if pseudo_label == own_pseudo:
                continue
            indices = [label_to_index[item] for item in websites]
            if indices:
                other_means.append(float(np.mean(distance_matrix[idx, indices])))
        b_value = min(other_means) if other_means else 0.0
        denominator = max(a_value, b_value)
        silhouette_values.append(0.0 if denominator == 0 else (b_value - a_value) / denominator)

    cluster_sizes = [float(value["website_count"]) for value in per_cluster.values()]
    cluster_sample_counts = [float(value["sample_count"]) for value in per_cluster.values()]
    return {
        "profile_method": profile_method,
        "num_websites": int(len(observed_labels)),
        "num_pseudo_labels": int(len(grouped)),
        "silhouette_mean": float(np.mean(silhouette_values)) if silhouette_values else 0.0,
        "silhouette_min": float(np.min(silhouette_values)) if silhouette_values else 0.0,
        "intra_pairwise_distance": _summarize(intra_means),
        "inter_centroid_distance": _summarize(inter_centroid_values),
        "cluster_website_count": _summarize(cluster_sizes),
        "cluster_sample_count": _summarize(cluster_sample_counts),
        "per_cluster": per_cluster,
    }


def _greedy_cluster(
    profiles: dict[int, np.ndarray],
    *,
    set_size: int,
    seed: int,
) -> list[list[int]]:
    """Greedy nearest-neighbour clustering over precomputed per-website profiles."""
    remaining = sorted(profiles.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    used: set[int] = set()
    total_sets: list[list[int]] = []
    for label in remaining:
        if label in used:
            continue
        pool = [lbl for lbl in remaining if lbl not in used and lbl != label]
        if not pool:
            anonymity_set = [label]
        else:
            distances = [
                (np.linalg.norm(profiles[label] - profiles[other]), other)
                for other in pool
            ]
            distances.sort(key=lambda x: x[0])
            neighbours = [label] + [lbl for _, lbl in distances[: set_size - 1]]
            anonymity_set = sorted(neighbours)

        for lbl in anonymity_set:
            used.add(lbl)
        total_sets.append(anonymity_set)
    return total_sets


def _result_from_sets(total_sets: list[list[int]]) -> ClusterResult:
    pseudo_label_to_websites = {
        pseudo_label: websites
        for pseudo_label, websites in enumerate(total_sets)
    }
    website_to_pseudo_label = {
        website: pseudo_label
        for pseudo_label, websites in pseudo_label_to_websites.items()
        for website in websites
    }
    labels = sorted(website for websites in total_sets for website in websites)
    return ClusterResult(
        website_to_pseudo_label=website_to_pseudo_label,
        pseudo_label_to_websites=pseudo_label_to_websites,
        labels=labels,
        total_sets=total_sets,
    )


def cluster_websites_from_profiles(
    profiles: dict[int, np.ndarray],
    *,
    set_size: int = 30,
    seed: int = 1,
) -> ClusterResult:
    """Cluster website labels using precomputed per-website profile vectors."""
    if not profiles:
        raise ValueError("No website profiles provided")
    return _result_from_sets(_greedy_cluster(profiles, set_size=set_size, seed=seed))


def cluster_websites(
    bursts: np.ndarray,
    labels: np.ndarray,
    *,
    set_size: int = 30,
    rounds: int = 1,
    seed: int = 1,
    exclude_labels: set[int] | None = None,
    profile_method: str = "super",
) -> ClusterResult:
    """Cluster website labels into anonymity sets by profile similarity.

    Returns compact pseudo labels for clusters of original website labels.
    """
    labels_int = labels_to_int(labels)
    exclude_labels = exclude_labels or set()
    if rounds != 1:
        raise ValueError("TOOB pseudo labels assign each website to one cluster; please keep rounds=1.")

    unique_labels = sorted(set(int(v) for v in labels_int) - exclude_labels)
    if not unique_labels:
        raise ValueError("No labels remaining after exclusions")

    profiles: dict[int, np.ndarray] = {}
    for label in unique_labels:
        mask = labels_int == label
        profiles[label] = _profile(bursts[mask], method=profile_method)

    return _result_from_sets(_greedy_cluster(profiles, set_size=set_size, seed=seed))


def build_website_profiles(
    bursts: np.ndarray,
    labels: np.ndarray,
    *,
    exclude_labels: set[int] | None = None,
    profile_method: str = "super",
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Build per-website profile vectors from burst data."""
    labels_int = labels_to_int(labels)
    exclude_labels = exclude_labels or set()
    site_labels = sorted(set(int(v) for v in labels_int) - exclude_labels)
    if not site_labels:
        raise ValueError("No labels remaining after exclusions")
    profiles = {
        label: _profile(bursts[labels_int == label], method=profile_method)
        for label in site_labels
    }
    return site_labels, profiles


def _kmeans_labels(
    x: np.ndarray,
    k: int,
    *,
    seed: int,
    restarts: int = 20,
    max_iter: int = 100,
) -> np.ndarray:
    """K-means clustering returning the best labels over several restarts."""
    n = x.shape[0]
    best_labels: np.ndarray | None = None
    best_inertia: float | None = None
    for offset in range(restarts):
        rng = np.random.default_rng(seed + offset * 1009)
        centers = x[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if len(members) == 0:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    centers[c] = x[farthest]
                else:
                    centers[c] = members.mean(axis=0)
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(distances, axis=1)))
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    assert best_labels is not None
    return best_labels


def cluster_websites_kmeans(
    profiles: dict[int, np.ndarray],
    *,
    num_clusters: int,
    seed: int = 1,
    restarts: int = 20,
    max_iter: int = 100,
) -> ClusterResult:
    """Cluster website profiles with K-means.

    Unlike the greedy set-size clustering, K-means is sensitive to the quality
    of the profile representation, so learned (encoder) representations can
    actually change the resulting website groups.
    """
    site_labels = sorted(profiles.keys())
    if num_clusters < 2:
        raise ValueError("num_clusters must be >= 2")
    if len(site_labels) < num_clusters:
        raise ValueError(f"num_clusters ({num_clusters}) exceeds website count ({len(site_labels)})")

    x = np.stack([np.asarray(profiles[s], dtype=np.float64) for s in site_labels], axis=0)
    labels = _kmeans_labels(x, num_clusters, seed=seed, restarts=restarts, max_iter=max_iter)

    total_sets = [
        sorted(site_labels[i] for i in np.flatnonzero(labels == c))
        for c in range(num_clusters)
    ]
    total_sets = [s for s in total_sets if s]
    return _result_from_sets(total_sets)
