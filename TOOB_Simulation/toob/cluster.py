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
    rng = np.random.default_rng(seed)

    unique_labels = sorted(set(int(v) for v in labels_int) - exclude_labels)
    if not unique_labels:
        raise ValueError("No labels remaining after exclusions")

    # Build per-website profiles
    profiles: dict[int, np.ndarray] = {}
    for label in unique_labels:
        mask = labels_int == label
        profiles[label] = _profile(bursts[mask], method=profile_method)

    remaining = list(unique_labels)
    total_sets: list[list[int]] = []

    for _round in range(rounds):
        rng.shuffle(remaining)
        used: set[int] = set()
        for label in remaining:
            if label in used:
                continue
            # Greedy nearest-neighbour clustering
            pool = [l for l in remaining if l not in used and l != label]
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

    pseudo_label_to_websites = {
        pseudo_label: websites
        for pseudo_label, websites in enumerate(total_sets)
    }
    website_to_pseudo_label = {
        website: pseudo_label
        for pseudo_label, websites in pseudo_label_to_websites.items()
        for website in websites
    }

    return ClusterResult(
        website_to_pseudo_label=website_to_pseudo_label,
        pseudo_label_to_websites=pseudo_label_to_websites,
        labels=unique_labels,
        total_sets=total_sets,
    )
