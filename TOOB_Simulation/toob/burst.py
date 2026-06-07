from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BurstStats:
    traces: int
    max_bursts: int
    nonzero_bursts: int
    max_abs_burst: float


def direction_to_burst(direction_data: np.ndarray, max_bursts: int | None = None) -> np.ndarray:
    """Convert packet direction sequences to signed burst-count sequences.

    Input rows contain positive values for outgoing packets, negative values for
    incoming packets, and zeros for padding. Magnitudes are ignored, so signed
    timestamp traces are handled as direction-only traces.
    """
    directions = np.asarray(direction_data)
    if directions.ndim != 2:
        raise ValueError(f"direction_data must be 2-D, got shape {directions.shape}")

    if max_bursts is None:
        max_bursts = directions.shape[1]

    bursts = np.zeros((directions.shape[0], max_bursts), dtype=np.float32)
    for row_idx, row in enumerate(directions):
        burst_idx = -1
        cur_sign = 0.0
        for value in row:
            sign = np.sign(value)
            if sign == 0:
                break
            if sign != cur_sign:
                burst_idx += 1
                cur_sign = sign
                if burst_idx >= max_bursts:
                    break
            bursts[row_idx, burst_idx] += sign
    return bursts


def direction_to_sign_sequence(direction_data: np.ndarray, max_trace_len: int) -> np.ndarray:
    """Convert signed packet traces to DF-style direction features."""
    directions = np.sign(np.asarray(direction_data, dtype=np.float32))
    if directions.ndim != 2:
        raise ValueError(f"direction_data must be 2-D, got shape {directions.shape}")
    if directions.shape[1] > max_trace_len:
        return directions[:, :max_trace_len].astype(np.float32, copy=False)
    if directions.shape[1] < max_trace_len:
        pad_width = ((0, 0), (0, max_trace_len - directions.shape[1]))
        directions = np.pad(directions, pad_width=pad_width, mode="constant", constant_values=0)
    return directions.astype(np.float32, copy=False)


def burst_to_direction(burst_data: np.ndarray, max_trace_len: int) -> np.ndarray:
    """Expand signed burst counts back into packet direction sequences."""
    bursts = np.asarray(burst_data)
    if bursts.ndim != 2:
        raise ValueError(f"burst_data must be 2-D, got shape {bursts.shape}")

    directions = np.zeros((bursts.shape[0], max_trace_len), dtype=np.float32)
    for row_idx, row in enumerate(bursts):
        cursor = 0
        for burst in row:
            count = int(round(abs(float(burst))))
            if count <= 0:
                continue
            sign = 1.0 if burst > 0 else -1.0
            end = min(max_trace_len, cursor + count)
            directions[row_idx, cursor:end] = sign
            cursor = end
            if cursor >= max_trace_len:
                break
    return directions


def apply_burst_perturbation(
    burst_batch: torch.Tensor,
    delta_batch: torch.Tensor,
    *,
    round_output: bool = False,
) -> torch.Tensor:
    """Apply non-negative burst padding in the original burst direction."""
    sign = torch.sign(burst_batch)
    adv = burst_batch + delta_batch * sign
    if round_output:
        adv = torch.round(adv)
    return adv


def soft_burst_to_direction(
    burst_batch: torch.Tensor,
    *,
    trace_len: int = 5000,
    tau: float = 1.5,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Differentiably project signed burst counts to direction sequences.

    A hard burst expansion repeats each burst sign `abs(count)` times. That
    operation is not differentiable because it involves rounding and indexing.
    This function uses smoothed interval indicators, so gradients can flow from
    a direction-sequence detector back into burst counts.
    """
    if burst_batch.ndim != 2:
        raise ValueError(f"burst_batch must be 2-D [N, K], got shape {tuple(burst_batch.shape)}")
    if tau <= 0:
        raise ValueError("tau must be positive")

    lengths = torch.abs(burst_batch).clamp_min(0)
    signs = torch.sign(burst_batch)
    ends = torch.cumsum(lengths, dim=1)
    starts = ends - lengths

    positions = (
        torch.arange(trace_len, device=burst_batch.device, dtype=burst_batch.dtype)
        .view(1, 1, trace_len)
        + 0.5
    )
    projected = torch.zeros(
        burst_batch.shape[0],
        trace_len,
        device=burst_batch.device,
        dtype=burst_batch.dtype,
    )

    for start_idx in range(0, burst_batch.shape[1], chunk_size):
        end_idx = min(start_idx + chunk_size, burst_batch.shape[1])
        chunk_starts = starts[:, start_idx:end_idx].unsqueeze(-1)
        chunk_ends = ends[:, start_idx:end_idx].unsqueeze(-1)
        chunk_signs = signs[:, start_idx:end_idx].unsqueeze(-1)
        occupancy = torch.sigmoid((positions - chunk_starts) / tau) - torch.sigmoid(
            (positions - chunk_ends) / tau
        )
        projected = projected + torch.sum(chunk_signs * occupancy, dim=1)

    return projected.clamp(-1.0, 1.0)


def hard_burst_to_direction(
    burst_batch: torch.Tensor,
    *,
    trace_len: int = 5000,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Project signed burst counts to hard DF-style direction sequences."""
    if burst_batch.ndim != 2:
        raise ValueError(f"burst_batch must be 2-D [N, K], got shape {tuple(burst_batch.shape)}")

    lengths = torch.round(torch.abs(burst_batch)).clamp_min(0)
    signs = torch.sign(burst_batch)
    ends = torch.cumsum(lengths, dim=1)
    starts = ends - lengths

    positions = (
        torch.arange(trace_len, device=burst_batch.device, dtype=burst_batch.dtype)
        .view(1, 1, trace_len)
        + 0.5
    )
    projected = torch.zeros(
        burst_batch.shape[0],
        trace_len,
        device=burst_batch.device,
        dtype=burst_batch.dtype,
    )

    for start_idx in range(0, burst_batch.shape[1], chunk_size):
        end_idx = min(start_idx + chunk_size, burst_batch.shape[1])
        chunk_starts = starts[:, start_idx:end_idx].unsqueeze(-1)
        chunk_ends = ends[:, start_idx:end_idx].unsqueeze(-1)
        chunk_signs = signs[:, start_idx:end_idx].unsqueeze(-1)
        occupancy = ((positions > chunk_starts) & (positions <= chunk_ends)).to(burst_batch.dtype)
        projected = projected + torch.sum(chunk_signs * occupancy, dim=1)

    return projected.clamp(-1.0, 1.0)


def straight_through_burst_to_direction(
    burst_batch: torch.Tensor,
    *,
    trace_len: int = 5000,
    tau: float = 1.5,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Use hard direction values in the forward pass and soft gradients backward."""
    soft = soft_burst_to_direction(
        burst_batch,
        trace_len=trace_len,
        tau=tau,
        chunk_size=chunk_size,
    )
    hard = hard_burst_to_direction(
        burst_batch,
        trace_len=trace_len,
        chunk_size=chunk_size,
    )
    return soft + (hard - soft).detach()


def overhead_ratio(original: torch.Tensor, defended: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return per-sample bandwidth overhead in burst space."""
    added = torch.sum(torch.abs(defended) - torch.abs(original), dim=1)
    base = torch.sum(torch.abs(original), dim=1).clamp_min(eps)
    return added / base


def summarize_bursts(bursts: np.ndarray) -> BurstStats:
    data = np.asarray(bursts)
    return BurstStats(
        traces=int(data.shape[0]),
        max_bursts=int(data.shape[1]),
        nonzero_bursts=int(np.count_nonzero(data)),
        max_abs_burst=float(np.max(np.abs(data))) if data.size else 0.0,
    )
