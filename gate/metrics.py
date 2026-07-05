"""Fair-comparison metrics computed on the common completed-request set.

Fairness (spec §6): plain/naive/proposed drop different leftover samples, so every
metric is restricted to the intersection of request IDs that completed in all
compared runtimes. Goodput uses the WALL-CLOCK definition, and the wall-clock
window is also measured on the common set (first arrival -> last completion among
common ids), so numerator and denominator share the same sample set.
"""
from __future__ import annotations

import numpy as np

from .runtimes import Schedule, simulate, simulate_starts


def _latency_seconds(sched: Schedule, arrivals: np.ndarray, origin: str):
    """Per-sample latency (seconds) + completion times.

    origin 'arrival'      : completion - arrival (response time).
    origin 'stage1_start' : completion - start of the sample's seg1/whole op
                            (service latency; excludes waiting behind the
                            backlog in saturated lambda=0 mode).
    """
    if origin == "stage1_start":
        completion, s1, _ = simulate_starts(sched, arrivals)
        with np.errstate(invalid="ignore"):          # dropped samples: inf - inf
            lat = completion - s1
        lat[~np.isfinite(completion)] = np.inf
        return lat, completion
    if origin == "arrival":
        completion, _ = simulate(sched, arrivals)
        return completion - arrivals, completion
    raise ValueError(f"latency origin must be 'arrival' or 'stage1_start', got {origin!r}")


def common_completed(schedules: list[Schedule]) -> np.ndarray:
    """Intersection of completed request ids across the given schedules.

    The dropped/completed partition depends only on the schedules (not lambda),
    so this set is stable across the load sweep.
    """
    ids = None
    for s in schedules:
        c = set(s.completed_ids().tolist())
        ids = c if ids is None else (ids & c)
    return np.array(sorted(ids), dtype=np.int64) if ids else np.array([], dtype=np.int64)


def wallclock(completion: np.ndarray, arrivals: np.ndarray, ids: np.ndarray) -> float:
    """Wall-clock span over the common set: last completion - first arrival."""
    return float(completion[ids].max() - arrivals[ids].min())


def goodput_vs_slo(sched: Schedule, arrivals: np.ndarray, common_ids: np.ndarray,
                   slo_ms_grid: np.ndarray, mode: str = "mean_throughput",
                   origin: str = "arrival") -> np.ndarray:
    """Goodput for each SLO in the grid, over the common set only.

    Two definitions (selected by `mode`):

    * "mean_throughput" (default): average per-sample throughput —
          goodput(SLO) = (1/N) * Σ_{i : latency_i <= SLO} (1 / latency_i)
      Each good sample contributes its own throughput (1/latency, in 1/s); samples
      that miss the SLO contribute 0, and N is the full common-set size, so misses
      drag the mean down. Low-latency samples (e.g. seg1 exits) are rewarded more.

    * "wallclock" (spec §6): goodput(SLO) = (# good samples) / (wall-clock span).
      wall-clock = last completion - first arrival over the common set.
    """
    lat, completion = _latency_seconds(sched, arrivals, origin)
    lat_c = lat[common_ids]                            # seconds, > 0
    N = len(common_ids)
    out = np.empty(len(slo_ms_grid), dtype=float)

    if mode == "wallclock":
        wc = wallclock(completion, arrivals, common_ids)
        for i, slo_ms in enumerate(slo_ms_grid):
            good = int(np.sum(lat_c <= slo_ms / 1000.0))
            out[i] = good / wc if wc > 0 else 0.0
        return out

    # mean_throughput
    inv = 1.0 / lat_c                                  # per-sample throughput (1/s)
    for i, slo_ms in enumerate(slo_ms_grid):
        good_mask = lat_c <= slo_ms / 1000.0
        out[i] = float(inv[good_mask].sum()) / N if N > 0 else 0.0
    return out


def latency_ms(sched: Schedule, arrivals: np.ndarray, common_ids: np.ndarray,
               origin: str = "arrival") -> np.ndarray:
    """Per-sample latency (ms) over the common set."""
    lat, _ = _latency_seconds(sched, arrivals, origin)
    return lat[common_ids] * 1000.0


def response_stats(sched: Schedule, arrivals: np.ndarray, common_ids: np.ndarray):
    """Return (mean_ms, p99_ms) response time over the common set."""
    lat = latency_ms(sched, arrivals, common_ids)
    return float(np.mean(lat)), float(np.percentile(lat, 99))
