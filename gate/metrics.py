"""Fair-comparison metrics computed on the common completed-request set.

Fairness (spec §6): plain/naive/proposed drop different leftover samples, so every
metric is restricted to the intersection of request IDs that completed in all
compared runtimes. Goodput uses the WALL-CLOCK definition, and the wall-clock
window is also measured on the common set (first arrival -> last completion among
common ids), so numerator and denominator share the same sample set.
"""
from __future__ import annotations

import numpy as np

from .runtimes import Schedule, simulate


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
                   slo_ms_grid: np.ndarray) -> np.ndarray:
    """Goodput (good samples / wall-clock) for each SLO in the grid.

    good sample := per-sample latency <= SLO, counted over common_ids only.
    """
    completion, _ = simulate(sched, arrivals)
    lat = completion - arrivals                       # seconds
    lat_c = lat[common_ids]
    wc = wallclock(completion, arrivals, common_ids)  # seconds
    out = np.empty(len(slo_ms_grid), dtype=float)
    for i, slo_ms in enumerate(slo_ms_grid):
        good = int(np.sum(lat_c <= slo_ms / 1000.0))
        out[i] = good / wc if wc > 0 else 0.0
    return out


def latency_ms(sched: Schedule, arrivals: np.ndarray, common_ids: np.ndarray) -> np.ndarray:
    """Per-sample latency (ms) over the common set."""
    completion, _ = simulate(sched, arrivals)
    lat = (completion - arrivals)[common_ids]
    return lat * 1000.0


def response_stats(sched: Schedule, arrivals: np.ndarray, common_ids: np.ndarray):
    """Return (mean_ms, p99_ms) response time over the common set."""
    lat = latency_ms(sched, arrivals, common_ids)
    return float(np.mean(lat)), float(np.percentile(lat, 99))
