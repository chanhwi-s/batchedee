"""Poisson arrival process on the simulation clock.

Inter-arrival times are Exp(rate=lambda). We draw the underlying standard
exponentials ONCE from a fixed seed and scale by 1/lambda, so:
  * the same seed gives all runtimes the identical arrival trace, and
  * across the load sweep, every lambda shares the same underlying randomness
    (higher lambda simply compresses the same arrival pattern in time).
"""
from __future__ import annotations

import numpy as np


def base_exponentials(n: int, seed: int) -> np.ndarray:
    """n iid Exp(1) inter-arrival gaps, fixed by seed."""
    rng = np.random.default_rng(int(seed))
    return rng.exponential(1.0, size=int(n))


def arrivals_from_base(base_exp: np.ndarray, lam: float) -> np.ndarray:
    """Arrival timestamps (seconds) for rate `lam` from precomputed Exp(1) gaps."""
    return np.cumsum(base_exp) / float(lam)


def poisson_arrivals(n: int, lam: float, seed: int) -> np.ndarray:
    """Convenience: arrival timestamps for n requests at rate lam."""
    return arrivals_from_base(base_exponentials(n, seed), lam)
