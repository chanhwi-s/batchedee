"""Shared utilities: config loading, ORT session helpers, timing."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config(dict):
    """Dict with attribute access and dotted-path get."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def get_path(self, dotted: str, default=None):
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def slo_grid_ms(cfg: Config) -> np.ndarray:
    s = cfg.slo
    # inclusive of stop
    return np.arange(s["start_ms"], s["stop_ms"] + s["step_ms"], s["step_ms"], dtype=float)


def lambda_grid(cfg: Config) -> np.ndarray:
    s = cfg.arrivals["lambda_sweep"]
    return np.arange(s["start"], s["stop"] + s["step"], s["step"], dtype=float)


def ensure_dirs(cfg: Config) -> None:
    for k in ("onnx_dir", "plots_dir", "results_dir"):
        os.makedirs(cfg.paths[k], exist_ok=True)


def setup_hf_cache(cfg: Config) -> str:
    """Point HuggingFace/timm caches at a writable dir.

    Fixes `PermissionError` when the system default (e.g. a shared /home/shared
    cache) is not writable. If HF_HOME is already set AND writable, it is kept.
    """
    env_home = os.environ.get("HF_HOME")
    if env_home and os.access(os.path.dirname(env_home) or "/", os.W_OK):
        try:
            os.makedirs(env_home, exist_ok=True)
            return env_home
        except OSError:
            pass
    cache = os.path.abspath(cfg.paths.get("hf_cache", "artifacts/hf_cache"))
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = os.path.join(cache, "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache, "hub")
    os.environ["TRANSFORMERS_CACHE"] = cache
    return cache


# --------------------------------------------------------------------------- #
# ONNX Runtime session
# --------------------------------------------------------------------------- #
_OPT_LEVELS = {
    "ORT_DISABLE_ALL": "ORT_DISABLE_ALL",
    "ORT_ENABLE_BASIC": "ORT_ENABLE_BASIC",
    "ORT_ENABLE_EXTENDED": "ORT_ENABLE_EXTENDED",
    "ORT_ENABLE_ALL": "ORT_ENABLE_ALL",
}


def make_session(onnx_path: str, cfg: Config):
    """Create an ORT InferenceSession honoring config providers/options."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    lvl = cfg.runtime.get("graph_optimization_level", "ORT_ENABLE_ALL")
    so.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, _OPT_LEVELS.get(lvl, "ORT_ENABLE_ALL")
    )
    n = int(cfg.runtime.get("intra_op_num_threads", 0))
    if n > 0:
        so.intra_op_num_threads = n

    providers = list(cfg.runtime.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    # Filter to what's actually available so local (CPU-only) dev doesn't crash.
    avail = set(ort.get_available_providers())
    providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess


class TimedSession:
    """Wraps an ORT session; runs inference and returns (outputs, elapsed_seconds).

    Timing uses a CUDA-synchronizing pattern: onnxruntime's run() call blocks
    until outputs are copied back to host, so wall-clock around run() captures
    real GPU service time. Warmup runs are executed once and excluded.
    """

    def __init__(self, onnx_path: str, cfg: Config, warmup_batch: dict | None = None):
        self.path = onnx_path
        self.cfg = cfg
        self.sess = make_session(onnx_path, cfg)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self._warmed = False
        if warmup_batch is not None:
            self.warmup(warmup_batch)

    def warmup(self, feed: dict) -> None:
        it = int(self.cfg.runtime.get("warmup_iters", 20))
        for _ in range(it):
            self.sess.run(self.output_names, feed)
        self._warmed = True

    def run_timed(self, feed: dict, iters: int | None = None):
        """Run inference; return (outputs_list, elapsed_seconds).

        Elapsed is the median over `iters` measured repeats (default from config).
        """
        if iters is None:
            iters = int(self.cfg.runtime.get("measure_iters", 1))
        iters = max(1, iters)
        times = np.empty(iters, dtype=float)
        outs = None
        for i in range(iters):
            t0 = time.perf_counter()
            outs = self.sess.run(self.output_names, feed)
            times[i] = time.perf_counter() - t0
        return outs, float(np.median(times))
