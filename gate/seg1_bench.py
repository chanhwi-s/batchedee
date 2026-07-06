"""Standalone seg1 kernel-time microbenchmark.

Sweeps the seg1 (EE model, patch-embed + blocks 1-6 + LPH) batch size over
powers of two and measures the per-op GPU kernel time, exec-stats style
(plot8 left panel, seg1 only). Exactly N_SAMPLES samples are pushed through
each batch size (N_SAMPLES/bs ops), so small sizes are averaged over many ops.

Timing is data-independent (dense model, no branching), so random tensors are
used — no ImageNet loading. Warmup runs precede every measurement.

Outputs
-------
  artifacts/plots/plot10_seg1_batch_sweep.{png,pdf}
  artifacts/results/seg1_batch_sweep.json
  aligned stdout table (per-op mean/std + per-sample time)
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import export
from . import plot_style as ps
from .plot_style import FIG_SINGLE, STAGE1_SWATCH
from .plots import _save
from .util import Config, TimedSession

ps.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_SAMPLES = 4096                 # fixed: quick-look benchmark


def run(cfg: Config, sizes: list[int] | None = None) -> list[dict]:
    sizes = sizes or SIZES
    export.export_seg1_sizes(cfg, sizes)     # cached after first run
    H = W = int(cfg.data.crop)

    rows = []
    for bs in sizes:
        sess = TimedSession(export.seg1_path_for(cfg, bs), cfg)
        feed = {sess.input_names[0]: np.random.randn(bs, 3, H, W).astype(np.float32)}
        sess.warmup(feed)
        n_ops = max(1, N_SAMPLES // bs)
        times = np.empty(n_ops, dtype=float)
        for i in range(n_ops):
            _, times[i] = sess.run_timed(feed)
        del sess                              # release GPU memory before next size
        rows.append({"batch_size": bs, "n_ops": n_ops,
                     "mean_ms": float(times.mean() * 1e3),
                     "std_ms": float(times.std() * 1e3),
                     "per_sample_ms": float(times.mean() * 1e3 / bs)})
        r = rows[-1]
        print(f"[seg1bench] bs={bs:>4}  ops={n_ops:>5}  "
              f"mean={r['mean_ms']:8.3f} ms  std={r['std_ms']:6.3f}  "
              f"per-sample={r['per_sample_ms']:6.3f} ms")

    d = cfg.paths["results_dir"]
    os.makedirs(d, exist_ok=True)
    jpath = os.path.join(d, "seg1_batch_sweep.json")
    with open(jpath, "w") as f:
        json.dump({"n_samples": N_SAMPLES, "rows": rows}, f, indent=2)
    print(f"[seg1bench] wrote {jpath}")

    _plot(cfg, rows)
    return rows


def _plot(cfg: Config, rows: list[dict]):
    x = np.arange(len(rows))
    means = [r["mean_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.bar(x, means, 0.7, color=STAGE1_SWATCH)
    ax.bar_label(bars, labels=[f"{m:.2f}" if m < 10 else f"{m:.1f}" for m in means],
                 fontsize=6, padding=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["batch_size"]) for r in rows])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time per op (ms)")
    ax.set_title("Seg1 Kernel Time vs Batch Size")
    ax.margins(y=0.2)
    return _save(fig, cfg, "plot10_seg1_batch_sweep")
