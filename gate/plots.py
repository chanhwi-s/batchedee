"""The four analysis plots. Each figure is saved as BOTH .png and .pdf."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import metrics  # noqa: E402
from .arrivals import poisson_arrivals  # noqa: E402
from .util import Config, lambda_grid, slo_grid_ms  # noqa: E402


def _save(fig, cfg: Config, name: str):
    d = cfg.paths["plots_dir"]
    os.makedirs(d, exist_ok=True)
    dpi = int(cfg.plots.get("dpi", 150))
    png = os.path.join(d, f"{name}.png")
    pdf = os.path.join(d, f"{name}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {png}")
    print(f"[plot] {pdf}")
    return png, pdf


def _kde(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    data = data[np.isfinite(data)]
    if len(data) < 2 or np.std(data) == 0:
        return np.zeros_like(grid)
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(data)(grid)
    except Exception:
        # Silverman-bandwidth Gaussian KDE fallback (no scipy).
        n = len(data)
        bw = 1.06 * np.std(data) * n ** (-1 / 5)
        bw = max(bw, 1e-6)
        u = (grid[:, None] - data[None, :]) / bw
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return k.sum(axis=1) / (n * bw)


# --------------------------------------------------------------------------- #
def plot_slo_goodput(cfg: Config, schedules: dict):
    """Plot 1: SLO vs Goodput, one curve per seg2_batch + plain + naive."""
    lam = float(cfg.arrivals["lambda"])
    n = schedules["plain"].n_requests
    arr = poisson_arrivals(n, lam, int(cfg.arrivals.seed))
    slo = slo_grid_ms(cfg)

    prop = schedules["proposed"]  # {B: Schedule}
    all_scheds = [schedules["plain"], schedules["naive"], *prop.values()]
    common = metrics.common_completed(all_scheds)
    mode = cfg.get_path("metrics.goodput_mode", "mean_throughput")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(slo, metrics.goodput_vs_slo(schedules["plain"], arr, common, slo, mode),
            "k--", lw=2, label="plain")
    ax.plot(slo, metrics.goodput_vs_slo(schedules["naive"], arr, common, slo, mode),
            color="0.45", ls=":", lw=2, label="naive")
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(prop)))
    for c, (B, sched) in zip(cmap, sorted(prop.items())):
        ax.plot(slo, metrics.goodput_vs_slo(sched, arr, common, slo, mode),
                color=c, lw=1.8, label=f"proposed seg2={B}")

    ylabel = ("Goodput  (1/N · Σ 1/latency, samples/s)" if mode == "mean_throughput"
              else "Goodput (good samples / sec)")
    ax.set_xlabel("Latency SLO (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"SLO vs Goodput  (λ={lam:g} req/s, N_common={len(common)})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, cfg, "plot1_slo_goodput")


def plot_latency_kde(cfg: Config, schedules: dict):
    """Plot 2: KDE of per-sample latency per runtime."""
    lam = float(cfg.arrivals["lambda"])
    n = schedules["plain"].n_requests
    arr = poisson_arrivals(n, lam, int(cfg.arrivals.seed))
    B = int(cfg.batching.seg2_batch)
    prop = schedules["proposed"][B]
    scheds = {"plain": schedules["plain"], "naive": schedules["naive"], f"proposed(seg2={B})": prop}
    common = metrics.common_completed(list(scheds.values()))

    lats = {name: metrics.latency_ms(s, arr, common) for name, s in scheds.items()}
    lo = min(l.min() for l in lats.values())
    hi = max(np.percentile(l, 99.5) for l in lats.values())
    grid = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"plain": "k", "naive": "0.45"}
    for name, l in lats.items():
        ax.plot(grid, _kde(l, grid), lw=2, label=name,
                color=colors.get(name, "C0"))
    ax.set_xlabel("Per-sample latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title(f"Latency distribution (KDE)  (λ={lam:g}, N_common={len(common)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, cfg, "plot2_latency_kde")


def plot_latency_cdf(cfg: Config, schedules: dict):
    """Plot 3: empirical CDF of per-sample latency per runtime."""
    lam = float(cfg.arrivals["lambda"])
    n = schedules["plain"].n_requests
    arr = poisson_arrivals(n, lam, int(cfg.arrivals.seed))
    B = int(cfg.batching.seg2_batch)
    prop = schedules["proposed"][B]
    scheds = {"plain": schedules["plain"], "naive": schedules["naive"], f"proposed(seg2={B})": prop}
    common = metrics.common_completed(list(scheds.values()))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"plain": "k", "naive": "0.45"}
    for name, s in scheds.items():
        l = np.sort(metrics.latency_ms(s, arr, common))
        y = np.arange(1, len(l) + 1) / len(l)
        ax.plot(l, y, lw=2, label=name, color=colors.get(name, "C0"))
    ax.set_xlabel("Per-sample latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Latency CDF  (λ={lam:g}, N_common={len(common)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, cfg, "plot3_latency_cdf")


def plot_load_latency(cfg: Config, schedules: dict):
    """Plot 4: Load (lambda) vs response time (mean + p99) per runtime."""
    n = schedules["plain"].n_requests
    B = int(cfg.batching.seg2_batch)
    prop = schedules["proposed"][B]
    scheds = {"plain": schedules["plain"], "naive": schedules["naive"], f"proposed(seg2={B})": prop}
    common = metrics.common_completed(list(scheds.values()))
    lams = lambda_grid(cfg)
    base_seed = int(cfg.arrivals.seed)

    means = {name: [] for name in scheds}
    p99s = {name: [] for name in scheds}
    for lam in lams:
        arr = poisson_arrivals(n, lam, base_seed)
        for name, s in scheds.items():
            m, p = metrics.response_stats(s, arr, common)
            means[name].append(m)
            p99s[name].append(p)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"plain": "k", "naive": "0.45"}
    for name in scheds:
        c = colors.get(name, "C0")
        ax.plot(lams, means[name], lw=2, color=c, label=f"{name} (mean)")
        ax.plot(lams, p99s[name], lw=1.5, ls="--", color=c, label=f"{name} (p99)")
    ax.set_xlabel("Arrival rate λ (req/s)")
    ax.set_ylabel("Response time (ms)")
    ax.set_title(f"Load vs Latency  (N_common={len(common)})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=len(scheds))
    return _save(fig, cfg, "plot4_load_latency")


from .runtimes import BREAKDOWN_KEYS, simulate_breakdown  # noqa: E402

_BD_LABELS = {
    "formation_wait":  "batch-formation wait",
    "gpu_wait":        "GPU-queue wait",
    "stage1_compute":  "stage-1 compute (seg1 / whole)",
    "seg2_queue_wait": "seg2 queue wait",
    "seg2_compute":    "seg2 compute",
}
_BD_COLORS = {
    "formation_wait":  "#4C72B0",   # blue  — arrivals too slow
    "gpu_wait":        "#C44E52",   # red   — GPU-bound
    "stage1_compute":  "#8C8C8C",   # grey
    "seg2_queue_wait": "#DD8452",   # orange
    "seg2_compute":    "#CCB974",   # tan
}


def _breakdown_curves(sched, lams, common, seed):
    """Return {component: mean-ms array over lams} for the common set."""
    n = sched.n_requests
    curves = {k: np.empty(len(lams)) for k in BREAKDOWN_KEYS}
    for j, lam in enumerate(lams):
        arr = poisson_arrivals(n, lam, seed)
        bd = simulate_breakdown(sched, arr)
        for k in BREAKDOWN_KEYS:
            curves[k][j] = bd[k][common].mean() * 1000.0   # ms
    return curves


def _stack_panel(ax, lams, curves, title):
    ys = [curves[k] for k in BREAKDOWN_KEYS]
    ax.stackplot(lams, *ys,
                 labels=[_BD_LABELS[k] for k in BREAKDOWN_KEYS],
                 colors=[_BD_COLORS[k] for k in BREAKDOWN_KEYS])
    ax.set_title(title)
    ax.set_xlabel("Arrival rate λ (req/s)")
    ax.grid(True, alpha=0.25)


def plot_latency_breakdown(cfg: Config, schedules: dict):
    """Plot 5/6: per-sample latency decomposed into wait/compute components vs λ.

    Reveals whether latency is batch-formation-dominated (blue) or GPU-bound
    (red), and how the seg2 queue wait (orange) grows with seg2_batch.
    """
    lams = lambda_grid(cfg)
    seed = int(cfg.arrivals.seed)
    prop = schedules["proposed"]
    B0 = int(cfg.batching.seg2_batch)
    common = metrics.common_completed([schedules["plain"], schedules["naive"], *prop.values()])

    # --- Figure 5: plain / naive / proposed(default B) ---
    panels = [("plain", schedules["plain"]),
              ("naive", schedules["naive"]),
              (f"proposed (seg2={B0})", prop[B0])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, sched) in zip(axes, panels):
        _stack_panel(ax, lams, _breakdown_curves(sched, lams, common, seed), name)
    axes[0].set_ylabel("Mean latency (ms)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Latency decomposition vs load  (N_common={len(common)})")
    _save(fig, cfg, "plot5_latency_breakdown")

    # --- Figure 6: proposed across the seg2_batch sweep ---
    Bs = sorted(prop.keys())
    ncol = len(Bs)
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 4.5), sharey=True)
    if ncol == 1:
        axes = [axes]
    for ax, B in zip(axes, Bs):
        _stack_panel(ax, lams, _breakdown_curves(prop[B], lams, common, seed), f"proposed seg2={B}")
    axes[0].set_ylabel("Mean latency (ms)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"proposed: latency decomposition vs load, per seg2_batch  (N_common={len(common)})")
    _save(fig, cfg, "plot6_breakdown_seg2sweep")


def plot_all(cfg: Config, schedules: dict):
    plot_slo_goodput(cfg, schedules)
    plot_latency_kde(cfg, schedules)
    plot_latency_cdf(cfg, schedules)
    plot_load_latency(cfg, schedules)
    plot_latency_breakdown(cfg, schedules)
