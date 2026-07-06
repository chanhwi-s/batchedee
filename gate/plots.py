"""The analysis figures (plot1–plot8), camera-ready.

Each figure is saved as PDF (vector, primary) and PNG (300 dpi). ALL styling
(fonts, sizes, colors, line styles) comes from gate.plot_style; this module
only computes the data and lays out the figures.
"""
from __future__ import annotations

import os

import numpy as np

from . import metrics
from . import plot_style as ps
from .arrivals import poisson_arrivals
from .plot_style import (COMPONENT_COLORS, COMPONENT_LABELS, FIG_DOUBLE,
                         FIG_SINGLE, IDLE_COLOR, RUNTIME_COLORS, RUNTIME_LABELS,
                         RUNTIME_ORDER, RUNTIME_STYLES, STAGE1_SWATCH,
                         STAGE2_SWATCH, b2_label, lighten, proposed_shades)
from .util import Config, lambda_grid, slo_grid_ms

ps.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def _save(fig, cfg: Config, name: str):
    d = cfg.paths["plots_dir"]
    os.makedirs(d, exist_ok=True)
    png = os.path.join(d, f"{name}.png")
    pdf = os.path.join(d, f"{name}.pdf")
    # figures are sized at final print size — save without bbox cropping
    fig.savefig(png, dpi=int(cfg.plots.get("dpi", 300)))
    fig.savefig(pdf)
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
        return gaussian_kde(data, bw_method=0.4)(grid)
    except Exception:
        # Silverman-bandwidth Gaussian KDE fallback (no scipy).
        n = len(data)
        bw = 1.06 * np.std(data) * n ** (-1 / 5)
        bw = max(bw, 1e-6)
        u = (grid[:, None] - data[None, :]) / bw
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return k.sum(axis=1) / (n * bw)


def _runtime_lambda(cfg: Config, runtime: str) -> float:
    """arrivals.lambda: scalar (shared by all runtimes) or a per-runtime
    mapping {plain: rate, naive: rate, proposed: rate} — e.g. each runtime's
    sustainable upper bound read off the load-vs-latency plot."""
    lam = cfg.arrivals["lambda"]
    if isinstance(lam, dict):
        if runtime not in lam:
            raise ValueError(f"arrivals.lambda mapping needs key {runtime!r} "
                             f"(has {sorted(lam)})")
        return float(lam[runtime])
    return float(lam)


def _single_arrivals(cfg: Config, n: int, runtime: str):
    """Arrival vector for every single-λ plot (all figures except the λ sweeps).

    lambda == 0 -> NO Poisson modeling: all n requests are queued at t=0
    (saturated backlog); latency is measured from each sample's seg1 input
    (service latency) since waiting behind the backlog is a setup artifact.
    lambda > 0  -> Poisson trace at that rate; latency = response time.
    The λ-sweep plots (load vs latency, breakdown) always use lambda_sweep.
    Returns (arrivals_seconds, description, latency_origin).
    """
    lam = _runtime_lambda(cfg, runtime)
    if lam <= 0:
        return (np.zeros(n, dtype=float),
                "saturated; latency from seg1 input", "stage1_start")
    return poisson_arrivals(n, lam, int(cfg.arrivals.seed)), f"λ={lam:g} req/s", "arrival"


def _arrivals_per_runtime(cfg: Config, n: int):
    """{runtime: (arr, desc, origin)} for the three runtimes."""
    return {r: _single_arrivals(cfg, n, r) for r in RUNTIME_ORDER}


# --------------------------------------------------------------------------- #
# Plot 1: SLO vs Goodput
# --------------------------------------------------------------------------- #
def plot_slo_goodput(cfg: Config, schedules: dict):
    """Plot 1: SLO vs Goodput, one curve per seg2_batch + plain + naive."""
    n = schedules["plain"].n_requests
    per = _arrivals_per_runtime(cfg, n)
    slo = slo_grid_ms(cfg)

    prop = schedules["proposed"]  # {B: Schedule}
    all_scheds = [schedules["plain"], schedules["naive"], *prop.values()]
    common = metrics.common_completed(all_scheds)
    mode = cfg.get_path("metrics.goodput_mode", "mean_throughput")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r in ("plain", "naive"):
        arr, _, origin = per[r]
        ax.plot(slo, metrics.goodput_vs_slo(schedules[r], arr, common, slo, mode, origin),
                color=RUNTIME_COLORS[r], label=RUNTIME_LABELS[r],
                markevery=3, **RUNTIME_STYLES[r])
    arr, _, origin = per["proposed"]
    shades = proposed_shades(len(prop))
    for c, (B, sched) in zip(shades, sorted(prop.items())):
        ax.plot(slo, metrics.goodput_vs_slo(sched, arr, common, slo, mode, origin),
                color=c, linestyle="-", label=b2_label(B))

    ax.set_xlabel("SLO (ms)")
    ax.set_ylabel("Goodput (samples/s)")
    ax.set_title("SLO vs Goodput")
    ax.legend(ncol=2, loc="lower right")
    return _save(fig, cfg, "plot1_slo_goodput")


# --------------------------------------------------------------------------- #
# Plots 2 & 3: latency distribution / CDF
# --------------------------------------------------------------------------- #
def _per_runtime_latencies(cfg: Config, schedules: dict):
    """[(runtime, latency_ms array)] — each runtime replayed against its own
    arrival trace — restricted to the common completed set."""
    n = schedules["plain"].n_requests
    per = _arrivals_per_runtime(cfg, n)
    B = int(cfg.batching.seg2_batch)
    entries = [("plain", schedules["plain"]),
               ("naive", schedules["naive"]),
               ("proposed", schedules["proposed"][B])]
    common = metrics.common_completed([s for _, s in entries])

    data = []
    for r, s in entries:
        arr, _, origin = per[r]
        data.append((r, metrics.latency_ms(s, arr, common, origin)))
    return data


def plot_latency_kde(cfg: Config, schedules: dict):
    """Plot 2: KDE of per-sample latency per runtime."""
    data = _per_runtime_latencies(cfg, schedules)
    lo = min(l.min() for _, l in data)
    hi = max(np.percentile(l, 99.5) for _, l in data)
    grid = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, l in data:
        ax.plot(grid, _kde(l, grid), color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Latency Distribution")
    ax.legend(loc="upper right")
    return _save(fig, cfg, "plot2_latency_kde")


def plot_latency_cdf(cfg: Config, schedules: dict):
    """Plot 3: empirical CDF of per-sample latency per runtime."""
    data = _per_runtime_latencies(cfg, schedules)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, l in data:
        l = np.sort(l)
        y = np.arange(1, len(l) + 1) / len(l)
        ax.plot(l, y, color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF")
    ax.legend(loc="lower right")
    return _save(fig, cfg, "plot3_latency_cdf")


# --------------------------------------------------------------------------- #
# Plot 4: load vs latency (+ divergence detection)
# --------------------------------------------------------------------------- #
def plot_load_latency(cfg: Config, schedules: dict):
    """Plot 4: Load (lambda) vs response time (mean + p99) per runtime.

    Also reports each runtime's divergence point — its service capacity
    (saturated throughput; arrival rates above it make the queue grow without
    bound) — plus the knee (latency minimum) of the sweep curve for reference,
    and returns the capacity-based divergence λ as a dict.
    """
    B = int(cfg.batching.seg2_batch)
    entries = [("plain", schedules["plain"]),
               ("naive", schedules["naive"]),
               ("proposed", schedules["proposed"][B])]
    common = metrics.common_completed([s for _, s in entries])
    lams = lambda_grid(cfg)
    base_seed = int(cfg.arrivals.seed)

    means, p99s, divergence = {}, {}, {}
    for r, s in entries:
        means[r], p99s[r] = metrics.load_latency_curves(s, lams, common, base_seed)
        divergence[r] = metrics.capacity_lambda(s, common)
        knee = metrics.knee_lambda(lams, means[r])
        knee_s = "-" if knee is None else f"{knee:g}"
        print(f"[plot4] {r}: divergence λ (capacity) = {divergence[r]:.1f} req/s"
              f" | knee (latency minimum) λ = {knee_s}")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, _ in entries:
        c = RUNTIME_COLORS[r]
        m = RUNTIME_STYLES[r]["marker"]
        ax.plot(lams, means[r], color=c, linestyle="-", marker=m, markevery=4)
        ax.plot(lams, p99s[r], color=c, linestyle="--", marker=m, markevery=4,
                markersize=2.6, linewidth=1.0)
    handles = ([Line2D([], [], color=RUNTIME_COLORS[r], linestyle="-",
                       marker=RUNTIME_STYLES[r]["marker"], label=RUNTIME_LABELS[r])
                for r in RUNTIME_ORDER]
               + [Line2D([], [], color="0.3", linestyle="-", label="mean"),
                  Line2D([], [], color="0.3", linestyle="--", label="p99")])
    ax.set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Load vs Latency")
    ax.legend(handles=handles, ncol=2, loc="upper left")
    _save(fig, cfg, "plot4_load_latency")
    return divergence


# --------------------------------------------------------------------------- #
# Plots 5 & 6: latency decomposition
# --------------------------------------------------------------------------- #
from .runtimes import BREAKDOWN_KEYS, simulate_breakdown  # noqa: E402


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
    ax.stackplot(lams, *ys, colors=[COMPONENT_COLORS[k] for k in BREAKDOWN_KEYS],
                 linewidth=0)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.margins(x=0)


def _component_legend(fig):
    handles = [Patch(facecolor=COMPONENT_COLORS[k], label=COMPONENT_LABELS[k])
               for k in BREAKDOWN_KEYS]
    fig.legend(handles=handles, ncol=5, loc="outside lower center")


def plot_latency_breakdown(cfg: Config, schedules: dict):
    """Plot 5/6: per-sample latency decomposed into wait/compute components vs λ."""
    lams = lambda_grid(cfg)
    seed = int(cfg.arrivals.seed)
    prop = schedules["proposed"]
    B0 = int(cfg.batching.seg2_batch)
    common = metrics.common_completed([schedules["plain"], schedules["naive"], *prop.values()])

    # --- Figure 5: plain / naive / proposed(default B) ---
    panels = [("plain", schedules["plain"]),
              ("naive", schedules["naive"]),
              ("proposed", prop[B0])]
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)
    for ax, (r, sched) in zip(axes, panels):
        _stack_panel(ax, lams, _breakdown_curves(sched, lams, common, seed),
                     RUNTIME_LABELS[r])
    axes[0].set_ylabel("Latency (ms)")
    axes[len(axes) // 2].set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    fig.suptitle("Latency Decomposition")
    _component_legend(fig)
    _save(fig, cfg, "plot5_latency_breakdown")

    # --- Figure 6: proposed across the seg2_batch sweep ---
    Bs = sorted(prop.keys())
    fig, axes = plt.subplots(1, len(Bs), figsize=FIG_DOUBLE, sharey=True)
    if len(Bs) == 1:
        axes = [axes]
    for ax, B in zip(axes, Bs):
        _stack_panel(ax, lams, _breakdown_curves(prop[B], lams, common, seed),
                     b2_label(B))
        ax.xaxis.set_major_locator(MaxNLocator(2))
    axes[0].set_ylabel("Latency (ms)")
    axes[len(axes) // 2].set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    fig.suptitle("Latency Decomposition")
    _component_legend(fig)
    _save(fig, cfg, "plot6_breakdown_seg2sweep")


# --------------------------------------------------------------------------- #
# Plot 7: GPU-stream timeline
# --------------------------------------------------------------------------- #
def _op_intervals(sched, arrivals: np.ndarray):
    """Replay the single-stream simulation and return [(start_s, end_s, kind)].

    kind ∈ {'wait', 'seg1', 'seg2'}; 'whole' (plain) maps to 'seg1'. Gaps where
    the GPU idles waiting for a batch to fill become 'wait' segments, so the
    concatenation is one contiguous bar from t=0 to the last completion.
    """
    segs = []
    gpu_free = 0.0
    for op in sched.ops:
        if op.gate_on_arrival and len(op.members):
            start = max(gpu_free, float(arrivals[op.members].max()))
        else:
            start = gpu_free
        if start > gpu_free:
            segs.append((gpu_free, start, "wait"))
        kind = "seg1" if op.kind in ("seg1", "whole") else "seg2"
        segs.append((start, start + op.duration, kind))
        gpu_free = start + op.duration
    return segs


def plot_timeline(cfg: Config, schedules: dict):
    """Plot 7: execution timeline per runtime on the simulation clock.

    One horizontal bar per runtime; x = simulation time. Each row uses its
    runtime's base color for stage-1 ops and a lighter tint for stage-2 ops;
    idle (arrival-wait) time is light gray. Works for both seg2 flush modes.
    """
    n = schedules["plain"].n_requests
    per = _arrivals_per_runtime(cfg, n)
    B = int(cfg.batching.seg2_batch)
    rows = [("plain", schedules["plain"]),
            ("naive", schedules["naive"]),
            ("proposed", schedules["proposed"][B])]

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    height = 0.6
    for y, (r, s) in enumerate(rows):
        arr = per[r][0]
        colors = {"seg1": RUNTIME_COLORS[r], "seg2": lighten(RUNTIME_COLORS[r]),
                  "wait": IDLE_COLOR}
        per_kind: dict[str, list] = {}
        for a, b, kind in _op_intervals(s, arr):
            per_kind.setdefault(kind, []).append((a * 1000.0, (b - a) * 1000.0))
        for kind, xranges in per_kind.items():
            ax.broken_barh(xranges, (y - height / 2, height),
                           facecolors=colors[kind], linewidth=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([RUNTIME_LABELS[r] for r, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    xlim = cfg.plots.get("timeline_xlim_ms", None)
    if xlim:
        ax.set_xlim(0, float(xlim))
    else:
        ax.set_xlim(left=0)
    ax.set_title("GPU Execution Timeline")
    ax.grid(False)
    handles = [Patch(facecolor=STAGE1_SWATCH, label="Stage 1"),
               Patch(facecolor=STAGE2_SWATCH, label="Stage 2"),
               Patch(facecolor=IDLE_COLOR, label="Idle")]
    fig.legend(handles=handles, ncol=3, loc="outside lower center")
    return _save(fig, cfg, "plot7_timeline")


# --------------------------------------------------------------------------- #
# Plot 8: per-runtime execution stats (mean service time + op count)
# --------------------------------------------------------------------------- #
def plot_exec_stats(cfg: Config, schedules: dict):
    """Plot 8: grouped bars — mean execution time per op and op count,
    per runtime and per stage (seg1/whole vs seg2).

    Stats are recomputed from the schedules (works with any pkl); `run.py`
    also stores the same numbers under schedules['op_stats'].
    """
    from .runtimes import op_stats

    B = int(cfg.batching.seg2_batch)
    entries = [("plain", schedules["plain"]),
               ("naive", schedules["naive"]),
               ("proposed", schedules["proposed"][B])]
    stats = [(r, op_stats(s)) for r, s in entries]

    x = np.arange(len(entries))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=FIG_SINGLE)
    panels = [("mean_ms", "Time per op (ms)", "%.1f"),
              ("count", "Op count", "%d")]
    for ax, (field, ylab, fmt) in zip(axes, panels):
        s1 = [st.get("seg1", st.get("whole", {})).get(field, 0) for _, st in stats]
        s2 = [st.get("seg2", {}).get(field, 0) for _, st in stats]
        c1 = [RUNTIME_COLORS[r] for r, _ in stats]
        c2 = [lighten(RUNTIME_COLORS[r]) for r, _ in stats]
        b1 = ax.bar(x - w / 2, s1, w, color=c1)
        b2 = ax.bar(x + w / 2, s2, w, color=c2)
        ax.bar_label(b1, labels=[fmt % v if v else "" for v in s1], fontsize=6, padding=1)
        ax.bar_label(b2, labels=[fmt % v if v else "" for v in s2], fontsize=6, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels([RUNTIME_LABELS[r] for r, _ in stats])
        ax.set_ylabel(ylab)
        ax.margins(y=0.15)
    handles = [Patch(facecolor=STAGE1_SWATCH, label="Stage 1"),
               Patch(facecolor=STAGE2_SWATCH, label="Stage 2")]
    fig.legend(handles=handles, ncol=2, loc="outside lower center")
    fig.suptitle("Execution Stats")
    return _save(fig, cfg, "plot8_exec_stats")


# --------------------------------------------------------------------------- #
# Plot 9: naive's dynamic seg2 batch-size distribution
# --------------------------------------------------------------------------- #
def plot_naive_seg2_sizes(cfg: Config, schedules: dict):
    """Plot 9: histogram of naive's seg2 batch sizes.

    naive forwards each seg1 batch's non-exiting samples to seg2 immediately,
    so its seg2 batch size = per-batch non-exit count — small and irregular.
    Prints summary stats and draws the integer histogram.
    """
    sizes = np.array([len(op.members) for op in schedules["naive"].ops
                      if op.kind == "seg2"], dtype=np.int64)
    if len(sizes) == 0:
        print("[plot9] naive has no seg2 ops; skipped")
        return None
    print(f"[plot9] naive seg2 sizes: n={len(sizes)}, mean={sizes.mean():.2f}, "
          f"median={np.median(sizes):g}, min={sizes.min()}, max={sizes.max()}")

    bins = np.arange(sizes.min(), sizes.max() + 2) - 0.5   # one bin per integer
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.hist(sizes, bins=bins, color=RUNTIME_COLORS["naive"],
            edgecolor="white", linewidth=0.4)
    ax.axvline(float(sizes.mean()), color="0.25", linestyle="--", linewidth=1.0)
    ax.annotate(f"mean {sizes.mean():.1f}", xy=(float(sizes.mean()), 1.0),
                xycoords=("data", "axes fraction"), xytext=(3, -10),
                textcoords="offset points", fontsize=7, color="0.25")
    ax.set_xlabel("Stage-2 batch size (samples)")
    ax.set_ylabel("Occurrences")
    ax.set_title("Naive Stage-2 Batch Sizes")
    return _save(fig, cfg, "plot9_naive_seg2_sizes")


def plot_all(cfg: Config, schedules: dict):
    plot_slo_goodput(cfg, schedules)
    plot_latency_kde(cfg, schedules)
    plot_latency_cdf(cfg, schedules)
    divergence = plot_load_latency(cfg, schedules)
    plot_latency_breakdown(cfg, schedules)
    plot_timeline(cfg, schedules)
    plot_exec_stats(cfg, schedules)
    plot_naive_seg2_sizes(cfg, schedules)
    return divergence
