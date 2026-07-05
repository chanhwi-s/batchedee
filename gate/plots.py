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
        return gaussian_kde(data, bw_method=0.4)(grid)
    except Exception:
        # Silverman-bandwidth Gaussian KDE fallback (no scipy).
        n = len(data)
        bw = 1.06 * np.std(data) * n ** (-1 / 5)
        bw = max(bw, 1e-6)
        u = (grid[:, None] - data[None, :]) / bw
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return k.sum(axis=1) / (n * bw)


def _prop_label(sched, B: int) -> str:
    """Human label for a proposed schedule, aware of the seg2 flush mode."""
    if getattr(sched, "flush_mode", "fixed") == "all":
        return f"proposed (flush-all, thr={B})"
    return f"proposed (seg2={B})"


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
    Returns (arrivals_seconds, description-for-title, latency_origin).
    """
    lam = _runtime_lambda(cfg, runtime)
    if lam <= 0:
        return (np.zeros(n, dtype=float),
                "saturated; latency from seg1 input", "stage1_start")
    return poisson_arrivals(n, lam, int(cfg.arrivals.seed)), f"λ={lam:g} req/s", "arrival"


def _arrivals_per_runtime(cfg: Config, n: int):
    """(arr, desc, origin) per runtime + shared-λ flag + title description."""
    per = {r: _single_arrivals(cfg, n, r) for r in ("plain", "naive", "proposed")}
    descs = {d for _, d, _ in per.values()}
    shared = len(descs) == 1
    title_desc = next(iter(descs)) if shared else "per-runtime arrivals"
    return per, shared, title_desc


# --------------------------------------------------------------------------- #
def plot_slo_goodput(cfg: Config, schedules: dict):
    """Plot 1: SLO vs Goodput, one curve per seg2_batch + plain + naive."""
    n = schedules["plain"].n_requests
    per, shared, title_desc = _arrivals_per_runtime(cfg, n)
    slo = slo_grid_ms(cfg)

    prop = schedules["proposed"]  # {B: Schedule}
    all_scheds = [schedules["plain"], schedules["naive"], *prop.values()]
    common = metrics.common_completed(all_scheds)
    mode = cfg.get_path("metrics.goodput_mode", "mean_throughput")

    def lbl(base, r):
        return base if shared else f"{base}  [{per[r][1]}]"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    arr, _, origin = per["plain"]
    ax.plot(slo, metrics.goodput_vs_slo(schedules["plain"], arr, common, slo, mode, origin),
            "k--", lw=2, label=lbl("plain", "plain"))
    arr, _, origin = per["naive"]
    ax.plot(slo, metrics.goodput_vs_slo(schedules["naive"], arr, common, slo, mode, origin),
            color="0.45", ls=":", lw=2, label=lbl("naive", "naive"))
    arr, _, origin = per["proposed"]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(prop)))
    for c, (B, sched) in zip(cmap, sorted(prop.items())):
        ax.plot(slo, metrics.goodput_vs_slo(sched, arr, common, slo, mode, origin),
                color=c, lw=1.8, label=lbl(_prop_label(sched, B), "proposed"))

    ylabel = ("Goodput  (1/N · Σ 1/latency, samples/s)" if mode == "mean_throughput"
              else "Goodput (good samples / sec)")
    ax.set_xlabel("Latency SLO (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"SLO vs Goodput  ({title_desc}, N_common={len(common)})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, cfg, "plot1_slo_goodput")


def _per_runtime_latencies(cfg: Config, schedules: dict):
    """Shared prep for the KDE/CDF plots: per-runtime latencies (each runtime
    replayed against its own arrival trace) + labels/axis/title strings."""
    n = schedules["plain"].n_requests
    per, shared, title_desc = _arrivals_per_runtime(cfg, n)
    B = int(cfg.batching.seg2_batch)
    prop = schedules["proposed"][B]
    entries = [("plain", "plain", schedules["plain"], "k"),
               ("naive", "naive", schedules["naive"], "0.45"),
               (_prop_label(prop, B), "proposed", prop, "C0")]
    common = metrics.common_completed([e[2] for e in entries])

    data = []          # (label, latency_ms array, color)
    for name, r, s, color in entries:
        arr, d, origin = per[r]
        label = name if shared else f"{name}  [{d}]"
        data.append((label, metrics.latency_ms(s, arr, common, origin), color))
    origins = {o for _, _, o in per.values()}
    xlabel = ("Per-sample service latency (ms, from seg1 input)"
              if origins == {"stage1_start"} else "Per-sample latency (ms)")
    return data, xlabel, title_desc, common


def plot_latency_kde(cfg: Config, schedules: dict):
    """Plot 2: KDE of per-sample latency per runtime."""
    data, xlabel, title_desc, common = _per_runtime_latencies(cfg, schedules)
    lo = min(l.min() for _, l, _ in data)
    hi = max(np.percentile(l, 99.5) for _, l, _ in data)
    grid = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, l, color in data:
        ax.plot(grid, _kde(l, grid), lw=2, label=label, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"Latency distribution (KDE)  ({title_desc}, N_common={len(common)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, cfg, "plot2_latency_kde")


def plot_latency_cdf(cfg: Config, schedules: dict):
    """Plot 3: empirical CDF of per-sample latency per runtime."""
    data, xlabel, title_desc, common = _per_runtime_latencies(cfg, schedules)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, l, color in data:
        l = np.sort(l)
        y = np.arange(1, len(l) + 1) / len(l)
        ax.plot(l, y, lw=2, label=label, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(f"Latency CDF  ({title_desc}, N_common={len(common)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, cfg, "plot3_latency_cdf")


def plot_load_latency(cfg: Config, schedules: dict):
    """Plot 4: Load (lambda) vs response time (mean + p99) per runtime.

    Also detects each runtime's divergence point — the λ at the minimum of the
    response-time curve (falling before it: formation-wait dominated; rising
    beyond it: queueing dominated) — prints it, and returns it as a dict.
    """
    B = int(cfg.batching.seg2_batch)
    prop = schedules["proposed"][B]
    scheds = {"plain": schedules["plain"], "naive": schedules["naive"], _prop_label(prop, B): prop}
    common = metrics.common_completed(list(scheds.values()))
    lams = lambda_grid(cfg)
    base_seed = int(cfg.arrivals.seed)

    means, p99s, divergence = {}, {}, {}
    for name, s in scheds.items():
        means[name], p99s[name] = metrics.load_latency_curves(s, lams, common, base_seed)
        dm = metrics.divergence_lambda(lams, means[name])
        dp = metrics.divergence_lambda(lams, p99s[name])
        divergence[name] = {"mean": dm, "p99": dp}

        def _fmt(d):
            if d is None:
                return f"not reached within sweep (still falling at λ={lams[-1]:g})"
            return (f"≤{lams[0]:g} (already rising at sweep start)" if d == lams[0]
                    else f"{d:g}")
        print(f"[plot4] {name}: divergence λ = {_fmt(dm)} (mean) | {_fmt(dp)} (p99)  [req/s]")

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
    _save(fig, cfg, "plot4_load_latency")
    return divergence


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
              (_prop_label(prop[B0], B0), prop[B0])]
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
        _stack_panel(ax, lams, _breakdown_curves(prop[B], lams, common, seed), _prop_label(prop[B], B))
    axes[0].set_ylabel("Mean latency (ms)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"proposed: latency decomposition vs load, per seg2_batch  (N_common={len(common)})")
    _save(fig, cfg, "plot6_breakdown_seg2sweep")


# --------------------------------------------------------------------------- #
# Plot 7: GPU-stream timeline (arrival wait / seg1 / seg2 as one contiguous bar)
# --------------------------------------------------------------------------- #
_TL_COLORS = {  # same hues as the breakdown plots: blue=wait, grey=stage-1, orange=seg2
    "wait": "#4C72B0",
    "seg1": "#8C8C8C",
    "seg2": "#DD8452",
}
_TL_LABELS = {
    "wait": "arrival wait (GPU idle)",
    "seg1": "seg1 / whole-model inference",
    "seg2": "seg2 inference",
}


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

    One horizontal bar per runtime; x = simulation time. Colors mark whether the
    GPU was idle (waiting for arrivals / batch formation), running seg1 (or the
    whole model for plain), or running seg2. Works for both seg2 flush modes.
    """
    from matplotlib.patches import Patch

    n = schedules["plain"].n_requests
    per, shared, title_desc = _arrivals_per_runtime(cfg, n)
    B = int(cfg.batching.seg2_batch)
    prop = schedules["proposed"][B]
    rows = [("plain", "plain", schedules["plain"]),
            ("naive", "naive", schedules["naive"]),
            (_prop_label(prop, B), "proposed", prop)]

    fig, ax = plt.subplots(figsize=(13, 3.8))
    height = 0.6
    labels = []
    for y, (name, r, s) in enumerate(rows):
        arr, d, _o = per[r]
        labels.append(name if shared else f"{name}\n[{d}]")
        per_kind: dict[str, list] = {}
        for a, b, kind in _op_intervals(s, arr):
            per_kind.setdefault(kind, []).append((a * 1000.0, (b - a) * 1000.0))
        for kind, xranges in per_kind.items():
            ax.broken_barh(xranges, (y - height / 2, height),
                           facecolors=_TL_COLORS[kind], linewidth=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Simulation time (ms)")
    xlim = cfg.plots.get("timeline_xlim_ms", None)
    if xlim:
        ax.set_xlim(0, float(xlim))
    else:
        ax.set_xlim(left=0)
    ax.set_title(f"GPU execution timeline  ({title_desc})")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(handles=[Patch(facecolor=_TL_COLORS[k], label=_TL_LABELS[k])
                       for k in ("wait", "seg1", "seg2")],
              fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
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
    prop = schedules["proposed"][B]
    rows = [("plain", schedules["plain"]),
            ("naive", schedules["naive"]),
            (_prop_label(prop, B), prop)]
    stats = [(name, op_stats(s)) for name, s in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("mean_ms", "Mean execution time per op (ms)", "%.2f"),
              ("count", "Execution count", "%d")]
    for ax, (field, ylab, fmt) in zip(axes, panels):
        s1 = [st.get("seg1", st.get("whole", {})).get(field, 0) for _, st in stats]
        s2 = [st.get("seg2", {}).get(field, 0) for _, st in stats]
        b1 = ax.bar(x - w / 2, s1, w, color=_TL_COLORS["seg1"], label="seg1 / whole")
        b2 = ax.bar(x + w / 2, s2, w, color=_TL_COLORS["seg2"], label="seg2")
        ax.bar_label(b1, labels=[fmt % v if v else "" for v in s1], fontsize=8)
        ax.bar_label(b2, labels=[fmt % v if v else "" for v in s2], fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _ in rows], fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-runtime execution stats (measured service times)")
    return _save(fig, cfg, "plot8_exec_stats")


def plot_all(cfg: Config, schedules: dict):
    plot_slo_goodput(cfg, schedules)
    plot_latency_kde(cfg, schedules)
    plot_latency_cdf(cfg, schedules)
    divergence = plot_load_latency(cfg, schedules)
    plot_latency_breakdown(cfg, schedules)
    plot_timeline(cfg, schedules)
    plot_exec_stats(cfg, schedules)
    return divergence
