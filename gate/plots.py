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


def _kde(data: np.ndarray, grid: np.ndarray, bw=None) -> np.ndarray:
    """Gaussian KDE. `bw` = bandwidth as a fraction of the data std (scipy's
    numeric bw_method semantics); None -> Scott's rule. Larger = smoother."""
    data = data[np.isfinite(data)]
    if len(data) < 2 or np.std(data) == 0:
        return np.zeros_like(grid)
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(data, bw_method=bw)(grid)
    except Exception:
        # Gaussian KDE fallback (no scipy); Silverman factor when bw is None.
        n = len(data)
        factor = float(bw) if bw is not None else 1.06 * n ** (-1 / 5)
        bw_abs = max(factor * np.std(data), 1e-6)
        u = (grid[:, None] - data[None, :]) / bw_abs
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return k.sum(axis=1) / (n * bw_abs)


def _capacity_step_lambda(cfg: Config, sched, common) -> float:
    """`sched`'s own capacity minus one sweep step, snapped to the sweep grid
    — the last stable load (same convention as plot1a/1b/Table B's λ1/λ3).
    Capacity itself is a near-critical (ρ→1) boundary; a finite-N replay
    right at it can already look unstable, so every single-λ figure backs
    off by one grid step to stay clearly inside the stable regime."""
    step = float(cfg.arrivals["lambda_sweep"]["step"])
    cap = metrics.capacity_lambda(sched, common)
    lams = lambda_grid(cfg)
    return float(lams[int(np.argmin(np.abs(lams - (cap - step))))])


def _capacity_arrivals(cfg: Config, n: int, sched, common):
    """(arrivals, desc, origin) for `sched` at its own capacity−step λ."""
    lam = _capacity_step_lambda(cfg, sched, common)
    return (poisson_arrivals(n, lam, int(cfg.arrivals.seed)),
            f"λ={lam:g} req/s (capacity−step)", "arrival")


def _arrivals_per_runtime(cfg: Config, schedules: dict, common):
    """{runtime: (arr, desc, origin)} for plain/naive/proposed@default bs2,
    each at its OWN capacity−step λ (last stable load)."""
    n = schedules["plain"].n_requests
    B = int(cfg.batching.seg2_batch)
    entries = {"plain": schedules["plain"], "naive": schedules["naive"],
               "proposed": schedules["proposed"][B]}
    return {r: _capacity_arrivals(cfg, n, s, common) for r, s in entries.items()}


# --------------------------------------------------------------------------- #
# Plot 1a/1b: Goodput under Latency SLOs — proposed (b2 sweep) vs ONE baseline per figure
# --------------------------------------------------------------------------- #
def _slo_goodput_pair(cfg: Config, schedules: dict, baseline: str, name: str):
    """One SLO-vs-goodput figure: `baseline` + the proposed b2 sweep, BOTH
    replayed on the same arrival trace at the figure's λ.

    λ selection (plots.slo_goodput_lambda.<baseline>):
      missing/"auto" -> derived from measured capacities, matching the e2e
        Table B rule: each figure sits at its OWN baseline's last stable
        load (D_baseline − step) — plain figure at D_plain − step, naive
        figure at D_naive − step. Symmetric across both figures.
      number > 0    -> manual override at that rate.
      0             -> saturated (all arrivals at t=0).
    """
    n = schedules["plain"].n_requests
    prop = schedules["proposed"]  # {B: Schedule}
    # common set over ALL runtimes so both figures share the same sample base
    all_scheds = [schedules["plain"], schedules["naive"], *prop.values()]
    common = metrics.common_completed(all_scheds)
    mode = cfg.get_path("metrics.goodput_mode", "mean_throughput")

    raw = cfg.get_path(f"plots.slo_goodput_lambda.{baseline}", None)
    if raw is None or raw == "auto":
        lams = lambda_grid(cfg)
        step = float(cfg.arrivals["lambda_sweep"]["step"])
        cap = metrics.capacity_lambda(schedules[baseline], common)
        lam = float(lams[int(np.argmin(np.abs(lams - (cap - step))))])
        src = f"auto: {baseline} capacity {cap:.1f} − step"
    else:
        lam = float(raw)
        src = "manual override"
    if lam <= 0:
        arr, origin, desc = np.zeros(n, dtype=float), "stage1_start", "saturated"
    else:
        arr, origin, desc = (poisson_arrivals(n, lam, int(cfg.arrivals.seed)),
                             "arrival", f"λ={lam:g} req/s")
    print(f"[{name}] {RUNTIME_LABELS[baseline]} vs Proposed at {desc} ({src})")
    slo = slo_grid_ms(cfg)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(slo, metrics.goodput_vs_slo(schedules[baseline], arr, common, slo, mode, origin),
            color=RUNTIME_COLORS[baseline], label=RUNTIME_LABELS[baseline],
            linestyle=RUNTIME_STYLES[baseline]["linestyle"])
    shades = proposed_shades(len(prop))
    for c, (B, sched) in zip(shades, sorted(prop.items())):
        ax.plot(slo, metrics.goodput_vs_slo(sched, arr, common, slo, mode, origin),
                color=c, linestyle="-", label=b2_label(B))

    ax.set_xlabel("SLO (ms)")
    ax.set_ylabel("Goodput (samples/s)")
    ax.set_title("Goodput under Latency SLOs")
    ax.legend(ncol=2, loc="lower right")
    return _save(fig, cfg, name)


def plot_slo_goodput(cfg: Config, schedules: dict):
    """Plot 1a: Plain vs Proposed; Plot 1b: Naive vs Proposed — each at its
    own configured λ (identical trace within a figure)."""
    a = _slo_goodput_pair(cfg, schedules, "plain", "plot1a_slo_goodput_vs_plain")
    b = _slo_goodput_pair(cfg, schedules, "naive", "plot1b_slo_goodput_vs_naive")
    return a, b


# --------------------------------------------------------------------------- #
# Plots 2 & 3: latency distribution / CDF
# --------------------------------------------------------------------------- #
def _per_runtime_latencies(cfg: Config, schedules: dict):
    """[(runtime, latency_ms array)] — each runtime replayed at its own
    capacity−step λ (last stable load) — restricted to the common completed
    set."""
    B = int(cfg.batching.seg2_batch)
    entries = [("plain", schedules["plain"]),
               ("naive", schedules["naive"]),
               ("proposed", schedules["proposed"][B])]
    common = metrics.common_completed([s for _, s in entries])
    per = _arrivals_per_runtime(cfg, schedules, common)

    data = []
    for r, s in entries:
        arr, desc, origin = per[r]
        print(f"[plot2/3] {RUNTIME_LABELS[r]}: {desc}")
        data.append((r, metrics.latency_ms(s, arr, common, origin)))
    return data


def _kde_hi(cfg: Config, lats: list, name: str) -> float:
    """Upper x-bound for the KDE plots. `plots.kde_xlim_ms` (fixed cutoff)
    takes precedence; otherwise `plots.kde_clip_percentile` of the plotted
    latencies (99.5 = legacy behavior). Lower either to keep a long tail from
    squeezing the bulk of the distribution."""
    xlim = cfg.get_path("plots.kde_xlim_ms", None)
    if xlim:
        hi = float(xlim)
        print(f"[{name}] x-range clipped at fixed {hi:g} ms (plots.kde_xlim_ms)")
        return hi
    pct = float(cfg.get_path("plots.kde_clip_percentile", 99.5))
    hi = max(float(np.percentile(l, pct)) for l in lats)
    print(f"[{name}] x-range 0–{hi:.1f} ms (p{pct:g} of plotted latencies)")
    return hi


def plot_latency_kde(cfg: Config, schedules: dict):
    """Plot 2: KDE of per-sample latency per runtime."""
    data = _per_runtime_latencies(cfg, schedules)
    bw = cfg.get_path("plots.kde_bandwidth", 0.4)
    lo = min(l.min() for _, l in data)
    hi = _kde_hi(cfg, [l for _, l in data], "plot2")
    grid = np.linspace(lo, hi, int(cfg.get_path("plots.kde_grid_points", 400)))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, l in data:
        ax.plot(grid, _kde(l, grid, bw), color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Latency Distribution")
    ax.legend(loc="upper right")
    return _save(fig, cfg, "plot2_latency_kde")


def _sweep_latencies(cfg: Config, schedules: dict, anchor: str = "capacity",
                     tag: str = ""):
    """Per-runtime latencies for the b2-sweep panel figures: plain/naive as
    fixed references + proposed per seg2_batch, all on the shared common set.
    EVERY configuration (plain, naive, each bs2) is replayed at its OWN
    operating point — never a value shared across configs:

    anchor="knee"     -> the λ minimizing mean response time (Table D).
    anchor="capacity" -> capacity−step, the last stable load (default; same
                         convention as plot1a/1b/Table B).
    Returns (lat_plain, lat_naive, {B: lat_proposed}, Bs).
    """
    n = schedules["plain"].n_requests
    prop = schedules["proposed"]
    Bs = sorted(prop.keys())
    common = metrics.common_completed(
        [schedules["plain"], schedules["naive"], *prop.values()])
    seed = int(cfg.arrivals.seed)
    lams = lambda_grid(cfg)

    if anchor == "knee":
        def pick(s, label):
            k, mean, _p99, edge = metrics.knee_stats(s, lams, common, seed)
            note = "  [WARNING: knee on sweep edge]" if edge else ""
            print(f"[{tag}] {label}: knee λ = {k:g} req/s "
                  f"(mean {mean:.2f} ms){note}")
            return metrics.latency_ms(s, poisson_arrivals(n, k, seed),
                                      common, "arrival")
    elif anchor == "capacity":
        def pick(s, label):
            lam = _capacity_step_lambda(cfg, s, common)
            print(f"[{tag}] {label}: capacity−step λ = {lam:g} req/s")
            return metrics.latency_ms(s, poisson_arrivals(n, lam, seed),
                                      common, "arrival")
    else:
        raise ValueError(f"anchor must be 'knee' or 'capacity', got {anchor!r}")

    lat_plain = pick(schedules["plain"], "Plain")
    lat_naive = pick(schedules["naive"], "Naive")
    lat_prop = {B: pick(prop[B], b2_label(B)) for B in Bs}
    return lat_plain, lat_naive, lat_prop, Bs


def _runtime_legend(fig):
    handles = [Line2D([], [], color=RUNTIME_COLORS[r],
                      linestyle=RUNTIME_STYLES[r]["linestyle"],
                      label=RUNTIME_LABELS[r]) for r in RUNTIME_ORDER]
    fig.legend(handles=handles, ncol=3, loc="outside lower center")


def plot_latency_kde_sweep(cfg: Config, schedules: dict):
    """Plot 2b: latency KDE per runtime, one panel per seg2_batch in the sweep.

    Plain/naive curves repeat in every panel as fixed references; the proposed
    curve changes with b2. Panels share both axes so shapes are comparable.
    Each configuration is replayed at its OWN knee λ (minimum-mean-latency
    operating point; logged to stdout — state these λ values in the caption).
    """
    bw = cfg.get_path("plots.kde_bandwidth", 0.4)
    pts = int(cfg.get_path("plots.kde_grid_points", 400))
    lat_plain, lat_naive, lat_prop, Bs = _sweep_latencies(
        cfg, schedules, anchor="knee", tag="plot2b")

    all_l = [lat_plain, lat_naive, *lat_prop.values()]
    lo = min(l.min() for l in all_l)
    hi = _kde_hi(cfg, all_l, "plot2b")
    grid = np.linspace(lo, hi, pts)

    # figure height follows the panel count so each panel keeps roughly the
    # single-figure (FIG_SINGLE) aspect ratio; the constant covers suptitle,
    # panel titles, xlabel, and the outside legend.
    w = FIG_DOUBLE[0]
    h = w / max(len(Bs), 1) * (FIG_SINGLE[1] / FIG_SINGLE[0]) + 0.95
    fig, axes = plt.subplots(1, len(Bs), figsize=(w, h),
                             sharex=True, sharey=True)
    if len(Bs) == 1:
        axes = [axes]
    for ax, B in zip(axes, Bs):
        for r, l in (("plain", lat_plain), ("naive", lat_naive),
                     ("proposed", lat_prop[B])):
            ax.plot(grid, _kde(l, grid, bw), color=RUNTIME_COLORS[r],
                    linestyle=RUNTIME_STYLES[r]["linestyle"], linewidth=1.1)
        ax.set_title(b2_label(B))
        ax.xaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_ylabel("Density")
    axes[len(axes) // 2].set_xlabel("Latency (ms)")
    fig.suptitle("Latency Distribution")
    _runtime_legend(fig)
    return _save(fig, cfg, "plot2b_latency_kde_sweep")


def plot_latency_cdf_sweep(cfg: Config, schedules: dict):
    """Plot 3b: one CDF figure with plain, naive, AND the whole proposed
    b2 sweep (sequential shades, light -> dark as b2 grows).

    CDFs handle long tails without distortion, so no x-clipping is applied.
    """
    lat_plain, lat_naive, lat_prop, Bs = _sweep_latencies(
        cfg, schedules, anchor="capacity", tag="plot3b")

    def _cdf(ax, l, **kw):
        l = np.sort(l)
        ax.plot(l, np.arange(1, len(l) + 1) / len(l), **kw)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, l in (("plain", lat_plain), ("naive", lat_naive)):
        _cdf(ax, l, color=RUNTIME_COLORS[r],
             linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    shades = proposed_shades(len(Bs))
    for c, B in zip(shades, Bs):
        _cdf(ax, lat_prop[B], color=c, linestyle="-", linewidth=1.1,
             label=b2_label(B))
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF")
    ax.legend(ncol=2, loc="lower right")
    return _save(fig, cfg, "plot3b_latency_cdf_sweep")


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
        ax.plot(lams, means[r], color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    ax.set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Load vs Latency")
    ax.legend(loc="upper left")
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
    B = int(cfg.batching.seg2_batch)
    rows = [("plain", schedules["plain"]),
            ("naive", schedules["naive"]),
            ("proposed", schedules["proposed"][B])]
    common = metrics.common_completed([s for _, s in rows])
    per = _arrivals_per_runtime(cfg, schedules, common)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    height = 0.6
    for y, (r, s) in enumerate(rows):
        arr, desc, _ = per[r]
        print(f"[plot7] {RUNTIME_LABELS[r]}: {desc}")
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


def plot_stage_time_bars(cfg: Config, schedules: dict):
    """Plot 7b: stacked vertical bars — total Stage-1 / Stage-2 / idle time
    per runtime, i.e. plot7's timeline collapsed from chronological position
    into a per-runtime sum. Same per-runtime arrival trace as plot7, so the
    two figures share the same run context and are directly comparable.
    """
    B = int(cfg.batching.seg2_batch)
    rows = [("plain", schedules["plain"]),
            ("naive", schedules["naive"]),
            ("proposed", schedules["proposed"][B])]
    common = metrics.common_completed([s for _, s in rows])
    per = _arrivals_per_runtime(cfg, schedules, common)

    totals = {}
    for r, s in rows:
        arr, desc, _ = per[r]
        sums = {"seg1": 0.0, "seg2": 0.0, "wait": 0.0}
        for a, b, kind in _op_intervals(s, arr):
            sums[kind] += (b - a)
        totals[r] = sums
        print(f"[plot7b] {r}: {desc} — stage1={sums['seg1']*1000:.1f} ms, "
              f"stage2={sums['seg2']*1000:.1f} ms, idle={sums['wait']*1000:.1f} ms")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(len(rows))
    width = 0.5
    bottom = np.zeros(len(rows))
    kind_colors = {
        "seg1": [RUNTIME_COLORS[r] for r, _ in rows],
        "seg2": [lighten(RUNTIME_COLORS[r]) for r, _ in rows],
        "wait": [IDLE_COLOR] * len(rows),
    }
    for kind in ("seg1", "seg2", "wait"):
        vals = np.array([totals[r][kind] * 1000.0 for r, _ in rows])   # ms
        ax.bar(x, vals, width, bottom=bottom, color=kind_colors[kind],
              edgecolor="black", linewidth=0.4)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([RUNTIME_LABELS[r] for r, _ in rows])
    ax.set_ylabel("Time (ms)")
    ax.set_title("GPU Time Breakdown")
    handles = [Patch(facecolor=STAGE1_SWATCH, label="Stage 1"),
               Patch(facecolor=STAGE2_SWATCH, label="Stage 2"),
               Patch(facecolor=IDLE_COLOR, label="Idle")]
    ax.legend(handles=handles, loc="upper right")
    return _save(fig, cfg, "plot7b_stage_time_bars")


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
    plot_latency_kde_sweep(cfg, schedules)
    plot_latency_cdf(cfg, schedules)
    plot_latency_cdf_sweep(cfg, schedules)
    divergence = plot_load_latency(cfg, schedules)
    plot_latency_breakdown(cfg, schedules)
    plot_timeline(cfg, schedules)
    plot_stage_time_bars(cfg, schedules)
    plot_exec_stats(cfg, schedules)
    plot_naive_seg2_sizes(cfg, schedules)
    return divergence
