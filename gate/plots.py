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
from .plot_style import (COMPONENT_COLORS, COMPONENT_LABELS, EXIT_CLASS_COLORS,
                         EXIT_CLASS_LABELS, EXIT_CLASS_ORDER, EXIT_CLASS_STYLES,
                         FIG_DOUBLE, FIG_SINGLE, IDLE_COLOR, RUNTIME_COLORS,
                         RUNTIME_LABELS, RUNTIME_ORDER, RUNTIME_STYLES,
                         STAGE1_SWATCH, STAGE2_SWATCH, b2_label, lighten,
                         proposed_shades)
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
    """`sched`'s own capacity × a safety margin (`plots.capacity_margin_frac`,
    default 0.90 — same convention as Nexus, SOSP'19: Poisson load at 90% of
    max throughput), snapped to the sweep grid. Capacity itself is a
    near-critical (ρ→1) boundary; backing off by a single fixed grid step
    (~0.3% of capacity here) was not enough margin — plain's latency
    distribution still showed near-critical burstiness (bimodal KDE) right
    at capacity−step. A percentage-of-capacity margin scales properly across
    runtimes with very different capacities, unlike a fixed absolute step."""
    frac = float(cfg.get_path("plots.capacity_margin_frac", 0.90))
    cap = metrics.capacity_lambda(sched, common)
    lams = lambda_grid(cfg)
    return float(lams[int(np.argmin(np.abs(lams - (cap * frac))))])


def _capacity_arrivals(cfg: Config, n: int, sched, common):
    """(arrivals, desc, origin) for `sched` at its own capacity×margin λ."""
    lam = _capacity_step_lambda(cfg, sched, common)
    return (poisson_arrivals(n, lam, int(cfg.arrivals.seed)),
            f"λ={lam:g} req/s (capacity×margin)", "arrival")


def _arrivals_per_runtime(cfg: Config, schedules: dict, common):
    """{runtime: (arr, desc, origin)} for plain/naive/proposed@default bs2,
    each at its OWN capacity×margin λ (last stable load)."""
    n = schedules["plain"].n_requests
    B = int(cfg.batching.seg2_batch)
    entries = {"plain": schedules["plain"], "naive": schedules["naive"],
               "proposed": schedules["proposed"][B]}
    return {r: _capacity_arrivals(cfg, n, s, common) for r, s in entries.items()}


# --------------------------------------------------------------------------- #
# Plot 1a/1b: Goodput under Latency SLOs — plain + naive + the proposed b2 sweep,
# the two figures differing only in the λ the whole panel is replayed at.
# --------------------------------------------------------------------------- #
def _slo_goodput_pair(cfg: Config, schedules: dict, anchor_runtime: str, name: str):
    """One SLO-vs-goodput figure: BOTH baselines (plain, naive) + the proposed
    b2 sweep, all replayed on the same arrival trace at the figure's λ.

    `anchor_runtime` no longer selects which curves are drawn — every runtime is
    drawn in both figures — it only picks the operating point:

    λ selection (plots.slo_goodput_lambda.<anchor_runtime>):
      missing/"auto" -> derived from measured capacities, matching the e2e
        Table B rule: each figure sits at its OWN anchor's last stable
        load (D_anchor − step) — plain figure at D_plain − step, naive
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

    raw = cfg.get_path(f"plots.slo_goodput_lambda.{anchor_runtime}", None)
    if raw is None or raw == "auto":
        lams = lambda_grid(cfg)
        step = float(cfg.arrivals["lambda_sweep"]["step"])
        cap = metrics.capacity_lambda(schedules[anchor_runtime], common)
        lam = float(lams[int(np.argmin(np.abs(lams - (cap - step))))])
        src = f"auto: {anchor_runtime} capacity {cap:.1f} − step"
    else:
        lam = float(raw)
        src = "manual override"
    if lam <= 0:
        arr, origin, desc = np.zeros(n, dtype=float), "stage1_start", "saturated"
    else:
        arr, origin, desc = (poisson_arrivals(n, lam, int(cfg.arrivals.seed)),
                             "arrival", f"λ={lam:g} req/s")
    print(f"[{name}] Plain vs Naive vs Proposed at {desc} ({src})")
    slo = slo_grid_ms(cfg)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r in ("plain", "naive"):
        ax.plot(slo, metrics.goodput_vs_slo(schedules[r], arr, common, slo, mode, origin),
                color=RUNTIME_COLORS[r], label=RUNTIME_LABELS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"])
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
    """Plot 1a and 1b both show Plain + Naive + the proposed b2 sweep; they
    differ only in the operating point (λ anchored on plain's capacity for 1a,
    on naive's for 1b). Within a figure every runtime shares one trace."""
    a = _slo_goodput_pair(cfg, schedules, "plain", "plot1a_slo_goodput_vs_plain")
    b = _slo_goodput_pair(cfg, schedules, "naive", "plot1b_slo_goodput_vs_naive")
    return a, b


# --------------------------------------------------------------------------- #
# Plots 2 & 3: latency distribution / CDF
# --------------------------------------------------------------------------- #
def _per_runtime_latencies(cfg: Config, schedules: dict):
    """[(runtime, latency_ms array)] — each runtime replayed at its own
    capacity×margin λ (last stable load) — restricted to the common
    completed set."""
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
    anchor="capacity" -> capacity × `plots.capacity_margin_frac` (default;
                         see `_capacity_step_lambda`). NOTE: plot1a/1b and
                         Table B/E intentionally still use the tighter
                         capacity−step convention — only the distribution-
                         shape figures (plot2/2b/3/3b/7/7b) use this margin.
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
            print(f"[{tag}] {label}: capacity×margin λ = {lam:g} req/s")
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
    Each configuration is replayed at its OWN operating point — knee (default)
    or capacity×margin, via `plots.kde_sweep_anchor` — logged to stdout;
    state these λ values (and which anchor) in the caption.
    """
    bw = cfg.get_path("plots.kde_bandwidth", 0.4)
    pts = int(cfg.get_path("plots.kde_grid_points", 400))
    anchor = cfg.get_path("plots.kde_sweep_anchor", "knee")
    lat_plain, lat_naive, lat_prop, Bs = _sweep_latencies(
        cfg, schedules, anchor=anchor, tag="plot2b")

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


def plot_latency_cdf_common(cfg: Config, schedules: dict):
    """Plot 11: empirical latency CDF with EVERY runtime replayed at ONE
    shared λ — the iso-load counterpart to plot3.

    plot3 puts each runtime at its own capacity×margin λ, so its curves mix two
    effects: the batching structure AND the different offered loads (proposed
    sustains ~11% more req/s than plain, and is therefore measured at a higher
    load). This figure removes the load term so the residual gap is attributable
    to batching alone — in particular to `proposed`'s seg2 queue wait, which
    naive does not pay because it flushes every seg1 batch's leftovers at once.

    λ comes from `plots.cdf_common_lambda` and is NEVER auto-derived: the value
    must be one every compared runtime can sustain (i.e. below the smallest
    capacity — see Table A / plot4), and that choice is the reader's to make and
    to state in the caption. Unset/null -> the figure is skipped. 0 -> saturated
    (all arrivals at t=0), latency measured from each sample's stage-1 input.
    """
    raw = cfg.get_path("plots.cdf_common_lambda", None)
    if raw is None:
        print("[plot11] plots.cdf_common_lambda unset; skipped")
        return None

    lam = float(raw)
    n = schedules["plain"].n_requests
    B = int(cfg.batching.seg2_batch)
    entries = [("plain", schedules["plain"]),
               ("naive", schedules["naive"]),
               ("proposed", schedules["proposed"][B])]
    common = metrics.common_completed([s for _, s in entries])

    if lam <= 0:
        arr, origin, desc = np.zeros(n, dtype=float), "stage1_start", "saturated"
    else:
        arr, origin, desc = (poisson_arrivals(n, lam, int(cfg.arrivals.seed)),
                             "arrival", f"λ={lam:g} req/s")
    print(f"[plot11] all runtimes at a SHARED {desc} "
          f"(proposed at bs2={B}); n={len(common)} common samples")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, s in entries:
        l = np.sort(metrics.latency_ms(s, arr, common, origin))
        print(f"[plot11] {RUNTIME_LABELS[r]}: p50={np.percentile(l, 50):.2f} ms, "
              f"p90={np.percentile(l, 90):.2f} ms, p99={np.percentile(l, 99):.2f} ms")
        ax.plot(l, np.arange(1, len(l) + 1) / len(l), color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Latency CDF at a Shared Load ({desc})")
    ax.legend(loc="lower right")
    return _save(fig, cfg, "plot11_latency_cdf_common_lambda")


# --------------------------------------------------------------------------- #
# Plot 12a/12b: latency HISTOGRAM (KDE-free view of the same distribution).
# The KDE (plot2) smears close modes together; a raw histogram keeps the naive
# exit vs non-exit split (offset = one seg2 service time) and proposed's seg2
# queue-wait mode visible without any smoothing.
#   12a: EVERY runtime at ONE shared λ  (plots.cdf_common_lambda; = plot11's λ)
#   12b: each runtime at its OWN capacity×margin λ            (= plot3's λ)
# --------------------------------------------------------------------------- #
def _latency_hist_fig(cfg: Config, data: list, title: str, name: str):
    """Overlaid per-runtime latency histogram on shared bins.

    `data` = [(runtime, latency_ms array)] (all restricted to the same common
    set, so equal N -> raw counts are directly comparable). Bins span [min, hi]
    where `hi` reuses the KDE x-clip (plots.kde_xlim_ms / kde_clip_percentile);
    latencies past `hi` fall outside the bin edges and are dropped (no edge
    pile-up), matching plot2's x-range. State any truncation in the caption.
    """
    bins = int(cfg.get_path("plots.hist_bins", 80))
    density = bool(cfg.get_path("plots.hist_density", False))
    lo = min(l.min() for _, l in data)
    hi = _kde_hi(cfg, [l for _, l in data], name)
    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, l in data:
        ax.hist(l, bins=edges, density=density, histtype="stepfilled",
                alpha=0.35, color=RUNTIME_COLORS[r], edgecolor=RUNTIME_COLORS[r],
                linewidth=1.1, label=RUNTIME_LABELS[r])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return fig


def plot_latency_hist_common(cfg: Config, schedules: dict):
    """Plot 12a: latency histogram, EVERY runtime replayed at ONE shared λ
    (plots.cdf_common_lambda) — histogram counterpart of plot11 (iso-load)."""
    raw = cfg.get_path("plots.cdf_common_lambda", None)
    if raw is None:
        print("[plot12a] plots.cdf_common_lambda unset; skipped")
        return None

    lam = float(raw)
    n = schedules["plain"].n_requests
    B = int(cfg.batching.seg2_batch)
    entries = [("plain", schedules["plain"]),
               ("naive", schedules["naive"]),
               ("proposed", schedules["proposed"][B])]
    common = metrics.common_completed([s for _, s in entries])

    if lam <= 0:
        arr, origin, desc = np.zeros(n, dtype=float), "stage1_start", "saturated"
    else:
        arr, origin, desc = (poisson_arrivals(n, lam, int(cfg.arrivals.seed)),
                             "arrival", f"λ={lam:g} req/s")
    print(f"[plot12a] all runtimes at a SHARED {desc} "
          f"(proposed at bs2={B}); n={len(common)} common samples")
    data = [(r, metrics.latency_ms(s, arr, common, origin)) for r, s in entries]
    fig = _latency_hist_fig(
        cfg, data, f"Latency Distribution at a Shared Load ({desc})", "plot12a")
    return _save(fig, cfg, "plot12a_latency_hist_common_lambda")


def plot_latency_hist(cfg: Config, schedules: dict):
    """Plot 12b: latency histogram, each runtime replayed at its OWN
    capacity×margin λ — histogram counterpart of plot3 (per-runtime load)."""
    data = _per_runtime_latencies(cfg, schedules)
    fig = _latency_hist_fig(cfg, data, "Latency Distribution", "plot12b")
    return _save(fig, cfg, "plot12b_latency_hist")


# --------------------------------------------------------------------------- #
# Plot 13a–13d: ONE early-exit runtime at a time — latency split by per-sample
# exit class (purple = exited at stage 1, red = also ran stage 2).
#   13a/13b: naive     (KDE / histogram)
#   13c/13d: proposed@batching.seg2_batch (KDE / histogram)
#
# Why the POOLED curves (plot2 / plot12b) look unimodal even though two
# populations exist. Both classes share the two dominant AND most variable
# latency terms — batch-formation wait and GPU queue wait — so the conditionals
# are the same distribution shifted by the non-exit class's extra stage-2 cost.
# What differs between the runtimes is the size of that shift:
#   * naive     — the extra cost is only its own batch's seg2 op, dispatched
#                 immediately after that batch's seg1 (dynamic size = per-batch
#                 non-exit count). Small and nearly constant, so the shift stays
#                 far below the shared spread and the modes never resolve.
#   * proposed  — the extra cost is seg2 QUEUE wait (until seg2_batch fills)
#                 plus a full static seg2 op, which is both larger and much more
#                 variable. The split is more visible, and at large seg2_batch
#                 the non-exit class grows a heavy right tail of its own.
# The shared spread has a floor either way (formation wait ~ (S-1)/2λ dominates
# at low λ, queue wait at high λ), so on an end-to-end clock the separation
# rarely reaches the ~2 pooled sd that a visibly bimodal curve needs — see the
# Cohen's d printed to stdout, and `plots.exit_split_lambda: 0` for the
# waits-removed view where it does.
# --------------------------------------------------------------------------- #
def _exit_mask(sched) -> np.ndarray:
    """Bool mask over request ids: True where the sample exits at stage 1.

    A sample COMPLETES at a seg1 op iff the LPH head was confident enough
    (conf >= early_exit.confidence_threshold); every other sample completes at
    a later seg2 op. Same rule for naive and proposed — only WHICH seg2 op
    (immediate vs queued flush) differs.
    """
    mask = np.zeros(sched.n_requests, dtype=bool)
    for op in sched.ops:
        if op.kind == "seg1" and len(op.completes):
            mask[op.completes] = True
    return mask


def _exit_split_lambda(cfg: Config, runtime: str, sched, common) -> tuple:
    """(arrivals, origin, desc, short) for the exit-split figures.
    `desc` is the full stdout description, `short` the compact figure form.

    `plots.exit_split_lambda` — a scalar applied to every runtime, or a
    per-runtime mapping ({naive: …, proposed: …}); either way:
      null / "auto" -> the runtime's OWN capacity×margin λ (the operating point
                       of plot2 / plot3 / plot12b, so the pooled curve here is
                       identical to that runtime's curve there)
      number > 0    -> manual override at that rate
      0             -> saturated (all arrivals at t=0; latency measured from
                       the stage-1 op start, as elsewhere). This is the only
                       setting that removes the shared formation + queue wait,
                       and hence the only one where the modes actually split.
    """
    raw = cfg.get_path("plots.exit_split_lambda", None)
    if isinstance(raw, dict):
        raw = raw.get(runtime, None)
    if raw is None or raw == "auto":
        lam = _capacity_step_lambda(cfg, sched, common)
        src = "capacity×margin"
    else:
        lam = float(raw)
        src = "manual"
    n = sched.n_requests
    if lam <= 0:
        return np.zeros(n, dtype=float), "stage1_start", "saturated", "saturated"
    return (poisson_arrivals(n, lam, int(cfg.arrivals.seed)), "arrival",
            f"lambda={lam:g} req/s ({src})", f"$\\lambda$={lam:g} req/s")


def _exit_split(cfg: Config, schedules: dict, runtime: str, name: str):
    """Split one runtime's per-sample latency by exit class.

    Restricted to the SAME common completed set as plot2/3/12, so the pooled
    array here is exactly that runtime's latency array there.
    Returns (data, pooled_ms, short_desc, label) where
    data = [("exit", ms), ("nonexit", ms)].
    """
    B = int(cfg.batching.seg2_batch)
    entries = {"naive": (schedules["naive"], RUNTIME_LABELS["naive"]),
               "proposed": (schedules["proposed"][B],
                            f"{RUNTIME_LABELS['proposed']} ({b2_label(B)})")}
    if runtime not in entries:
        raise ValueError(f"exit-split runtime must be one of {list(entries)}, "
                         f"got {runtime!r}")
    sched, label = entries[runtime]
    common = metrics.common_completed(
        [schedules["plain"], schedules["naive"], schedules["proposed"][B]])
    arr, origin, desc, short = _exit_split_lambda(cfg, runtime, sched, common)

    pooled = metrics.latency_ms(sched, arr, common, origin)
    is_exit = _exit_mask(sched)[common]
    data = [("exit", pooled[is_exit]), ("nonexit", pooled[~is_exit])]

    ex, nx = data[0][1], data[1][1]
    print(f"[{name}] {label} at {desc}; n={len(pooled)} common samples "
          f"(exit {len(ex)} = {100 * len(ex) / max(len(pooled), 1):.1f}%, "
          f"non-exit {len(nx)})")
    if len(ex) > 1 and len(nx) > 1:
        gap = float(nx.mean() - ex.mean())
        pooled_sd = float(np.sqrt((ex.var() + nx.var()) / 2))
        print(f"[{name}] exit mean={ex.mean():.2f} ms (sd {ex.std():.2f}), "
              f"non-exit mean={nx.mean():.2f} ms (sd {nx.std():.2f})")
        print(f"[{name}] mode gap={gap:.2f} ms, pooled sd={pooled_sd:.2f} ms, "
              f"Cohen's d={gap / pooled_sd if pooled_sd else float('nan'):.2f} "
              f"(d >~ 2 needed for a visibly bimodal pooled curve)")
    return data, pooled, short, label


def _exit_class_marks(ax, data):
    """Dotted vertical line at each class mean (plots.exit_split_marks)."""
    for c, l in data:
        if len(l):
            ax.axvline(float(l.mean()), color=EXIT_CLASS_COLORS[c],
                       linestyle=":", linewidth=0.8, alpha=0.8, zorder=1)


def _exit_split_title(label: str, desc: str) -> str:
    """Two-line title — the runtime label plus λ does not fit on one line at
    FIG_SINGLE width."""
    return f"{label} Latency by Exit Class\n({desc})"


def _exit_split_kde_fig(cfg: Config, runtime: str, schedules: dict, name: str):
    """Shared body of 13a/13c.

    `plots.exit_split_normalize`:
      "mixture" (default) -- each class KDE is scaled by its share of the
        samples, so the two curves ADD UP to the pooled density (drawn in
        gray). This is the view that explains a missing bimodality: the sum of
        two heavily overlapping components has no dip.
      "each" -- each class normalized to unit area instead; shapes are easier
        to compare but the curves no longer sum to the pooled density.
    """
    data, pooled, desc, label = _exit_split(cfg, schedules, runtime, name)
    bw = cfg.get_path("plots.kde_bandwidth", 0.4)
    norm = str(cfg.get_path("plots.exit_split_normalize", "mixture")).lower()
    if norm not in ("mixture", "each"):
        raise ValueError("plots.exit_split_normalize must be 'mixture' or "
                         f"'each', got {norm!r}")
    show_pooled = bool(cfg.get_path("plots.exit_split_show_pooled", True))

    lo = float(pooled.min())
    hi = _kde_hi(cfg, [pooled], name)
    grid = np.linspace(lo, hi, int(cfg.get_path("plots.kde_grid_points", 400)))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    if show_pooled and norm == "mixture":
        ax.plot(grid, _kde(pooled, grid, bw), color=EXIT_CLASS_COLORS["pooled"],
                linestyle=EXIT_CLASS_STYLES["pooled"]["linestyle"],
                linewidth=1.0, label=EXIT_CLASS_LABELS["pooled"], zorder=2)
    for c, l in data:
        if len(l) < 2:
            continue
        w = len(l) / len(pooled) if norm == "mixture" else 1.0
        ax.plot(grid, w * _kde(l, grid, bw), color=EXIT_CLASS_COLORS[c],
                linestyle=EXIT_CLASS_STYLES[c]["linestyle"],
                label=EXIT_CLASS_LABELS[c], zorder=3)
    if bool(cfg.get_path("plots.exit_split_marks", True)):
        _exit_class_marks(ax, data)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title(_exit_split_title(label, desc))
    ax.legend(loc="upper right")
    return fig


def _exit_split_hist_fig(cfg: Config, runtime: str, schedules: dict, name: str):
    """Shared body of 13b/13d — the same split with no smoothing at all.

    `plots.exit_split_hist_stacked` (default true): the classes are disjoint
    subsets of the same population, so stacking them reproduces the runtime's
    pooled histogram exactly and shows how the components fill in each other's
    gaps. Set false for an overlaid (alpha-blended) comparison of the two
    shapes instead. Bins/x-range follow plot12 (plots.hist_bins, the KDE
    x-clip); samples past the clip fall outside the edges and are dropped.
    """
    data, pooled, desc, label = _exit_split(cfg, schedules, runtime, name)
    bins = int(cfg.get_path("plots.hist_bins", 80))
    density = bool(cfg.get_path("plots.hist_density", False))
    stacked = bool(cfg.get_path("plots.exit_split_hist_stacked", True))

    lo = float(pooled.min())
    hi = _kde_hi(cfg, [pooled], name)
    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    order = list(EXIT_CLASS_ORDER)
    by_class = dict(data)
    if stacked:
        _, _, containers = ax.hist(
            [by_class[c] for c in order], bins=edges, density=density,
            stacked=True, histtype="stepfilled",
            color=[EXIT_CLASS_COLORS[c] for c in order],
            label=[EXIT_CLASS_LABELS[c] for c in order],
            edgecolor="white", linewidth=0.3)
        # per-class hatch (ax.hist takes no hatch list) — keeps the stack
        # readable when the figure is printed in grayscale
        for c, cont in zip(order, containers):
            h = EXIT_CLASS_STYLES[c]["hatch"]
            if h:
                for patch in np.atleast_1d(cont):
                    patch.set_hatch(h)
    else:
        for c, l in data:
            ax.hist(l, bins=edges, density=density, histtype="stepfilled",
                    alpha=0.35, color=EXIT_CLASS_COLORS[c],
                    edgecolor=EXIT_CLASS_COLORS[c], linewidth=1.1,
                    hatch=EXIT_CLASS_STYLES[c]["hatch"],
                    label=EXIT_CLASS_LABELS[c])
    if bool(cfg.get_path("plots.exit_split_marks", True)):
        _exit_class_marks(ax, data)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(_exit_split_title(label, desc))
    ax.legend(loc="upper right")
    return fig


def plot_naive_exit_kde(cfg: Config, schedules: dict):
    """Plot 13a: naive latency KDE, split by exit class."""
    fig = _exit_split_kde_fig(cfg, "naive", schedules, "plot13a")
    return _save(fig, cfg, "plot13a_naive_exit_kde")


def plot_naive_exit_hist(cfg: Config, schedules: dict):
    """Plot 13b: naive latency histogram, split by exit class."""
    fig = _exit_split_hist_fig(cfg, "naive", schedules, "plot13b")
    return _save(fig, cfg, "plot13b_naive_exit_hist")


def plot_proposed_exit_kde(cfg: Config, schedules: dict):
    """Plot 13c: proposed@seg2_batch latency KDE, split by exit class.

    Same construction as 13a; the contrast is the point — proposed's non-exit
    class carries the seg2 QUEUE wait on top of the seg2 op, so its component
    sits further right and is wider than naive's.
    """
    fig = _exit_split_kde_fig(cfg, "proposed", schedules, "plot13c")
    return _save(fig, cfg, "plot13c_proposed_exit_kde")


def plot_proposed_exit_hist(cfg: Config, schedules: dict):
    """Plot 13d: proposed@seg2_batch latency histogram, split by exit class."""
    fig = _exit_split_hist_fig(cfg, "proposed", schedules, "plot13d")
    return _save(fig, cfg, "plot13d_proposed_exit_hist")


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
    plot_latency_cdf_common(cfg, schedules)
    plot_latency_hist_common(cfg, schedules)
    plot_latency_hist(cfg, schedules)
    plot_naive_exit_kde(cfg, schedules)
    plot_naive_exit_hist(cfg, schedules)
    plot_proposed_exit_kde(cfg, schedules)
    plot_proposed_exit_hist(cfg, schedules)
    divergence = plot_load_latency(cfg, schedules)
    plot_latency_breakdown(cfg, schedules)
    plot_timeline(cfg, schedules)
    plot_stage_time_bars(cfg, schedules)
    plot_exec_stats(cfg, schedules)
    plot_naive_seg2_sizes(cfg, schedules)
    return divergence
