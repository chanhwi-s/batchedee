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
from .plot_style import (COMPONENT_COLORS, COMPONENT_LABELS,
                         COMPONENT_LEGEND_LABELS, EXIT_CLASS_COLORS,
                         EXIT_CLASS_LABELS, EXIT_CLASS_LABELS_SHORT,
                         EXIT_CLASS_ORDER, EXIT_CLASS_STYLES, FIG_DOUBLE,
                         FIG_QUAD, FIG_SINGLE, IDLE_COLOR, RUNTIME_COLORS,
                         RUNTIME_LABELS, RUNTIME_ORDER, RUNTIME_STYLES,
                         SLO_COLOR, SLO_FILL_ALPHA, STAGE1_SWATCH,
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


def _manual_arrivals(cfg: Config, n: int, lam: float):
    """(arrivals, origin, desc, mathtext) for an explicitly chosen λ — `desc`
    is the plain-text stdout form, `mathtext` the figure form. 0/negative ->
    saturated: all arrivals at t=0, latency measured from the stage-1 op start
    (waiting behind the t=0 backlog is a setup artifact, not a runtime
    property)."""
    if lam <= 0:
        return np.zeros(n, dtype=float), "stage1_start", "saturated", "saturated"
    return (poisson_arrivals(n, lam, int(cfg.arrivals.seed)), "arrival",
            f"lambda={lam:g} req/s", f"$\\lambda$={lam:g} req/s")


def _config_entries(cfg: Config, schedules: dict):
    """([(runtime, sched)], common_ids, B) — plain / naive / proposed@seg2_batch
    on their shared common completed set. The base of plot2c and plot11a/11b."""
    B = int(cfg.batching.seg2_batch)
    entries = [("plain", schedules["plain"]),
               ("naive", schedules["naive"]),
               ("proposed", schedules["proposed"][B])]
    return entries, metrics.common_completed([s for _, s in entries]), B


def _manual_arrivals_per_runtime(cfg: Config, entries, common, name: str):
    """{runtime: (arr, origin, desc, mathtext)} from `arrivals.lambda`, or None
    if that key is absent (caller should skip its figure).

    Shared by plot2c and plot11b — the "each runtime at the load IT would
    actually be operated at" figures. `arrivals.lambda` may be a per-runtime
    mapping or a scalar; per runtime a number is a rate, 0 is saturated, and a
    missing/null entry falls back to that runtime's capacity×margin λ (the
    plot2/3 convention) with a note on stdout.
    """
    raw = cfg.get_path("arrivals.lambda", None)
    if raw is None:
        print(f"[{name}] arrivals.lambda unset; skipped")
        return None
    n = entries[0][1].n_requests
    per = {}
    for r, s in entries:
        lam = raw.get(r, None) if isinstance(raw, dict) else raw
        if lam is None:
            lam = _capacity_step_lambda(cfg, s, common)
            print(f"[{name}] arrivals.lambda.{r} unset -> falling back to "
                  f"{r}'s capacity×margin λ = {lam:g} req/s")
        per[r] = _manual_arrivals(cfg, n, float(lam))
    return per


# --------------------------------------------------------------------------- #
# Plot 1a/1b: Goodput under Latency SLOs — plain + naive + the proposed b2 sweep,
# the two figures differing only in the λ the whole panel is replayed at.
# --------------------------------------------------------------------------- #
def _slo_goodput_anchor(cfg: Config, schedules: dict, anchor_runtime: str,
                        common):
    """(schedule, note) whose capacity sets a plot1a/1b/1c figure's λ.

    plain and naive have a single schedule. `proposed` is a {seg2_batch:
    Schedule} sweep, so plot1c has to pick one — the operating point it wants is
    the highest load ANY proposed configuration sustains, i.e. the b2 with the
    largest measured capacity. `plots.slo_goodput_b2` pins a specific b2 instead
    ("config" = `batching.seg2_batch`, a number = that b2). Whichever rule is
    used, the chosen b2 and how it compares with the sweep is logged, because
    the figure's λ is only meaningful next to that choice.
    """
    if anchor_runtime != "proposed":
        return schedules[anchor_runtime], anchor_runtime

    prop = schedules["proposed"]
    caps = {B: metrics.capacity_lambda(s, common) for B, s in prop.items()}
    best = max(caps, key=caps.get)
    pick = cfg.get_path("plots.slo_goodput_b2", "max_capacity")
    if pick == "max_capacity":
        B = best
    elif pick == "config":
        B = int(cfg.batching.seg2_batch)
    else:
        B = int(pick)
    if B not in prop:
        raise ValueError(f"plots.slo_goodput_b2 -> bs2={B}, not in the sweep "
                         f"{sorted(prop)}")
    print("[plot1c] proposed capacities: " +
          ", ".join(f"bs2={b}: {caps[b]:.0f}" for b in sorted(caps)))
    note = f"proposed bs2={B}"
    if B != best:
        note += f" (bs2={best} sustains more: {caps[best]:.0f} req/s)"
    return prop[B], note


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
        anchor_sched, anchor_note = _slo_goodput_anchor(cfg, schedules,
                                                        anchor_runtime, common)
        cap = metrics.capacity_lambda(anchor_sched, common)
        lam = float(lams[int(np.argmin(np.abs(lams - (cap - step))))])
        src = f"auto: {anchor_note} capacity {cap:.1f} − step"
    else:
        lam = float(raw)
        src = "manual override"
    if lam <= 0:
        arr, origin, desc = np.zeros(n, dtype=float), "stage1_start", "saturated"
    else:
        arr, origin, desc = (poisson_arrivals(n, lam, int(cfg.arrivals.seed)),
                             "arrival", f"λ={lam:g} req/s")
    print(f"[{name}] " + " vs ".join(RUNTIME_LABELS[r] for r in RUNTIME_ORDER)
          + f" at {desc} ({src})")
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
    """Plots 1a/1b/1c: identical figures — Plain + Naive + the whole proposed b2
    sweep, every curve on one shared arrival trace — differing ONLY in which
    runtime's capacity sets the operating point:

      1a → plain's      (the load plain can just barely still serve)
      1b → naive's
      1c → proposed's   (the highest load any configuration here sustains)

    Read as a series they answer "who still meets which SLO as the offered load
    is raised to each design's own ceiling in turn". 1c is the extreme end: at
    proposed's capacity both baselines are past their divergence point, so their
    goodput collapses across the whole SLO range while the proposed curves stay
    up — which is the claim the paper is making, stated at its strongest.
    """
    a = _slo_goodput_pair(cfg, schedules, "plain", "plot1a_slo_goodput_vs_plain")
    b = _slo_goodput_pair(cfg, schedules, "naive", "plot1b_slo_goodput_vs_naive")
    c = _slo_goodput_pair(cfg, schedules, "proposed",
                          "plot1c_slo_goodput_vs_proposed")
    return a, b, c


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


def plot_latency_kde_per_runtime(cfg: Config, schedules: dict):
    """Plot 2c: the same KDE with each runtime at its OWN manually chosen λ,
    read from `arrivals.lambda` — the KDE counterpart of plot11b.

    plot2 derives every λ automatically (each runtime's capacity×margin); this
    one uses the rates written in the config, for when the comparison you want
    is "each runtime at the load it would actually be operated at". Because the
    offered loads differ, the gaps here are NOT attributable to batching alone.
    See `_manual_arrivals_per_runtime` for how `arrivals.lambda` is read;
    absent -> the figure is skipped. Each runtime's λ goes to stdout, not into
    the legend — state them in the caption.
    """
    entries, common, B = _config_entries(cfg, schedules)
    per = _manual_arrivals_per_runtime(cfg, entries, common, "plot2c")
    if per is None:
        return None

    bw = cfg.get_path("plots.kde_bandwidth", 0.4)
    lats = []
    for r, s in entries:
        arr, origin, desc, _ = per[r]
        l = metrics.latency_ms(s, arr, common, origin)
        print(f"[plot2c] {RUNTIME_LABELS[r]} @ {desc}: mean={l.mean():.2f} ms, "
              f"p50={np.percentile(l, 50):.2f} ms, "
              f"p99={np.percentile(l, 99):.2f} ms")
        lats.append((r, l))
    print(f"[plot2c] per-runtime loads (proposed at bs2={B}); "
          f"n={len(common)} common samples")

    lo = min(l.min() for _, l in lats)
    hi = _kde_hi(cfg, [l for _, l in lats], "plot2c")
    grid = np.linspace(lo, hi, int(cfg.get_path("plots.kde_grid_points", 400)))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, l in lats:
        ax.plot(grid, _kde(l, grid, bw), color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Latency Distribution at Per-Runtime Loads")
    ax.legend(loc="upper right")
    return _save(fig, cfg, "plot2c_latency_kde_per_runtime_lambda")


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


def _latency_cdf_fig(cfg: Config, entries, common, per: dict, title: str,
                     name: str):
    """Shared body of plot11a/11b: one empirical CDF per runtime.

    `per` = {runtime: (arrivals, origin, desc, mathtext)}. The legend carries
    the plain runtime name only — each runtime's operating point goes to
    stdout, so state the λ values in the caption (they are NOT self-evident
    from 11b, where the curves are not iso-load).
    """
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, s in entries:
        arr, origin, desc, _ = per[r]
        l = np.sort(metrics.latency_ms(s, arr, common, origin))
        print(f"[{name}] {RUNTIME_LABELS[r]} @ {desc}: "
              f"p50={np.percentile(l, 50):.2f} ms, "
              f"p90={np.percentile(l, 90):.2f} ms, "
              f"p99={np.percentile(l, 99):.2f} ms")
        ax.plot(l, np.arange(1, len(l) + 1) / len(l), color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"],
                label=RUNTIME_LABELS[r])
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, cfg, name)


def plot_latency_cdf_common(cfg: Config, schedules: dict):
    """Plot 11a: empirical latency CDF with EVERY runtime replayed at ONE
    shared λ — the iso-load counterpart to plot3.

    plot3 puts each runtime at its own capacity×margin λ, so its curves mix two
    effects: the batching structure AND the different offered loads (proposed
    sustains ~6% more req/s than plain, and is therefore measured at a higher
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
        print("[plot11a] plots.cdf_common_lambda unset; skipped")
        return None

    entries, common, B = _config_entries(cfg, schedules)
    n = schedules["plain"].n_requests
    got = _manual_arrivals(cfg, n, float(raw))
    print(f"[plot11a] all runtimes at a SHARED {got[2]} "
          f"(proposed at bs2={B}); n={len(common)} common samples")
    per = {r: got for r, _ in entries}
    return _latency_cdf_fig(cfg, entries, common, per,
                            f"Latency CDF at a Shared Load ({got[3]})",
                            "plot11a_latency_cdf_common_lambda")


def plot_latency_cdf_per_runtime(cfg: Config, schedules: dict):
    """Plot 11b: the same CDF with each runtime at its OWN manually chosen λ,
    read from `arrivals.lambda` — the anti-iso-load counterpart to 11a.

    Use this when the comparison you want is "each runtime at the load IT would
    actually be operated at" (e.g. each one's sustainable rate read off plot4),
    rather than a single rate the slowest runtime dictates. Because the offered
    loads differ, the gaps here are NOT attributable to batching alone — that is
    exactly what 11a is for; the two figures are meant to be read as a pair.

    See `_manual_arrivals_per_runtime` for how `arrivals.lambda` is read;
    absent -> the figure is skipped. Each runtime's λ is printed to stdout (not
    drawn in the legend) — the curves are NOT comparable without those values,
    so state them in the caption. plot2c is the KDE counterpart.
    """
    entries, common, B = _config_entries(cfg, schedules)
    per = _manual_arrivals_per_runtime(cfg, entries, common, "plot11b")
    if per is None:
        return None
    print(f"[plot11b] per-runtime loads (proposed at bs2={B}); "
          f"n={len(common)} common samples")
    return _latency_cdf_fig(cfg, entries, common, per,
                            "Latency CDF at Per-Runtime Loads",
                            "plot11b_latency_cdf_per_runtime_lambda")


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


def _exit_split_raw(cfg: Config, schedules: dict, runtime: str, lam=None):
    """Everything the exit-split figures need, with no printing.

    Restricted to the SAME common completed set as plot2/3/12, so `pooled` is
    exactly that runtime's latency array there. `lam` overrides the operating
    point with an explicit rate (plot14, where every runtime shares one λ);
    None keeps the `plots.exit_split_lambda` rule (plot13).
    Returns (sched, label, common, arr, origin, desc, short, pooled, is_exit).
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
    if lam is None:
        arr, origin, desc, short = _exit_split_lambda(cfg, runtime, sched, common)
    else:
        arr, origin, desc, short = _manual_arrivals(cfg, sched.n_requests,
                                                    float(lam))
        desc, short = f"{desc} (shared)", f"{short}, shared"
    pooled = metrics.latency_ms(sched, arr, common, origin)
    return (sched, label, common, arr, origin, desc, short, pooled,
            _exit_mask(sched)[common])


def _exit_split(cfg: Config, schedules: dict, runtime: str, name: str, lam=None):
    """Split one runtime's per-sample latency by exit class (13a–13d, 14a–14d).

    Returns (data, pooled_ms, short_desc, label) where
    data = [("exit", ms), ("nonexit", ms)].
    """
    _, label, _, _, _, desc, short, pooled, is_exit = _exit_split_raw(
        cfg, schedules, runtime, lam)
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


def _exit_split_hi(cfg: Config, pooled: np.ndarray, name: str, xmax=None):
    """Upper x-bound for an exit-split figure.

    `xmax` (plot14's `plots.exit_split_xlim_ms`) pins the axis to a fixed value
    so all four panels share one scale and the tail is shown rather than
    clipped; None falls back to the usual `_kde_hi` percentile clip (plot13).
    Reports whatever still falls outside, since histogram bins drop it.
    """
    if xmax is None:
        return _kde_hi(cfg, [pooled], name)
    hi = float(xmax)
    over = float((pooled > hi).mean())
    note = f", {100 * over:.2f}% beyond it" if over else ", nothing beyond it"
    print(f"[{name}] x-range fixed, upper bound {hi:g} ms "
          f"(plots.exit_split_xlim_ms){note}")
    return hi


def _slo_values(slo_ms):
    """`slo_ms` (a number, a list, or None) -> sorted list of floats."""
    if slo_ms is None:
        return []
    seq = slo_ms if isinstance(slo_ms, (list, tuple)) else [slo_ms]
    return sorted(float(v) for v in seq)


def _slo_marks(ax, slo_ms, pooled=None, annotate=False, show_count=False):
    """Red deadline rule(s) plus, with `annotate`, the violated share in text
    (plot14). The violating MASS is filled separately — `_slo_fill_area` for a
    KDE, `_slo_color_bars` for a histogram — so that what is coloured is the
    same quantity as the number printed here. `show_count` (plot14e/14f) adds
    the raw violating/total counts alongside the percentage."""
    values = _slo_values(slo_ms)
    if not values:
        return
    for i, v in enumerate(values):
        ax.axvline(v, color=SLO_COLOR, linestyle="-", linewidth=1.1,
                   alpha=0.9, zorder=4, label="SLO" if i == 0 else None)
    if annotate and pooled is not None and len(pooled):
        n = len(pooled)
        if show_count:
            txt = "\n".join(
                f"{100 * (pooled > v).mean():.1f}% ({int((pooled > v).sum())}"
                f"/{n}) > {v:g} ms" for v in values)
        else:
            txt = "\n".join(f"{100 * (pooled > v).mean():.1f}% > {v:g} ms"
                            for v in values)
        ax.text(0.98, 0.62, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=SLO_COLOR, zorder=5)


def _slo_fill_area(ax, slo_ms, grid, density):
    """Fill the area under the pooled density beyond the deadline (plot14a/14c).

    That area is exactly the violated fraction, so the red region and the
    printed percentage are the same number rendered two ways. Filled under the
    POOLED curve, not the per-class ones: the class densities are mixture
    components and their areas would not add up to the reported share. Drawn
    below the curves so nothing is obscured.
    """
    values = _slo_values(slo_ms)
    if not values:
        return
    m = grid >= values[0]
    if not m.any():
        return
    ax.fill_between(grid[m], 0, density[m], color=SLO_COLOR,
                    alpha=SLO_FILL_ALPHA, linewidth=0, zorder=1)


def _slo_color_bars(containers, edges, slo_ms):
    """Recolour every histogram bar past the deadline red (plot14b/14d).

    The KDE counterpart of `_slo_fill_area`. Needs `histtype="bar"` — with
    "stepfilled" matplotlib returns one polygon per dataset, not one patch per
    bin, and individual bars cannot be addressed.
    """
    values = _slo_values(slo_ms)
    if not values:
        return
    centers = 0.5 * (edges[:-1] + edges[1:])
    for cont in containers:
        for c, patch in zip(centers, cont):
            if c > values[0]:
                patch.set_facecolor(SLO_COLOR)
                patch.set_edgecolor(SLO_COLOR)
                patch.set_hatch(None)


def _exit_split_title(label: str, desc: str, simple: bool) -> str:
    """plot13 spells out the figure and its operating point; plot14 (`simple`)
    carries the runtime name only — λ, the SLO and the class definitions all
    belong in the caption, and four panels of repeated boilerplate is noise."""
    if simple:
        return label
    return f"{label} Latency by Exit Class\n({desc})"


def _exit_class_label(cls: str, simple: bool) -> str:
    return (EXIT_CLASS_LABELS_SHORT if simple else EXIT_CLASS_LABELS)[cls]


def _exit_split_kde_fig(cfg: Config, runtime: str, schedules: dict, name: str,
                        lam=None, slo_ms=None, xmax=None, simple=False):
    """Shared body of 13a/13c (and 14a/14c, which pass `lam` + `slo_ms`).

    `plots.exit_split_normalize`:
      "mixture" (default) -- each class KDE is scaled by its share of the
        samples, so the two curves ADD UP to the pooled density (drawn in
        gray). This is the view that explains a missing bimodality: the sum of
        two heavily overlapping components has no dip.
      "each" -- each class normalized to unit area instead; shapes are easier
        to compare but the curves no longer sum to the pooled density.
    """
    data, pooled, desc, label = _exit_split(cfg, schedules, runtime, name, lam)
    bw = cfg.get_path("plots.kde_bandwidth", 0.4)
    norm = str(cfg.get_path("plots.exit_split_normalize", "mixture")).lower()
    if norm not in ("mixture", "each"):
        raise ValueError("plots.exit_split_normalize must be 'mixture' or "
                         f"'each', got {norm!r}")
    show_pooled = bool(cfg.get_path("plots.exit_split_show_pooled", True))

    lo = float(pooled.min())
    hi = _exit_split_hi(cfg, pooled, name, xmax)
    grid = np.linspace(lo, hi, int(cfg.get_path("plots.kde_grid_points", 400)))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    pooled_density = _kde(pooled, grid, bw)
    _slo_fill_area(ax, slo_ms, grid, pooled_density)
    _slo_marks(ax, slo_ms, pooled=pooled, annotate=simple)
    if show_pooled and norm == "mixture":
        ax.plot(grid, pooled_density, color=EXIT_CLASS_COLORS["pooled"],
                linestyle=EXIT_CLASS_STYLES["pooled"]["linestyle"],
                linewidth=1.0, zorder=2,
                label="All" if simple else EXIT_CLASS_LABELS["pooled"])
    for c, l in data:
        if len(l) < 2:
            continue
        w = len(l) / len(pooled) if norm == "mixture" else 1.0
        ax.plot(grid, w * _kde(l, grid, bw), color=EXIT_CLASS_COLORS[c],
                linestyle=EXIT_CLASS_STYLES[c]["linestyle"],
                label=_exit_class_label(c, simple), zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title(_exit_split_title(label, desc, simple))
    ax.legend(loc="upper right")
    return fig


def _exit_split_hist_fig(cfg: Config, runtime: str, schedules: dict, name: str,
                         lam=None, slo_ms=None, xmax=None, simple=False):
    """Shared body of 13b/13d (and 14b/14d) — the same split, no smoothing.

    `plots.exit_split_hist_stacked` (default true): the classes are disjoint
    subsets of the same population, so stacking them reproduces the runtime's
    pooled histogram exactly and shows how the components fill in each other's
    gaps. Set false for an overlaid (alpha-blended) comparison of the two
    shapes instead. Bins/x-range follow plot12 (plots.hist_bins, the KDE
    x-clip); samples past the clip fall outside the edges and are dropped.
    """
    data, pooled, desc, label = _exit_split(cfg, schedules, runtime, name, lam)
    bins = int(cfg.get_path("plots.hist_bins", 80))
    density = bool(cfg.get_path("plots.hist_density", False))
    stacked = bool(cfg.get_path("plots.exit_split_hist_stacked", True))

    lo = float(pooled.min())
    hi = _exit_split_hi(cfg, pooled, name, xmax)
    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    _slo_marks(ax, slo_ms, pooled=pooled, annotate=simple)
    order = list(EXIT_CLASS_ORDER)
    by_class = dict(data)
    # "bar" gives one patch per bin, which _slo_color_bars needs; "stepfilled"
    # collapses each dataset into a single polygon.
    htype = "bar" if _slo_values(slo_ms) else "stepfilled"
    containers = []
    if stacked:
        _, _, containers = ax.hist(
            [by_class[c] for c in order], bins=edges, density=density,
            stacked=True, histtype=htype,
            color=[EXIT_CLASS_COLORS[c] for c in order],
            label=[_exit_class_label(c, simple) for c in order],
            edgecolor="white", linewidth=0.3)
        # per-class hatch (ax.hist takes no hatch list) — keeps the stack
        # readable in grayscale. plot14 drops it: the green/amber pair already
        # separates by luminance and the texture only adds clutter there.
        for c, cont in zip(order, containers):
            h = None if simple else EXIT_CLASS_STYLES[c]["hatch"]
            if h:
                for patch in cont:
                    patch.set_hatch(h)
    else:
        for c, l in data:
            _, _, cont = ax.hist(
                l, bins=edges, density=density, histtype=htype,
                alpha=0.35, color=EXIT_CLASS_COLORS[c],
                edgecolor=EXIT_CLASS_COLORS[c], linewidth=1.1,
                hatch=None if simple else EXIT_CLASS_STYLES[c]["hatch"],
                label=_exit_class_label(c, simple))
            containers.append(cont)
    _slo_color_bars(containers, edges, slo_ms)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(_exit_split_title(label, desc, simple))
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
# Plot 14a–14d: 13a–13d again, but pinned to ONE config-given λ and annotated
# with the SLO deadline.
#
# 13a–13d put each runtime at its OWN capacity×margin λ, which is right for
# "what does this runtime's latency look like when pushed near its ceiling" but
# makes naive and proposed non-comparable — they are measured at different
# offered loads. 14a–14d replay BOTH at the single rate in
# `plots.exit_split_common_lambda`, so 14a/14b and 14c/14d sit side by side at
# iso-load and the only difference left is the batching structure. The red
# vertical line(s) from `plots.exit_split_slo_ms` mark the deadline, making the
# SLO-violating mass readable straight off the figure — which is the whole
# argument for proposed: its non-exit component may sit further right, but the
# question that matters is how much of it crosses the line.
#
# λ is NEVER auto-derived here (that is 13's job): pick a rate below the
# smallest capacity among the compared runtimes — see Table A / plot4 — and
# state it, and the SLO, in the caption. Unset -> all four are skipped.
#
# The x-axis is also pinned (`plots.exit_split_xlim_ms`, default 100 ms) rather
# than clipped at a percentile the way plot13 is: all four panels must share
# one scale to be comparable, and the tail past the SLO is the part being
# argued about, so cutting it at p99 would hide the evidence.
# --------------------------------------------------------------------------- #
def _iso_exit_split_lambda(cfg: Config, schedules: dict, name: str):
    """The shared λ for plot14 (14a-14g), or None when unset (caller skips).

    `plots.exit_split_common_lambda`:
      null   -> skip every plot14 figure
      "auto" -> naive's capacity minus one λ-sweep step, snapped to the grid
                — the SAME "last stable load" formula e2e_table.py's Table B
                uses for naive's own reference λ (λ2), computed here against
                the plain/naive/proposed@seg2_batch common set so plot14 and
                its companion Table F (e2e_table.py) read the identical λ.
      number -> fixed rate (manual override); 0 = saturated.
    """
    raw = cfg.get_path("plots.exit_split_common_lambda", None)
    if raw is None:
        print(f"[{name}] plots.exit_split_common_lambda unset; skipped")
        return None
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        B = int(cfg.batching.seg2_batch)
        common = metrics.common_completed(
            [schedules["plain"], schedules["naive"], schedules["proposed"][B]])
        lams = lambda_grid(cfg)
        step = float(cfg.arrivals["lambda_sweep"]["step"])
        lam = metrics.capacity_step_lambda(schedules["naive"], common, lams, step)
        print(f"[{name}] exit_split_common_lambda=auto -> λ={lam:g} req/s "
              f"(naive capacity − 1 sweep step)")
        return lam
    return float(raw)


def _iso_exit_split_slo(cfg: Config):
    """`plots.exit_split_slo_ms` — a number, a list of numbers, or None."""
    return cfg.get_path("plots.exit_split_slo_ms", None)


def _iso_exit_split_fig(cfg: Config, runtime: str, schedules: dict, name: str,
                        out: str, kind: str):
    lam = _iso_exit_split_lambda(cfg, schedules, name)
    if lam is None:
        return None
    slo = _iso_exit_split_slo(cfg)
    xmax = cfg.get_path("plots.exit_split_xlim_ms", 100)
    build = _exit_split_kde_fig if kind == "kde" else _exit_split_hist_fig
    fig = build(cfg, runtime, schedules, name, lam=lam, slo_ms=slo, xmax=xmax,
                simple=True)
    if slo is not None:
        # how much of each class misses the deadline — the number these figures
        # are drawn to support, so print it for the caption
        *_, pooled, is_exit = _exit_split_raw(cfg, schedules, runtime, lam)
        values = slo if isinstance(slo, (list, tuple)) else [slo]
        for v in sorted(float(x) for x in values):
            print(f"[{name}] SLO {v:g} ms violated by: "
                  f"all {100 * (pooled > v).mean():.2f}%, "
                  f"exit {100 * (pooled[is_exit] > v).mean():.2f}%, "
                  f"non-exit {100 * (pooled[~is_exit] > v).mean():.2f}%")
    return _save(fig, cfg, out)


def plot_naive_exit_kde_iso(cfg: Config, schedules: dict):
    """Plot 14a: naive latency KDE by exit class, at the shared λ + SLO line."""
    return _iso_exit_split_fig(cfg, "naive", schedules, "plot14a",
                               "plot14a_naive_exit_kde_iso", "kde")


def plot_naive_exit_hist_iso(cfg: Config, schedules: dict):
    """Plot 14b: naive latency histogram by exit class, shared λ + SLO line."""
    return _iso_exit_split_fig(cfg, "naive", schedules, "plot14b",
                               "plot14b_naive_exit_hist_iso", "hist")


def plot_proposed_exit_kde_iso(cfg: Config, schedules: dict):
    """Plot 14c: proposed@seg2_batch latency KDE by exit class, shared λ."""
    return _iso_exit_split_fig(cfg, "proposed", schedules, "plot14c",
                               "plot14c_proposed_exit_kde_iso", "kde")


def plot_proposed_exit_hist_iso(cfg: Config, schedules: dict):
    """Plot 14d: proposed@seg2_batch latency histogram by exit class, shared λ."""
    return _iso_exit_split_fig(cfg, "proposed", schedules, "plot14d",
                               "plot14d_proposed_exit_hist_iso", "hist")


# --------------------------------------------------------------------------- #
# Plot 13e/13f: the SAME histogram, but every bar split by where its samples
# actually spent their time.
#
# 13b/13d answer "how many samples land at this latency"; these answer "and WHY
# are they there". For each bin we take the samples in it, average each latency
# component over them, and cut the bar (whose height is still the bin count)
# into those proportions — so bar height still traces the distribution while the
# colors show composition. Reading left to right along x is then a direct
# picture of what turns a fast sample into a slow one: for naive the growing
# band is formation + GPU wait (seg2 compute stays a thin constant sliver, which
# is exactly why 13a/13b show no split), while for proposed the non-exit panel's
# right half is dominated by seg2 queue wait.
#
# Note `gpu_wait` is included even though it is not one of the "interesting"
# terms: the five components are additive and must sum to the latency, so
# dropping it would make the bars lie.
# --------------------------------------------------------------------------- #
def _composition_heights(lat: np.ndarray, comps: dict, edges: np.ndarray):
    """Per-bin stacked heights (counts, split by component share).

    For bin b: height_k[b] = n[b] * mean_k[b] / mean_total[b]
                           = n[b] * sum_k[b] / sum_total[b]
    so the segments of a bar sum to the bin's sample count, and their ratio is
    the mean composition of that bin's latency. Samples outside `edges` (the
    KDE x-clip tail) are dropped, exactly as in 13b/13d.
    Returns (centers, width, {component: heights}, counts).
    """
    nb = len(edges) - 1
    idx = np.digitize(lat, edges) - 1
    inside = (idx >= 0) & (idx < nb)
    idx = idx[inside]
    counts = np.bincount(idx, minlength=nb).astype(float)
    sums = {k: np.bincount(idx, weights=v[inside], minlength=nb)
            for k, v in comps.items()}
    total = np.sum(list(sums.values()), axis=0)
    safe = np.where(total > 0, total, 1.0)
    heights = {k: counts * s / safe for k, s in sums.items()}
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, float(edges[1] - edges[0]), heights, counts


def _composition_panel(ax, lat, comps, edges, title):
    centers, width, heights, counts = _composition_heights(lat, comps, edges)
    bottom = np.zeros(len(centers))
    for k in BREAKDOWN_KEYS:                     # stacked in pipeline order
        h = heights[k]
        if not np.any(h > 0):
            continue                             # e.g. naive's seg2 queue wait
        ax.bar(centers, h, width=width, bottom=bottom, align="center",
               color=COMPONENT_COLORS[k], linewidth=0)
        bottom += h
    ax.set_xlim(edges[0], edges[-1])
    ax.set_title(title)
    return counts


def _composition_pair(cfg: Config, runtime: str, schedules: dict, name: str,
                      axes, lam=None, slo_ms=None, xmax=None, iso=False,
                      title_prefix: str = "", lo=None, hi=None):
    """Draws the exit | non-exit composition panels for ONE runtime onto the
    given pair of axes. Shared by `_latency_composition_fig` (13e/13f,
    14e/14f — one runtime per figure) and the combined plot14g (both runtimes
    in one 4-panel figure, where `title_prefix` disambiguates which pair is
    which).

    `plots.composition_bins` (default 40) is deliberately coarser than
    `hist_bins`: 80 thin bars cut into five colors is unreadable at print
    size. Skipped in saturated mode, where latency is measured from the
    stage-1 op start and therefore no longer equals the sum of the
    arrival-referenced components.

    `lo`/`hi` override the bin-edge bounds — plot14g precomputes ONE shared
    (lo, hi) across both runtimes before drawing either pair, since its 4
    axes share x-limits (`sharex=True`): without a common bound, the
    second-drawn pair would silently override the first's view window.
    Defaults (None) fall back to this runtime's own pooled min / `_exit_split_hi`,
    matching the single-runtime figures.

    Returns (label, short) for the caller's title, or None if skipped.
    """
    (sched, label, common, arr, origin, desc, short, pooled,
     is_exit) = _exit_split_raw(cfg, schedules, runtime, lam)
    if origin != "arrival":
        print(f"[{name}] {label}: λ is saturated; the additive decomposition "
              f"is not defined against a stage-1-start clock — skipped")
        return None

    bd = simulate_breakdown(sched, arr)
    comps = {k: bd[k][common] * 1000.0 for k in BREAKDOWN_KEYS}
    bins = int(cfg.get_path("plots.composition_bins", 40))
    lo = float(pooled.min()) if lo is None else float(lo)
    hi = _exit_split_hi(cfg, pooled, name, xmax) if hi is None else float(hi)
    edges = np.linspace(lo, hi, bins + 1)

    print(f"[{name}] {label} at {desc}; {bins} bins over {lo:.1f}–{hi:.1f} ms")
    for ax, (cls, sel) in zip(axes, [("exit", is_exit), ("nonexit", ~is_exit)]):
        sub = {k: v[sel] for k, v in comps.items()}
        cls_title = (EXIT_CLASS_LABELS_SHORT[cls] if iso else
                    f"{EXIT_CLASS_LABELS[cls]} (n={int(sel.sum())})")
        _composition_panel(ax, pooled[sel], sub, edges, title_prefix + cls_title)
        _slo_marks(ax, slo_ms, pooled=pooled[sel], annotate=iso,
                  show_count=iso)
        means = {k: float(v.mean()) for k, v in sub.items()}
        tot = sum(means.values()) or 1.0
        print(f"[{name}]   {cls:8s} mean latency {tot:6.2f} ms = " + ", ".join(
            f"{COMPONENT_LABELS[k]} {means[k]:.2f} ({100 * means[k] / tot:.0f}%)"
            for k in BREAKDOWN_KEYS if means[k] > 0))
    return label, short


def _latency_composition_fig(cfg: Config, runtime: str, schedules: dict,
                             name: str, out: str, lam=None, slo_ms=None,
                             xmax=None, iso=False):
    """Shared body of 13e/13f (and 14e/14f, which pass lam/slo_ms/xmax): two
    panels (exit | non-exit) for ONE runtime, via `_composition_pair`.

    `iso=True` (14e/14f) keeps the title to a bare runtime name — the shared
    λ / SLO operating point is already fixed by config and not worth stating
    on every panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharex=True, sharey=True)
    result = _composition_pair(cfg, runtime, schedules, name, axes, lam=lam,
                               slo_ms=slo_ms, xmax=xmax, iso=iso)
    if result is None:
        plt.close(fig)
        return None
    label, short = result
    axes[0].set_ylabel("Count")
    axes[len(axes) // 2].set_xlabel("Latency (ms)")
    title = (f"{RUNTIME_LABELS[runtime]} Latency Decomposition" if iso
             else f"{label} Latency Composition by Bin ({short})")
    fig.suptitle(title)
    _component_legend(fig, slo_ms)
    return _save(fig, cfg, out)


def plot_naive_latency_composition(cfg: Config, schedules: dict):
    """Plot 13e: naive — per-bin latency composition, exit | non-exit panels."""
    return _latency_composition_fig(cfg, "naive", schedules, "plot13e",
                                    "plot13e_naive_latency_composition")


def plot_proposed_latency_composition(cfg: Config, schedules: dict):
    """Plot 13f: proposed@seg2_batch — per-bin latency composition."""
    return _latency_composition_fig(cfg, "proposed", schedules, "plot13f",
                                    "plot13f_proposed_latency_composition")


def _iso_latency_composition_fig(cfg: Config, runtime: str, schedules: dict,
                                 name: str, out: str):
    """13e/13f pinned to plot14's shared λ, fixed x-axis and SLO line."""
    lam = _iso_exit_split_lambda(cfg, schedules, name)
    if lam is None:
        return None
    return _latency_composition_fig(
        cfg, runtime, schedules, name, out, lam=lam,
        slo_ms=_iso_exit_split_slo(cfg),
        xmax=cfg.get_path("plots.exit_split_xlim_ms", 100), iso=True)


def plot_naive_latency_composition_iso(cfg: Config, schedules: dict):
    """Plot 14e: naive per-bin latency composition, shared λ + SLO line."""
    return _iso_latency_composition_fig(
        cfg, "naive", schedules, "plot14e",
        "plot14e_naive_latency_composition_iso")


def plot_proposed_latency_composition_iso(cfg: Config, schedules: dict):
    """Plot 14f: proposed@seg2_batch per-bin composition, shared λ + SLO."""
    return _iso_latency_composition_fig(
        cfg, "proposed", schedules, "plot14f",
        "plot14f_proposed_latency_composition_iso")


def plot_latency_composition_iso_combined(cfg: Config, schedules: dict):
    """Plot 14g: 14e (naive) and 14f (GATE) side by side as ONE 4-panel
    figure — naive-exit | naive-nonexit | GATE-exit | GATE-nonexit — for a
    single glance at all four distributions at plot14's shared λ/SLO
    operating point. Panel titles carry the runtime tag (14e/14f standalone
    keep the bare "Exit"/"Non-exit" titles since the figure title alone names
    the runtime there)."""
    name = "plot14g"
    lam = _iso_exit_split_lambda(cfg, schedules, name)
    if lam is None:
        return None
    slo_ms = _iso_exit_split_slo(cfg)
    xmax = cfg.get_path("plots.exit_split_xlim_ms", 100)

    # Precompute both runtimes' pooled latency once so all 4 panels share
    # identical bin edges — the figure's sharex=True ties every panel's view
    # limits together, so without a common (lo, hi) the second-drawn pair
    # would silently override the first's window.
    pooled_by_runtime = {}
    for runtime in ("naive", "proposed"):
        raw = _exit_split_raw(cfg, schedules, runtime, lam)
        if raw[4] != "arrival":
            print(f"[{name}] {raw[1]}: λ is saturated; the additive "
                  f"decomposition is not defined against a stage-1-start "
                  f"clock — skipped")
            return None
        pooled_by_runtime[runtime] = raw[7]
    lo = min(float(p.min()) for p in pooled_by_runtime.values())
    hi = max(_exit_split_hi(cfg, p, name, xmax) for p in pooled_by_runtime.values())

    # sharey only WITHIN each runtime's pair (matching 14e/14f, each its own
    # figure with its own y-scale) — NOT across naive/GATE: their bin counts
    # can differ enough in magnitude that a figure-wide shared y-axis flattens
    # the smaller runtime's bars into near-invisibility.
    fig, axes = plt.subplots(1, 4, figsize=FIG_QUAD, sharex=True, sharey=False)
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])
    for runtime, sub_axes in (("naive", axes[:2]), ("proposed", axes[2:])):
        result = _composition_pair(
            cfg, runtime, schedules, name, sub_axes, lam=lam, slo_ms=slo_ms,
            xmax=xmax, iso=True, title_prefix=f"{RUNTIME_LABELS[runtime]} – ",
            lo=lo, hi=hi)
        if result is None:
            plt.close(fig)
            return None
    axes[0].set_ylabel("Count")
    axes[2].set_ylabel("Count")
    axes[len(axes) // 2].set_xlabel("Latency (ms)")
    fig.suptitle(f"{RUNTIME_LABELS['naive']}/{RUNTIME_LABELS['proposed']} "
                f"Latency Decomposition")
    _component_legend(fig, slo_ms)
    return _save(fig, cfg, "plot14g_latency_composition_combined")


# --------------------------------------------------------------------------- #
# Plot 4: load vs latency (+ divergence detection)
# --------------------------------------------------------------------------- #
def _curve_crossing(lams: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Where curve `a` overtakes curve `b` over the λ grid: the LAST sign
    change of (a-b), linearly interpolated between the straddling grid
    points. The last (not first) crossing is picked so incidental finite-N
    noise near the low-λ end — where both curves sit close together — is not
    mistaken for the real crossover near the runtimes' divergence points.
    Returns (lam_x, y_x), or None if the curves never cross.
    """
    diff = a - b
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        return None
    i = int(idx[-1])
    d0, d1 = diff[i], diff[i + 1]
    t = d0 / (d0 - d1)
    lam_x = lams[i] + t * (lams[i + 1] - lams[i])
    y_x = a[i] + t * (a[i + 1] - a[i])
    return float(lam_x), float(y_x)


def _load_latency_fig(cfg: Config, schedules: dict, name: str, out: str,
                      mark_crossover: bool = False):
    """Shared body of plot4/plot4b: Load (lambda) vs mean response time per
    runtime.

    Also reports each runtime's divergence point — its service capacity
    (saturated throughput; arrival rates above it make the queue grow without
    bound) — plus the knee (latency minimum) of the sweep curve for reference,
    and returns the capacity-based divergence λ as a dict.

    `mark_crossover=True` (plot4b) additionally marks the λ where naive's
    mean latency overtakes GATE's (`_curve_crossing`) — otherwise identical
    to plot4.
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
        print(f"[{name}] {r}: divergence λ (capacity) = {divergence[r]:.1f} req/s"
              f" | knee (latency minimum) λ = {knee_s}")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for r, _ in entries:
        ax.plot(lams, means[r], color=RUNTIME_COLORS[r],
                linestyle=RUNTIME_STYLES[r]["linestyle"], label=RUNTIME_LABELS[r])
    if mark_crossover:
        cross = _curve_crossing(lams, means["naive"], means["proposed"])
        if cross is None:
            print(f"[{name}] naive/GATE mean-latency curves never cross over "
                  f"the swept λ range")
        else:
            lam_x, y_x = cross
            ax.plot([lam_x], [y_x], marker="x", color="black", markersize=6,
                    markeredgewidth=1.5, zorder=5, label="Naive/GATE crossover")
            ax.annotate(f"λ={lam_x:.0f}", (lam_x, y_x),
                       textcoords="offset points", xytext=(4, 6), fontsize=6)
            print(f"[{name}] naive/GATE mean-latency crossover at "
                  f"λ={lam_x:.1f} req/s, latency={y_x:.2f} ms")
    ax.set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Load vs Latency")
    ax.legend(loc="upper left")
    _save(fig, cfg, out)
    return divergence


def plot_load_latency(cfg: Config, schedules: dict):
    """Plot 4: Load (lambda) vs response time (mean + p99) per runtime."""
    return _load_latency_fig(cfg, schedules, "plot4", "plot4_load_latency")


def plot_load_latency_crossover(cfg: Config, schedules: dict):
    """Plot 4b: plot4, identical, plus the naive/GATE mean-latency crossover
    point marked — the λ past which GATE's decoupled batching overtakes
    naive's per-batch seg2 dispatch on mean response time."""
    return _load_latency_fig(cfg, schedules, "plot4b",
                             "plot4b_load_latency_crossover",
                             mark_crossover=True)


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


def _component_legend(fig, slo_ms=None):
    """Shared component legend. `slo_ms` (plot14e/14f) appends the deadline
    line so the red rule in the panels is explained."""
    handles = [Patch(facecolor=COMPONENT_COLORS[k],
                     label=COMPONENT_LEGEND_LABELS[k])
               for k in BREAKDOWN_KEYS]
    if slo_ms is not None:
        handles.append(Line2D([], [], color=SLO_COLOR, linewidth=1.1,
                              label="SLO"))
    fig.legend(handles=handles, ncol=len(handles), loc="outside lower center")


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
    ps.configure(cfg)
    plot_slo_goodput(cfg, schedules)
    plot_latency_kde(cfg, schedules)
    plot_latency_kde_per_runtime(cfg, schedules)
    plot_latency_kde_sweep(cfg, schedules)
    plot_latency_cdf(cfg, schedules)
    plot_latency_cdf_sweep(cfg, schedules)
    plot_latency_cdf_common(cfg, schedules)
    plot_latency_cdf_per_runtime(cfg, schedules)
    plot_latency_hist_common(cfg, schedules)
    plot_latency_hist(cfg, schedules)
    plot_naive_exit_kde(cfg, schedules)
    plot_naive_exit_hist(cfg, schedules)
    plot_proposed_exit_kde(cfg, schedules)
    plot_proposed_exit_hist(cfg, schedules)
    plot_naive_latency_composition(cfg, schedules)
    plot_proposed_latency_composition(cfg, schedules)
    plot_naive_exit_kde_iso(cfg, schedules)
    plot_naive_exit_hist_iso(cfg, schedules)
    plot_proposed_exit_kde_iso(cfg, schedules)
    plot_proposed_exit_hist_iso(cfg, schedules)
    plot_naive_latency_composition_iso(cfg, schedules)
    plot_proposed_latency_composition_iso(cfg, schedules)
    plot_latency_composition_iso_combined(cfg, schedules)
    divergence = plot_load_latency(cfg, schedules)
    plot_load_latency_crossover(cfg, schedules)
    plot_latency_breakdown(cfg, schedules)
    plot_timeline(cfg, schedules)
    plot_stage_time_bars(cfg, schedules)
    plot_exec_stats(cfg, schedules)
    plot_naive_seg2_sizes(cfg, schedules)
    return divergence
