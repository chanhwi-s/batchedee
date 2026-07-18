"""End-to-end comparison tables (Table A / Table B) for the paper.

Everything is derived from the measured schedules + cheap event-driven replay
(no GPU work here). Inputs: schedules.pkl produced by `run.py run` (which also
stores the per-sample correctness arrays for accuracy).

Table A (λ-independent, one row per runtime):
  accuracy (%), saturated throughput (samples/s, λ=0 backlog drain), and the
  divergence λ. Divergence is CAPACITY-based: the service capacity (=saturated
  throughput, req/s) above which the queue grows without bound and latency
  diverges. The sweep-curve knee (latency minimum) is kept in meta as
  reference only — it is the sweet spot, not the instability point.

Table B (runtime × λ grid): three λ values derived deterministically from the
capacity-based divergence points — each runtime's own last stable load:
  λ1 = D_plain − step     (below every capacity: all three stable)
  λ2 = D_naive − step     (above D_plain: plain overloaded; naive at its own
                           ceiling, proposed still stable)
  λ3 = D_proposed − step  (above D_naive too: only proposed is stable)
Collisions collapse to the distinct achievable subset (recorded in meta).
The two SLOs are plain's mean / p99 response time at λ1, rounded to the
nearest 10 ms, then held fixed across all rows.

Table C (optional): same columns and the SAME fixed SLOs as Table B, but on
the user-chosen λ values from `plots.slo_goodput_lambda` (the per-figure λ of
plot1a/1b) — so the table matches the operating points shown in the SLO-vs-
goodput figures. Non-positive (saturated) entries are skipped.

Table D (knee operating points): for plain, naive, and proposed at EVERY
seg2_batch in the sweep, the λ minimizing mean response time (the knee of the
load curve) plus the mean/p99 latency at that point and the capacity-based
divergence λ. Computed on the common set over ALL configurations. plot2b
draws each latency distribution at these knee λ values.

Peak goodput (plot1c + peak_goodput.{json,csv}): for each runtime and each of
the two fixed SLOs, goodput is scanned over the whole lambda_sweep grid and
the maximum (with its argmax λ) is reported — each runtime at its OWN
goodput-maximizing operating point, complementing the fixed-λ Table B view.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np

from . import metrics
from .arrivals import poisson_arrivals
from .runtimes import simulate
from .util import Config, lambda_grid

RUNTIMES = ("plain", "naive", "proposed")


def _snap(lams: np.ndarray, x: float) -> float:
    """Nearest point on the sweep grid."""
    return float(lams[int(np.argmin(np.abs(lams - x)))])


def _round10(x: float) -> int:
    return int(round(x / 10.0) * 10)



def _check(checks: list, name: str, ok: bool, detail: str):
    status = "PASS" if ok else "FAIL"
    print(f"[e2e] {status}  {name}: {detail}")
    checks.append({"name": name, "status": status, "detail": detail})


# --------------------------------------------------------------------------- #
def _peak_goodput(cfg: Config, entries: dict, common, div: dict,
                  slo_values: list[int], seed: int, lams: np.ndarray):
    """Peak-goodput scan: for each (runtime, SLO), goodput over the whole λ
    grid via the existing replay path; report the max and its argmax λ.
    Returns (rows, per-runtime goodput curves)."""
    n = next(iter(entries.values())).n_requests
    slo_grid = np.array(slo_values, dtype=float)
    curves = {r: np.empty((len(lams), len(slo_values))) for r in entries}
    print(f"[peak] scanning goodput over {len(lams)} λ points "
          f"({lams[0]:g}–{lams[-1]:g}) ...")
    for j, lam in enumerate(lams):
        arr = poisson_arrivals(n, float(lam), seed)
        for r, s in entries.items():
            curves[r][j] = metrics.goodput_vs_slo(s, arr, common, slo_grid,
                                                  "wallclock")

    rows = []
    for si, slo in enumerate(slo_values):
        for r in entries:
            c = curves[r][:, si]
            i = int(np.argmax(c))
            lam_star, peak = float(lams[i]), float(c[i])
            rows.append({"slo_ms": int(slo), "runtime": r,
                         "argmax_lambda": lam_star,
                         "peak_goodput_sps": round(peak, 1),
                         "capacity_lambda": round(div[r], 1),
                         "diverged_at_argmax": bool(lam_star >= div[r])})
            # sanity: argmax within capacity / not on the sweep boundary
            if lam_star >= div[r]:
                print(f"[peak] WARN  ({r}, SLO={slo}ms): argmax λ {lam_star:g} "
                      f"exceeds capacity {div[r]:.1f}")
            if i in (0, len(lams) - 1):
                print(f"[peak] WARN  ({r}, SLO={slo}ms): argmax λ {lam_star:g} "
                      f"sits on the sweep boundary; peak may not be captured")
            # sanity: unimodal (rise -> peak -> collapse); tolerate <2% ripple
            post = c[i:]
            pre = c[:i + 1]
            tol = 0.02 * peak
            if len(post) > 1 and np.any(np.diff(post) > tol):
                print(f"[peak] WARN  ({r}, SLO={slo}ms): goodput re-rises after "
                      f"the peak by >2% — curve not unimodal")
            if len(pre) > 1 and np.any(np.diff(pre) < -tol):
                print(f"[peak] WARN  ({r}, SLO={slo}ms): goodput dips before "
                      f"the peak by >2% — curve not unimodal")
    return rows, curves


def _peak_goodput_figure(cfg: Config, rows: list[dict], slo_values: list[int],
                         bs2_values: list[int]):
    from . import plot_style as ps
    from .plots import _save
    import matplotlib.pyplot as plt

    labels_order = ["plain", "naive"] + [f"proposed(bs2={b})" for b in bs2_values]
    shades = ps.proposed_shades(len(bs2_values))
    colors = {"plain": ps.RUNTIME_COLORS["plain"], "naive": ps.RUNTIME_COLORS["naive"],
              **{f"proposed(bs2={b})": c for b, c in zip(bs2_values, shades)}}
    legend_labels = {"plain": ps.RUNTIME_LABELS["plain"],
                     "naive": ps.RUNTIME_LABELS["naive"],
                     **{f"proposed(bs2={b})": ps.b2_label(b) for b in bs2_values}}

    n_slo = len(slo_values)
    n_bars = len(labels_order)
    x = np.arange(n_slo)
    w = 0.6 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * w
    # Wide enough that rotated value labels (one per bar) don't collide;
    # exact argmax λ per bar lives in peak_goodput.{json,csv}, not the figure.
    fig_w = max(ps.FIG_DOUBLE[0], 0.32 * n_slo * n_bars)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))
    for k, label in enumerate(labels_order):
        vals = []
        for slo in slo_values:
            row = next(q for q in rows
                       if q["runtime"] == label and q["slo_ms"] == slo)
            vals.append(row["peak_goodput_sps"])
        bars = ax.bar(x + offsets[k], vals, w, color=colors[label],
                      label=legend_labels[label], edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in vals],
                     fontsize=4.6, padding=1, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in slo_values])
    ax.set_xlabel("SLO (ms)")
    ax.set_ylabel("Goodput (completions/s)")
    ax.set_title("Peak Goodput")
    ax.margins(y=0.6)
    ax.legend(ncol=2, loc="upper left", fontsize=6)
    return _save(fig, cfg, "plot1c_peak_goodput_bars")


def generate(cfg: Config, scheds: dict) -> dict:
    B = int(cfg.batching.seg2_batch)
    entries = {"plain": scheds["plain"], "naive": scheds["naive"],
               "proposed": scheds["proposed"][B]}
    # One common set for the whole function (plain, naive, and EVERY bs2
    # sweep config) so every table measures capacity — and everything
    # downstream of it — on the identical sample set. Table A/B/C still only
    # REPORT the 3 headline configs (`entries`); Table D/E report the full
    # bs2 sweep (`configs`, below) — only the output scope differs, not the
    # measurement basis.
    all_prop = scheds["proposed"]
    common = common_all = metrics.common_completed(
        [scheds["plain"], scheds["naive"], *all_prop.values()])
    n = entries["plain"].n_requests
    seed = int(cfg.arrivals.seed)
    lams = lambda_grid(cfg)
    step = float(cfg.arrivals["lambda_sweep"]["step"])
    notes: list[str] = []
    checks: list[dict] = []

    # ---- divergence λ per runtime = service capacity (saturated throughput);
    #      the sweep-curve knee (latency minimum) is kept as reference only ----
    div, knee = {}, {}
    for r, s in entries.items():
        div[r] = metrics.capacity_lambda(s, common)
        m, _p = metrics.load_latency_curves(s, lams, common, seed)
        knee[r] = metrics.knee_lambda(lams, m)

    # ---- Table A: accuracy ----
    corr = scheds.get("correct")
    if corr is None:
        raise SystemExit(
            "[e2e] schedules.pkl has no per-sample correctness arrays; "
            "re-run `python run.py run` (the accuracy pass stores them) first.")
    acc = {"plain": 100.0 * float(corr["plain"][common].mean()),
           "naive": 100.0 * float(corr["ee"][common].mean()),
           "proposed": 100.0 * float(corr["ee"][common].mean())}

    # ---- Table A: saturated throughput == capacity-based divergence λ ----
    sat = dict(div)      # identical by definition (samples/s vs req/s)

    table_a = [{"runtime": r,
                "accuracy_pct": round(acc[r], 2),
                "saturated_throughput_sps": round(sat[r], 1),
                "divergence_lambda": round(div[r], 1)} for r in RUNTIMES]

    # ---- Table B: deterministic λ1/λ2/λ3 — each runtime's OWN last stable
    #      load (capacity − step), same convention for all three ----
    D = {r: div[r] for r in RUNTIMES}
    raw = {"lambda1": _snap(lams, D["plain"] - step),
           "lambda2": _snap(lams, D["naive"] - step),
           "lambda3": _snap(lams, D["proposed"] - step)}
    if raw["lambda1"] < lams[0]:
        raw["lambda1"] = float(lams[0])
        notes.append("λ1 clamped to sweep start")

    chosen: list[float] = []
    for name in ("lambda1", "lambda2", "lambda3"):
        v = raw[name]
        if chosen and v <= chosen[-1]:
            notes.append(f"{name}={v:g} does not exceed the previous λ "
                         f"({chosen[-1]:g}); collapsed")
            continue
        chosen.append(v)

    # ---- SLOs: plain's response time at λ1, rounded to nearest 10 ms ----
    arr1 = poisson_arrivals(n, chosen[0], seed)
    raw_mean, raw_p99 = metrics.response_stats(entries["plain"], arr1, common)
    slo_avg, slo_p99 = _round10(raw_mean), _round10(raw_p99)

    # ---- Table B / C rows (same columns, same fixed SLOs) ----
    slo_grid = np.array([slo_avg, slo_p99], dtype=float)
    throughputs = {}                      # (runtime, λ) -> completed throughput

    def _rows_at(lam_values: list[float]) -> list[dict]:
        rows = []
        for lam in lam_values:
            arr = poisson_arrivals(n, lam, seed)
            for r, s in entries.items():
                mean, p99 = metrics.response_stats(s, arr, common)
                g = metrics.goodput_vs_slo(s, arr, common, slo_grid, "wallclock")
                completion, _ = simulate(s, arr)
                throughputs[(r, lam)] = len(common) / metrics.wallclock(completion, arr, common)
                rows.append({"runtime": r, "lambda": lam,
                             "avg_ms": round(mean, 2), "p99_ms": round(p99, 2),
                             "goodput_slo_avg": round(float(g[0]), 1),
                             "goodput_slo_p99": round(float(g[1]), 1),
                             "diverged": bool(lam >= div[r])})   # λ ≥ capacity
        return rows

    table_b = _rows_at(chosen)

    # ---- Table C: user-configured λ values (plots.slo_goodput_lambda) ----
    user_map = dict(cfg.get_path("plots.slo_goodput_lambda", {}) or {})
    user_lams = sorted({float(v) for v in user_map.values() if float(v) > 0})
    table_c = _rows_at(user_lams)

    # ---- Table D: knee operating point per configuration (incl. bs2 sweep) --
    configs = ([("plain", scheds["plain"]), ("naive", scheds["naive"])]
               + [(f"proposed(bs2={b})", all_prop[b]) for b in sorted(all_prop)])
    # ---- Table E: EVERY configuration replayed AT its own measured capacity
    #      (not capacity − step) — mean/p99 response time and completed
    #      throughput right at the boundary. Capacity is a near-critical
    #      (ρ→1) point; a finite-N Poisson replay exactly there can already
    #      show elevated/noisy latency (this is why plot1a/1b/Table B and the
    #      capacity-anchored plots all back off by one sweep step instead) —
    #      these numbers describe the boundary itself, not a safe operating
    #      point. Flag any row where measured throughput undershoots the
    #      nominal capacity by more than 5%, a sign this replay is already
    #      queue-unstable within N.
    table_d = []
    table_e = []
    for label, s in configs:
        k, mean, p99, edge = metrics.knee_stats(s, lams, common_all, seed)
        cap = metrics.capacity_lambda(s, common_all)
        if edge:
            notes.append(f"Table D: {label} knee sits on the sweep edge "
                         f"({k:g}); extend lambda_sweep")
        table_d.append({"config": label, "knee_lambda": k,
                        "mean_ms": round(mean, 2), "p99_ms": round(p99, 2),
                        "divergence_lambda": round(cap, 1)})

        arr_cap = poisson_arrivals(s.n_requests, cap, seed)
        mean_cap, p99_cap = metrics.response_stats(s, arr_cap, common_all)
        completion_cap, _ = simulate(s, arr_cap)
        thr_cap = len(common_all) / metrics.wallclock(completion_cap, arr_cap, common_all)
        rel = abs(thr_cap - cap) / cap
        if rel > 0.05:
            notes.append(f"Table E: {label} measured throughput at capacity "
                         f"({thr_cap:.1f}) is {100*rel:.1f}% off nominal "
                         f"capacity ({cap:.1f}) — likely already unstable "
                         f"within N at this λ")
        # reference λ = capacity − 1 sweep step, grid-snapped — the "last
        # stable load" convention Table B/plot1a/1b actually use downstream
        # (Table E itself sits AT capacity; this column shows where that
        # boundary maps to once backed off to a safe operating point).
        ref_lambda = _snap(lams, cap - step)
        table_e.append({"config": label, "capacity_lambda": round(cap, 1),
                        "mean_ms": round(mean_cap, 2), "p99_ms": round(p99_cap, 2),
                        "throughput_sps": round(thr_cap, 1),
                        "reference_lambda": ref_lambda})

    # ---- sanity checks ----
    lam1 = chosen[0]
    g_plain = next(row for row in table_b
                   if row["runtime"] == "plain" and row["lambda"] == lam1)
    # by construction of SLO_p99, ~99% of plain's completions fit the SLO, so
    # goodput ≈ 0.99 × its completed throughput at λ1 (not 0.99·λ1, which
    # drifts when λ1 sits close to plain's capacity).
    target = 0.99 * throughputs[("plain", lam1)]
    rel = abs(g_plain["goodput_slo_p99"] - target) / target
    _check(checks, "plain goodput@SLO_p99 ≈ 0.99·throughput(λ1)", rel <= 0.05,
           f"got {g_plain['goodput_slo_p99']:g} vs 0.99·thr={target:.1f} "
           f"(λ1={lam1:g}; rel. diff {100*rel:.1f}%)")

    for row in table_b:
        if row["diverged"]:
            continue
        thr = throughputs[(row["runtime"], row["lambda"])]
        rel = abs(thr - row["lambda"]) / row["lambda"]
        _check(checks, f"throughput≈λ ({row['runtime']}, λ={row['lambda']:g})",
               rel <= 0.05, f"completed throughput {thr:.1f} vs λ={row['lambda']:g} "
                            f"(rel. diff {100*rel:.1f}%)")

    _check(checks, "naive accuracy == proposed accuracy",
           acc["naive"] == acc["proposed"],
           f"naive={acc['naive']:.2f}%, proposed={acc['proposed']:.2f}%")

    drop_pct = 100.0 * (n - len(common)) / n
    _check(checks, "common set close to N", drop_pct <= 5.0,
           f"common={len(common)} of N={n} ({drop_pct:.2f}% dropped)")

    # ---- meta + outputs ----
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "N": n, "seed": seed,
        "seg1_batch": int(cfg.batching.seg1_batch), "seg2_batch": B,
        "seg2_flush_mode": str(cfg.batching.get("seg2_flush_mode", "fixed")),
        "confidence_threshold": float(cfg.early_exit.confidence_threshold),
        "goodput_mode": "wallclock",
        "common_set_size": len(common), "dropped_pct": round(drop_pct, 3),
        "lambda_sweep": dict(cfg.arrivals["lambda_sweep"]),
        "divergence_lambda_capacity": div,
        "knee_lambda_reference": knee,   # sweep-curve latency minimum (NOT divergence)
        "D_used_for_selection": D,
        "lambda_selection": {"raw": raw, "chosen": chosen, "notes": notes},
        "slo": {"raw_mean_ms": round(raw_mean, 3), "raw_p99_ms": round(raw_p99, 3),
                "slo_avg_ms": slo_avg, "slo_p99_ms": slo_p99},
        "user_lambda_table": {"source": "plots.slo_goodput_lambda",
                              "configured": user_map, "values_used": user_lams},
        "sanity_checks": checks,
    }
    result = {"meta": meta, "table_a": table_a, "table_b": table_b,
              "table_c": table_c, "table_d": table_d, "table_e": table_e}

    d = cfg.paths["results_dir"]
    os.makedirs(d, exist_ok=True)
    written = []

    def _csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        written.append(path)

    _csv(os.path.join(d, "e2e_table_a.csv"), table_a)
    _csv(os.path.join(d, "e2e_table_b.csv"), table_b)
    if table_c:
        _csv(os.path.join(d, "e2e_table_c.csv"), table_c)
    _csv(os.path.join(d, "e2e_table_d.csv"), table_d)
    _csv(os.path.join(d, "e2e_table_e.csv"), table_e)

    # ---- peak goodput: plain, naive, and EVERY proposed bs2, each at its own
    #      goodput-maximizing λ, for every 10 ms SLO from SLO_avg up to
    #      SLO_p99. Uses common_all (Table D's set) so all bs2 curves are
    #      compared on the identical common set. ----
    slo_values = list(range(slo_avg, slo_p99 + 1, 10)) or [slo_avg]
    bs2_values = sorted(all_prop)
    default_label = f"proposed(bs2={B})"
    peak_entries = {"plain": scheds["plain"], "naive": scheds["naive"],
                    **{f"proposed(bs2={b})": all_prop[b] for b in bs2_values}}
    peak_div = {label: metrics.capacity_lambda(s, common_all)
               for label, s in peak_entries.items()}
    peak_rows, _curves = _peak_goodput(cfg, peak_entries, common_all, peak_div,
                                       slo_values, seed, lams)
    r90 = {q["runtime"]: q["peak_goodput_sps"] for q in peak_rows
           if q["slo_ms"] == slo_p99}
    cap_ratio = peak_div[default_label] / peak_div["naive"]
    peak_ratio = r90[default_label] / r90["naive"]
    print(f"[peak] proposed(bs2={B})/naive peak ratio @SLO={slo_p99}ms = "
          f"{peak_ratio:.3f} (capacity ratio {cap_ratio:.3f})")
    peak_out = {"meta": {"slo_source": "meta.slo (plain@λ1, rounded to 10 ms)",
                         "slo_values_ms": slo_values, "seed": seed,
                         "lambda_grid": dict(cfg.arrivals["lambda_sweep"]),
                         "bs2_values": bs2_values, "default_bs2": B,
                         "common_set": "common_all (plain, naive, every bs2)",
                         "peak_ratio_p99_proposed_over_naive": round(peak_ratio, 4),
                         "capacity_ratio_proposed_over_naive": round(cap_ratio, 4)},
                "rows": peak_rows}
    ppath = os.path.join(d, "peak_goodput.json")
    with open(ppath, "w") as f:
        json.dump(peak_out, f, indent=2, default=float)
    written.append(ppath)
    _csv(os.path.join(d, "peak_goodput.csv"), peak_rows)
    _peak_goodput_figure(cfg, peak_rows, slo_values, bs2_values)
    result["peak_goodput"] = peak_out

    jpath = os.path.join(d, "e2e_table.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2, default=float)
    written.insert(0, jpath)

    _print_tables(table_a, table_b, meta, table_c, table_d, table_e)
    for p in written:
        print(f"[e2e] wrote {p}")
    return result


def _print_rows(rows):
    hdr = (f"{'runtime':<10} {'λ':>7} {'avg(ms)':>9} {'p99(ms)':>9} "
           f"{'gp@SLOavg':>10} {'gp@SLOp99':>10} {'diverged':>9}")
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(f"{row['runtime']:<10} {row['lambda']:>7g} {row['avg_ms']:>9.2f} "
              f"{row['p99_ms']:>9.2f} {row['goodput_slo_avg']:>10.1f} "
              f"{row['goodput_slo_p99']:>10.1f} {str(row['diverged']):>9}")


def _print_tables(table_a, table_b, meta, table_c=None, table_d=None, table_e=None):
    print("\n[e2e] Table A — λ-independent metrics")
    hdr = f"{'runtime':<10} {'acc(%)':>8} {'sat.thr(s/s)':>13} {'divλ(capacity)':>15}"
    print(hdr)
    print("-" * len(hdr))
    for row in table_a:
        print(f"{row['runtime']:<10} {row['accuracy_pct']:>8.2f} "
              f"{row['saturated_throughput_sps']:>13.1f} "
              f"{row['divergence_lambda']:>15.1f}")

    slo = meta["slo"]
    slo_desc = f"SLO_avg={slo['slo_avg_ms']} ms, SLO_p99={slo['slo_p99_ms']} ms"
    print(f"\n[e2e] Table B — auto-derived λ grid  ({slo_desc})")
    _print_rows(table_b)
    if table_c:
        print(f"\n[e2e] Table C — user λ grid from plots.slo_goodput_lambda  "
              f"(same SLOs: {slo_desc})")
        _print_rows(table_c)

    if table_d:
        print("\n[e2e] Table D — knee operating points (λ at minimum mean latency)")
        hdr = (f"{'config':<18} {'kneeλ':>8} {'mean(ms)':>9} {'p99(ms)':>9} "
               f"{'divλ(capacity)':>15}")
        print(hdr)
        print("-" * len(hdr))
        for row in table_d:
            print(f"{row['config']:<18} {row['knee_lambda']:>8g} "
                  f"{row['mean_ms']:>9.2f} {row['p99_ms']:>9.2f} "
                  f"{row['divergence_lambda']:>15.1f}")

    if table_e:
        print("\n[e2e] Table E — AT capacity (boundary, not a safe operating "
              "point; see notes for instability warnings)")
        hdr = (f"{'config':<18} {'capacityλ':>10} {'mean(ms)':>9} {'p99(ms)':>9} "
               f"{'throughput':>11} {'refλ(-step)':>12}")
        print(hdr)
        print("-" * len(hdr))
        for row in table_e:
            print(f"{row['config']:<18} {row['capacity_lambda']:>10.1f} "
                  f"{row['mean_ms']:>9.2f} {row['p99_ms']:>9.2f} "
                  f"{row['throughput_sps']:>11.1f} {row['reference_lambda']:>12g}")
    print()
