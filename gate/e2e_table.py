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
capacity-based divergence points:
  λ1 = D_plain − step                 (below every capacity: all three stable)
  λ2 = grid-snapped midpoint(D_plain, D_naive)  (plain overloaded;
                                       naive/proposed stable)
  λ3 = D_proposed − step              (with a fine sweep step this lies above
                                       D_naive: only proposed is stable)
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
def generate(cfg: Config, scheds: dict) -> dict:
    B = int(cfg.batching.seg2_batch)
    entries = {"plain": scheds["plain"], "naive": scheds["naive"],
               "proposed": scheds["proposed"][B]}
    common = metrics.common_completed(list(entries.values()))
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

    # ---- Table B: deterministic λ1/λ2/λ3 from the divergence points ----
    D = {r: div[r] for r in RUNTIMES}
    raw = {"lambda1": _snap(lams, D["plain"] - step),
           "lambda2": _snap(lams, (D["plain"] + D["naive"]) / 2.0),
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
    all_prop = scheds["proposed"]
    common_all = metrics.common_completed(
        [scheds["plain"], scheds["naive"], *all_prop.values()])
    configs = ([("plain", scheds["plain"]), ("naive", scheds["naive"])]
               + [(f"proposed(bs2={b})", all_prop[b]) for b in sorted(all_prop)])
    table_d = []
    for label, s in configs:
        k, mean, p99, edge = metrics.knee_stats(s, lams, common_all, seed)
        cap = metrics.capacity_lambda(s, common_all)
        if edge:
            notes.append(f"Table D: {label} knee sits on the sweep edge "
                         f"({k:g}); extend lambda_sweep")
        table_d.append({"config": label, "knee_lambda": k,
                        "mean_ms": round(mean, 2), "p99_ms": round(p99, 2),
                        "divergence_lambda": round(cap, 1)})

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
              "table_c": table_c, "table_d": table_d}

    d = cfg.paths["results_dir"]
    os.makedirs(d, exist_ok=True)
    jpath = os.path.join(d, "e2e_table.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2, default=float)
    written = [jpath]

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

    _print_tables(table_a, table_b, meta, table_c, table_d)
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


def _print_tables(table_a, table_b, meta, table_c=None, table_d=None):
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
    print()
