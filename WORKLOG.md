# Work Log

Numbers are as of the `schedules.pkl` of the day. The conclusions (orderings, ratios) hold, but
always re-derive paper values from the latest run.

---

## 2026-08-10 — exit-class latency analysis + plots 2c / 11a-b / 13a-f / 14a-f

### Starting question: why is there no bimodality in naive's KDE?

The only extra cost a non-exiting sample pays in naive is **its own batch's seg2 op**
(`seg1_batch`=16, 7.1 non-exit per batch on average → 2.71 ms). At λ=1500:

- exit 13.78 ± 4.08 ms, non-exit 16.65 ± 4.17 ms
- gap **2.88 ms**, pooled sd **4.12 ms** → **Cohen's d = 0.70**
- a 2-component mixture looks bimodal only if gap > 2σ = **8.2 ms** → 2.9× short

The variance comes not from the Poisson process itself but from the **waits both classes share**:
formation wait 5.01±3.6, GPU wait 1.86±2.7, stage-1 6.91±0.1 (constant).
Those three alone are **88% of the total variance**. The two conditionals are one distribution
shifted by 2.88 ms.

Formation wait checks out in closed form (a sample at a random position in the batch → Erlang):

```
sd = (1/λ)·sqrt((S-1)/2 + (S²-1)/12) = 5.36/λ     # 3.61 ms at λ=1485, matches the measurement
```

**It grows with S while the gap — seg2(S/2) — barely does, because of kernel-launch overhead.**
So the failure to separate is structural. Changing λ does not help either: the variance is U-shaped
(λ↓ formation ∝1/λ, λ↑ queue wait as ρ→1), and d peaks at 0.75 over the whole sweep.

- **proposed differs**: its non-exit class also pays the seg2 queue wait, so gap 15.04 ms,
  **d = 2.33** → the modes really do split. That is why only proposed shows two humps in plot2.
- **Removing the shared waits splits naive too**: with `exit_split_lambda: 0` (saturated),
  d goes 0.70 → **4.62**.

### Follow-up: why does the front of non-exit overlap exit in proposed's histogram?

The overlap is real, not a rendering artifact — **overlap coefficient 0.26**. The cause is the seg2
queue wait's **p10 = 0 ms** (the last samples to join a flush never wait). non-exit min 12.2 <
exit p50 15.4, exit max 35.4 > non-exit p50 29.7; 15.9% of non-exit sits below exit's p90.

Three reasons the KDE hid it: (1) the mixture weighting flattens non-exit's left shoulder,
(2) curves invite reading peak positions, so overlapping area goes unnoticed, (3) a bandwidth of
0.3×sd ≈ 2.3 ms smears the hard cutoff at 12.2 ms into a slope. **The histogram is the honest one.**

---

### Figures added

| figure | what | λ source |
|---|---|---|
| `plot1c_slo_goodput_vs_proposed` | third figure of the 1a/1b series, anchored on proposed's capacity. **The old `plot1c_peak_goodput_bars` is now `plot1d_`** | `slo_goodput_lambda.proposed` |
| `plot2c_..._per_runtime_lambda` | plot2 at manual λ (KDE twin of plot11b) | `arrivals.lambda` |
| `plot11a_..._common_lambda` | the old plot11. **renamed `plot11_`→`plot11a_`** | `plots.cdf_common_lambda` |
| `plot11b_..._per_runtime_lambda` | CDF, each runtime at its own λ | `arrivals.lambda` |
| `plot13a/13b` | naive latency split exit (purple) / non-exit (red), KDE / hist | capacity×margin |
| `plot13c/13d` | same for proposed@`seg2_batch` | capacity×margin |
| `plot13e/13f` | each bin cut by its **mean latency composition**, exit\|non-exit panels | same as 13 |
| `plot14a`–`14d` | 13a–13d pinned to a **shared λ** + red SLO line + fixed x-axis | `exit_split_common_lambda` |
| `plot14e/14f` | 13e/13f under the same treatment | `exit_split_common_lambda` |

Design rules: 13/14 share plot2/3/12b's common set and λ convention so the pooled curves match those
figures exactly / KDE defaults to `"mixture"` (the class curves sum to the pooled density) /
histograms default to stacked (they sum to the pooled histogram) / the composition figures must
include `gpu_wait`, or the five components stop summing to the latency and the bars lie / 14 never
auto-derives λ and pins the x-axis instead of clipping at a percentile — the panels need one shared
scale and the tail past the SLO is the evidence.

### Config keys

```yaml
plots:
  exit_split_lambda: null          # 13 only. null=capacity×margin, a number, 0=saturated, or a per-runtime map
  exit_split_normalize: "mixture"  # | "each"
  exit_split_show_pooled: true
  exit_split_hist_stacked: true
  composition_bins: 40             # 13e/f, 14e/f only — coarser than hist_bins (80)
  exit_split_common_lambda: 1650   # 14 only, the shared λ. null skips all of 14
  exit_split_slo_ms: 50            # a number or a list → red vertical line(s)
  exit_split_xlim_ms: 100          # 14's fixed upper x-bound
arrivals:
  lambda: {plain: …, naive: …, proposed: …}   # plot2c / plot11b only
```

`arrivals.lambda` was revived from "unused". `naive_exit_*` renamed to `exit_split_*`.
The per-class mean dotted lines (`exit_split_marks`) were removed.

### Measurements (08/10 15:30 pkl)

| | capacity | ×0.90 | op times |
|---|---|---|---|
| plain | 1606 | 1445 | whole 9.96 ms |
| naive | 1664 | 1500 | seg1 6.91 + seg2 2.71 |
| proposed@16 | 1725 | 1550 | seg1 6.91 + seg2 5.30 |

Exit rate 55.4% (threshold 0.7), common set n=16379.
Violations at λ=1650, SLO 50 ms — naive 41.9% (exit 39.5 / non-exit 44.9), proposed 1.1% (0.0 / 2.5).
Composition at the same λ: naive exit 28.96 ms is **60% GPU wait**; proposed non-exit 30.82 ms is
29% seg2 queue wait + 17% compute. So both miss the same SLO for different reasons — a collapsing
queue vs. waiting for a batch to fill.

---

### Open issues

**1. The configured λ values are near-critical (important).** `arrivals.lambda` is 1600/1650/1650 and
`exit_split_common_lambda` is 1650, against capacities of 1606 (plain) and 1664 (naive) →
**ρ = 0.996 / 0.991**. plain and naive going bimodal in plot2c is **queue burstiness on the edge of
divergence**, not batching structure, and naive's 60% GPU wait in 14e is the same symptom. This is
exactly why `capacity_margin_frac: 0.90` exists. **Use ~1445 / 1500 / 1550.**

**2. The intro claim "naive violates the SLO more at high load" is unsafe as stated.**
Violation rate (%) at SLO 50 ms:

| λ | 1400 | 1500 | 1600 | 1650 | 1700 |
|---|---|---|---|---|---|
| plain | 0.00 | 0.00 | 45.98 | 77.30 | 93.36 |
| naive | 0.00 | 0.00 | 0.00 | 41.89 | 87.44 |
| proposed | 0.98 | 0.51 | 0.35 | 1.12 | 27.57 |

At λ ≤ 1550 **naive is at 0.00% and proposed is the worse one**. At a 30 ms SLO proposed sits at
20–29% across the whole range and loses to naive until λ=1650. A single-λ table invites
"but at 1500 it is the other way round".
What the data actually says is **"it pushes back the load at which violations begin"** (the cliff
moves 1606 → 1664 → 1725), which matches the goodput argument and is hard to refute.
→ **TODO `plot15_slo_violation_vs_load`**: violation rate vs λ per runtime + capacity rules.
Candidate intro figure.
→ The story needs an SLO of ≥50 ms (proposed's non-exit tail reaches ~40 ms). Justify the choice in
the caption.

**3. Tail clipping.** With `kde_clip_percentile: 99` the cut is the max over the plotted arrays' p99,
so **in practice only proposed is clipped**: plot2/2b/12b cut at 46.8 ms (1.00% of proposed lost),
plot13c/13d lose 2.24% of the non-exit class, plot3/3b/11 are unclipped, plot14 is fixed at 100 ms and
loses nothing. Note the asymmetry — **a KDE is fitted on all the data and only the drawing grid stops,
so the curve is correct**, whereas **a histogram drops samples outside the bin edges** (a real loss).
The bandwidth is `0.3 × the full sd`, so the clipped tail also inflates it. Since the 2.24% lost in
13c/13d are precisely the worst seg2-queue-wait samples, the evidence for "proposed grows a heavy
non-exit tail" sits outside the figure → raise to `kde_clip_percentile: 99.9` (cut 53.0 ms, 0.1% lost)
or state the truncation in the caption.

**4. Read plot3 with care.** Each runtime sits at its own λ, so the gaps mix batching structure with
offered load. The iso-load comparison is plot11a's job, currently skipped because
`cdf_common_lambda: null`. Set it to ~1445 to enable.

**5. Operational.** `artifacts/` is tracked, so rendering on both machines produces a binary conflict
every time — **render on the server only**. Mixing Windows and WSL makes `config.yaml` show as
modified on CRLF alone (`* text=auto eol=lf` in `.gitattributes` would fix it; not applied).
