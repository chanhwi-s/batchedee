# Goodput-Evaluation Runtime — Early-Exit ViT-B/16 (LPH) with 2-Stage Batching

Benchmark harness that measures the **goodput** of three inference runtimes for a
ViT-B/16 with a Local-Perception exit head (LPH) after block 6, and renders four
analysis plots. Target: single **NVIDIA RTX 5090**, ONNX Runtime **CUDA EP**.

The centerpiece `proposed` runtime combines **early exit** with **two-stage
decoupled batching** (independent seg1 / seg2 batch sizes), compared against
`plain` (full model, no exit) and `naive` (early exit, no decoupled batching).

## Layout

```
config.yaml               All knobs (paths, batch sizes, threshold, λ, SLO, seed…)
models/ee_vit_b16_lph.py  The EE-ViT-B/16 LPH model (provided)
gate/
  util.py       config loader + ORT session/timing helpers
  model_split.py  seg1 / seg2 wrappers + plain builder + checkpoint loader
  export.py     ONNX exporters (static & dynamic) with on-disk cache
  data.py       ImageNet-val loader (timm standard transform), N-request sampling
  arrivals.py   Poisson arrival trace (fixed seed)
  runtimes.py   GPU execution pass (measured) + CPU event simulator
  metrics.py    common-set intersection, goodput, latency stats
  plots.py      the four figures (png + pdf each)
run.py          CLI: export / run / plot / all
```

## Model split

- **seg1** = patch-embed + blocks 1–6 + LPH → outputs `(hidden_tokens, lph_logits)`.
  `lph_logits` drive the per-sample exit decision; `hidden_tokens` feed seg2.
- **seg2** = blocks 7–12 + final norm/head → `final_logits`.
- **plain** = ImageNet-pretrained `timm` ViT-B/16, whole model, no exit.

A sample **exits at seg1** iff its max-softmax confidence ≥ `confidence_threshold`
(default 0.7, fixed across the SLO sweep). Exit is decided **per sample**.

## Setup (on the server)

```bash
pip install -r requirements.txt      # use onnxruntime-gpu on the 5090
```

Edit `config.yaml`:

- `data.imagenet_val_dir` → ImageNet val (ImageFolder: `val/<class>/*.JPEG`).
- `model.best_ckpt_path` → trained `best.pth` (used by `naive`/`proposed`;
  `plain` uses timm pretrained weights, not this checkpoint).
- Tune `arrivals.lambda`, `arrivals.lambda_sweep`, and `batching.*` to your GPU.

## Run

```bash
python run.py export      # export + cache all ONNX graphs
python run.py run         # GPU pass: measure service times + accuracy, build schedules
python run.py plot        # render the figures
python run.py e2e         # end-to-end comparison tables (Table A/B, json + csv)
python run.py seg1bench   # seg1 kernel-time sweep over batch sizes 1..512 (plot10)
# or the whole pipeline:
python run.py all
```

Outputs:

- ONNX graphs → `artifacts/onnx/`
- Schedules (pickled) → `artifacts/results/schedules.pkl` (ops + measured
  service times, op stats, per-sample top-1 correctness for accuracy)
- End-to-end tables → `artifacts/results/e2e_table.json`, `e2e_table_a.csv`,
  `e2e_table_b.csv`, `e2e_table_c.csv`. Table A: accuracy / saturated
  throughput (λ=0) / divergence λ per runtime (capacity-based: divergence λ =
  the runtime's service capacity, above which the queue grows without bound).
  Table B: mean & p99 response time and goodput at two fixed SLOs, on three λ
  values derived deterministically from the capacity points (λ1/λ2/λ3;
  derivation recorded in the JSON `meta` block). Table C: same columns and
  SLOs on the user-chosen λ values from `plots.slo_goodput_lambda`. Built-in
  sanity checks print PASS/FAIL.
- Figures (png + pdf) → `artifacts/plots/`
  1. `plot1a_slo_goodput_vs_plain` / `plot1b_slo_goodput_vs_naive` — SLO vs
     goodput pair figures: BOTH baselines (plain, naive) plus the proposed
     `seg2_batch` sweep in each figure, all replayed on the same arrival trace
     at that figure's λ. The two figures differ only in the operating point.
     λ is AUTO-derived from measured capacities by default (plot1a at
     D_plain − step, plot1b at D_proposed − step — the same rule as e2e
     Table B); `plots.slo_goodput_lambda.{plain,naive}` overrides manually
     (0 = saturated).

  Most distribution figures derive their λ automatically (each runtime's own
  capacity × `plots.capacity_margin_frac`). `arrivals.lambda` is the manual
  escape hatch, read only by **plot2c and plot11b**: a per-runtime mapping
  (`lambda: {plain: 1400, naive: 1650, proposed: 1700}`) benchmarks each runtime
  at its own chosen load, `0` for a runtime means saturated (all requests queued
  at t=0), and a null entry falls back to that runtime's capacity×margin λ. The
  λ-sweep figures (load vs latency, breakdown) always use
  `arrivals.lambda_sweep`.
  2. `plot2_latency_kde` — KDE of per-sample latency per runtime
     (`plots.kde_bandwidth` / `kde_grid_points`; x-axis clipped via
     `plots.kde_xlim_ms` or `kde_clip_percentile` when a long tail squeezes
     the bulk). `plot2c_latency_kde_per_runtime_lambda` is the same figure at
     the manual `arrivals.lambda` rates instead of the derived ones — the KDE
     counterpart of plot11b, skipped when that key is absent. Its λ values go
     to stdout, not the legend, so put them in the caption; and mind that a λ
     within a few percent of a runtime's capacity puts it in the near-critical
     (ρ→1) regime, where the KDE grows a second hump that is queue burstiness,
     not batching structure.
  3. `plot2b_latency_kde_sweep` — same KDE, one panel per `seg2_batch` in the
     sweep (plain/naive repeated as references), horizontally concatenated.
  4. `plot3_latency_cdf` — empirical CDF of per-sample latency per runtime.
  5. `plot3b_latency_cdf_sweep` — single-axes CDF: plain + naive + the whole
     proposed b2 sweep as sequential shades.
  6. `plot4_load_latency` — mean response time vs λ per runtime; also prints
     each runtime's capacity-based divergence λ (+ knee for reference) and
     records it in the pkl (`schedules['divergence']`).
  7. `plot5/6` — per-component latency decomposition vs λ (formation wait /
     GPU wait / stage-1 compute / stage-2 queue wait / stage-2 compute).
  8. `plot7_timeline` — GPU execution timeline per runtime on the simulation
     clock: one contiguous bar colored by state (arrival wait / seg1 or whole /
     seg2). `plots.timeline_xlim_ms` clips the x-axis for zooming; works in both
     seg2 flush modes.
  9. `plot8_exec_stats` — per-runtime bars: mean measured service time per op
     and op count, split by stage (seg1/whole vs seg2). The same numbers are
     printed by `run.py run` and stored in the pkl under
     `schedules['op_stats']`.
  10. `plot9_naive_seg2_sizes` — histogram of naive's dynamic seg2 batch sizes
     (per-batch non-exit counts); summary stats printed to stdout.
  11. `plot1c_peak_goodput_bars` — peak goodput per runtime, each at its OWN
     goodput-maximizing λ, for every 10 ms SLO from SLO_avg to SLO_p99
     (generated by `run.py e2e`; data in `results/peak_goodput.{json,csv}`,
     argmax λ annotated on every bar).
  12. `plot11a_latency_cdf_common_lambda` / `plot11b_latency_cdf_per_runtime_lambda`
     — the latency CDF of plain / naive / proposed@`seg2_batch` at manually
     chosen loads, as a pair. **11a is iso-load**: ONE shared λ from
     `plots.cdf_common_lambda` (null skips the figure, 0 = saturated). plot3
     puts each runtime at its own capacity×margin λ, so its curves mix batching
     structure with the different offered loads; 11a removes the load term so
     the remaining gap is attributable to batching alone (mainly proposed's
     seg2 queue wait). λ is never auto-derived — pick a rate below the smallest
     capacity in Table A and state it in the caption. **11b is per-runtime**:
     each runtime at its OWN λ from `arrivals.lambda.{plain,naive,proposed}`
     (per-runtime rate; 0 = that runtime saturated; null = fall back to its
     capacity×margin λ; a scalar applies to all; remove the key to skip the
     figure) — the "each runtime at the load it would actually be operated at"
     view, so the gaps there are NOT attributable to batching alone. 11b's
     legend carries the runtime names only; its per-runtime λ values go to
     stdout and belong in the caption. p50/p90/p99 per runtime are printed to
     stdout by both.
  13. `plot13a_naive_exit_kde` / `plot13b_naive_exit_hist` and
     `plot13c_proposed_exit_kde` / `plot13d_proposed_exit_hist` — ONE early-exit
     runtime per figure (13a/b naive, 13c/d proposed@`seg2_batch`), per-sample
     latency split by exit class (purple = exited at stage 1, red = also ran
     stage 2), as a KDE and as a raw histogram. Both classes share the
     formation + queue wait, which carries almost all the variance, so each
     pair of conditionals is one distribution shifted by the non-exit class's
     extra stage-2 cost — and that shift is what differs between the runtimes:
     naive's seg2 runs right after its own seg1 (gap ≈ 2.9 ms, Cohen's
     d ≈ 0.7 → no visible bimodality in plot2/plot12b), while proposed's
     non-exit samples also wait for the seg2 queue to fill (gap ≈ 14.5 ms,
     d ≈ 2.3 → the split is real and visible). The `"mixture"` normalization
     scales each class KDE by its sample share so the two sum to the pooled
     density (gray); the histograms stack the two disjoint classes so they
     reproduce the pooled histogram. λ defaults to each runtime's own
     capacity×margin (same point as plot2/3/12b); override per runtime with
     `plots.exit_split_lambda`. Mean gap, pooled sd and Cohen's d are printed
     to stdout — quote them in the caption. Setting `exit_split_lambda: 0`
     (saturated: latency measured from the stage-1 op start, dropping the
     shared waits) is what separates naive's modes too — its d goes ~0.7 →
     ~4.6 — useful as a "the two populations really are there" companion.
     `plot13e_naive_latency_composition` / `plot13f_proposed_latency_composition`
     extend the same histogram with WHY each bar is where it is: bar height is
     still the bin count, but each bar is cut into that bin's **mean latency
     composition** (formation wait / GPU wait / stage-1 / stage-2 queue wait /
     stage-2 compute, plot5's palette), with exit | non-exit as two panels.
     Reading left to right shows what turns a fast sample into a slow one —
     for naive the growing band is formation + GPU wait while stage-2 compute
     stays a thin constant sliver (exactly why 13a/13b show no split), whereas
     proposed's non-exit panel is taken over by stage-2 queue wait on the
     right. `plots.composition_bins` (default 40) is coarser than `hist_bins`
     on purpose; `gpu_wait` is included because the five components must sum to
     the latency. Skipped in saturated mode. Per-class component means and
     percentages are printed to stdout.
  14. `plot14a`–`plot14d` (`..._naive_exit_kde_iso` / `..._naive_exit_hist_iso` /
     `..._proposed_exit_kde_iso` / `..._proposed_exit_hist_iso`) — 13a–13d
     again, pinned to ONE shared λ and annotated with the SLO deadline.
     13a–13d give each runtime its own capacity×margin λ, which measures naive
     and proposed at different offered loads and so cannot support a direct
     comparison; 14a–14d replay both at `plots.exit_split_common_lambda`, so
     14a/14b and 14c/14d sit side by side at iso-load with only the batching
     structure differing. The red vertical line(s) from
     `plots.exit_split_slo_ms` (a number or a list) mark the deadline, and the
     per-class violation rates at each SLO are printed to stdout — that is the
     number the pair argues about: proposed's non-exit component sits further
     right, but what matters is how much of it crosses the line. λ is never
     auto-derived here — pick a rate below the smallest capacity among the
     compared runtimes and state it, with the SLO, in the caption. Unset λ
     skips all four. The x-axis is pinned by `plots.exit_split_xlim_ms`
     (default 100 ms) instead of clipped at a percentile the way plot13 is:
     all four panels need one shared scale, and the tail past the SLO is the
     evidence, so a p99 cut would hide it.
     `plot14e_naive_latency_composition_iso` /
     `plot14f_proposed_latency_composition_iso` are 13e/13f under the same
     treatment — per-bin composition at the shared λ, fixed x-axis, SLO line on
     both panels — so you can read *which component* pushes each runtime's
     samples past the deadline.
  14. `plot10_seg1_batch_sweep` — seg1 kernel time per op over batch sizes
     1..512 (`run.py seg1bench`; 4096 random samples per size; numbers also in
     `artifacts/results/seg1_batch_sweep.json`).

## Methodology notes

- **Queuing is simulated; inference is measured.** Arrival timestamps live on a
  simulation clock; seg1/seg2/whole-model service times are measured by actually
  running the ONNX graphs on the GPU (with warmup excluded). Because batch
  composition and exit masks are independent of λ, the GPU graphs are run **once**
  to build a schedule of measured ops, then each λ is a cheap CPU replay.
- **seg1** is executed on every full batch (yields real exit masks + timing).
  **naive** seg2 (dynamic batch) is measured on **every** batch to capture
  per-size kernel/allocation overhead. **proposed** seg2 (static) and **plain**
  whole-model are measured once and cached (`cache_static_service_times`).
- **Single-stream GPU:** ops execute serially in dispatch order; a seg1/whole op
  waits for the last of its members to arrive, seg2 ops start as soon as the GPU
  frees (inputs already produced).
- **Batch formation:** seg1 waits indefinitely until `seg1_batch` fills. The
  `proposed` seg2 queue has two flush modes (`batching.seg2_flush_mode`):
  `"fixed"` (default) flushes exactly `seg2_batch` through the static seg2 graph
  once the queue reaches `seg2_batch`; `"all"` flushes the **entire queue**
  through the dynamic seg2 graph (like naive, timed per flush since the size
  varies). In both modes, never-filled leftovers are dropped.
- **Fair comparison:** every metric is restricted to the **intersection of request
  IDs completed by all compared runtimes**; goodput's wall-clock window is also
  measured on that common set.

## Notes

- Fixed seeds throughout; measured GPU times still carry small run-to-run variance.
- All parameters are overridable via `config.yaml`.
