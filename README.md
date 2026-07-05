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
python run.py run         # GPU pass: measure service times, build schedules
python run.py plot        # render the four figures
# or the whole pipeline:
python run.py all
```

Outputs:

- ONNX graphs → `artifacts/onnx/`
- Schedules (pickled) → `artifacts/results/schedules.pkl`
- Figures (png + pdf) → `artifacts/plots/`
  1. `plot1_slo_goodput` — SLO 0–200 ms vs goodput; one curve per `seg2_batch` in
     `{2,4,8,16,32}`, plus `plain` and `naive`.

  All single-λ figures (goodput, KDE, CDF, timeline) share `arrivals.lambda`:
  `0` disables Poisson modeling entirely (all requests queued at t=0, saturated
  drain), `> 0` uses the Poisson trace at that rate. It can also be a
  per-runtime mapping (`lambda: {plain: 1400, naive: 1650, proposed: 1700}`) to
  benchmark each runtime at its own sustainable load (e.g. read off the
  load-vs-latency plot); labels then show each runtime's λ. The λ-sweep figures
  (load vs latency, breakdown) always use `arrivals.lambda_sweep`.
  2. `plot2_latency_kde` — KDE of per-sample latency per runtime.
  3. `plot3_latency_cdf` — empirical CDF of per-sample latency per runtime.
  4. `plot4_load_latency` — response time (mean **and** p99) vs λ per runtime.
  5. `plot7_timeline` — GPU execution timeline per runtime on the simulation
     clock: one contiguous bar colored by state (arrival wait / seg1 or whole /
     seg2). `plots.timeline_xlim_ms` clips the x-axis for zooming; works in both
     seg2 flush modes.
  6. `plot8_exec_stats` — per-runtime bars: mean measured service time per op
     and op count, split by stage (seg1/whole vs seg2). The same numbers are
     printed by `run.py run` and stored in the pkl under
     `schedules['op_stats']`.

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
