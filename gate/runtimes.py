"""The three runtimes: GPU execution pass (measured) + CPU event simulator.

Key design — measurement is decoupled from queuing:
  * What actually runs on the GPU (batch composition, exit masks, per-op service
    times) depends ONLY on the fixed request sequence, NOT on the arrival rate.
    So we run the graphs ONCE to build a `Schedule` (an ordered list of GPU ops
    with measured durations + which requests complete at each op).
  * The arrival rate (lambda) only shifts WHEN each op may start. So per lambda we
    replay a cheap CPU event simulation over the same Schedule.

This keeps "inference is measured, queuing is simulated" exactly, and makes the
load sweep fast and low-variance.

Timing notes:
  * seg1 is executed for real on every full batch — it yields the real LPH exit
    masks AND its measured service time.
  * seg2 is dense (no data-dependent branching); its service time is determined by
    input shape, so we time it on correctly-shaped tensors.
      - `proposed` seg2 is static (size = seg2_batch): measured once and cached
        (cache_static_service_times).
      - `naive` seg2 is dynamic (variable non-exit count): measured on EVERY batch
        to capture per-size kernel/allocation overhead (never cached).
  * `plain` whole-model (static, size = seg1_batch): measured once and cached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import export
from .data import iter_batches
from .util import Config, TimedSession


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Op:
    kind: str                       # 'whole' | 'seg1' | 'seg2'
    members: np.ndarray             # request ids that are INPUT to this op
    completes: np.ndarray           # request ids that COMPLETE when this op finishes
    duration: float                 # measured GPU service time (seconds)
    gate_on_arrival: bool           # True for seg1/whole (wait for members to arrive)


@dataclass
class Schedule:
    runtime: str
    ops: list[Op]
    n_requests: int
    seg2_batch: int | None = None
    dropped: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def completed_ids(self) -> np.ndarray:
        ids = [op.completes for op in self.ops if len(op.completes)]
        return np.concatenate(ids) if ids else np.array([], dtype=np.int64)


# --------------------------------------------------------------------------- #
# GPU execution pass
# --------------------------------------------------------------------------- #
@dataclass
class BatchResult:
    batch_ids: np.ndarray
    exit_ids: np.ndarray
    nonexit_ids: np.ndarray
    seg1_dur: float


def _softmax_maxconf(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    return p.max(axis=1)


def run_seg1_pass(cfg: Config, images: np.ndarray, seg1_sess: TimedSession) -> list[BatchResult]:
    """Run seg1 on every full batch; return per-batch exit/non-exit ids + timing.

    Shared by `naive` and every `proposed` seg2_batch variant (masks are identical).
    """
    S = int(cfg.batching.seg1_batch)
    thr = float(cfg.early_exit.confidence_threshold)
    results: list[BatchResult] = []
    for s, batch in iter_batches(images, S, drop_last=True):
        feed = {seg1_sess.input_names[0]: batch.astype(np.float32)}
        outs, dur = seg1_sess.run_timed(feed)
        # outputs: ["hidden", "lph_logits"] (order from export). Find lph by shape.
        lph = _pick_logits(seg1_sess.output_names, outs)
        conf = _softmax_maxconf(lph)
        mask = conf >= thr
        ids = np.arange(s, s + S, dtype=np.int64)
        results.append(BatchResult(
            batch_ids=ids,
            exit_ids=ids[mask],
            nonexit_ids=ids[~mask],
            seg1_dur=dur,
        ))
    return results


def _pick_logits(names, outs):
    for n, o in zip(names, outs):
        if n == "lph_logits":
            return o
    # fallback: the 2-D output is the logits
    for o in outs:
        if o.ndim == 2:
            return o
    return outs[-1]


# --------------------------------------------------------------------------- #
# Schedule builders
# --------------------------------------------------------------------------- #
def build_plain(cfg: Config, images: np.ndarray) -> Schedule:
    S = int(cfg.batching.seg1_batch)
    sess = TimedSession(export.plain_path(cfg), cfg)
    # warmup + measure whole-model service time once (static batch).
    probe = images[:S].astype(np.float32)
    feed = {sess.input_names[0]: probe}
    sess.warmup(feed)
    _, dur = sess.run_timed(feed)

    ops: list[Op] = []
    n = images.shape[0]
    completed = []
    for s, _ in iter_batches(images, S, drop_last=True):
        ids = np.arange(s, s + S, dtype=np.int64)
        ops.append(Op("whole", ids, ids, dur, gate_on_arrival=True))
        completed.append(ids)
    done = np.concatenate(completed) if completed else np.array([], dtype=np.int64)
    dropped = np.setdiff1d(np.arange(n), done)
    return Schedule("plain", ops, n, seg2_batch=None, dropped=dropped)


def build_naive(cfg: Config, images: np.ndarray, seg1_pass: list[BatchResult]) -> Schedule:
    seg2 = TimedSession(export.seg2_dynamic_path(cfg), cfg)
    S = int(cfg.batching.seg1_batch)
    hidden_dim = int(cfg.model.hidden_dim)
    # warm the dynamic graph once (full-batch shape).
    seg2.warmup({seg2.input_names[0]: np.random.randn(S, 197, hidden_dim).astype(np.float32)})

    ops: list[Op] = []
    n = images.shape[0]
    for br in seg1_pass:
        ops.append(Op("seg1", br.batch_ids, br.exit_ids, br.seg1_dur, gate_on_arrival=True))
        k = len(br.nonexit_ids)
        if k > 0:
            # measure THIS batch's seg2 (dynamic size k) — never cached.
            feed = {seg2.input_names[0]: np.random.randn(k, 197, hidden_dim).astype(np.float32)}
            _, dur = seg2.run_timed(feed)
            ops.append(Op("seg2", br.nonexit_ids, br.nonexit_ids, dur, gate_on_arrival=False))
    done = np.concatenate([op.completes for op in ops if len(op.completes)]) if ops else np.array([], np.int64)
    dropped = np.setdiff1d(np.arange(n), done)
    return Schedule("naive", ops, n, seg2_batch=None, dropped=dropped)


def build_proposed(cfg: Config, images: np.ndarray, seg1_pass: list[BatchResult],
                   seg2_batch: int) -> Schedule:
    B = int(seg2_batch)
    hidden_dim = int(cfg.model.hidden_dim)
    seg2 = TimedSession(export.seg2_static_path(cfg, B), cfg)
    cache_static = bool(cfg.runtime.get("cache_static_service_times", True))

    # measure static seg2(B) once (warmup + timed); reuse if caching.
    probe = {seg2.input_names[0]: np.random.randn(B, 197, hidden_dim).astype(np.float32)}
    seg2.warmup(probe)
    _, seg2_dur_cached = seg2.run_timed(probe)

    def seg2_time() -> float:
        if cache_static:
            return seg2_dur_cached
        _, d = seg2.run_timed({seg2.input_names[0]: np.random.randn(B, 197, hidden_dim).astype(np.float32)})
        return d

    ops: list[Op] = []
    n = images.shape[0]
    queue: list[int] = []
    for br in seg1_pass:
        ops.append(Op("seg1", br.batch_ids, br.exit_ids, br.seg1_dur, gate_on_arrival=True))
        queue.extend(br.nonexit_ids.tolist())
        while len(queue) >= B:
            flush = np.array(queue[:B], dtype=np.int64)
            del queue[:B]
            ops.append(Op("seg2", flush, flush, seg2_time(), gate_on_arrival=False))
    dropped_queue = np.array(queue, dtype=np.int64)          # leftover seg2 queue dropped
    done = np.concatenate([op.completes for op in ops if len(op.completes)]) if ops else np.array([], np.int64)
    dropped = np.setdiff1d(np.arange(n), done)
    return Schedule("proposed", ops, n, seg2_batch=B, dropped=dropped)


# --------------------------------------------------------------------------- #
# Event simulator (CPU) — per lambda
# --------------------------------------------------------------------------- #
def simulate(sched: Schedule, arrivals: np.ndarray):
    """Replay the schedule on a single-stream GPU against an arrival trace.

    Returns (completion[N], completed_mask[N]). Dropped/uncompleted -> inf/False.
    """
    n = sched.n_requests
    completion = np.full(n, np.inf, dtype=float)
    gpu_free = 0.0
    for op in sched.ops:
        if op.gate_on_arrival and len(op.members):
            ready = float(arrivals[op.members].max())   # last member to arrive
            start = max(gpu_free, ready)
        else:
            start = gpu_free                             # inputs already produced
        end = start + op.duration
        gpu_free = end
        if len(op.completes):
            completion[op.completes] = end
    completed_mask = np.isfinite(completion)
    return completion, completed_mask


def latencies(sched: Schedule, arrivals: np.ndarray):
    """Per-sample latency array (inf where not completed)."""
    completion, mask = simulate(sched, arrivals)
    lat = completion - arrivals
    return lat, mask


# --------------------------------------------------------------------------- #
# Orchestration: build all schedules with a single GPU pass
# --------------------------------------------------------------------------- #
def build_all_schedules(cfg: Config, images: np.ndarray,
                        seg2_batches: list[int] | None = None) -> dict:
    """Build plain, naive, and proposed(B) schedules. One seg1 GPU pass shared.

    Returns {'plain': Schedule, 'naive': Schedule,
             'proposed': {B: Schedule, ...}}.
    """
    if seg2_batches is None:
        seg2_batches = [int(cfg.batching.seg2_batch)]

    out: dict = {}
    # plain (independent of seg1 pass)
    out["plain"] = build_plain(cfg, images)

    # shared seg1 pass for naive + proposed
    seg1_sess = TimedSession(export.seg1_path(cfg), cfg)
    seg1_sess.warmup({seg1_sess.input_names[0]:
                      images[:int(cfg.batching.seg1_batch)].astype(np.float32)})
    seg1_pass = run_seg1_pass(cfg, images, seg1_sess)

    out["naive"] = build_naive(cfg, images, seg1_pass)
    out["proposed"] = {B: build_proposed(cfg, images, seg1_pass, B) for B in seg2_batches}
    return out
