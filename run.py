#!/usr/bin/env python3
"""Entry point for the goodput-evaluation runtime.

Subcommands
-----------
  export   Export + cache all ONNX graphs (plain / seg1 / seg2 static+dynamic).
  run      Run the GPU execution pass (+ accuracy pass), build schedules, save.
  plot     Load saved schedules and render the figures (png + pdf).
  e2e      Generate the end-to-end comparison tables (Table A/B, json + csv).
  all      export -> run -> plot -> e2e.

Usage
-----
  python run.py export            --config config.yaml
  python run.py run               --config config.yaml
  python run.py plot              --config config.yaml
  python run.py all               --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

from gate import export as export_mod
from gate import plots as plots_mod
from gate import runtimes
from gate.data import load_requests
from gate.util import ensure_dirs, load_config, setup_hf_cache


def _sched_path(cfg) -> str:
    return os.path.join(cfg.paths["results_dir"], "schedules.pkl")


def cmd_export(cfg, args):
    ensure_dirs(cfg)
    which = tuple(args.runtimes)
    export_mod.export_all(cfg, which=which, force=args.force)


def cmd_run(cfg, args):
    ensure_dirs(cfg)
    print("[run] loading requests ...")
    images, labels, paths = load_requests(cfg)
    print(f"[run] N={images.shape[0]} requests; building schedules ...")
    sweep = [int(b) for b in cfg.batching.get("seg2_batch_sweep", [])]
    if int(cfg.batching.seg2_batch) not in sweep:
        sweep.append(int(cfg.batching.seg2_batch))
    scheds = runtimes.build_all_schedules(cfg, images, seg2_batches=sorted(set(sweep)))

    # drop-stat summary
    for name in ("plain", "naive"):
        s = scheds[name]
        print(f"[run] {name}: completed={len(s.completed_ids())}, dropped={len(s.dropped)}")
    for B, s in sorted(scheds["proposed"].items()):
        print(f"[run] proposed(seg2={B}): completed={len(s.completed_ids())}, dropped={len(s.dropped)}")

    # per-runtime op stats (count + mean service time), stored in the pkl
    def _fmt(st):
        return "  ".join(f"{k}: n={v['count']}, mean={v['mean_ms']:.2f}ms" for k, v in st.items())

    scheds["op_stats"] = {
        "plain": runtimes.op_stats(scheds["plain"]),
        "naive": runtimes.op_stats(scheds["naive"]),
        "proposed": {B: runtimes.op_stats(s) for B, s in scheds["proposed"].items()},
    }
    print(f"[run] plain : {_fmt(scheds['op_stats']['plain'])}")
    print(f"[run] naive : {_fmt(scheds['op_stats']['naive'])}")
    for B, st in sorted(scheds["op_stats"]["proposed"].items()):
        print(f"[run] proposed(seg2={B}): {_fmt(st)}")

    # accuracy pass: real (untimed) inference -> per-sample top-1 correctness
    print("[run] accuracy pass (real inference, untimed) ...")
    scheds["correct"] = runtimes.collect_correctness(cfg, images, labels)
    done = scheds["plain"].completed_ids()
    print(f"[run] top-1 accuracy over batched samples: "
          f"plain={100 * scheds['correct']['plain'][done].mean():.2f}%  "
          f"ee={100 * scheds['correct']['ee'][done].mean():.2f}%  "
          f"(exit rate {100 * scheds['correct']['exit'][done].mean():.1f}%)")

    with open(_sched_path(cfg), "wb") as f:
        pickle.dump(scheds, f)
    print(f"[run] saved schedules -> {_sched_path(cfg)}")


def cmd_plot(cfg, args):
    ensure_dirs(cfg)
    p = _sched_path(cfg)
    if not os.path.exists(p):
        sys.exit(f"[plot] no schedules at {p}; run `python run.py run` first.")
    with open(p, "rb") as f:
        scheds = pickle.load(f)
    divergence = plots_mod.plot_all(cfg, scheds)
    if divergence:
        scheds["divergence"] = divergence          # capacity-based divergence λ per runtime
        with open(p, "wb") as f:
            pickle.dump(scheds, f)
        print(f"[plot] divergence λ recorded in {p} (schedules['divergence'])")


def cmd_e2e(cfg, args):
    ensure_dirs(cfg)
    p = _sched_path(cfg)
    if not os.path.exists(p):
        sys.exit(f"[e2e] no schedules at {p}; run `python run.py run` first.")
    with open(p, "rb") as f:
        scheds = pickle.load(f)
    from gate import e2e_table
    e2e_table.generate(cfg, scheds)


def cmd_all(cfg, args):
    cmd_export(cfg, args)
    cmd_run(cfg, args)
    cmd_plot(cfg, args)
    cmd_e2e(cfg, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "run", "plot", "all", "e2e"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="re-export ONNX even if cached")
    ap.add_argument("--runtimes", nargs="+", default=["plain", "naive", "proposed"],
                    help="which runtimes to export")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = setup_hf_cache(cfg)          # writable HF/timm cache before timm import
    print(f"[env] HF cache -> {cache}")
    {"export": cmd_export, "run": cmd_run, "plot": cmd_plot,
     "all": cmd_all, "e2e": cmd_e2e}[args.command](cfg, args)


if __name__ == "__main__":
    main()
