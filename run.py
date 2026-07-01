#!/usr/bin/env python3
"""Entry point for the goodput-evaluation runtime.

Subcommands
-----------
  export   Export + cache all ONNX graphs (plain / seg1 / seg2 static+dynamic).
  run      Run the GPU execution pass, build schedules, save them to disk.
  plot     Load saved schedules and render the four figures (png + pdf).
  all      export -> run -> plot.

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
from gate.util import ensure_dirs, load_config


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
    plots_mod.plot_all(cfg, scheds)


def cmd_all(cfg, args):
    cmd_export(cfg, args)
    cmd_run(cfg, args)
    cmd_plot(cfg, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "run", "plot", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="re-export ONNX even if cached")
    ap.add_argument("--runtimes", nargs="+", default=["plain", "naive", "proposed"],
                    help="which runtimes to export")
    args = ap.parse_args()

    cfg = load_config(args.config)
    {"export": cmd_export, "run": cmd_run, "plot": cmd_plot, "all": cmd_all}[args.command](cfg, args)


if __name__ == "__main__":
    main()
