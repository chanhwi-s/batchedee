"""Export + cache ONNX graphs for plain / naive / proposed.

Graphs
------
plain_bs{S}.onnx        : image[S,3,H,W]           -> logits[S,1000]         (static)
seg1_bs{S}.onnx         : image[S,3,H,W]           -> hidden[S,197,768],
                                                       lph_logits[S,1000]     (static)
seg2_static_bs{B}.onnx  : hidden[B,197,768]        -> logits[B,1000]          (static)
seg2_dynamic.onnx       : hidden[N,197,768]        -> logits[N,1000]          (dynamic N)

`plain`/`proposed` use static graphs (one per batch size).
`naive` seg2 receives a variable non-exit count, so it uses the dynamic graph.
seg1 for naive can be static (batch is always seg1_batch).
"""
from __future__ import annotations

import os

from .util import Config
# NOTE: torch / model_split are imported lazily inside export functions so that
# importing this module (for its path helpers) does not require torch. This lets
# the CPU-only simulator/metrics/plots be used without a torch install.

OPSET = 17


def _seg2_sizes(cfg: Config) -> list[int]:
    sizes = set(int(b) for b in cfg.batching.get("seg2_batch_sweep", []))
    sizes.add(int(cfg.batching.seg2_batch))
    return sorted(sizes)


def _p(cfg: Config, name: str) -> str:
    return os.path.join(cfg.paths["onnx_dir"], name)


def plain_path(cfg: Config) -> str:
    return _p(cfg, f"plain_bs{int(cfg.batching.seg1_batch)}.onnx")


def seg1_path(cfg: Config) -> str:
    return _p(cfg, f"seg1_bs{int(cfg.batching.seg1_batch)}.onnx")


def seg2_static_path(cfg: Config, b: int) -> str:
    return _p(cfg, f"seg2_static_bs{int(b)}.onnx")


def seg2_dynamic_path(cfg: Config) -> str:
    return _p(cfg, "seg2_dynamic.onnx")


# --------------------------------------------------------------------------- #
def _export(module, args, path, input_names, output_names, dynamic_axes=None, force=False):
    import torch

    if os.path.exists(path) and not force:
        print(f"[export] cached: {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
        )
    print(f"[export] wrote: {path}")
    return path


def export_all(cfg: Config, which: tuple[str, ...] = ("plain", "naive", "proposed"), force: bool = False):
    import torch

    from .model_split import Seg1, Seg2, build_plain, load_ee_model

    os.makedirs(cfg.paths["onnx_dir"], exist_ok=True)
    S = int(cfg.batching.seg1_batch)
    H = W = int(cfg.data.crop)
    dummy_img = torch.randn(S, 3, H, W)
    hidden = int(cfg.model.hidden_dim)
    exit_after = int(cfg.model.exit_after_block)

    need_ee = any(r in which for r in ("naive", "proposed"))
    ee = load_ee_model(cfg.model.best_ckpt_path, int(cfg.model.num_classes)) if need_ee else None

    written = {}

    if "plain" in which:
        plain = build_plain(cfg.model.timm_model)
        written["plain"] = _export(
            plain, (dummy_img,), plain_path(cfg),
            input_names=["image"], output_names=["logits"], force=force,
        )

    if need_ee:
        seg1 = Seg1(ee, exit_after=exit_after)
        written["seg1"] = _export(
            seg1, (dummy_img,), seg1_path(cfg),
            input_names=["image"], output_names=["hidden", "lph_logits"], force=force,
        )

        # proposed: one static seg2 per flush size
        if "proposed" in which:
            seg2 = Seg2(ee, exit_after=exit_after)
            written["seg2_static"] = {}
            for b in _seg2_sizes(cfg):
                dummy_h = torch.randn(b, 197, hidden)
                written["seg2_static"][b] = _export(
                    seg2, (dummy_h,), seg2_static_path(cfg, b),
                    input_names=["hidden"], output_names=["logits"], force=force,
                )

        # naive: dynamic-batch seg2
        if "naive" in which:
            seg2d = Seg2(ee, exit_after=exit_after)
            dummy_h = torch.randn(S, 197, hidden)  # any batch; axis 0 is dynamic
            written["seg2_dynamic"] = _export(
                seg2d, (dummy_h,), seg2_dynamic_path(cfg),
                input_names=["hidden"], output_names=["logits"],
                dynamic_axes={"hidden": {0: "n"}, "logits": {0: "n"}}, force=force,
            )

    return written
