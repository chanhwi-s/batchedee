"""Split EE-ViT-B/16 LPH into seg1 / seg2, and build the `plain` model.

seg1 = patch_embed + pos + blocks[0:6] + lph_head
       -> outputs (hidden_tokens [B,197,768], lph_logits [B,1000])
       hidden_tokens feed seg2; lph_logits drive the per-sample exit decision.

seg2 = blocks[6:12] + final_norm + final_fc  (input: hidden_tokens)
       -> outputs final_logits [B,1000]

plain = ImageNet-pretrained timm vit_base_patch16_224, whole model, no exit.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Make repo root importable so `models` package resolves regardless of cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.ee_vit_b16_lph import EEViTB16LPH  # noqa: E402


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ee_model(ckpt_path: str, num_classes: int = 1000) -> EEViTB16LPH:
    """Instantiate EE-ViT-B/16 LPH and load the trained checkpoint (eval mode).

    eval() is essential: LPH contains BatchNorm/Dropout, so exit confidences are
    only correct in eval mode.
    """
    model = EEViTB16LPH(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only tolerate benign mismatches; surface anything unexpected.
    hard_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if hard_missing:
        raise RuntimeError(f"Missing keys when loading {ckpt_path}: {hard_missing[:10]} ...")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {ckpt_path}: {unexpected[:10]} ...")
    model.eval()
    return model


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    # strip a possible "module." prefix from DataParallel
    return { (k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items() }


# --------------------------------------------------------------------------- #
# Wrappers (export targets)
# --------------------------------------------------------------------------- #
class Seg1(nn.Module):
    """image -> (hidden_tokens, lph_logits)."""

    def __init__(self, ee: EEViTB16LPH, exit_after: int = 6):
        super().__init__()
        self.patch_embed = ee.patch_embed
        self.cls_token = ee.cls_token
        self.pos_embed = ee.pos_embed
        self.pos_drop = ee.pos_drop
        self.blocks = ee.blocks
        self.lph_head = ee.lph_head
        self.exit_after = exit_after

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i in range(self.exit_after):
            x = self.blocks[i](x)
        lph_logits = self.lph_head(x)
        return x, lph_logits


class Seg2(nn.Module):
    """hidden_tokens -> final_logits (blocks[exit_after:12] + final head)."""

    def __init__(self, ee: EEViTB16LPH, exit_after: int = 6, total_blocks: int = 12):
        super().__init__()
        self.blocks = ee.blocks
        self.final_norm = ee.final_norm
        self.final_fc = ee.final_fc
        self.exit_after = exit_after
        self.total_blocks = total_blocks

    def forward(self, h: torch.Tensor):
        x = h
        for i in range(self.exit_after, self.total_blocks):
            x = self.blocks[i](x)
        return self.final_fc(self.final_norm(x[:, 0]))


def build_plain(timm_model: str = "vit_base_patch16_224") -> nn.Module:
    """ImageNet-pretrained full ViT-B/16 (no early exit). eval mode."""
    import timm

    m = timm.create_model(timm_model, pretrained=True)
    m.eval()
    return m
