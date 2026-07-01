"""
EE-ViT-B/16 with LPH (Local Perception Head) exit

구조:
  Exit 1 (block 6)  : LocalPerceptionHead — LGViT (ACM MM 2023) 정확한 구현
                       conv1×1→GELU→BN → DW3×3→GELU→BN → conv1×1→BN
                       + residual → AdaptiveAvgPool1d → (pooled + CLS) → Linear
  Exit 2 (block 12) : Pretrained ViT-B/16 classifier (frozen)
                       timm vit.norm + vit.head 그대로 사용

Backbone (blocks 0~11): Frozen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

_PATCH_H = _PATCH_W = 14   # 224 / 16 = 14


class LocalPerceptionHead(nn.Module):
    """
    LPH: Local Perception Head — LGViT (ACM MM 2023) 원문 구조.
    highway_conv1_1 + DeiTHighway_v2 완전 동일 구현.

    입력 x : [B, 197, hidden_dim]  (CLS + 196 patch 토큰)
    출력   : [B, num_classes] logits
    """

    def __init__(self, hidden_dim: int, num_classes: int,
                 patch_h: int = _PATCH_H, patch_w: int = _PATCH_W,
                 drop: float = 0.0):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

        # 1×1 conv → GELU → BN
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=1e-5),
        )
        # DW 3×3 → GELU → BN
        self.proj     = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.proj_act = nn.GELU()
        self.proj_bn  = nn.BatchNorm2d(hidden_dim, eps=1e-5)
        # 1×1 conv → BN
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True),
            nn.BatchNorm2d(hidden_dim, eps=1e-5),
        )

        self.drop = nn.Dropout(drop)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls     = x[:, 0]       # [B, C]
        patches = x[:, 1:]      # [B, 196, C]
        B, N, C = patches.shape

        x0   = patches          # residual

        feat = patches.permute(0, 2, 1).reshape(B, C, self.patch_h, self.patch_w)
        feat = self.conv1(feat)
        feat = self.drop(feat)
        feat = self.proj(feat)
        feat = self.proj_act(feat)
        feat = self.proj_bn(feat)
        feat = self.conv2(feat)

        feat = feat.flatten(2).permute(0, 2, 1) + x0   # [B, 196, C] + residual
        feat = self.drop(feat)

        pooled = self.pool(feat.permute(0, 2, 1)).squeeze(-1)  # [B, C]
        return self.fc(pooled + cls)


class EEViTB16LPH(nn.Module):
    """
    ViT-B/16 기반 2-exit early exit 모델.

    학습 대상: lph_head (LocalPerceptionHead, block 6 exit) 만.
    고정:      backbone (blocks 0~11) + final_norm + final_fc (pretrained).

    forward(x, threshold=None):
      threshold=None  → 학습 모드. [logits_lph, logits_final] 반환.
      threshold=float → 추론 모드. (logits, exit_block) 반환.

    forward_lph_only(x):
      학습 전용 경량 경로. blocks 0~5 + LPH만 실행. 불필요한 blocks 6~11
      forward 없이 LPH loss만 계산할 때 사용.
    """

    TOTAL_BLOCKS = 12
    HIDDEN_DIM   = 768
    NUM_BLOCKS   = 2      # trainer 호환

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        vit = timm.create_model("vit_base_patch16_224", pretrained=True)

        # ── Backbone (frozen) ─────────────────────────────────────────────
        self.patch_embed = vit.patch_embed
        self.cls_token   = vit.cls_token
        self.pos_embed   = vit.pos_embed
        self.pos_drop    = vit.pos_drop
        self.blocks      = vit.blocks

        for p in self.patch_embed.parameters(): p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad  = False
        for p in self.blocks.parameters():     p.requires_grad = False

        # ── Exit 1: LPH at block 6 (trainable) ───────────────────────────
        self.lph_head = LocalPerceptionHead(self.HIDDEN_DIM, num_classes)

        # ── Exit 2: Pretrained final head (trainable, pretrained 초기화) ──
        # backbone은 frozen이지만 exit head는 joint training 대상.
        # pretrained 가중치에서 시작하므로 수렴이 빠르고 combined acc 방어 가능.
        self.final_norm = vit.norm
        self.final_fc   = vit.head

    @property
    def exit_block_labels(self) -> list:
        return ['B6', 'B12']

    @property
    def model_name(self) -> str:
        return "ee_vit_b16_lph_2exit"

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B   = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        return self.pos_drop(x + self.pos_embed)

    def forward_lph_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        학습 전용 경량 경로.
        blocks 0~5 + LPH만 실행하므로 blocks 6~11의 불필요한 forward를 건너뜀.
        LPH 파라미터에만 gradient가 흐름.
        """
        with torch.no_grad():
            x = self._embed(x)
            for i in range(6):
                x = self.blocks[i](x)
        return self.lph_head(x)

    def forward(self, x: torch.Tensor, threshold=None):
        x = self._embed(x)

        for i in range(6):
            x = self.blocks[i](x)

        if threshold is None:
            # 학습 / 평가 모드: 두 exit 모두 계산
            logits1 = self.lph_head(x)
            for i in range(6, 12):
                x = self.blocks[i](x)
            logits2 = self.final_fc(self.final_norm(x[:, 0]))
            return [logits1, logits2]

        else:
            # 추론 모드: confidence >= threshold 이면 즉시 종료
            logits1 = self.lph_head(x)
            conf    = F.softmax(logits1, dim=1).max(dim=1).values
            if conf.min().item() >= threshold:
                return logits1, 6
            for i in range(6, 12):
                x = self.blocks[i](x)
            logits2 = self.final_fc(self.final_norm(x[:, 0]))
            return logits2, 12


def build_model(num_classes: int = 1000) -> EEViTB16LPH:
    return EEViTB16LPH(num_classes=num_classes)


def print_trainable_params(model: EEViTB16LPH) -> None:
    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n{'─' * 64}")
    print(f"{'Trainable Parameters (EE-ViT-B/16 LPH)':^64}")
    print(f"{'─' * 64}")
    for name, shape in trainable:
        print(f"  {name:<50} {str(list(shape)):>10}")
    print(f"{'─' * 64}")
    print(f"  Backbone (blocks 0~11) : frozen      ({n_frozen:,} params)")
    print(f"  LPH head  (block  6)   : trainable  (random init)")
    print(f"  Final head (block 12)  : trainable  (pretrained init)")
    print(f"  Total trainable        : {n_train:,} params")
    print(f"{'─' * 64}\n")
