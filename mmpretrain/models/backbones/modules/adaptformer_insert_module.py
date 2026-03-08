"""
adaptformer_insert_module.py

AdaptFormer-style *insertable* adapter for SwinBlock-style code paths like:

    x = x + Attn(norm1(x))
    x = Adapter1(x, hw_shape)
    x = FFN(norm2(x), identity=x)
    x = Adapter2(x, hw_shape)

This matches the insertion pattern shown in your screenshot (two post-sub-layer inserts),
and keeps the call signature Adapter(x, hw_shape) for drop-in replacement of Mona_PathoMSF.

IMPORTANT NAMING NOTE
---------------------
"AdaptFormer" in the original paper is an FFN-parallel adapter (FFN(x) + s*adapter(x)).
What you are asking for here is *an AdaptFormer-style bottleneck MLP inserted as a residual
module after attention and after FFN* (2 inserts per block). This is a valid PEFT baseline,
but to avoid confusion in papers, you may want to name it "AdaptFormer-2x-insert" or similar.

Module definition (residual form):
    y = x + s * Up(ReLU(Down(x)))

Shapes:
- x is typically (B, L, C) for ViT/Swin token sequences.
- hw_shape is accepted for compatibility but not used.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class InsertAdapterCfg:
    embed_dims: int
    inner_dim: int = 64          # bottleneck
    scale: float = 0.1           # s
    dropout: float = 0.0
    learnable_scale: bool = False
    zero_init_up: bool = True


class AdaptFormerInsert(nn.Module):
    """
    Residual bottleneck adapter that can be inserted anywhere in the block.
    Signature: forward(x, hw_shape=None) -> x'
    """
    def __init__(self, cfg: InsertAdapterCfg):
        super().__init__()
        self.cfg = cfg

        self.down = nn.Linear(cfg.embed_dims, cfg.inner_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=cfg.dropout) if cfg.dropout and cfg.dropout > 0 else nn.Identity()
        self.up = nn.Linear(cfg.inner_dim, cfg.embed_dims, bias=True)

        if cfg.learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(cfg.scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(cfg.scale)), persistent=False)

        if cfg.zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, hw_shape=None) -> torch.Tensor:
        # hw_shape is intentionally ignored; kept only for API compatibility with Mona_PathoMSF
        y = self.down(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.up(y)
        return x + self.scale * y


def mark_only_inserts_trainable(model: nn.Module, train_head: bool = True) -> None:
    """
    Freeze all params, then unfreeze:
      - AdaptFormerInsert modules
      - optional classifier head (heuristic: 'head'/'classifier'/'fc')
    """
    for p in model.parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, AdaptFormerInsert):
            for p in m.parameters():
                p.requires_grad = True

    if train_head:
        for name, m in model.named_modules():
            if name.endswith(("head", "classifier", "fc")):
                for p in m.parameters():
                    p.requires_grad = True


def get_trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)
