# -*- coding: utf-8 -*-
"""
Mona_PathoMSF_v4_ablation (config-agnostic, Python<=3.9 compatible)

This file is based on your v4_compat, but adds **ablation flags** so you can run
the following settings by only changing constructor args (no config edits):

1) Full (default)
2) w/o high-pass          -> enable_highpass=False
3) w/o low-rank mixer     -> enable_mixer=False
4) w/o 7x7 augmentation   -> enable_7x7_aug=False
5) selector uniform       -> selector_type="uniform"  (no learnable selector)

Drop-in usage:
    from .Mona_PathoMSF_v4_ablation import Mona_PathoMSF
(or rename this file to match your import path)

Recommended naming in logs:
    model.backbone...adapter_cfg = dict(enable_highpass=..., ...)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mmcv.cnn import BaseModule  # type: ignore
except Exception:
    BaseModule = nn.Module


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth per sample."""
    if drop_prob <= 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    return x.div(keep_prob) * binary_mask


class DWAnisotropic(nn.Module):
    """Depthwise (1 x k) + (k x 1) anisotropic conv (depthwise)."""
    def __init__(self, channels: int, k: int = 7):
        super().__init__()
        pad = k // 2
        self.h = nn.Conv2d(channels, channels, kernel_size=(1, k), padding=(0, pad),
                           groups=channels, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=(k, 1), padding=(pad, 0),
                           groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.h(x))


class SKChannelSelectorHybrid(nn.Module):
    """
    Hybrid selector (learnable, default):
      logits = base_logits + ctx_scale * mlp_delta(desc)
      weights = softmax(logits / tau, dim=branch)

    - base_logits: [K, C] learnable per-branch per-channel bias (robust under AdamW weight_decay).
    - mlp_delta: last layer is zero-initialized => safe start (uniform mixing).
    - desc: computed from fused sum of branches, LN-normalized.
    """
    def __init__(self, channels: int, num_branches: int, reduction: int = 4, tau: float = 1.5):
        super().__init__()
        self.channels = channels
        self.num_branches = num_branches
        self.tau = float(tau)

        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels * num_branches, bias=True)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.base_logits = nn.Parameter(torch.zeros(num_branches, channels))
        self.ctx_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, feats):
        assert len(feats) == self.num_branches
        B, C, _, _ = feats[0].shape

        u = feats[0]
        for i in range(1, self.num_branches):
            u = u + feats[i]
        s = u.mean(dim=(2, 3))          # [B, C]
        s = F.layer_norm(s, (C,))

        z = self.act(self.fc1(s))
        delta = self.fc2(z).view(B, self.num_branches, C)  # [B, K, C]

        logits = self.base_logits.unsqueeze(0) + self.ctx_scale * delta
        attn = torch.softmax(logits / self.tau, dim=1)      # over branches
        attn = attn.unsqueeze(-1).unsqueeze(-1)             # [B, K, C, 1, 1]
        return [attn[:, i] for i in range(self.num_branches)]


class UniformSelector(nn.Module):
    """Ablation: uniform weights (no learnable selector)."""
    def __init__(self, num_branches: int):
        super().__init__()
        self.num_branches = int(num_branches)

    def forward(self, feats):
        assert len(feats) == self.num_branches
        B, C, _, _ = feats[0].shape
        w = torch.full((B, C, 1, 1), 1.0 / self.num_branches, device=feats[0].device, dtype=feats[0].dtype)
        return [w for _ in range(self.num_branches)]


class MonaOp_PathoMSF(BaseModule):
    """
    Multi-scale depthwise conv operator with ablation flags.

    Branches (core):
      - DW 3x3
      - DW 5x5
      - DW 7x7

    Optional (enable_7x7_aug):
      - + alpha_aniso * anisotropic(x)
      - + alpha_dil   * dilated_dw3(x)

    Optional (enable_highpass):
      - + alpha_hp * (x - avgpool(x))

    Optional (enable_mixer):
      - + low-rank pointwise mixer: C -> r -> C (up-proj zero-init)

    selector_type:
      - "hybrid"  (default): SKChannelSelectorHybrid
      - "uniform": UniformSelector
    """
    def __init__(
        self,
        in_features: int,
        dilation: int = 2,
        aniso_k: int = 7,
        progressive: bool = True,
        tau: float = 1.5,
        mixer_rank: Optional[int] = None,
        enable_7x7_aug: bool = True,
        enable_highpass: bool = True,
        enable_mixer: bool = True,
        selector_type: str = "hybrid",
    ):
        super().__init__()
        C = int(in_features)
        self.progressive = bool(progressive)

        self.enable_7x7_aug = bool(enable_7x7_aug)
        self.enable_highpass = bool(enable_highpass)
        self.enable_mixer = bool(enable_mixer)

        selector_type = str(selector_type).lower()
        if selector_type not in ("hybrid", "uniform"):
            raise ValueError(f"selector_type must be 'hybrid' or 'uniform', got: {selector_type}")
        self.selector_type = selector_type

        # Core isotropic branches
        self.dw3 = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.dw5 = nn.Conv2d(C, C, kernel_size=5, padding=2, groups=C, bias=False)
        self.dw7 = nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C, bias=False)

        # Optional 7x7 augmentation paths
        if self.enable_7x7_aug:
            self.dwaniso = DWAnisotropic(C, k=aniso_k)
            self.dwdil = nn.Conv2d(
                C, C, kernel_size=3, padding=dilation, dilation=dilation,
                groups=C, bias=False
            )
            self.alpha_aniso = nn.Parameter(torch.tensor(0.0))
            self.alpha_dil = nn.Parameter(torch.tensor(0.0))
        else:
            self.dwaniso = None
            self.dwdil = None
            self.register_buffer("alpha_aniso", torch.tensor(0.0), persistent=False)
            self.register_buffer("alpha_dil", torch.tensor(0.0), persistent=False)

        # Optional high-pass residual (no conv params)
        if self.enable_highpass:
            self.alpha_hp = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("alpha_hp", torch.tensor(0.0), persistent=False)

        # Selector (3 branches)
        self.K = 3
        if self.selector_type == "hybrid":
            self.selector = SKChannelSelectorHybrid(C, num_branches=self.K, reduction=4, tau=tau)
        else:
            self.selector = UniformSelector(num_branches=self.K)

        # Optional low-rank mixer
        if self.enable_mixer:
            r = int(mixer_rank) if mixer_rank is not None else max(8, C // 8)
            self.pw_down = nn.Conv2d(C, r, kernel_size=1, bias=True)
            self.pw_up = nn.Conv2d(r, C, kernel_size=1, bias=True)
            nn.init.zeros_(self.pw_up.weight)
            nn.init.zeros_(self.pw_up.bias)
        else:
            self.pw_down = None
            self.pw_up = None

        # GLU projector
        self.projector = nn.Conv2d(C, C * 2, kernel_size=1, bias=True)

    @staticmethod
    def _highpass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Core multiscale
        if self.progressive:
            y3 = self.dw3(x)
            y5 = self.dw5(x + y3)
            y7_base = self.dw7(x + y5)
        else:
            y3 = self.dw3(x)
            y5 = self.dw5(x)
            y7_base = self.dw7(x)

        # Optional 7x7 augmentation
        if self.enable_7x7_aug:
            y7 = y7_base + self.alpha_aniso * self.dwaniso(x) + self.alpha_dil * self.dwdil(x)
        else:
            y7 = y7_base

        feats = [y3, y5, y7]
        w3, w5, w7 = self.selector(feats)

        fused = y3 * w3 + y5 * w5 + y7 * w7
        x = fused + identity

        # Optional high-pass
        if self.enable_highpass:
            x = x + self.alpha_hp * self._highpass(identity)

        # Optional low-rank mixer
        if self.enable_mixer:
            x = x + self.pw_up(F.gelu(self.pw_down(x)))

        # GLU residual
        identity2 = x
        a, b = self.projector(x).chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        return identity2 + x


class Mona_PathoMSF(BaseModule):
    """
    Token-space adapter with ablation flags (mirrors MonaOp_PathoMSF flags).

    Default behavior (Full) matches v4_compat logic + drop_path.

    Flags:
      - enable_highpass, enable_mixer, enable_7x7_aug
      - selector_type: "hybrid" or "uniform"
    """
    def __init__(
        self,
        in_dim: int,
        inner_dim: int = 64,
        drop: float = 0.1,
        dilation: int = 2,
        aniso_k: int = 7,
        progressive: bool = True,
        tau: float = 1.5,
        use_norm_gating: bool = True,
        mixer_rank: Optional[int] = None,
        drop_path_prob: Optional[float] = None,
        # ablation flags
        enable_7x7_aug: bool = True,
        enable_highpass: bool = True,
        enable_mixer: bool = True,
        selector_type: str = "hybrid",
    ):
        super().__init__()
        self.use_norm_gating = bool(use_norm_gating)

        self.project1 = nn.Linear(in_dim, inner_dim)
        self.project2 = nn.Linear(inner_dim, in_dim)
        self.dropout = nn.Dropout(p=drop)

        self.adapter_conv = MonaOp_PathoMSF(
            inner_dim,
            dilation=dilation,
            aniso_k=aniso_k,
            progressive=progressive,
            tau=tau,
            mixer_rank=mixer_rank,
            enable_7x7_aug=enable_7x7_aug,
            enable_highpass=enable_highpass,
            enable_mixer=enable_mixer,
            selector_type=selector_type,
        )

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

        if drop_path_prob is None:
            drop_path_prob = min(0.1, float(drop) * 0.5)
        self.drop_path_prob = float(drop_path_prob)

    def forward(self, x: torch.Tensor, hw_shapes=None) -> torch.Tensor:
        identity = x

        if self.use_norm_gating:
            x = self.norm(x) * self.gamma + x * self.gammax

        y = self.project1(x)  # [B, N, C]
        b, n, c = y.shape
        h, w = hw_shapes
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]

        y = self.adapter_conv(y)

        y = y.permute(0, 2, 3, 1).reshape(b, n, c)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.project2(y)

        y = drop_path(y, drop_prob=self.drop_path_prob, training=self.training)
        return identity + y


def _sanity():
    m = Mona_PathoMSF(in_dim=768, inner_dim=64)
    x = torch.randn(2, 196, 768)
    y = m(x, (14, 14))
    return y.shape

if __name__ == "__main__":
    print(_sanity())
