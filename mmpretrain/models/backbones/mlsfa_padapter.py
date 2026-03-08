# -*- coding: utf-8 -*-
"""
MLSFA Adapter with integrated Prompt-style bottleneck (PAdapter-inspired).

NOTE: This file was generated based on the user's unpublished adapter.
It is intended for *your own* research/experimentation only.
Do NOT use it for training or distribution without the author's permission.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# BaseModule compatibility
# ---------------------------------------------------------------------------
try:
    from mmengine.model import BaseModule  # type: ignore
except Exception:
    # Fallback stub so this file is self-contained.
    # In your real project, you should import BaseModule from the correct place.
    class BaseModule(nn.Module):
        pass


# ---------------------------------------------------------------------------
# NOTE: We assume Config and QuantizeConv2d are defined elsewhere in your codebase,
# exactly as in your original project. Do NOT redefine them here to avoid conflicts.
#
# from your_quant_module import Config, QuantizeConv2d
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Prompt-style bottleneck (PAdapter-inspired)
# ---------------------------------------------------------------------------
class PromptBottleneck(nn.Module):
    """
    A simplified PAdapter-style bottleneck.

    - Input: x, shape (B, N, dim)
    - First project to a lower dimension r (bottleneck).
    - Concatenate with a learnable prompt vector in the bottleneck space.
    - Non-linear transform, then project back to dim.
    - Residual connection with optional learnable scale.

    This is designed to sit inside your MLSFA inner_dim bottleneck.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int = 64,
        dropout: float = 0.0,
        layernorm_option: str = "none",  # "none" / "in" / "out"
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.r = bottleneck
        self.dropout = dropout
        self.layernorm_option = layernorm_option

        if layernorm_option in ["in", "out"]:
            self.ln = nn.LayerNorm(dim)
        else:
            self.ln = None

        # Low-rank projection down / up.
        self.down_proj = nn.Linear(dim, self.r)
        # We will concatenate [down, prompt] along the channel dim -> 2 * r.
        self.up_proj = nn.Linear(self.r * 2, dim)

        # Global prompt in bottleneck space, later expanded to (B, N, r).
        self.prompt = nn.Parameter(torch.empty(1, 1, self.r))
        nn.init.xavier_uniform_(self.prompt)

        self.act = nn.ReLU()

        # Residual scaling.
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.tensor(1.0))

        # LoRA-style initialization: down random, up ~ 0 so we start as near-identity.
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, dim)
        Returns:
            Tensor of shape (B, N, dim) with residual connection applied.
        """
        residual = x

        if self.layernorm_option == "in" and self.ln is not None:
            x = self.ln(x)

        # (B, N, r)
        down = self.down_proj(x)
        B, N, _ = down.shape

        # Expand prompt to match (B, N, r)
        prompt = self.prompt.expand(B, N, self.r)

        # Concatenate along the channel dimension: (B, N, 2r)
        h = torch.cat([down, prompt], dim=-1)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Back to original dimension with a small residual effect initially.
        up = self.up_proj(h) * self.scale

        if self.layernorm_option == "out" and self.ln is not None:
            up = self.ln(up)

        return residual + up


# ---------------------------------------------------------------------------
# Original multi-scale spatial fusion adapter (feature-level, CNN-style)
# ---------------------------------------------------------------------------
class MLSFAFusion(BaseModule):
    """
    A more compact multi-scale convolutional adapter with quantized depthwise convs.

    Args:
        in_features: Number of channels for the inner bottleneck features.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()

        # Quantization config (assumed to be defined in your project).
        qcfg = Config()  # type: ignore[name-defined]
        qcfg.weight_bits = 8
        qcfg.input_bits = 8
        qcfg.clip_val = 2.0
        qcfg.recu = False

        # Depthwise quantized convolutions with different kernel sizes.
        # Original non-quantized versions (for reference):
        # self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        # self.conv5 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        # self.conv7 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)

        self.conv3 = QuantizeConv2d(  # type: ignore[name-defined]
            in_features,
            in_features,
            3,
            1,
            1,
            1,
            in_features,
            bias=True,
            config=qcfg,
        )
        self.conv5 = QuantizeConv2d(
            in_features,
            in_features,
            5,
            1,
            4,
            2,
            in_features,
            bias=True,
            config=qcfg,
        )
        self.conv7 = QuantizeConv2d(
            in_features,
            in_features,
            7,
            1,
            9,
            3,
            in_features,
            bias=True,
            config=qcfg,
        )

        # Learnable mixing weights for the three branches (scalar logits).
        # Softmax ensures they sum to 1; initialized as equal-weight mixture.
        self.mix_logits = nn.Parameter(torch.zeros(3))

        # 1x1 projection to fuse the multi-scale responses back into inner_dim channels.
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

        # Small residual scaling for stability (learnable).
        self.res_scale = nn.Parameter(torch.tensor(1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H, W) with scaled multi-scale residual.
        """
        u3 = self.conv3(x)
        u5 = self.conv5(x)
        u7 = self.conv7(x)

        w = F.softmax(self.mix_logits, dim=0)  # (3,)
        y = w[0] * u3 + w[1] * u5 + w[2] * u7

        y = self.projector(y)

        return x + self.res_scale * y


# ---------------------------------------------------------------------------
# MLSFA Adapter with integrated PromptBottleneck
# ---------------------------------------------------------------------------
class MLSFA(BaseModule):
    """
    Multi-Level Spatial Fusion Adapter for Transformer features, with an additional
    PAdapter-style PromptBottleneck in the inner_dim bottleneck.

    Overall structure (for x: (B, N, C), H * W == N):
        1) LayerNorm in token space.
        2) fc1: C -> inner_dim, then GELU (token-wise).
        3) Reshape to (B, inner_dim, H, W) and apply MLSFAFusion (multi-scale conv).
        4) Inner residual in bottleneck space with small learnable alpha.
        5) PromptBottleneck (PAdapter-inspired) in bottleneck space.
        6) GELU + Dropout + fc2 back to C, then outer residual.

    Args:
        in_dim:   Input channel dimension C of tokens.
        factor:   Reduction factor for inner_dim = max(in_dim // factor, 16).
        drop:     Dropout probability after bottleneck and before fc2.
        padapt_r: Bottleneck dimension r used inside PromptBottleneck.
        padapt_drop: Dropout probability used inside PromptBottleneck.
    """

    def __init__(
        self,
        in_dim: int,
        factor: int = 4,
        drop: float = 0.1,
        padapt_r: int = 64,
        padapt_drop: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = max(in_dim // factor, 16)

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, inner_dim)
        self.act = nn.GELU()

        # Spatial multi-scale fusion in CNN space.
        self.adapter_conv = MLSFAFusion(inner_dim)

        self.dropout = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(inner_dim, in_dim)

        # Small initialization for fc2 to avoid large early magnitudes.
        nn.init.normal_(self.fc2.weight, std=1e-3)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

        # Inner residual scaling in bottleneck space (very small, learnable).
        self.inner_alpha = nn.Parameter(torch.tensor(1e-3))

        # PAdapter-style PromptBottleneck applied in inner_dim space.
        self.prompt_bottleneck = PromptBottleneck(
            dim=inner_dim,
            bottleneck=padapt_r,
            dropout=padapt_drop,
            layernorm_option="none",  # You already have self.norm outside.
            learnable_scale=True,
        )

    def forward(self, x: torch.Tensor, hw_shapes: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x:  Tensor of shape (B, N, C), where N = H * W.
            hw_shapes: Tuple (H, W) specifying the spatial layout.
        Returns:
            Tensor of shape (B, N, C) with outer residual.
        """
        B, N, C = x.shape
        H, W = hw_shapes
        assert H * W == N, f"hw_shapes={hw_shapes} 与 tokens={N} 不匹配"

        identity = x

        # 1) Pre LayerNorm in token space.
        x = self.norm(x)

        # 2) Channel reduction + activation in token space.
        u = self.fc1(x)            # (B, N, inner_dim)
        u = self.act(u)

        # 3) Spatial multi-scale mixing via AdapterConv.
        u_skip = u                 # pre-spatial token representation
        u = u.view(B, H, W, -1).permute(0, 3, 1, 2)   # (B, inner_dim, H, W)
        u = self.adapter_conv(u)   # (B, inner_dim, H, W)
        u = u.permute(0, 2, 3, 1).contiguous().view(B, N, -1)

        # 4) Inner small residual in bottleneck space.
        u = u_skip + self.inner_alpha * u             # (B, N, inner_dim)

        # 4.5) PAdapter-style PromptBottleneck in bottleneck space.
        u = self.prompt_bottleneck(u)                 # (B, N, inner_dim)

        # 5) Activation + Dropout + projection back to original dim.
        u = self.act(u)
        u = self.dropout(u)
        u = self.fc2(u)

        # 6) Outer residual to maintain compatibility with Transformer block.
        return identity + u
