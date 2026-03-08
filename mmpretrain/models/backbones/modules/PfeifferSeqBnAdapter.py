import torch
import torch.nn as nn
import torch.nn.functional as F


# class PfeifferSeqBnAdapter(nn.Module):
#     def __init__(self, hidden_size, inner_dim=64, non_linearity="relu", dropout=0.0,
#                  bias=True, init_up_as_zero=True):
#         super().__init__()
#         self.down = nn.Linear(hidden_size, inner_dim, bias=bias)
#         self.up = nn.Linear(inner_dim, hidden_size, bias=bias)
#         self.act = F.relu if non_linearity == "relu" else F.gelu
#         self.drop = nn.Dropout(dropout)
#
#         if init_up_as_zero:
#             nn.init.zeros_(self.up.weight)
#             if self.up.bias is not None:
#                 nn.init.zeros_(self.up.bias)
#
#     def forward(self, x):
#         return x + self.drop(self.up(self.act(self.down(x))))




class PfeifferSeqBnAdapter(nn.Module):
    """
    Pfeiffer / SeqBn-style bottleneck adapter:
      h <- h + W_up( act( W_down(h) ) )
    Default hyperparams aligned with common SeqBn usage:
      reduction_factor=16, non_linearity='relu', dropout=0.0, no new LayerNorm.
    """
    def __init__(
        self,
        hidden_size: int,
        reduction_factor: int = 16,
        non_linearity: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
        init_up_as_zero: bool = True,   # makes adapter start as (almost) identity
    ):
        super().__init__()
        assert reduction_factor > 0

        bottleneck_dim = max(1, hidden_size // reduction_factor)

        if non_linearity == "relu":
            act = F.relu
        elif non_linearity == "gelu":
            act = F.gelu
        elif non_linearity == "swish":
            act = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError("non_linearity must be one of: relu/gelu/swish")

        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=bias)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=bias)
        self.act = act
        self.drop = nn.Dropout(dropout)

        # Common stable init: start adapter near identity by zeroing up-proj.
        if init_up_as_zero:
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up(self.act(self.down(x)))
        z = self.drop(z)
        return x + z
