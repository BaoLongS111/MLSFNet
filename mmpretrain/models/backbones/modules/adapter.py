import torch
import torch.nn as nn


class HoulsbyAdapter(nn.Module):
    """
    Houlsby et al. (2019) bottleneck adapter (serial adapter).
    Output: x + W_up( f( W_down(x) ) )
    - Internal residual/skip connection inside the adapter
    - Initialize W_up ~ 0 so the adapter starts as near-identity
    """
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 48,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        act = act.lower()
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu" or act == "swish":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.GELU()

        # Key trick: near-identity at init
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, hw_shape=None) -> torch.Tensor:
        # x: (B, N, C) or (B, C)
        h = self.down(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + h
