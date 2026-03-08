
import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 更合理的轻量 ARConv：保留方向建模特性 + 控制参数规模
class EfficientARConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        inter_c = in_c // 2

        # 横向 5x1 和纵向 1x5 卷积
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_c, inter_c, kernel_size=(5, 1), padding=(2, 0), groups=1),
            nn.BatchNorm2d(inter_c),
            nn.GELU()
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_c, inter_c, kernel_size=(1, 5), padding=(0, 2), groups=1),
            nn.BatchNorm2d(inter_c),
            nn.GELU()
        )

        # 深度可分离卷积捕捉局部特征
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.GELU()
        )

        # 融合后线性映射
        self.proj = nn.Conv2d(in_c + inter_c * 2, out_c, kernel_size=1)

    def forward(self, x, **kwargs):
        x_h = self.conv_h(x)
        x_v = self.conv_v(x)
        x_dw = self.dwconv(x)
        x_cat = torch.cat([x_dw, x_h, x_v], dim=1)
        x_out = self.proj(x_cat)
        return x_out
