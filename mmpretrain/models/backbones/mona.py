from mmengine.model import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F

# INNER_DIM = 64
class MonaOp(BaseModule):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

class Mona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        inner_dim = in_dim // factor

        self.project1 = nn.Linear(in_dim, inner_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(inner_dim, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(inner_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2
# ---------------------------- 插入模式 -----------------------------
# 此处省略部分 Swin 组件实现，仅提供 Mona 插入模式。
# class SwinBlock(BaseModule):
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  feedforward_channels,
#                  window_size=7,
#                  shift=False,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN'),
#                  with_cp=False,
#                  init_cfg=None):
#
#         super(SwinBlock, self).__init__()
#
#         self.init_cfg = init_cfg
#         self.with_cp = with_cp
#
#         self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
#         self.attn = ShiftWindowMSA(
#             embed_dims=embed_dims,
#             num_heads=num_heads,
#             window_size=window_size,
#             shift_size=window_size // 2 if shift else 0,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop_rate=attn_drop_rate,
#             proj_drop_rate=drop_rate,
#             dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#             init_cfg=None)
#
#         self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
#         self.ffn = FFN(
#             embed_dims=embed_dims,
#             feedforward_channels=feedforward_channels,
#             num_fcs=2,
#             ffn_drop=drop_rate,
#             dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#             act_cfg=act_cfg,
#             add_identity=True,
#             init_cfg=None)
#
#         self.mona1 = Mona(embed_dims, 8)
#         self.mona2 = Mona(embed_dims, 8)
    # def forward(self, x, hw_shape):
    #
    #     def _inner_forward(x):
    #         identity = x
    #         x = self.norm1(x)
    #         x = self.attn(x, hw_shape)
    #
    #         x = x + identity
    #
    #         x = self.mona1(x, hw_shape)
    #
    #         identity = x
    #         x = self.norm2(x)
    #         x = self.ffn(x, identity=identity)
    #
    #         x = self.mona2(x, hw_shape)
    #
    #         return x
    #
    #     if self.with_cp and x.requires_grad:
    #         x =self.with_cp.checkpoint(_inner_forward, x)
    #     else:
    #         x = _inner_forward(x)
    #
    #     return x


#---------------------------------------------------------