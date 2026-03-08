import torch
import torch.nn as nn
import torch.nn.functional as F

class PreCM(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 groups: int=1,
                 bias: int=0
                 ):
        super(PreCM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        weight_tensor = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor)
        self.convtest = nn.Conv2d(in_channels // groups, out_channels, kernel_size, bias=False)


    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2)
        pl = int(prl // 2)
        pa = pab - pb
        pr = prl - pl
        padding = (pa, pb, pl, pr)
        input = torch.cat([input,
                           torch.rot90(input, k=-1, dims=(2, 3)),
                           torch.rot90(input, k=-2, dims=(2, 3)),
                           torch.rot90(input, k=-3, dims=(2, 3))], dim=0)
        return F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, groups=self.groups)



# ----------------- 测试 -----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入张量尺寸 B, C, H, W
    x = torch.randn(1, 32, 256, 256, device=device)

    # 实例化 PreCM 模块
    net = PreCM(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
    ).to(device)

    # 前向计算
    y = net(x, output_shape=(256,256))

    # 打印结果
    print(net)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家复现! \n")
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
