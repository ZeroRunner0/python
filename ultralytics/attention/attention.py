import torch
from torch import nn

"""
SE注意力机制
"""


class SE(nn.Module):
    def __init__(self, c1, c2, r=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        print(x.size())
        """
        b: batch size
        c: channels number
        _: width
        _: high

        可以这样理解：
        获取有多少个特征图，一个特征图有多少个通道，宽和高
        """
        b, c, _, _ = x.size()

        # 每个通道进行全局平均化，最后结果只有一个数值
        # 有多少个通道，就有多少个数值
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        # y.expand_as(x)表示生成每一个通道的权重矩阵
        return x * y.expand_as(x)


"""
CBAM注意力机制
"""


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        # 目的是：卷积后，特征图的大小不变，只是通道数改变了
        padding = 3 if (kernel_size == 7) else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"size of x: {x.size()} type:{x.type()}")

        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(f"max_pool: {avg_out.size()} type:{avg_out.type()}")
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(f"max_out: {max_out.size()} type:{max_out.type()}")
        x = torch.cat([avg_out, max_out], dim=1)
        # print(f"x1: {x.size()} type:{x.type()}")
        # 2*h*w
        x = self.conv(x)
        # print(f"x2: {x.size()} type:{x.type()}")
        # 1*h*w
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # print(f"in_planes:{in_planes}, in_planes//ratio: {in_planes//ratio}")
        self.l1 = nn.Linear(in_planes, in_planes//ratio, bias=False)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(in_planes//ratio, in_planes, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.l1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.l2(avg_out)

        # print(f"ChannelAttention avg_out: {avg_out.size()} type:{avg_out.type()}")

        max_out = self.max_pool(x).view(b, c)
        max_out = self.l1(max_out)
        max_out = self.relu(max_out)
        max_out = self.l2(max_out)

        # print(f"ChannelAttention max_out: {max_out.size()} type:{max_out.type()}")

        out = self.sigmoid(avg_out + max_out)
        out = out.view(b, c, 1, 1)

        # print(f"ChannelAttention out: {out.size()} type:{out.type()}")
        return out.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, c1, c2, r=16, k_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes=c1, ratio=r)
        self.spatial_attention = SpatialAttention(kernel_size=k_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print(f"CBAM out1:{out.size()} type:{out.type()}")
        out = self.spatial_attention(out) * out
        # print(f"CBAM out2:{out.size()} type:{out.type()}")
        return out


"""
ECA注意力机制
"""


class ECA(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.squeeze(-1)
        # 理解成矩阵转置，变成行向量
        out = out.transpose(-1, -2)
        out = self.conv(out)
        out = out.transpose(-1, -2)
        out = out.unsqueeze(-1)
        out = self.sigmoid(out)
        out = out.expand_as(x)

        return out * x
