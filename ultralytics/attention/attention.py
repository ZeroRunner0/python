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


"""
CA注意力机制
"""


class CA(nn.Module):
    def __init__(self, c1, c2, ratio=3):
        super(CA, self).__init__()
        self.h_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.w_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_11 = nn.Conv2d(c1, c1//ratio, kernel_size=1, stride=1, bias=False)
        self.f_h = nn.Conv2d(c1//ratio, c1, kernel_size=1, stride=1, bias=False)
        self.f_w = nn.Conv2d(c1//ratio, c1, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm2d(c1//ratio)
        self.relu = nn.ReLU()
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # h方向avg_pool: b*c*1*w
        h_avg = self.h_avg_pool(x)

        # w方向avg_pool: b*c*h*1
        w_avg = self.w_avg_pool(x)
        # 维度交换: b*c*h*1->b*c*1*h
        w_avg = w_avg.permute(0, 1, 3, 2)

        # 拼接 & 通道压缩
        avg_out = torch.cat([h_avg, w_avg], dim=3)
        avg_out = self.conv_11(avg_out)

        # BN & relu
        avg_out = self.relu(self.bn(avg_out))

        # split
        h_split, w_split = torch.split(avg_out, [w, h], dim=3)

        # 维度交换: b*c*1*h->b*c*h*1
        w_split = w_split.permute(0, 1, 3, 2)

        # 通道数还原 & sigmoid
        w_out = self.sigmoid_w(self.f_w(w_split))
        h_out = self.sigmoid_h(self.f_h(h_split))

        return x * h_out * w_out


"""
SimAM注意力机制
"""


class SimAM(nn.Module):
    def __init__(self, c1, c2):
        super(SimAM, self).__init__()
        self.lamdba = 0.0001
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # spatial size
        n = X.shape[2] * X.shape[3] - 1

        """
        要添加keepdim=True, 
        不添加的话，结果的维度会变成2维的，和原来的4维不对应
        例如：原来b*c*h*w ---> b*c
        """
        # square of (t - u)
        x_mean = X.mean(dim=[2, 3], keepdim=True)
        d = (X - x_mean)
        d = d.pow(2)

        """
        要添加keepdim=True, 
        不添加的话，结果的维度会变成2维的，和原来的4维不对应
        例如：原来b*c*h*w ---> b*c
        """
        # d.sum() / n is channel variance
        v = d.sum(dim=[2, 3], keepdim=True)
        v = v / n

        # E_inv groups all importance of X
        E_inv = d / (4 * (v + self.lamdba)) + 0.5

        # return attended features
        return X * self.sigmoid(E_inv)


"""
Split Attention机制
"""


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = self.softmax(x)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAtt(nn.Module):
    def __init__(self, in_channel, out_channel, radix=2, groups=1, kernel_size=(3, 3),
                 reduction_factor=4, padding=(1, 1)):
        super(SplitAtt, self).__init__()

        inter_channels = max(in_channel * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channel, out_channel*radix, kernel_size=kernel_size, 
                              stride=1, padding=padding, groups=groups*radix, bias=False)
        self.bn_chn_radix = nn.BatchNorm2d(out_channel*radix)
        self.bn_chn_inter = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_chn_radix = nn.Conv2d(inter_channels, out_channel*radix, 1, groups=self.cardinality)
        self.fc_chn_inter = nn.Conv2d(out_channel, inter_channels, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

        # print(f"in_chn: {in_channel},"
        #       f"out_chn: {out_channel},"
        #       f"inter_channels: {inter_channels}"
        #       )

    def forward(self, x):

        # print(f"x size:{x.size()}")

        x_tmp = self.conv(x)

        # print(f"x_tmp size: {x_tmp.size()}")

        x_tmp = self.bn_chn_radix(x_tmp)
        x_tmp = self.relu(x_tmp)

        batch, rchannel = x_tmp.shape[:2]
        splited = torch.split(x_tmp, int(rchannel//self.radix), dim=1)

        # print(f"splited num: {len(splited)}, splited size: {splited[0].shape}")

        gap = sum(splited)

        gap = self.avg_pool(gap)

        # print(f"gap size: {gap.size()}")

        gap = self.fc_chn_inter(gap)

        # print(f"gap size: {gap.size()}")

        # gap = self.bn_chn_inter(gap)
        gap = self.relu(gap)

        atten = self.fc_chn_radix(gap)

        # print(f"atten size: {atten.size()}")

        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        # print(f"atten size: {atten.size()}")

        attens = torch.split(atten, rchannel//self.radix, dim=1)

        # print(f"attens num: {len(attens)}, attens size: {attens[0].shape}")

        out = sum([att*split for (att, split) in zip(attens, splited)])

        # print(f"out size: {out.size()}")

        return out.contiguous()


"""
S2-MLPv2机制
"""


class S2_MLPv2_SplitAtt(nn.Module):
    def __init__(self, channel, k=3):
        super(S2_MLPv2_SplitAtt, self).__init__()
        self.channel = channel
        self.k = k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gelu = nn.GELU()
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.mlp2 = nn.Linear(channel, channel*k, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(x_all, 1)
        a = torch.sum(a, 1)

        hat_a = self.mlp1(a)
        hat_a = self.gelu(hat_a)
        hat_a = self.mlp2(hat_a)
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


def spatial_shift1(x):
    b, w, h, c = x.size()
    # x_tmp = x.clone()

    # h方向右移
    x[:, :, 1:, 0:c//4] = x[:, :, :h - 1, 0:c//4]
    # print(x[:, :, :, 0:c//4])
    # h方向左移
    x[:, :, :h - 1, c//4:c//2] = x[:, :, 1:, c//4:c//2]
    # print(x[:, :, :, c//4:c//2])
    # w方向右移
    x[:, 1:, :, c//2:c*3//4] = x[:, :w - 1, :, c//2:c*3//4]
    # print(x[:, :, :, c//2:c*3//4])
    # w方向左移
    x[:, :w - 1, :, c*3//4:c] = x[:, 1:, :, c*3//4:c]
    # print(x[:, :, :, c*3//4:c])

    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    # x_tmp = x.clone()

    # w方向右移
    x[:, 1:, :, 0:c//4] = x[:, :w - 1, :, 0:c//4]
    # print(x[:, :, :, 0:c//4])
    # w方向左移
    x[:, :w - 1, :, c//4:c//2] = x[:, 1:, :, c//4:c//2]
    # print(x[:, :, :, c//4:c//2])
    # h方向右移
    x[:, :, 1:, c//2:c*3//4] = x[:, :, :h - 1, c//2:c*3//4]
    # print(x[:, :, :, c//2:c*3//4])
    # h方向左移
    x[:, :, :h - 1, c*3//4:c] = x[:, :, 1:, c*3//4:c]
    # print(x[:, :, :, c*3//4:c])

    return x


class S2_MLPv2(nn.Module):
    def __init__(self, channel, out_channel):
        super(S2_MLPv2, self).__init__()
        self.SplitAtt = S2_MLPv2_SplitAtt(channel)
        self.shift1 = spatial_shift1
        self.shift2 = spatial_shift2
        self.mlp1 = nn.Linear(channel, channel * 3)
        self.mlp2 = nn.Linear(channel, channel)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = self.shift1(x[:, :, :, 0:c])
        x2 = self.shift2(x[:, :, :, c:c*2])
        x3 = x[:, :, :, c*2:c*3]

        x_cat = torch.stack((x1, x2, x3), dim=1)
        out = self.SplitAtt(x_cat)
        x = self.mlp2(out)
        x = x.permute(0, 3, 1, 2)
        return x





