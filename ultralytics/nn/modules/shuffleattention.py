import torch
from torch import nn
from torch.nn.parameter import Parameter


class ShuffleAttention(nn.Module):
    def __init__(self, channel, groups=64):
        super(ShuffleAttention, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_weight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.spatial_weight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.channel_bias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.spatial_bias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.group_norm = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def channel_shuffle(x, groups):
        b, _, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, _, h, w = x.shape
        res = x.clone()

        x_grouped = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x_grouped.chunk(2, dim=1)

        x_channel = self.avg_pool(x_0)
        x_channel = self.channel_weight * x_channel + self.channel_bias
        x_channel = x_0 * self.sigmoid(x_channel)

        x_spatial = self.group_norm(x_1)
        x_spatial = self.spatial_weight * x_spatial + self.spatial_bias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        out += res
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SABlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, width)
        self.bn3 = norm_layer(planes)
        self.sa = ShuffleAttention(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sa(out)

        out += identity
        out = self.relu(out)

        return out


class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()

    def forward(self, x):
        return x
