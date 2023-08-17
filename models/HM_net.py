import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.core import HPCA


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=strd,
        padding=padding,
        bias=bias,
        dilation=dilation,
    )


def conv_in_gelu(
    in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, dilation=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation,
        ),
        nn.InstanceNorm2d(out_dim),
        nn.GELU(),
    )


def dconv_in_gelu(
    in_dim, out_dim, kernel_size=3, stride=1, padding=1, output_padding=1
):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_dim, out_dim, kernel_size, stride, padding, output_padding
        ),
        nn.InstanceNorm2d(out_dim),
        nn.GELU(),
    )


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(
            int(out_planes / 2), int(out_planes / 4), padding=1, dilation=1
        )
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(
            int(out_planes / 4), int(out_planes / 4), padding=1, dilation=1
        )
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        # out3=self.fcvit(x)
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module("b1_" + str(level), ConvBlock(256, 256))

        self.add_module("b2_" + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module("b2_plus_" + str(level), ConvBlock(256, 256))

        self.add_module("b3_" + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules["b1_" + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules["b2_" + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules["b2_plus_" + str(level)](low2)

        low3 = low2
        low3 = self._modules["b3_" + str(level)](low3)

        up2 = F.upsample(low3, scale_factor=2, mode="nearest")

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class SeqGuidNET(nn.Module):
    def __init__(self, num_modules=1, end_relu=False):
        super(SeqGuidNET, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = conv3x3(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            if hg_module == 0:
                first_one = True
            else:
                first_one = False
            self.add_module("m" + str(hg_module), HourGlass(1, 3, 256, first_one))
            self.add_module("top_m_" + str(hg_module), ConvBlock(256, 256))
            self.add_module(
                "conv_last" + str(hg_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            )
            self.add_module("bn_end" + str(hg_module), nn.BatchNorm2d(256))
            self.add_module(
                "HPB" + str(hg_module),
                HPCA.HPB(dim=256, attn_height_top_k=16, attn_width_top_k=16),
            )

            if hg_module < self.num_modules - 1:
                self.add_module(
                    "l" + str(hg_module),
                    nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
                )
                self.add_module(
                    "bl" + str(hg_module),
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                )
                self.add_module(
                    "al" + str(hg_module),
                    nn.Conv2d(1, 256, kernel_size=1, stride=1, padding=0),
                )
            else:
                self.add_module(
                    "l" + str(hg_module),
                    nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0),
                )

    def SG_net(self, x):
        x = self.bn1(x)
        x = F.relu(x, True)

        x = self.conv1(x)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv4(x)

        previous = x
        heatmap_outputs = []
        tmp_out = None
        for i in range(self.num_modules):
            hg = self._modules["m" + str(i)](previous)
            ll = hg
            ll = self._modules["top_m_" + str(i)](ll)

            ll = F.relu(
                self._modules["bn_end" + str(i)](
                    self._modules["conv_last" + str(i)](ll)
                ),
                True,
            )

            tmp_out = self._modules["l" + str(i)](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out)

            if i < self.num_modules - 1:
                # previous:上一层Hourglass的输入
                # ll:上一层的特征层
                # tmp_out:上一层的输出
                ll = self._modules["bl" + str(i)](ll)
                tmp_out_ = self._modules["al" + str(i)](tmp_out)
                # 下一层的输入
                previous = self._modules["HPB" + str(i)](
                    previous, [tmp_out_, ll, previous]
                )
                previous = previous + ll + tmp_out_
            tmp_out = F.upsample(tmp_out, scale_factor=2, mode="nearest")
            heatmap_outputs.append(tmp_out)
        return heatmap_outputs

    def forward(self, x):
        heatmap_outputs = self.SG_net(x)
        return heatmap_outputs
