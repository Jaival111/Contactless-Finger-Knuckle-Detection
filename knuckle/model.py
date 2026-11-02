"""Residual Feature Network (RFN) backbone, Fig. 1 of the paper.

    input I x I (1 channel)
    conv1 : 32  ch, stride 2           -> I/2
    conv2 : 64  ch, stride 2           -> I/4
    conv3 : 128 ch                     -> I/4
    res1..res4 : 2 x (128 ch) + skip   -> I/4
    conv4 : 64 ch                      -> I/4
    conv5 : 1  ch (template)           -> I/4

For a 128 x 128 ROI this yields the 32 x 32 template reported in Table II.
Setting ``deformable=True`` replaces the convolutions of the residual blocks
by deformable convolutions, i.e. the "DefConvNet" baseline of Sec. IV.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, stride=1, act=True, deformable=False):
        super().__init__()
        pad = k // 2
        if deformable:
            self.offset = nn.Conv2d(cin, 2 * k * k, k, stride, pad)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.conv = DeformConv2d(cin, cout, k, stride, pad, bias=False)
        else:
            self.offset = None
            self.conv = nn.Conv2d(cin, cout, k, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x, self.offset(x)) if self.offset is not None else self.conv(x)
        return self.act(self.bn(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels, deformable=False):
        super().__init__()
        self.conv_a = ConvBNAct(channels, channels, 3, deformable=deformable)
        self.conv_b = ConvBNAct(channels, channels, 3, act=False, deformable=deformable)

    def forward(self, x):
        return F.relu(x + self.conv_b(self.conv_a(x)), inplace=True)


class RFN(nn.Module):
    """Fully convolutional residual feature network producing a 1-channel template."""

    def __init__(self, in_channels=1, deformable=False, num_res_blocks=4):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, 32, 7, stride=2)
        self.conv2 = ConvBNAct(32, 64, 5, stride=2)
        self.conv3 = ConvBNAct(64, 128, 3)
        self.res = nn.Sequential(*[ResidualBlock(128, deformable) for _ in range(num_res_blocks)])
        self.conv4 = ConvBNAct(128, 64, 3)
        self.conv5 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.res(x)
        return self.conv5(self.conv4(x))


def build_model(name="rfn", in_channels=1):
    if name == "rfn":
        return RFN(in_channels)
    if name == "defconv":
        return RFN(in_channels, deformable=True)
    raise ValueError(f"Unknown backbone '{name}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
