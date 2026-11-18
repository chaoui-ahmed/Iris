"""Tiny U-Net for iris segmentation (4 classes: background, sclera, iris, pupil).

Depthwise-separable convolutions keep it around 0.2M parameters (~1 MB as ONNX), so it runs in a few
milliseconds in the browser.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each followed by BatchNorm and ReLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.a = DSConv(cin, cout, stride)
        self.b = DSConv(cout, cout)

    def forward(self, x):
        return self.b(self.a(x))


class IrisNet(nn.Module):
    def __init__(self, n_classes=4, w=(16, 24, 40, 64, 96)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        self.e1 = Block(w[0], w[0])            # 1/2 (stem already halves)
        self.e2 = Block(w[0], w[1], 2)         # 1/2
        self.e3 = Block(w[1], w[2], 2)         # 1/4
        self.e4 = Block(w[2], w[3], 2)         # 1/8
        self.e5 = Block(w[3], w[4], 2)         # 1/16 (wide context: is this an eye at all?)
        self.d4 = Block(w[4] + w[3], w[3])
        self.d3 = Block(w[3] + w[2], w[2])
        self.d2 = Block(w[2] + w[1], w[1])
        self.d1 = Block(w[1] + w[0], w[0])
        self.head = nn.Conv2d(w[0], n_classes, 1)

    @staticmethod
    def up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s = self.stem(x)
        e1 = self.e1(s)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        d4 = self.d4(torch.cat([self.up(e5, e4), e4], 1))
        d3 = self.d3(torch.cat([self.up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2, e1), e1], 1))
        return F.interpolate(self.head(d1), size=x.shape[-2:], mode="bilinear", align_corners=False)


if __name__ == "__main__":
    m = IrisNet()
    print(sum(p.numel() for p in m.parameters()), "parameters")
    print(m(torch.zeros(1, 3, 128, 128)).shape)
