import torch.nn as nn
import math


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class MobileNet_V2(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilenetv2_bottleneck_setting,
                 arcface=None):
        super(MobileNet_V2, self).__init__()
        self.conv1 = ConvBlock(1, 32, 3, 2, 1)
        self.inplanes = 32
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 1280, 1, 1, 0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_out = nn.Linear(1280, num_class)
        self.arcface = arcface
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        # input(bs,1,128,313)
        x = self.conv1(x)  # (bs,64,64,157)

        x = self.blocks(x)  # (bs, 128,8,20)
        x = self.conv2(x)  # (bs, 512,8,20)

        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)  # (bs,128)
        if self.arcface is not None:
            out = self.arcface(feature, label)
        else:
            out = self.fc_out(feature)
        return out, feature


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 embedding_size=128,
                 arcface=None):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(1, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, embedding_size, 1, 1, 0, linear=True)
        self.fc_out = nn.Linear(embedding_size, num_class)
        self.arcface = arcface
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        # input(bs,1,128,313)
        x = self.conv1(x)  # (bs,64,64,157)
        x = self.dw_conv1(x)  # (bs, 64,64,157)
        x = self.blocks(x)  # (bs, 128,8,20)
        x = self.conv2(x)  # (bs, 512,8,20)
        x = self.linear7(x)  # (32,512,1,1)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)  # (bs,128)
        if self.arcface is not None:
            out = self.arcface(feature, label)
        else:
            out = self.fc_out(feature)
        return out, feature

class MFN_Classifier(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 embedding_size=128,
                 arcface=None):
        super(MFN_Classifier, self).__init__()
        self.mobilenetfacenet = MobileFaceNet(num_class=num_class,
                                              bottleneck_setting=bottleneck_setting,
                                              embedding_size=embedding_size,
                                              arcface=arcface)

    def forward(self, x_mel, label):
        out, feature = self.mobilenetfacenet(x_mel, label)
        return out, feature