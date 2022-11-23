import torch
import torch.nn as nn
import math
from nnet.wavegram import Wavegram


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
    def __init__(self, inp, oup, k, s, p, dw=False, twostage=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False),
                nn.BatchNorm2d(oup)
            )
        elif twostage:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, oup, k, s, p, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup, oup, k, 1, p, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, oup, k, s, p, bias=False),
                nn.BatchNorm2d(oup)
            )
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Simple_Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 input_channel=1,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 embedding_size=128,
                 arcface=None):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(input_channel, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, embedding_size, 1, 1, 0, linear=True)
        self.fc_out = nn.Linear(embedding_size, num_class)
        self.arcface = arcface
        self.dropout = nn.Dropout()
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
        # x = x.transpose(2, 3)
        # input(bs,1,128,313)

        x = self.conv1(x)  # (bs,64,64,157)
        x = self.dw_conv1(x)  # (bs, 64,64,157)
        x = self.blocks(x)  # (bs, 128,8,20)
        x = self.conv2(x)  # (bs, 512,8,20)
        x = self.linear7(x)  # (32,512,1,1)
        x = self.linear1(x)
        x = self.dropout(x)
        feature = x.view(x.size(0), -1)  # (bs,128)
        if self.arcface is not None:
            out = self.arcface(feature, label)
        else:
            out = self.fc_out(feature)
        return out, feature


class SimpleMobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Simple_Mobilefacenet_bottleneck_setting,
                 embedding_size=128,
                 arcface=None):
        super(SimpleMobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(1, 64, 3, 2, 1)
        self.conv2 = ConvBlock(64, 64, 3, 1, 1)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv3 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, embedding_size, 1, 1, 0, linear=True)
        self.drouout = nn.Dropout()
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
        # x = x.transpose(2, 3)
        # input(bs,1,128,313)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.blocks(x)
        x = self.conv3(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = self.drouout(x)
        feature = x.view(x.size(0), -1)  # (bs,128)
        if self.arcface is not None:
            out = self.arcface(feature, label)
        else:
            out = self.fc_out(feature)
        return out, feature


class STgramMFN(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting,
                 arcface=None):
        super(STgramMFN, self).__init__()
        self.wavegram = Wavegram()
        self.conv = ConvBlock(1, 16, 3, 1, 1, twostage=True)
        self.MFN = MobileFaceNet(
            num_class=num_class,
            input_channel=16,
            bottleneck_setting=bottleneck_setting,
            arcface=arcface
        )

    def forward(self, wav, mel, label):
        x_wav = self.wavegram(wav)
        x_mel = self.conv(mel)
        input = torch.cat((x_wav, x_mel), dim=1)
        out, feature = self.MFN(input, label)
        return out, feature

