import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

class ConvMaxPoolDropout(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            nn.MaxPool2d(2, stride=2),
        )
        self.dropout = dropout

    def forward(self, input):
        out = self.layer(input)
        if self.dropout > 0.:
            out = F.dropout(out, p=self.dropout)
        return out

class BashFC(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
        )
        self.dropout = dropout
        self.layer2 = nn.Sequential(nn.Linear(256, 2,bias=False),)

    def forward(self, input):
        out = self.layer1(input)

        if self.dropout > 0.:
            out = F.dropout(out, p=self.dropout)
        out = self.layer2(out)
        return out


class SfcFC(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=53,bias=False),
            #nn.Conv2d(in_channels, out_channels, kernel_size=(287, 197), bias=False),
        )
        self.dropout = dropout
        self.layer2 = nn.Sequential(nn.Conv2d(out_channels, 2,kernel_size=1,bias=False),)

    def forward(self, input):
        out = self.layer1(input)

        if self.dropout > 0.:
            out = F.dropout(out, p=self.dropout)
        out = self.layer2(out)
        return out



class SFCNET(nn.Module):
    def __init__(self, args):
        super().__init__()



        blocks = []
        blocks += [("conv_{}".format(1), ConvMaxPoolDropout(3, 32, args.dropout))]
        blocks += [("conv_{}".format(2), ConvMaxPoolDropout(32, 64, args.dropout))]

        blocks += [("fc", SfcFC(64,256,0))]

        self.net = nn.Sequential(OrderedDict(blocks))

    def forward(self, input):
        out = self.net(input)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
