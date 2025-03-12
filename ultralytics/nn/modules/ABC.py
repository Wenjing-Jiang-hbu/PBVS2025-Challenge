import torch
import torch.nn as nn
from .gcn.layers.GConv2 import GConv2
from .SPCCAB import SSPCAB

import torch.nn.functional as F


class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch,kernel):
        super(DEPTHWISECONV, self).__init__()


        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=kernel,
                                    stride=1,
                                    padding="same",
                                    groups=in_ch)
    def forward(self,input):
        out = self.depth_conv(input.to(self.depth_conv.bias.dtype))
        return out


class ChannelSelect2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelSelect2, self).__init__()

        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.conv = DEPTHWISECONV(in_planes*2,in_planes*2,1)
    def forward(self, x, y):

        xy = torch.cat([x, y], dim=1)
        ca = F.softmax(self.conv(self.advavg(xy)), dim=1).sigmoid()
        cax,cay = torch.split(ca, ca.size(1) // 2, dim=1)

        output = x*cax+y*cay

        return output


class CDFF(nn.Module):
    def __init__(self, in_dim):
        super(CDFF, self).__init__()
        self.wdim = [in_dim, in_dim]


        self.sfe = GConv2(int(in_dim / 4), int(in_dim / 4), 5, M=4, padding=2)
        self.ffe = SSPCAB(int(in_dim / 4))
        self.fusion = ChannelSelect2(in_dim)

    def forward(self, x):

        fftx=torch.fft.fftn(x, dim=(-2, -1)).to(torch.float32).to(x.device)
        channels_per_group1 = fftx.shape[1] // 4
        feature_hl_1, feature_hl_2, feature_hl_3, feature_hl_4= torch.split(fftx, channels_per_group1, 1)
        trans_fea11 = self.ffe(feature_hl_1)
        trans_fea12 = self.ffe(feature_hl_2)
        trans_fea13 = self.ffe(feature_hl_3)
        trans_fea14 = self.ffe(feature_hl_4)
        trans_fea2 = torch.cat((trans_fea11, trans_fea12, trans_fea13,trans_fea14), dim=1)



        transy =torch.fft.ifftn(trans_fea2, dim=(-2, 1)).to(torch.float32).to(x.device)
        convsy = self.sfe(x)
        out = self.fusion(transy, convsy)
        return out





