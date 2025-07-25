import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):  # EnhancedDepthwiseConv
    def __init__(self, c1, c2, k=3, s=1, act=True, depth_multiplier=2):
        super(DSConv, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(c1, c1*depth_multiplier, kernel_size=k, stride=s, padding=k//2, groups=c1, bias=False),
            nn.BatchNorm2d(c1 * depth_multiplier),
            nn.GELU() if act else nn.Identity(),
            nn.Conv2d(c1*depth_multiplier, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU() if act else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)

#class PixelSliceConcat(nn.Module):
#    def forward(self, x):
#        return torch.cat([
#            x[..., ::2, ::2],
#            x[..., 1::2, ::2],
#            x[..., ::2, 1::2],
#            x[..., 1::2, 1::2],
#        ], dim=1)

class ESSamp(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, depth_multiplier=2):
        super(ESSamp, self).__init__()
        self.dsconv = DSConv(c1 * 4, c2, k=k, s=s, act=act,depth_multiplier=depth_multiplier)
        self.slices = nn.PixelUnshuffle(2)
        #self.slices = PixelSliceConcat()


    def forward(self, x):
        x = self.slices(x)
        return self.dsconv(x)

