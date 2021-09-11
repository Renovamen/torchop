import torch
from torch import nn
from collections import OrderedDict

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 1) -> None:
        super(ChannelAttention, self).__init__()

        self.attn = nn.Sequential()
        channels = [in_channels] + [out_channels] * n_layers + [in_channels]

        for i in range(len(channels) - 1):
            self.attn.add_module('ca_fc_%d' % i, nn.Conv2d(channels[i], channels[i + 1], kernel_size=1))
            if i != len(channels) - 2:
                self.attn.add_module('ca_bn_%d' % (i + 1), nn.BatchNorm2d(channels[i + 1]))
                self.attn.add_module('ca_relu_%d' % (i + 1), nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_c = self.avg_pool(x)
        out = self.attn(f_c)  # (batch_size, in_channels, height, width)
        return out


class SpatialAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_convs: int = 2,
        dilation: int = 4
    ) -> None:
        super(SpatialAttention, self).__init__()

        # dimension reduction
        self.attn = nn.Sequential(OrderedDict([
            ('sa_reduce_conv_0', nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            ('sa_reduce_bn_0', nn.BatchNorm2d(out_channels)),
            ('sa_reduce_relu_0', nn.ReLU())
        ]))  # (batch_size, out_channels, height, width)

        # dilated convolutions
        for i in range(n_convs):
            self.attn.add_module(
                'sa_dc_conv_%d' % i,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
            )  # (batch_size, out_channels, height, width)
            self.attn.add_module('sa_dc_bn_%d' % i, nn.BatchNorm2d(out_channels))
            self.attn.add_module('sa_dc_relu_%d' % i, nn.ReLU())

        self.attn.add_module( 'sa_final_conv', nn.Conv2d(out_channels, 1, kernel_size=1))  # (batch_size, 1, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(x).expand_as(x)
        return out


class BAM(nn.Module):
    """
    Implementation of Bottleneck Attention Module (BAM) proposed in [1].

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.

    reduction : int, optional, default=16
        Reduction ratio to control the intermediate channel dimension.

    n_ca_layers : bool, optional, default=1
        Number of mlp layers in channel attention module.

    n_sa_convs : int, optional, default=2
        Number of dilated convolution layers in spatial attention module.

    sa_dilation : bool, optiona, defualt=4
        Dilation size for spatial attention module.

    References
    ----------
    1. "`BAM: Bottleneck Attention Module. <https://arxiv.org/abs/1807.06514>`_" Jongchan Park, et al. BMVC 2018.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        n_ca_layers: int = 1,
        n_sa_convs: int = 2,
        sa_dilation: int = 4
    ) -> None:
        super(BAM, self).__init__()
        out_channels = in_channels // reduction
        self.channel_attn = ChannelAttention(in_channels, out_channels, n_ca_layers)
        self.spatial_attn = SpatialAttention(in_channels, out_channels, n_sa_convs, sa_dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.

        Returns
        -------
        out : torch.Tensor (batch_size, out_channels, height, width)
            Output of BAM.
        """
        channel_att = self.channel_attn(x)
        spatial_att = self.spatial_attn(x)
        att = 1 + self.sigmoid(channel_att + spatial_att)
        return att * x
