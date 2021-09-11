import torch
from torch import nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7, use_bn: bool = True, use_relu: bool = False) -> None:
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(1, momentum=0.01) if use_bn else nn.Identity(),
            nn.ReLU() if use_relu else nn.Identity()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.

        Returns
        -------
        out : torch.Tensor (batch_size, out_channels, height, width)
            Output tensor of spatial attention module.
        """
        # channel pooling
        f_max = torch.max(x, 1)[0].unsqueeze(1)  # (batch_size, 1, height, width)
        f_avg = torch.mean(x, 1).unsqueeze(1)  # (batch_size, 1, height, width)
        pooled_feat = torch.cat((f_max, f_avg), dim=1)  # (batch_size, 2, height, width)

        out = self.conv(pooled_feat)
        att = self.sigmoid(out)

        return x * att


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        out_channels = in_channels // reduction
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.

        Returns
        -------
        out : torch.Tensor (batch_size, out_channels, height, width)
            Output tensor of channel attention module.
        """
        f_avg = self.avg_pool(x)  # (batch_size, in_channels, 1, 1)
        f_avg = self.mlp(f_avg)  # (batch_size, in_channels, 1, 1)

        f_max = self.max_pool(x)  # (batch_size, in_channels, 1, 1)
        f_max = self.mlp(f_max)  # (batch_size, in_channels, 1, 1)

        att = self.sigmoid(f_avg + f_max)  # (batch_size, in_channels, 1, 1)

        return x * att


class CBAM(nn.Module):
    """
    Implementation of Convolutional Block Attention Module (CBAM) proposed in [1].

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.

    use_spatial : bool, optional, default=True
        Use spatial attention or not.

    kernel_size : int, optional, default=7
        Kernel size for spatial attention module.

    use_bn : bool, optiona, defualt=True
        Use Batch Normalization in spatial attention module or not.

    use_relu : bool, optiona, defualt=False
        Use ReLU in spatial attention module or not.

    reduction : int, optional, default=16
        Reduction ratio to control the intermediate channel dimension of channel attention module.

    References
    ----------
    1. "`CBAM: Convolutional Block Attention Module. <https://arxiv.org/abs/1807.06521>`_" \
        Sanghyun Woo, et al. ECCV 2018.
    """

    def __init__(
        self,
        in_channels: int,
        use_spatial: bool = True,
        kernel_size: int = 7,
        use_bn: bool = True,
        use_relu: bool = False,
        reduction: int = 16
    ) -> None:
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size, use_bn, use_relu) if use_spatial else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.

        Returns
        -------
        out : torch.Tensor (batch_size, out_channels, height, width)
            Output of CBAM.
        """
        out = self.channel_attn(x)
        out = self.spatial_attn(out) if self.spatial_attn is not None else out
        return out
