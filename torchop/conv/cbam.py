import torch
from torch import nn
from torch.nn import functional as F

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
        f_mean = torch.mean(x, 1).unsqueeze(1)  # (batch_size, 1, height, width)
        pooled_feat = torch.cat((f_max, f_mean), dim=1)  # (batch_size, 2, height, width)

        out = self.conv(pooled_feat)
        attn = self.sigmoid(out)

        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super(ChannelAttention, self).__init__()
        out_channels = in_channels // reduction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, in_channels)
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
        batch_size, _, h, w = x.shape

        f_mean = F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))  # (batch_size, in_channels, 1, 1)
        f_mean = f_mean.view(batch_size, -1)  # (batch_size, in_channels)
        f_mean = self.mlp(f_mean)  # (batch_size, in_channels)

        f_max = F.max_pool2d(x, kernel_size=(h, w), stride=(h, w))  # (batch_size, in_channels, 1, 1)
        f_max = f_max.view(batch_size, -1)  # (batch_size, in_channels)
        f_max = self.mlp(f_max)  # (batch_size, in_channels)

        attn = self.sigmoid(f_mean + f_max)  # (batch_size, in_channels)
        attn = attn.unsqueeze(2).unsqueeze(3).expand_as(x)  # (batch_size, in_channels, h, w)

        return x * attn


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
