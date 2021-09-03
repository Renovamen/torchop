import torch
from torch import nn

class Involution(nn.Module):
    """
    Implementation of the Involution operator proposed in [1].

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.

    kernels : int
        List of kernel sizes for each branch.

    stride : int
        Stride for the sliding blocks.

    reduction : int, optional, default=4
        Reduction ratio to control the intermediate channel dimension.

    group_channels : int, optional, default=16
        Number of channels in a group. Each group shares the same involution kernel.

    References
    ----------
    1. "`Involution: Inverting the Inherence of Convolution for Visual Recognition. \
        <https://arxiv.org/abs/2103.06255>`_" Duo Li, et al. CVPR 2021.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        reduction: int = 4,
        group_channels: int = 16
    ) -> None:
        super(Involution, self).__init__()

        out_channels = in_channels // reduction
        paddingg = (kernel_size - 1) // 2

        self.in_channels = in_channels
        self.stride = stride
        self.group_channels = group_channels
        self.groups = in_channels // group_channels
        self.kernel_size = kernel_size

        self.pool = nn.AvgPool2d(kernel_size=stride) if stride > 1 else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, (kernel_size ** 2) * self.groups, kernel_size=1, stride=1)
        )
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=paddingg, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.

        Returns
        -------
        out : torch.Tensor (batch_size, in_channels, H = height / stride, W = width / stride)
            Output of the involution layer.
        """
        height, width = x.size(2), x.size(3)
        assert height % self.stride == 0 and width % self.stride == 0

        # ----- generate kernal -----
        kernel = self.conv(self.pool(x))  # (batch_size, G * K * K, H = height / stride, W = width / stride)

        batch_size, _, h, w = kernel.shape
        kernel = kernel.view(batch_size, self.groups, self.kernel_size ** 2, h, w)
        kernel = kernel.unsqueeze(2)  # (batch_size, G, 1, K * K, H, W)

        # ----- involution -----
        unfolded_x = self.unfold(x)  # (batch_size, in_channels * K * K, H * W)
        unfolded_x = unfolded_x.view(batch_size, self.groups, self.group_channels, self.kernel_size ** 2, h, w)

        out = (kernel * unfolded_x).sum(dim=3)  # (batch_size, G, group_channels, H, W)
        out = out.view(batch_size, self.in_channels, h, w)

        return out
