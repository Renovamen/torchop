import torch
from torch import nn

class SEBlock(nn.Module):
    """
    Implementation of the Squeeze-and-Excitation (SE) block proposed in [1].

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.

    reduction : int, optional, default=16
        Reduction ratio to control the intermediate channel dimension.

    References
    ----------
    1. "`Squeeze-and-Excitation Networks. <https://arxiv.org/abs/1709.01507>`_" Jie Hu, et al. CVPR 2018.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16
    ) -> None:
        super(SEBlock, self).__init__()

        out_channels = in_channels // reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.

        Returns
        -------
        out : torch.Tensor (batch_size, in_channels, height, width)
            Output of the SK convolution layer.
        """
        z = self.squeeze(x)  # (batch_size, in_channels, 1, 1), eq.2
        s = self.excitation(z)  # (batch_size, in_channels, 1, 1), eq.3
        out = x * s  # channel-wise multiplication, eq. 4
        return out
