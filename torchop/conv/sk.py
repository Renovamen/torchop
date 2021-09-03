import torch
from torch import nn
from typing import List, Optional

class SKConv(nn.Module):
    """
    Implementation of the Selective Kernel (SK) Convolution proposed in [1].

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.

    out_channels : int
        Number of channels produced by the convolution.

    kernels : List[int], optional, default=[3, 5]
        List of kernel sizes for each branch.

    r : int, optional, default=16
        Reduction ratio to control the dimension of "compact feature" ``z`` (see eq.4).

    L : int, optional, default=32
        Minimal value of the dimension of "compact feature" ``z`` (see eq.4).

    groups : int, optional, default=32
        Hyperparameter for ``torch.nn.Conv2d``.

    References
    ----------
    1. "`Selective Kernel Networks. <https://arxiv.org/abs/1903.06586>`_" Xiang Li, et al. CVPR 2019.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernels: List[int] = [3, 5],
        r: int = 16,
        L: int = 32,
        groups: int = 32
    ) -> None:
        super(SKConv, self).__init__()

        d = max(in_channels // r, L)  # eq.4

        self.M = len(kernels)

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size = k,
                    stride = 1,
                    padding = (k - 1) // 2,
                    groups = groups
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            for k in kernels
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_z = nn.Sequential(
            nn.Linear(out_channels, d),
            nn.BatchNorm1d(d),
            nn.ReLU()
        )
        self.fc_attn = nn.Linear(d, out_channels * self.M)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, width, height)
            Input tensor.

        Returns
        -------
        out : torch.Tensor (batch_size, out_channels, width, height)
            Output of the SK convolution layer.
        """
        # ----- split -----
        feats = torch.cat([conv(x).unsqueeze(1) for conv in self.convs], dim=1)  # (batch_size, M, out_channels, width, height)

        # ----- fuse -----
        # eq.1
        U = torch.sum(feats, dim=1)  # (batch_size, out_channels, width, height)
        # channel-wise statistics, eq.2
        s = self.pool(U).squeeze(-1).squeeze(-1)  # (batch_size, out_channels)
        # compact feature, eq.3
        z = self.fc_z(s)  # (batch_size, d)

        # ----- select -----
        batch_size, out_channels = s.shape

        # attention map, eq.5
        score = self.fc_attn(z)  # (batch_size, M * out_channels)
        score = score.view(batch_size, self.M, out_channels, 1, 1)  # (batch_size, M, out_channels, 1, 1)
        att = self.softmax(score)

        # fuse multiple branches, eq.6
        out = torch.sum(att * feats, dim=1)  # (batch_size, out_channels, width, height)
        return out
