from typing import Tuple, Optional
import torch
from torch import nn

class SimpleSelfAttention(nn.Module):
    """
    Implementation of Simple Self-Attention proposed in [1].

    Parameters
    ----------
    in_channels : int
        Number of channels of the input tensor.

    reduce_dim : int, optional, default=8
        Factor to reduce the channel number.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Self-Attention Generative Adversarial Networks. <https://arxiv.org/abs/1805.08318>`_" \
            Han Zhan, et al. ICML 2019.
    """

    def __init__(
        self, in_channels: int, reduce_dim: int = 8, dropout: Optional[float] = None
    ) -> None:
        super(SimpleSelfAttention, self).__init__()

        out_channels = in_channels // reduce_dim

        self.W_Q = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.W_K = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.W_V = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, width, height)
            Input data

        Returns
        -------
        out : torch.Tensor (batch_size, in_channels, width, height)
            Output of simple self-attention network

        att: torch.Tensor (batch_size, width * height, width * height)
            Attention weights
        """
        batch_size, in_channels, width, height = x.size()
        flat_x = x.view(batch_size, in_channels, width * height)  # (batch_size, in_channels, width * height)

        Q = self.W_Q(flat_x).permute(0, 2, 1)  # (batch_size, width * height, out_channels)
        K = self.W_K(flat_x)  # (batch_size, out_channels, width * height)
        V = self.W_V(flat_x)  # (batch_size, in_channels, width * height)

        score = Q @ K
        att = self.softmax(score)  # (batch_size, width * height, width * height)
        context = V @ att.permute(0, 2, 1)  # (batch_size, out_channels, width * height)

        out = context.view(batch_size, in_channels, width, height)
        out = out if self.dropout is None else self.dropout(out)

        out = self.gamma * out + x  # residual connection

        return out, att
