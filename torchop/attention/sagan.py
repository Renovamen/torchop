from typing import Tuple, Optional
import torch
from torch import nn

from ..modules.utils import *

class SAGANAttention(nn.Module):
    """
    Implementation of the attention layer proposed in [1].

    Parameters
    ----------
    dim : int
        Dimension of the input features.

    reduction : int, optional, default=8
        Factor to reduce the channel number.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Self-Attention Generative Adversarial Networks. <https://arxiv.org/abs/1805.08318>`_" \
            Han Zhan, et al. ICML 2019.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 8,
        n_heads: int = 1,
        dropout: Optional[float] = None
    ) -> None:
        super(SAGANAttention, self).__init__()

        out_size = dim // reduction

        assert dim % n_heads == 0 and out_size % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.d_out_head = out_size // n_heads

        self.W_Q = nn.Linear(dim, out_size)
        self.W_K = nn.Linear(dim, out_size)
        self.W_V = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, length, dim)
            Input data, where ``length`` is the length (number of features) of the input and
            ``dim`` is the dimension of the features.

        mask : torch.Tensor, optional (batch_size, length)
            Mask metrix, ``None`` if it is not needed.

        Returns
        -------
        out : torch.Tensor (batch_size, length, dim)
            Output of the attention layer.

        att: torch.Tensor (batch_size, length, length)
            Attention weights.
        """
        Q = self.W_Q(x)  # (batch_size, length, out_size)
        K = self.W_K(x)  # (batch_size, length, out_size)
        V = self.W_V(x)  # (batch_size, length, dim)

        Q = split_heads(Q, self.n_heads)  # (batch_size, n_heads, length, d_out_head)
        K = split_heads(K, self.n_heads)
        V = split_heads(V, self.n_heads)  # (batch_size, n_heads, length, d_head)

        score = Q @ K.transpose(2, 3)  # (batch_size, n_heads, length, length)
        score = add_mask(score, mask)

        att = self.softmax(score)  # (batch_size, n_heads, length, length)
        att = att if self.dropout is None else self.dropout(att)

        context = att @ V  # (batch_size, n_heads, length, d_head)
        context = combine_heads(context)  # (batch_size, length, dim = n_heads * d_head)

        out = self.gamma * context
        out = out + x  # residual connection

        return out, att
