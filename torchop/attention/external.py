from typing import Tuple, Optional
import torch
from torch import nn

from ..modules import *

class ExternalAttention(nn.Module):
    """
    Implementation of External Attention proposed in [1].

    External Attention computes attention between the input pixels and external memory unit M_k and
    M_v (as the key and value) to reduce the computational complexity. Also, an external memory unit
    acts as a memory of the whole training dataset thus takes relationships between elements in
    different samples into consideration.

    Parameters
    ----------
    dim : int
        Dimension of the input features.

    s : int, optional, default=64
        Demension of the memory unit, see the original paper for details.

    n_heads : int, optional, default=1
        Number of attention heads.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks. \
        <https://arxiv.org/abs/2105.02358>`_" Meng-Hao Guo, et al. arXiv 2021.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 1,
        s: int = 64,
        dropout: Optional[float] = None
    ) -> None:
        super(ExternalAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads

        self.W_Q = nn.Linear(dim, n_heads * self.d_head)
        self.M_K = nn.Linear(self.d_head, s, bias=False)  # memory unit
        self.M_V = nn.Linear(s, self.d_head, bias=False)  # memory unit
        self.M_V.weight.data = self.M_K.weight.data.transpose(0, 1)  # initialization of M_V and M_K should be the same

        self.fc = nn.Linear(dim, dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        self.relu = nn.ReLU()

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
        Q = self.W_Q(x)  # (batch_size, length, n_heads * d_head)
        Q = split_heads(Q, self.n_heads)  # (batch_size, n_heads, length, d_head)

        score = self.M_K(Q).transpose(2, 3)  # (batch_size, n_heads, s, length)
        score = add_mask(score, mask)

        att = self.double_norm(score).transpose(2, 3)  # (batch_size, n_heads, length, s)
        att = att if self.dropout is None else self.dropout(att)

        context = self.M_V(att)  # (batch_size, n_heads, length, d_head)
        context = combine_heads(context)  # (batch_size, length, n_heads * d_head)

        out = self.fc(context).transpose(1, 2)  # (batch_size, dim, length)
        out = self.batch_norm(out).transpose(1, 2)  # BatchNorm (batch_size, length, dim)
        out = out + x  # residual connection
        out = self.relu(out)

        return out, att

    def double_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Double-normalization mentioned in section 3.2 of the paper."""
        eps = 1e-9
        x = self.softmax(x)
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x
