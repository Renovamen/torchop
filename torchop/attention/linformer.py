from typing import Tuple, Optional
import torch
from torch import nn

from ..modules.utils import *


class LinAttention(nn.Module):
    """
    Implementation of the attention layer proposed in Linformer [1].

    Parameters
    ----------
    dim : int
        Dimension of the input features.

    length : int
        Number of the input features.

    n_heads : int, optional, default=1
        Number of attention heads.

    k : int, optional, default=256
        Output dimension of ``E`` and ``F``.

    share_kv_head : bool, optional, default=True
        Share attention head for key and value or not.

    share_kv : bool, optional, default=True
        Share linear projection (``W_Q = W_K``, ``E = F``) for key and value or not.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Linformer: Self-Attention with Linear Complexity. <https://arxiv.org/abs/2006.04768>`_" \
        Sinong Wang, et al. arXiv 2020.
    """

    def __init__(
        self,
        dim: int,
        length: int,
        n_heads: int = 8,
        k: int = 256,
        share_kv_head: bool = True,
        share_kv: bool = True,
        dropout: Optional[float] = None
    ) -> None:
        super(LinAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale = self.d_head ** 0.5

        self.W_Q = nn.Linear(dim, n_heads * self.d_head)

        self.share_kv_head = share_kv_head
        d_kv = self.d_head if share_kv_head else (n_heads * self.d_head)

        self.W_K = nn.Linear(dim, d_kv)
        self.E = nn.Linear(length, k)
        torch.nn.init.xavier_normal_(self.E.weight)

        self.share_kv = share_kv
        if not share_kv:
            self.W_V = nn.Linear(dim, d_kv)
            self.F = nn.Linear(length, k)
            torch.nn.init.xavier_normal_(self.F.weight)

        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(n_heads * self.d_head, dim)
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
            Output of simple self-attention network.
        """
        Q = self.W_Q(x)  # (batch_size, length, n_heads * d_head)
        K = self.W_K(x)  # (batch_size, length, d_kv)
        V = self.W_V(x) if not self.share_kv else K

        # ----- linear projection: length -> k -----
        F = self.E if self.share_kv else self.F
        K = self.E(K.transpose(1, 2)).transpose(1, 2)  # (batch_size, k, d_kv)
        V =      F(V.transpose(1, 2)).transpose(1, 2)

        # ----- split heads -----
        kv_n_heads = 1 if self.share_kv_head else self.n_heads
        Q = split_heads(Q, self.n_heads)  # (batch_size, n_heads, length, d_head)
        K, V = split_heads(K, kv_n_heads), split_heads(V, kv_n_heads)  # (batch_size, kv_n_head, k, d_head)
        K, V = K.expand(-1, self.n_heads, -1, -1), V.expand(-1, self.n_heads, -1, -1)  # (batch_size, self.n_heads, k, d_head)

        # ----- self-attention -----
        score = torch.matmul(Q / self.scale, K.transpose(2, 3))  # (batch_size, n_heads, length, k)
        score = add_mask(score, mask)

        att = self.softmax(score)  # (batch_size, n_heads, length, k)
        att = att if self.dropout is None else self.dropout(att)

        context = att @ V  # (batch_size, n_heads, length, d_head)
        context = combine_heads(context)  # (batch_size, length, dim = n_heads * d_head)

        out = self.fc(context)

        return out
