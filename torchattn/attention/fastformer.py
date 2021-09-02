from typing import Tuple, Optional
import numpy as np
import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """
    Additive Attention

    Parameters
    ----------
    d_head : int
        Dimension of each attention head.

    scale : float
        Scale factor (``sqrt(d_head)``).

    dropout : float, optional
        Dropout, ``None`` if no dropout layer.
    """
    def __init__(self, d_head: int, scale: float, dropout: float = 0.5) -> None:
        super(AdditiveAttention, self).__init__()

        self.attention = nn.Linear(d_head, 1, bias=False)
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, n_heads, length, d_head)
            Input tensor.

        mask : torch.Tensor (batch_size, 1, 1, length)
            Mask metrix, ``None`` if it is not needed.

        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, 1, d_head)
            Global attention vector.

        att : torch.Tensor (batch_size, n_heads, 1, length)
            Attention weights.
        """
        score = self.attention(x).transpose(2, 3) / self.scale  # (batch_size, n_heads, 1, length)

        if mask is not None:
            score = score.masked_fill(mask.bool(), -np.inf)

        att = self.softmax(score)
        att = att if self.dropout is None else self.dropout(att)

        # global attention query vector
        context = att @ x  # (batch_size, n_heads, 1, d_head)

        return context, att


class FastAttention(nn.Module):
    """
    Implementation of the attention layer proposed in Fastformer [1].

    Parameters
    ----------
    dim : int
        Dimension of the input features.

    n_heads : int, optional, default=1
        Number of attention heads.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Fastformer: Additive Attention Can Be All You Need. <https://arxiv.org/abs/2108.09084>`_" \
        Meng-Hao Guo, et al. arXiv 2021.
    """

    def __init__(
        self,
        dim: int,
        n_heads: Optional[int] = 8,
        dropout: Optional[float] = None
    ) -> None:
        super(FastAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads

        # linear transformation layer to transform the input into the Q, K, V
        self.W_Q = nn.Linear(dim, n_heads * self.d_head)
        self.W_K = nn.Linear(dim, n_heads * self.d_head)
        self.W_V = nn.Linear(dim, n_heads * self.d_head)

        # additive attention
        scale = self.d_head ** 0.5  # scale factor
        self.attn_Q = AdditiveAttention(self.d_head, scale)
        self.attn_K = AdditiveAttention(self.d_head, scale)

        self.to_R = nn.Linear(self.d_head, self.d_head)
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

        att: torch.Tensor (batch_size, length, length)
            Attention weights.
        """
        batch_size = x.size(0)

        Q = self.W_Q(x)  # (batch_size, length, n_heads * d_head)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_head)  # (batch_size, length, n_heads, d_head)
        K = K.view(batch_size, -1, self.n_heads, self.d_head)
        V = V.view(batch_size, -1, self.n_heads, self.d_head)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # (batch_size, n_heads, length, d_head)

        # for n_heads axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, length)

        # global attention query vector
        global_Q, _ = self.attn_Q(Q, mask)  # (batch_size, n_heads, 1, d_head)

        # global context-aware query-key matrix
        K = K * global_Q  # (batch_size, n_heads, length, d_head)

        # global attention key vector
        global_K, _ = self.attn_K(Q, mask)  # (batch_size, n_heads, 1, d_head)

        # global context-aware key-value matrix
        V = V * global_K  # (batch_size, n_heads, length, d_head)

        # linear transformation to get R
        R = self.to_R(V)  # (batch_size, n_heads, length, d_head)
        R = R if self.dropout is None else self.dropout(R)

        out = R + Q  # residual connection
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_head * self.n_heads)  # (batch_size, length, dim = n_heads * d_head)
        out = self.fc(out)

        return out
