from typing import Tuple, Optional
import numpy as np
import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Parameters
    ----------
    scale : float
        Scale factor (``sqrt(d_k)``)

    dropout : float, optional
        Dropout, ``None`` if no dropout layer
    """
    def __init__(self, scale: float, dropout: float = 0.5) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Parameters
        ----------
        Q : torch.Tensor (batch_size, n_heads, length, d_k)
            Query

        K : torch.Tensor
            Key

        V : torch.Tensor
            Value

        mask : torch.Tensor (batch_size, 1, 1, length)
            Mask metrix, None if it is not needed

        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, length, d_k)
            Context vector

        att : torch.Tensor (batch_size, n_heads, length, length)
            Attention weights
        """
        # Q·K^T / sqrt(d_k)
        score = torch.matmul(Q / self.scale, K.transpose(2, 3))  # (batch_size, n_heads, length, length)

        # mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        if mask is not None:
            score = score.masked_fill(mask.bool(), -np.inf)

        # eq.1: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k))·V
        att = self.softmax(score)  # (batch_size, n_heads, length, length)
        att = att if self.dropout is None else self.dropout(att)
        context = att @ V  # (batch_size, n_heads, length, d_k)

        return context, att


class SelfAttention(nn.Module):
    """
    Implementation of Multi-Head Self-Attention proposed in [1].

    Parameters
    ----------
    input_size : int
        Size of the input tensor

    n_heads : int
        Number of attention heads

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Neural Machine Translation by Jointly Learning to Align and Translate. \
            <https://arxiv.org/abs/1409.0473>`_" Dzmitry Bahdanau, et al. ICLR 2015.
    """
    def __init__(
        self, input_size: int, n_heads: int, dropout: Optional[float] = None
    ) -> None:
        super(SelfAttention, self).__init__()

        assert input_size % n_heads == 0

        # we assume d_v always equals d_k
        self.d_k = input_size // n_heads  # d_k: size of each head
        self.n_heads = n_heads

        # linear projections
        self.W_Q = nn.Linear(input_size, n_heads * self.d_k)
        self.W_K = nn.Linear(input_size, n_heads * self.d_k)
        self.W_V = nn.Linear(input_size, n_heads * self.d_k)

        # scaled dot-product attention
        scale = self.d_k ** 0.5  # scale factor
        self.attention = ScaledDotProductAttention(scale=scale, dropout=dropout)

        self.layer_norm = nn.LayerNorm(input_size)
        self.fc = nn.Linear(n_heads * self.d_k, input_size)

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, length, input_size)
            Input data

        mask : torch.Tensor, optional (batch_size, 1, length)
            Mask metrix, None if it is not needed

        Returns
        -------
        out : torch.Tensor (batch_size, length, input_size)
            Output of multi-head self-attention network

        att: torch.Tensor (batch_size, n_heads, length, length)
            Attention weights
        """
        batch_size = x.size(0)

        Q = self.W_Q(x)  # (batch_size, length, n_heads * d_k)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)  # (batch_size, length, n_heads, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # (batch_size, n_heads, length, d_k)

        # for n_heads axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, 1, d_k)

        context, att = self.attention(Q, K, V, mask=mask)  # (batch_size, n_heads, length, d_k)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.n_heads)  # (batch_size, length, n_heads * d_k)

        out = self.fc(context)  # (batch_size, length, input_size)
        out = out if self.dropout is None else self.dropout(out)

        out = out + x  # residual connection
        out = self.layer_norm(out)  # LayerNorm

        return out, att
