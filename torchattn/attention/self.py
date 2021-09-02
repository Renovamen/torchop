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
        Scale factor (``sqrt(d_head)``).

    dropout : float, optional
        Dropout, ``None`` if no dropout layer.
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
        Q : torch.Tensor (batch_size, n_heads, length, d_head)
            Query

        K : torch.Tensor
            Key

        V : torch.Tensor
            Value

        mask : torch.Tensor (batch_size, 1, 1, length)
            Mask metrix, None if it is not needed

        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, length, d_head)
            Context vector.

        att : torch.Tensor (batch_size, n_heads, length, length)
            Attention weights.
        """
        # Q·K^T / sqrt(d_head)
        score = torch.matmul(Q / self.scale, K.transpose(2, 3))  # (batch_size, n_heads, length, length)

        # mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        if mask is not None:
            score = score.masked_fill(mask.bool(), -np.inf)

        # eq.1: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_head))·V
        att = self.softmax(score)  # (batch_size, n_heads, length, length)
        att = att if self.dropout is None else self.dropout(att)
        context = att @ V  # (batch_size, n_heads, length, d_head)

        return context, att


class SelfAttention(nn.Module):
    """
    Implementation of Multi-Head Self-Attention proposed in [1].

    Parameters
    ----------
    dim : int
        Dimension of the input features.

    n_heads : int
        Number of attention heads.

    simplified : bool, optional, default=False
        Use the simplified version of the Multi-Head Self-Attention or not. See
        :class:`torchattn.SimplifiedSelfAttention` for details.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer.

    References
    ----------
    1. "`Attention Is All You Need. <https://arxiv.org/abs/1706.03762>`_" Ashish Vaswani, et al. NIPS 2017.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        simplified: bool = False,
        dropout: Optional[float] = None
    ) -> None:
        super(SelfAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads

        self.simplified = simplified

        if not simplified:
            # linear projections
            self.W_Q = nn.Linear(dim, n_heads * self.d_head)
            self.W_K = nn.Linear(dim, n_heads * self.d_head)
            self.W_V = nn.Linear(dim, n_heads * self.d_head)

        # scaled dot-product attention
        scale = self.d_head ** 0.5  # scale factor
        self.attention = ScaledDotProductAttention(scale=scale, dropout=dropout)

        self.layer_norm = nn.LayerNorm(dim)
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

        mask : torch.Tensor, optional (batch_size, 1, length)
            Mask metrix, ``None`` if it is not needed.

        Returns
        -------
        out : torch.Tensor (batch_size, length, dim)
            Output of multi-head self-attention network.

        att: torch.Tensor (batch_size, n_heads, length, length)
            Attention weights.
        """
        batch_size = x.size(0)

        if self.simplified:
            Q = K = V = x
        else:
            Q = self.W_Q(x)  # (batch_size, length, n_heads * d_head)
            K = self.W_K(x)
            V = self.W_V(x)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_head)  # (batch_size, length, n_heads, d_head)
        K = K.view(batch_size, -1, self.n_heads, self.d_head)
        V = V.view(batch_size, -1, self.n_heads, self.d_head)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # (batch_size, n_heads, length, d_head)

        # for n_heads axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, 1, length)

        context, att = self.attention(Q, K, V, mask=mask)  # (batch_size, n_heads, length, d_head)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_head * self.n_heads)  # (batch_size, length, n_heads * d_head)

        out = self.fc(context)  # (batch_size, length, dim)
        out = out if self.dropout is None else self.dropout(out)

        out = out + x  # residual connection
        out = self.layer_norm(out)  # LayerNorm

        return out, att


class SimplifiedSelfAttention(SelfAttention):
    """
    Implementation of a common simplified version of Multi-Head Self-Attention, which drops the
    linear projection layers and directly calculates an attention map from the input feature to
    reduce the computational complexity.

    Parameters
    ----------
    dim : int
        Dimension of the input features.

    n_heads : int
        Number of attention heads.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer.
    """
    def __init__(
        self, dim: int, n_heads: int, dropout: Optional[float] = None
    ) -> None:
        super(SimplifiedSelfAttention, self).__init__(
            dim = dim,
            n_heads = n_heads,
            simplified = True,
            dropout = dropout
        )
