from typing import Tuple, Optional
import torch
from torch import nn

class ExternalAttention(nn.Module):
    """
    Implementation of External Attention proposed in [1].

    External Attention computes attention between the input pixels and external memory unit M_k and
    M_v (as the key and value) to reduce the computational complexity. Also, an external memory unit
    acts as a memory of the whole training dataset thus takes relationships between elements in
    different samples into consideration.

    Parameters
    ----------
    input_size : int
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
        input_size: int,
        n_heads: Optional[int] = 1,
        s: int = 64,
        dropout: Optional[float] = None
    ) -> None:
        super(ExternalAttention, self).__init__()

        assert input_size % n_heads == 0

        self.n_heads = n_heads
        self.dim_head = input_size // n_heads

        self.W_Q = nn.Linear(input_size, n_heads * self.dim_head)
        self.M_K = nn.Linear(self.dim_head, s, bias=False)  # memory unit
        self.M_V = nn.Linear(s, self.dim_head, bias=False)  # memory unit
        self.M_V.weight.data = self.M_K.weight.data.transpose(0, 1)  # initialization of M_V and M_K should be the same

        self.fc = nn.Linear(input_size, input_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(input_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, length, input_size)
            Input data, where ``length`` is the length (number of features) of the input and
            ``input_size`` is the dimension of the features.

        Returns
        -------
        out : torch.Tensor (batch_size, length, input_size)
            Output of simple self-attention network

        att: torch.Tensor (batch_size, length, length)
            Attention weights
        """
        batch_size = x.size(0)

        Q = self.W_Q(x)  # (batch_size, length, n_heads * dim_head)
        Q = Q.view(batch_size, -1, self.n_heads, self.dim_head)  # (batch_size, length, n_heads, dim_head)
        Q = Q.transpose(1, 2)  # (batch_size, n_heads, length, dim_head)

        score = self.M_K(Q)  # (batch_size, n_heads, length, s)
        att = self.double_norm(score)  # (batch_size, n_heads, length, s)
        att = att if self.dropout is None else self.dropout(att)

        context = self.M_V(att)  # (batch_size, n_heads, length, dim_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_head * self.n_heads)  # (batch_size, length, n_heads * dim_head)

        out = self.fc(context).transpose(1, 2)  # (batch_size, input_size, length)
        out = self.batch_norm(out).transpose(1, 2)  # BatchNorm (batch_size, length, input_size)
        out = out + x  # residual connection
        out = self.relu(out)

        return out, att

    def double_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Double-normalization mentioned in section 3.2 of the paper."""
        eps = 1e-9
        x = self.softmax(x)
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x
