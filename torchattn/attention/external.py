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

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks. \
        <https://arxiv.org/abs/2105.02358>`_" Meng-Hao Guo, et al. arXiv 2021.
    """

    def __init__(
        self, input_size: int, s: int = 64, dropout: Optional[float] = None
    ) -> None:
        super(ExternalAttention, self).__init__()

        self.W_Q = nn.Conv1d(input_size, input_size, kernel_size=1)
        self.M_K = nn.Conv1d(input_size, s, kernel_size=1, bias=False)  # memory unit
        self.M_V = nn.Conv1d(s, input_size, kernel_size=1, bias=False)  # memory unit
        self.M_V.weight.data = self.M_K.weight.data.transpose(0, 1)  # initialization of M_V and M_K should be the same

        self.fc = nn.Conv1d(input_size, input_size, kernel_size=1, bias=False)
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
        x = x.transpose(1, 2)  # (batch_size, input_size, length)

        Q = self.W_Q(x)  # (batch_size, input_size, length)

        score = self.M_K(Q)  # (batch_size, s, length)
        att = self.double_norm(score)  # (batch_size, s, length)
        att = att if self.dropout is None else self.dropout(att)

        context = self.M_V(att)  # (batch_size, input_size, length)

        out = self.fc(context)  # (batch_size, input_size, length)
        out = self.batch_norm(out)  # BatchNorm
        out = out + x  # residual connection
        out = self.relu(out).transpose(1, 2)  # (batch_size, length, input_size)

        return out, att

    def double_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Double-normalization mentioned in section 3.2 of the paper."""
        eps = 1e-9
        x = self.softmax(x)
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x
