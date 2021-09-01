from typing import Tuple, Optional
import torch
from torch import nn

class SAGANAttention(nn.Module):
    """
    Implementation of attention layer proposed in [1].

    Parameters
    ----------
    input_size : int
        Dimension of the input features.

    reduce_coef : int, optional, default=8
        Factor to reduce the channel number.

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Self-Attention Generative Adversarial Networks. <https://arxiv.org/abs/1805.08318>`_" \
            Han Zhan, et al. ICML 2019.
    """

    def __init__(
        self, input_size: int, reduce_coef: int = 8, dropout: Optional[float] = None
    ) -> None:
        super(SAGANAttention, self).__init__()

        out_size = input_size // reduce_coef

        self.W_Q = nn.Conv1d(input_size, out_size, kernel_size=1)
        self.W_K = nn.Conv1d(input_size, out_size, kernel_size=1)
        self.W_V = nn.Conv1d(input_size, input_size, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.dropout = None if dropout is None else nn.Dropout(dropout)

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

        Q = self.W_Q(x).transpose(1, 2)  # (batch_size, length, out_size)
        K = self.W_K(x)  # (batch_size, out_size, length)
        V = self.W_V(x)  # (batch_size, input_size, length)

        score = Q @ K  # (batch_size, length, length)
        att = self.softmax(score)  # (batch_size, length, length)
        att = att if self.dropout is None else self.dropout(att)

        out = V @ att.transpose(1, 2)  # (batch_size, input_size, length)
        out = self.gamma * out + x  # residual connection
        out = out.transpose(1, 2)  # (batch_size, length, input_size)

        return out, att
