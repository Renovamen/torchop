from typing import Optional
import torch
from torch import nn

from ..modules import add_mask

class VanillaAttention(nn.Module):
    """
    Implementation of the attention network proposed in [1] and [2].

    Parameters
    ----------
    dim : int
        Size of the input tensor

    align_function : str, optional, default="general"
        Type of fuction for computing alignment scores, a string in "dot" / "general"

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Neural Machine Translation by Jointly Learning to Align and Translate. \
            <https://arxiv.org/abs/1409.0473>`_" Dzmitry Bahdanau, et al. ICLR 2015.

    2. "`Effective Approaches to Attention-based Neural Machine Translation. \
            <https://arxiv.org/abs/1508.04025>`_" Minh-Thang Luong, et al. EMNLP 2015.
    """
    def __init__(
        self,
        dim: int,
        align_function: str = "general",
        dropout: Optional[float] = None
    ) -> None:
        super(VanillaAttention, self).__init__()

        self.align_function = align_function

        if align_function == "general":
            self.fc_align = nn.Linear(dim, dim)
        elif align_function != 'dot':
            raise ValueError('Invalid alignment score function: {0}'.format(align_function))

        self.fc_query = nn.Linear(dim, dim)
        self.fc_value = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : torch.Tensor (batch_size, dim)
            Query

        value : torch.Tensor (batch_size, length, dim)
            Value

        mask : torch.Tensor, optional (batch_size, length)
            Mask metrix, None if it is not needed

        Returns
        -------
        out : torch.Tensor (batch_size, dim)
            Output tensor

        att: torch.Tensor (batch_size, length)
            Attention weights
        """

        # alignment scores
        query = query if self.align_function == "dot" else self.fc_align(query)  # (batch_size, dim)
        score = (value @ query.unsqueeze(2)).squeeze(2)  # (batch_size, length)
        score = add_mask(score, mask)

        # attention weights
        att = self.softmax(score)  # (batch_size, length)
        att = att if self.dropout is None else self.dropout(att)

        # context vector (weighted value)
        context = (att.unsqueeze(1) @ value).squeeze(1)  # (batch_size, dim)

        # attention result
        out = self.tanh(self.fc_value(context) + self.fc_query(query))

        return out, att
