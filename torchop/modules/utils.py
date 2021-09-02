import torch
import numpy as np
from typing import Optional

__all__ = [
    'split_heads',
    'combine_heads',
    'add_mask'
]

def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor (batch_size, length, dim)
        Input tensor.

    n_heads : int
        Number of attention heads.
    """
    batch_size, dim = x.size(0), x.size(-1)
    x = x.view(batch_size, -1, n_heads, dim // n_heads)  # (batch_size, length, n_heads, d_head)
    x = x.transpose(1, 2)  # (batch_size, n_heads, length, d_head)
    return x

def combine_heads(x: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor (batch_size, n_heads, length, d_head)
        Input tensor.
    """
    batch_size, n_heads, d_head = x.size(0), x.size(1), x.size(3)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, d_head * n_heads)  # (batch_size, length, n_heads * d_head)
    return x

def add_mask(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mask away by setting such weights to a large negative number, so that they evaluate to 0
    under the softmax.

    Parameters
    ----------
    x : torch.Tensor (batch_size, n_heads, *, length) or (batch_size, length)
        Input tensor.

    mask : torch.Tensor, optional (batch_size, length)
        Mask metrix, ``None`` if it is not needed.
    """
    if mask is not None:
        if len(x.size()) == 4:
            expanded_mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, length)
        x = x.masked_fill(expanded_mask.bool(), -np.inf)
    return x
