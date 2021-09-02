import torch

__all__ = [
    'split_heads',
    'combine_heads'
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
