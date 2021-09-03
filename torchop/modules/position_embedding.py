import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'RelativePositionEmbedding'
]


class RelativePositionEmbedding(nn.Module):
    """
    Implementation of Relative Position Embedding proposed in [1].
    Adopted from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/halo_attn.py

    References
    ----------
    1. "`Attention Augmented Convolutional Networks. <https://arxiv.org/abs/1904.09925>`_" \
        Irwan Bello, et al. ICCV 2019.
    """
    def __init__(
        self,
        block_size: int,
        window_size: int,
        d_head: int
    ) -> None:
        super(RelativePositionEmbedding, self).__init__()

        self.block_size = block_size
        self.window_size = window_size

        scale = d_head ** 0.5
        self.rel_height = nn.Parameter(torch.randn(window_size * 2 - 1, d_head) / scale)
        self.rel_width = nn.Parameter(torch.randn(window_size * 2 - 1, d_head) / scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, length, d_head = x.shape  # length = block_size ** 2
        assert length == self.block_size ** 2

        x = x.reshape(-1, self.block_size, self.block_size, d_head)  # (batch_size * n_heads, block_size, block_size, d_head)
        rel_logits_w = self.rel_logits_1d(x, self.rel_width)  # (batch_size * n_heads, block_size, window_size, block_size, window_size)
        rel_logits_w = rel_logits_w.permute(0, 1, 3, 2, 4)  # (batch_size * n_heads, block_size, block_size, window_size, window_size)

        x = x.transpose(1, 2)  # (batch_size * n_heads, block_size, block_size, d_head)
        rel_logits_h = self.rel_logits_1d(x, self.rel_height)  # (batch_size * n_heads, block_size, window_size, block_size, window_size)
        rel_logits_h = rel_logits_h.permute(0, 3, 1, 4, 2)  # (batch_size * n_heads, block_size, block_size, window_size, window_size)

        rel_logits = rel_logits_w + rel_logits_h
        rel_logits = rel_logits.reshape(batch_size, n_heads, length, -1)  # (batch_size, n_heads, length, window_size ** 2)

        return rel_logits

    def rel_logits_1d(self, x: torch.Tensor, rel) -> torch.Tensor:
        batch_size, h, w, _ = x.shape
        rel_size = rel.shape[0]

        logits = x @ rel.transpose(0, 1)  # (batch_size, h, w, rel_size)
        logits = logits.view(-1, w, rel_size)  # (batch_size * h, w, rel_size)

        logits = self.rel_to_abs(logits)  # (batch_size * h, w, window_size)
        logits = logits.reshape(batch_size, h, 1, w, self.window_size)
        logits = logits.expand(-1, -1, self.window_size, -1, -1)  # (batch_size, h, window_size, w, window_size)

        return logits

    def rel_to_abs(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, rel_size = x.shape

        # pad to shift from relative to absolute indexing
        x = F.pad(x, pad=[0, 1])  # (batch_size, length, rel_size + 1)
        flat_x = x.reshape(batch_size, -1)  # (batch_size, length * (rel_size + 1))
        flat_padded_x = F.pad(flat_x, pad=[0, rel_size - length])  # (batch_size, (length + 1) * rel_size)

        # reshape and slice out the padded elements
        out = flat_padded_x.reshape(batch_size, length + 1, rel_size)
        out = out[:, :length, -self.window_size:]  # (batch_size, length, window_size)

        return out
