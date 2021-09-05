from typing import Optional
import torch
from torch import nn

from ..modules.utils import *
from ..modules import RelativePositionEmbedding


class HaloAttention(nn.Module):
    """
    Implementation of Halo Attention (or Blocked Local Self-Attention) proposed in [1].

    Notes
    -----
    In this paper, the authors claimed: "we find that empirically, using a non-masked block
    local self-attention actually improves the accuracy of the model (see Section 4.3)". So I
    didn't implement mask operation here either. But maybe I should investigate.

    Parameters
    ----------
    dim : int
        Dimension of the input features.

    n_heads : int, optional, default=1
        Number of attention heads.

    block_size : int, optional, default=8
        Size of each image block.

    halo_size: int, optional, default=3
        Size of the padded boundary (halo).

    dropout : float, optional
        Dropout, ``None`` if no dropout layer

    References
    ----------
    1. "`Scaling Local Self-Attention for Parameter Efficient Visual Backbones. \
        <https://arxiv.org/abs/2103.12731>`_" Ashish Vaswani, et al. CVPR 2021.

    2. https://github.com/lucidrains/halonet-pytorch
    """
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        block_size: int = 8,
        halo_size: int = 3,
        dropout: Optional[float] = None
    ) -> None:
        super(HaloAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale = self.d_head ** 0.5  # scale factor

        self.block_size = block_size
        self.halo_size = halo_size

        self.W_Q = nn.Conv2d(dim, n_heads * self.d_head, kernel_size=1, bias=False)
        self.W_KV = nn.Conv2d(dim, n_heads * self.d_head * 2, kernel_size=1, bias=False)

        self.window_size = block_size + halo_size * 2  #
        self.unfold_Q = nn.Unfold(block_size, stride=block_size)
        self.unfold_KV = nn.Unfold(self.window_size, stride=block_size, padding=halo_size)

        self.fc = nn.Linear(n_heads * self.d_head, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if dropout is None else nn.Dropout(dropout)

        self.pos_embed = RelativePositionEmbedding(
            block_size = block_size,
            window_size = self.window_size,
            d_head = self.d_head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, dim, height, width)
            Input data, where ``dim`` is the number of channels of the input tensors.

        Returns
        -------
        out : torch.Tensor (batch_size, dim, height, width)
            Output of the Halo Attention.
        """

        batch_size, dim, height, width = x.shape
        assert height % self.block_size == 0 and width % self.block_size == 0, 'feature map dimensions must be divisible by the block size'

        Q = self.W_Q(x)  # (batch_size, n_heads * d_head = dim, height, width)
        KV = self.W_KV(x)  # (batch_size, n_heads * d_head * 2 = dim * 2, height, width)

        # ----- number of blocks -----
        n_blocks_h, n_blocks_w = height // self.block_size, width // self.block_size
        n_blocks = n_blocks_h * n_blocks_w

        # ----- get blocked image -----
        Q = self.unfold_Q(Q)  # (batch_size, dim * (block_size ** 2), n_blocks)
        Q = Q.reshape(batch_size, dim, self.block_size ** 2, n_blocks).transpose(1, 3)  # (batch_size, n_blocks, block_size ** 2, dim)
        Q = Q.reshape(batch_size * n_blocks, self.block_size ** 2, dim)

        # ----- get neighborhood windows -----
        KV = self.unfold_KV(KV)  # (batch_size, dim * 2 * (window_size ** 2), n_blocks)
        KV = KV.reshape(batch_size, dim * 2, self.window_size ** 2, n_blocks).transpose(1, 3)  # (batch_size, n_blocks, window_size ** 2, dim * 2)
        KV = KV.reshape(batch_size * n_blocks, self.window_size ** 2, dim * 2)

        K, V = torch.split(KV, [dim, dim], dim=-1)  # (batch_size * n_blocks, window_size ** 2, dim = n_heads * d_head)

        # ----- split heads -----
        Q, K, V = split_heads(Q, self.n_heads), split_heads(K, self.n_heads), split_heads(V, self.n_heads)
        # Q:    (batch_size * n_blocks, n_heads, block_size ** 2 , d_head)
        # K, V: (batch_size * n_blocks, n_heads, window_size ** 2, d_head)

        # ----- self-attention -----
        score = torch.matmul(Q / self.scale, K.transpose(2, 3))  # (batch_size * n_blocks, n_heads, block_size ** 2, window_size ** 2)

        # add position embeddings
        score += self.pos_embed(Q)

        att = self.softmax(score)
        att = att if self.dropout is None else self.dropout(att)

        context = att @ V  # (batch_size * n_blocks, n_heads, block_size ** 2, d_head)
        context = combine_heads(context)  # (batch_size * n_blocks, block_size ** 2, dim = n_heads * d_head)

        out = self.fc(context)  # (batch_size * n_blocks, block_size ** 2, dim)
        out = out if self.dropout is None else self.dropout(out)

        # ----- merge blocks -----
        out = out\
            .reshape(batch_size, n_blocks_h, n_blocks_w, self.block_size, self.block_size, dim)\
            .permute(0, 5, 1, 3, 2, 4)\
            .reshape(batch_size, dim, height, width)

        return out
