import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torchattn

BATCH_SIZE = 128
LENGTH = 50
INPUT_SIZE = 512
IN_CHANNELS = 32
IMAGE_W = 14
IMAGE_H = 14
N_HEADS = 8

class TestAttention(unittest.TestCase):
    def test_vanilla(self):
        query = torch.randn(BATCH_SIZE, INPUT_SIZE)
        value = torch.randn(BATCH_SIZE, LENGTH, INPUT_SIZE)

        for align_function in ["dot", "general"]:
            attention = torchattn.VanillaAttention(INPUT_SIZE, align_function)
            out, _ = attention(query, value)
            assert out.size() == torch.Size([BATCH_SIZE, INPUT_SIZE])

    def test_multi_head_self_attention(self):
        x = torch.randn(BATCH_SIZE, LENGTH, INPUT_SIZE)

        attention = torchattn.SelfAttention(INPUT_SIZE, N_HEADS)
        out, _ = attention(x)
        assert out.size() == torch.Size([BATCH_SIZE, LENGTH, INPUT_SIZE])

        attention = torchattn.SimplifiedSelfAttention(INPUT_SIZE, N_HEADS)
        out, _ = attention(x)
        assert out.size() == torch.Size([BATCH_SIZE, LENGTH, INPUT_SIZE])

    def test_simple_self_attention(self):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMAGE_W, IMAGE_H)
        attention = torchattn.SimpleSelfAttention(IN_CHANNELS)
        out, _ = attention(x)
        assert out.size() == torch.Size([BATCH_SIZE, IN_CHANNELS, IMAGE_W, IMAGE_H])

if __name__ == '__main__':
    unittest.main()
