import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torchop

BATCH_SIZE = 128
IN_CHANNELS = 128
OUT_CHANNELS = 128
WIDTH = 56
HEIGHT = 56

INPUT = torch.randn(BATCH_SIZE, IN_CHANNELS, WIDTH, HEIGHT)


class TestAttention(unittest.TestCase):
    def test_skconv(self):
        conv = torchop.SKConv(IN_CHANNELS)
        out = conv(INPUT)
        assert out.size() == torch.Size([BATCH_SIZE, IN_CHANNELS, WIDTH, HEIGHT])

    def test_involution(self):
        STRIDE = 4
        conv = torchop.Involution(IN_CHANNELS, kernel_size=3, stride=STRIDE)
        out = conv(INPUT)
        assert out.size() == torch.Size([BATCH_SIZE, IN_CHANNELS, WIDTH // STRIDE, HEIGHT // STRIDE])

    def test_seblock(self):
        conv = torchop.SEBlock(IN_CHANNELS)
        out = conv(INPUT)
        assert out.size() == torch.Size([BATCH_SIZE, IN_CHANNELS, WIDTH, HEIGHT])


if __name__ == '__main__':
    unittest.main()
