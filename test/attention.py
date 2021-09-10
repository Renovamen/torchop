import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torchop

BATCH_SIZE = 128
HEIGHT = 16
WIDTH = 16
LENGTH = HEIGHT * WIDTH
INPUT_SIZE = 512
N_HEADS = 8

INPUT = torch.randn(BATCH_SIZE, LENGTH, INPUT_SIZE)
INPUT_IMAGE = torch.randn(BATCH_SIZE, INPUT_SIZE, HEIGHT, WIDTH)


def check_size(x):
    assert x.size() == torch.Size([BATCH_SIZE, LENGTH, INPUT_SIZE])

def check_size_image(x):
    assert x.size() == torch.Size([BATCH_SIZE, INPUT_SIZE, HEIGHT, WIDTH])


class TestAttention(unittest.TestCase):
    def test_vanilla(self):
        query = torch.randn(BATCH_SIZE, INPUT_SIZE)
        value = torch.randn(BATCH_SIZE, LENGTH, INPUT_SIZE)

        for align_function in ["dot", "general"]:
            attention = torchop.VanillaAttention(INPUT_SIZE, align_function)
            out, _ = attention(query, value)
            assert out.size() == torch.Size([BATCH_SIZE, INPUT_SIZE])

    def test_multi_head_self_attention(self):
        attention = torchop.SelfAttention(INPUT_SIZE, N_HEADS)
        out = attention(INPUT)
        check_size(out)

        attention = torchop.SimplifiedSelfAttention(INPUT_SIZE, N_HEADS)
        out = attention(INPUT)
        check_size(out)

    def test_sagan_attention(self):
        attention = torchop.SAGANAttention(INPUT_SIZE)
        out = attention(INPUT)
        check_size(out)

    def test_external_attention(self):
        attention = torchop.ExternalAttention(INPUT_SIZE, N_HEADS)
        out = attention(INPUT)
        check_size(out)

    def test_fast_attention(self):
        attention = torchop.FastAttention(INPUT_SIZE, N_HEADS)
        out = attention(INPUT)
        check_size(out)

    def test_halo_attention(self):
        attention = torchop.HaloAttention(INPUT_SIZE, N_HEADS)
        out = attention(INPUT_IMAGE)
        check_size_image(out)

    def test_lin_attention(self):
        attention = torchop.LinAttention(INPUT_SIZE, LENGTH, N_HEADS)
        out = attention(INPUT)
        check_size(out)

if __name__ == '__main__':
    unittest.main()
