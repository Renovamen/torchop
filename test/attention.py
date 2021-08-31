import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torchattn

BATCH_SIZE = 128
LENGTH = 196
INPUT_SIZE = 512
N_HEADS = 8

INPUT = torch.randn(BATCH_SIZE, LENGTH, INPUT_SIZE)


def check_self_attention_size(x):
    assert x.size() == torch.Size([BATCH_SIZE, LENGTH, INPUT_SIZE])


class TestAttention(unittest.TestCase):
    def test_vanilla(self):
        query = torch.randn(BATCH_SIZE, INPUT_SIZE)
        value = torch.randn(BATCH_SIZE, LENGTH, INPUT_SIZE)

        for align_function in ["dot", "general"]:
            attention = torchattn.VanillaAttention(INPUT_SIZE, align_function)
            out, _ = attention(query, value)
            assert out.size() == torch.Size([BATCH_SIZE, INPUT_SIZE])

    def test_multi_head_self_attention(self):
        attention = torchattn.SelfAttention(INPUT_SIZE, N_HEADS)
        out, _ = attention(INPUT)
        check_self_attention_size(out)

        attention = torchattn.SimplifiedSelfAttention(INPUT_SIZE, N_HEADS)
        out, _ = attention(INPUT)
        check_self_attention_size(out)

    def test_simple_self_attention(self):
        attention = torchattn.SimpleSelfAttention(INPUT_SIZE)
        out, _ = attention(INPUT)
        check_self_attention_size(out)

if __name__ == '__main__':
    unittest.main()
