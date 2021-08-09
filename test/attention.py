import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torchattn

BATCH_SIZE = 128
INPUT_SIZE = 512
LENGTH = 50

class TestAttention(unittest.TestCase):
    def test_vanilla(self):
        query = torch.randn(BATCH_SIZE, INPUT_SIZE)
        value = torch.randn(BATCH_SIZE, LENGTH, INPUT_SIZE)

        for align_function in ["dot", "general"]:
            attn = torchattn.attention.VanillaAttention(INPUT_SIZE, align_function)
            out = attn(query, value)
            assert out.size() == torch.Size([BATCH_SIZE, INPUT_SIZE])

if __name__ == '__main__':
    unittest.main()
