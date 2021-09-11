# torchop

A collection of some attention/convolution operators implemented using PyTorch.

这里是一篇介绍它们的[博客](https://zxh.io/post/2021/08/31/attention-conv/)

**WIP**


&nbsp;

## Installation (optional)

```bash
git clone https://github.com/Renovamen/torchop.git
cd torchop
python setup.py install
```

or

```bash
pip install git+https://github.com/Renovamen/torchop.git --upgrade
```


&nbsp;

## Implemented Networks

### Attention

- Vanilla Attention

  [Neural Machine Translation by Jointly Learning to Align and Translate.](https://arxiv.org/abs/1409.0473) ICLR 2015.

  [Effective Approaches to Attention-based Neural Machine Translation.](https://arxiv.org/abs/1508.04025) EMNLP 2015.

- Self-Attention, Simplified Self-Attention

  [Attention Is All You Need.](https://arxiv.org/abs/1706.03762) NIPS 2017.

- SAGAN Attention

  [Self-Attention Generative Adversarial Networks.](https://arxiv.org/abs/1805.08318) ICML 2019.

- External Attention

  [Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks.](https://arxiv.org/abs/2105.02358) arXiv 2021.

- Fast Attention (proposed in Fastformer)

  [Fastformer: Additive Attention Can Be All You Need.](https://arxiv.org/abs/2108.09084) arXiv 2021.

- Halo Attention (or Blocked Local Self-Attention)

  [Scaling Local Self-Attention for Parameter Efficient Visual Backbones.](https://arxiv.org/abs/2103.12731) CVPR 2019.

- LinAttention (proposed in Linformer)

  [Linformer: Self-Attention with Linear Complexity.](https://arxiv.org/abs/2006.04768) arXiv 2020.


## Convolution

- Selective Kernel (SK) Convolution

  [Selective Kernel Networks.](https://arxiv.org/abs/1903.06586) CVPR 2019.

- Involution

  [Involution: Inverting the Inherence of Convolution for Visual Recognition.](https://arxiv.org/abs/2103.06255) CVPR 2021.

- Squeeze-and-Excitation (SE) Block

  [Squeeze-and-Excitation Networks.](https://arxiv.org/abs/1709.01507) CVPR 2018.

- CBAM

  [CBAM: Convolutional Block Attention Module.](https://arxiv.org/abs/1807.06521) ECCV 2018.
