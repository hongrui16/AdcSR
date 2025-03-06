# (CVPR 2025) Adversarial Diffusion Compression for Real-World Image Super-Resolution [PyTorch]

[![icon](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.13383) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Guaishou74851.AdcSR)

[Bin Chen](https://scholar.google.com/citations?user=aZDNm98AAAAJ)<sup>1,3,\*</sup>
| Gehui Li<sup>1,\*</sup>
| [Rongyuan Wu](https://scholar.google.com/citations?user=A-U8zE8AAAAJ)<sup>2,3,\*</sup>
| [Xindong Zhang](https://scholar.google.com/citations?user=q76RnqIAAAAJ)<sup>3</sup>
| [Jie Chen](https://aimia-pku.github.io/)<sup>1,†</sup>
| [Jian Zhang](https://jianzhang.tech/)<sup>1,†</sup>
| [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>2,3</sup>

<sup>1</sup>*Peking University*, <sup>2</sup>*The Hong Kong Polytechnic University*, <sup>3</sup>*OPPO Research Institute*

<sup>*</sup> Equal Contribution. <sup>†</sup> Corresponding Authors.

:star: If AdcSR is helpful to you, please star this repo. Thanks! :hugs:

## Overview

### Highlights

- **Adversarial Diffusion Compression (ADC).** We remove and prune redundant modules from a one-step diffusion network [OSEDiff](https://github.com/cswry/OSEDiff) and then perform adversarial distillation to retain generation capability despite reduced capacity.
- **Real-Time [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1)-Based Image Super-Resolution.** AdcSR can super-resolve a 128×128 low-resolution image to a 512×512 output **in just 0.03 seconds** on an A100 GPU.
- **Competitive Visual Quality.** Despite its significantly lower complexity (74% fewer parameters vs. [OSEDiff](https://github.com/cswry/OSEDiff)), AdcSR achieves **competitive visual quality and metrics** (PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, etc.) across multiple synthetic and real-world benchmarks.

### Framework

1. Structural Compression
   - **Removable modules** (VAE encoder, textual prompt extractor, cross-attention, time embeddings) are removed.
   - **Prunable modules** (UNet, VAE decoder) are **channel-pruned** to balance efficiency and performance.

<p align="center">
   <img src="figs/teaser.png" alt="teaser" style="width: 50%;" />
</p>

2. Two-Stage Training
   1. **Pretrain the Pruned VAE Decoder** to ensure it can decode latent codes accurately.
   2. **Adversarial Distillation** in the feature space to align the compressed network's features with both the teacher (e.g., [OSEDiff](https://github.com/cswry/OSEDiff)) and ground truth images.

<img src="figs/method.png" alt="method" />

## Installation

```shell
git clone https://github.com/Guaishou74851/AdcSR
cd AdcSR
conda create -n AdcSR python=3.10
conda activate AdcSR
pip install --upgrade pip
pip install -r requirements.txt
```

## Citation

If you find our work or code helpful, please cite the following paper:

```latex
@article{chen2025invertible,
  title={Invertible Diffusion Models for Compressed Sensing},
  author={Chen, Bin and Zhang, Zhenyu and Li, Weiqi and Zhao, Chen and Yu, Jiwen and Zhao, Shijie and Chen, Jie and Zhang, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
}
```
