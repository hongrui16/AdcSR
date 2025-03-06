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

- **Adversarial Diffusion Compression (ADC).** We remove and prune redundant modules from a one-step diffusion network and then perform adversarial distillation to retain generation capability despite reduced capacity.
- **Real-time Image Super-Resolution.** AdcSR can super-resolve a 128×128 low-resolution image to a 512×512 output **in just 0.03 seconds** on an A100 GPU.
- **Competitive Visual Quality.** Despite its significantly lower complexity (74% fewer parameters vs. OSEDiff), AdcSR achieves **competitive visual quality and metrics** (PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, etc.) across multiple synthetic and real-world benchmarks.

### Architecture

1. Structural Compression
   - **Removable modules** (VAE encoder, textual prompt extractor, cross-attention, time embeddings) are removed.
   - **Prunable modules** (UNet, VAE decoder) are **channel-pruned** to balance efficiency and performance.

![teaser](figs/teaser.png)

2. Two-Stage Training
   1. **Pretrain the Pruned VAE Decoder** to ensure it can decode latent codes accurately.
   2. **Adversarial Distillation** in the feature space to align the compressed network’s features with both the teacher (e.g., OSEDiff) and ground truth images.

![method](figs/method.png)

## Environment

```shell
torch==2.3.1+cu121
diffusers==0.30.2
transformers==4.44.2
numpy==1.26.3
opencv-python==4.10.0
scikit-image==0.24.0
```

## Test

Download the pretrained models ([Google Drive](https://drive.google.com/file/d/1UzUg0lFqwWfmeXi8gqAOeQA3yt4HpPNy/view?usp=sharing), [PKU Disk 北大网盘](https://disk.pku.edu.cn/link/AA0B0294E9BCF64185B677BDF0951A7D54)) and put the `weight` directory into `./`, then run the following command:

```shell
python test.py --cs_ratio=0.1/0.3/0.5 --testset_name=Set11/CBSD68/Urban100/DIV2K
```

The reconstructed images will be in `./result`.

The test sets CBSD68, Urban100, and DIV2K are available at https://github.com/Guaishou74851/SCNet/tree/main/data.

For easy comparison, test results of various existing image CS methods are available on [Google Drive](https://drive.google.com/drive/folders/1Lif_7N_bCyILFLac5JcOtJ9cWpGBNVCd) and [PKU Disk](https://disk.pku.edu.cn/link/AA1C2D8A08050744449CBFCAB51A846B2D).

## Train

Download the dataset of [Waterloo Exploration Database](https://kedema.org/project/exploration/index.html) ([Google Drive](https://drive.google.com/file/d/1TOg7BZE1XsJ7l2VzMoqFRAETk7OLcv75/view?usp=drive_link), [PKU Disk 北大网盘](https://disk.pku.edu.cn/link/AAD0DCBBD65D744526921B334ED2AB4F76)) and put the `pristine_images` directory (containing 4744 `.bmp` image files) into `./data`, then run the following command:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=23333 train.py --cs_ratio=0.1/0.3/0.5
```

The log and model files will be in `./log` and `./weight`, respectively.

## Results

![res1](figs/res1.png)

![res2](figs/res2.png)

## Citation

If you find the code helpful in your research or work, please cite the following paper:

```latex
@article{chen2025invertible,
  title={Invertible Diffusion Models for Compressed Sensing},
  author={Chen, Bin and Zhang, Zhenyu and Li, Weiqi and Zhao, Chen and Yu, Jiwen and Zhao, Shijie and Chen, Jie and Zhang, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
}
```
