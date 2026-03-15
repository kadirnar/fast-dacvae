# Fast-DACVAE

Optimized inference for [DACVAE](https://github.com/facebookresearch/dacvae) (Descript Audio Codec with VAE bottleneck). **14.3x faster** on NVIDIA H100 GPU — **3,859x real-time** audio processing.

## Benchmark

**Hardware**: NVIDIA H100 PCIe (80GB)

| Audio Duration | Baseline (FP32) | **Optimized** | Speedup | Real-time Factor |
|---------------|-----------------|--------------|---------|-----------------|
| 10s (441K samples) | 37ms | **2.8ms** | 13.2x | 3,588x |
| 100s (4.4M samples) | 377ms | **26.4ms** | 14.3x | 3,859x |

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from dacvae import DACVAE
from dacvae.optimize import optimize_dacvae

model = DACVAE().cuda().eval()
audio = torch.randn(1, 1, 441000, device="cuda")

replay_fn, description, _ = optimize_dacvae(model, audio, backend="inductor")

with torch.no_grad():
    output = replay_fn()
```

```bash
python benchmark.py --duration 101.8 --backend inductor
```

## How It Works

DACVAE uses Snake activation (`x + (1/α)·sin²(α·x)`) after every convolution. Each Snake reads the conv output from global memory and writes back, creating ~210GB of memory traffic per forward pass.

A custom Inductor compiler pass pattern-matches conv+snake sequences in the FX graph and replaces them with cuDNN v9 fused kernels that compute snake inside the conv epilogue — eliminating intermediate memory writes. 50 patterns are fused per compilation (26 conv+snake, 24 conv+residual).

## Requirements

- PyTorch 2.9+
- NVIDIA GPU (Ampere/Hopper)
- cuDNN 9.1+ with Python frontend (`pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend`)

## Acknowledgments

Based on [DACVAE](https://github.com/facebookresearch/dacvae) by Meta Research.

## License

Apache 2.0
