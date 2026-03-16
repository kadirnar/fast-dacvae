# Fast-DACVAE

Fast inference engine for [DACVAE](https://github.com/facebookresearch/dacvae), a neural audio codec that compresses and reconstructs audio using a convolutional encoder-decoder with a VAE bottleneck. This library accelerates DACVAE inference up to **11.2x** on NVIDIA GPUs through graph-level optimizations — with no custom kernels, no quality loss at FP32, and no changes to model weights.

## Benchmark

NVIDIA H100 PCIe | `facebook/dacvae-watermarked` (107.7M params) | 100s audio @ 48kHz

| Method | FP32 | FP16 | BF16 |
|--------|:----:|:----:|:----:|
| PyTorch (baseline) | 1,047 ms | 775 ms | 775 ms |
| + channels_last | 549 ms | 307 ms | 305 ms |
| + torch.compile + CUDA graph | **209 ms** | **93 ms** | **100 ms** |


## Quick Start

```bash
pip install -e .
```

```python
from dacvae import DACVAE
from dacvae.optimize import optimize_dacvae
import torch

model = DACVAE.load("facebook/dacvae-watermarked").cuda().eval()
audio = torch.randn(1, 1, 4800000, device="cuda")

# FP16 (fastest, high quality)
replay_fn, _, _ = optimize_dacvae(model, audio, backend="inductor")
output = replay_fn()  # ~93ms
```


## Requirements

- PyTorch 2.9+
- NVIDIA GPU (Hopper/Ampere)

## License

Apache 2.0
