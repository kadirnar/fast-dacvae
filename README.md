# Fast-DACVAE

Optimized inference for [DACVAE](https://github.com/facebookresearch/dacvae) (Descript Audio Codec with VAE bottleneck).

## Benchmark

NVIDIA H100 PCIe | `facebook/dacvae-watermarked` (107.7M params) | 100s audio @ 48kHz

### Full Precision (FP32) — Zero Quality Loss

| Method | Latency | Speedup | Real-time Factor |
|--------|:-------:|:-------:|:----------------:|
| PyTorch FP32 | 1,047 ms | 1.0x | 96x |
| + channels_last + wn_off | 549 ms | 1.9x | 182x |
| **+ torch.compile + graph** | **209 ms** | **5.0x** | **478x** |

### Half Precision (FP16 / BF16)

| Method | Latency | Speedup | RTF | SNR vs FP32 |
|--------|:-------:|:-------:|:---:|:-----------:|
| PyTorch FP16 | 775 ms | 1.4x | 129x | 40.4 dB |
| + channels_last + wn_off | 307 ms | 3.4x | 326x | 40.2 dB |
| **+ torch.compile + graph (FP16)** | **93 ms** | **11.2x** | **1,071x** | **40.2 dB** |
| + torch.compile + graph (BF16) | 100 ms | 10.5x | 1,004x | 29.8 dB |


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

## Optimizations

All optimizations are applied automatically via `optimize_dacvae()`:

- **Conv1d → Conv2d channels_last** — cuDNN NHWC fast path (1.9x alone)
- **Weight norm removal** — eliminates per-call recomputation
- **Deterministic VAE bottleneck** — pre-initialized noise for CUDA graph
- **torch.compile** — Inductor fusion + freezing + fullgraph (5x with FP32)
- **CUDA graph capture** — zero kernel launch overhead
- **Original snake activation** — exact `sin²(αx)`, no approximation

All optimizations use standard `torch.compile` — no custom kernels required.

## Requirements

- PyTorch 2.9+
- NVIDIA GPU (Hopper/Ampere)

## License

Apache 2.0
