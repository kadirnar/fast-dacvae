# Fast-DACVAE

Optimized inference for [DACVAE](https://github.com/facebookresearch/dacvae) (Descript Audio Codec with VAE bottleneck). **6.4x faster** on NVIDIA H100 GPU using cuDNN v9 runtime fusion.

## Benchmark Results

**Hardware**: NVIDIA H100 PCIe (80GB) | **Audio**: 101.8s (4,489,380 samples at 44.1kHz)

| Backend | p50 Latency | Min Latency | Speedup |
|---------|-------------|-------------|---------|
| Baseline (PyTorch FP32) | 377ms | - | 1.0x |
| Inductor BF16 (torch.compile) | 224ms | 223ms | 1.7x |
| **Inductor BF16 + cuDNN v9 Fusion** | **58.7ms** | **56.2ms** | **6.4x** |

> The cuDNN v9 fusion pass automatically fuses conv+snake and conv+residual patterns into single GPU kernels, eliminating intermediate memory traffic.

## Optimizations

### Core: cuDNN v9 Runtime Fusion (NEW)

A custom [Inductor compiler pass](dacvae/inductor_fusion.py) that pattern-matches conv+snake sequences in the compiled FX graph and replaces them with cuDNN v9 fused kernels:

- **26 conv+snake fusions** — snake activation computed inside the conv epilogue (zero intermediate memory writes)
- **24 conv+residual fusions** — k=1 convolution + residual addition in a single kernel
- **50 total fused patterns** per model compilation

This eliminates ~165ms of memory traffic overhead per forward pass.

### Additional Optimizations

1. **Conv1d to Conv2d channels_last** — unlocks cuDNN NHWC fast-path algorithms
2. **Polynomial Snake activation** — SFU-free sin² approximation for non-fused snakes
3. **Weight norm removal** — eliminates per-call weight recomputation
4. **torch.compile + CUDA graph** — Inductor kernel fusion + zero launch overhead
5. **BF16 precision** — halves memory traffic vs FP32
6. **DecoderBlock precomputation** — avoids dynamic `nn.Sequential` construction
7. **Deterministic VAE bottleneck** — pre-allocated noise for CUDA graph compatibility
8. **Static ResidualUnit shortcuts** — eliminates dynamic shape computation

## Requirements

- PyTorch 2.9+
- NVIDIA GPU with compute capability 8.0+ (Ampere/Hopper)
- cuDNN 9.1+ with Python frontend (`pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend`)
- Triton 3.0+

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from dacvae import DACVAE, optimize_dacvae

# Load model
model = DACVAE().cuda().eval()

# Create sample audio (10 seconds at 44.1kHz)
audio = torch.randn(1, 1, 441000, device="cuda")

# Optimize — cuDNN v9 fusion is applied automatically
replay_fn, description, original_length = optimize_dacvae(model, audio, backend="inductor")

# Run optimized inference (CUDA graph replay)
with torch.no_grad():
    output = replay_fn()
```

## Benchmark

```bash
# Benchmark with generated audio
python benchmark.py --duration 101.8

# Benchmark with a WAV file
python benchmark.py --audio your_audio.wav

# Benchmark specific backend
python benchmark.py --backend inductor
```

## How It Works

### The Problem

DACVAE uses **Snake activation** (`x + (1/α)·sin²(α·x)`) after every convolution — 57 times in the model. Each Snake reads the conv output from global memory and writes the result back, creating ~210GB of memory traffic per forward pass.

### The Solution

cuDNN v9's [Runtime Fusion Engine](https://docs.nvidia.com/cudnn/) can fuse arbitrary pointwise operations into convolution kernels. We teach PyTorch's Inductor compiler to use this:

1. **Pattern matching**: A custom `post_grad_custom_pre_pass` scans the FX graph for `aten.convolution → polynomial_snake` sequences
2. **Graph rewriting**: Matched patterns are replaced with `dacvae::conv_snake_fused` custom ops
3. **cuDNN execution**: Each custom op calls a pre-built `cudnn.pygraph` that computes conv + bias + sin²(αx)/α in a single kernel
4. **Zero overhead**: Inductor handles all remaining operations (standalone snakes, bias, etc.) with its own Triton kernel fusion

```
Before: Conv → [write to memory] → Snake → [write to memory] → next op
After:  Conv+Snake → [single write to memory] → next op
```

### Architecture

```
Audio [B, 1, T] → Encoder (Conv2d NHWC, 512x downsample) → VAE Bottleneck → Decoder (ConvTranspose2d, 512x upsample) → Audio [B, 1, T]
```

- **91M parameters** (encoder + decoder + watermark)
- **Encoder**: 4 blocks × 3 ResidualUnits (dilated conv + Snake) + stride convolutions
- **Decoder**: 4 blocks × 3 ResidualUnits + transposed convolutions
- **Bottleneck**: VAE with 1x1 convolution projections (negligible cost)

## File Structure

```
dacvae/
├── optimize.py          # Main optimization pipeline
├── inductor_fusion.py   # cuDNN v9 Inductor fusion pass (conv+snake, conv+add)
├── triton_snake.py      # Fast Triton snake kernel (standalone use)
├── cudnn_forward.py     # Experimental fully-manual cuDNN forward
├── model/dacvae.py      # Model architecture
└── nn/layers.py         # Snake1d, NormConv1d definitions
benchmark.py             # Benchmark harness
```

## Acknowledgments

Based on [DACVAE](https://github.com/facebookresearch/dacvae) by Meta Research.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
