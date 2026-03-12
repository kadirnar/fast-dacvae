# Fast-DACVAE

Optimized inference for [DACVAE](https://github.com/facebookresearch/dacvae) (Descript Audio Codec with VAE bottleneck). **3.6x faster** on NVIDIA H100 GPU.

## Benchmark Results

**Hardware**: NVIDIA H100 PCIe (80GB) | **Audio**: 101.8s (4,491,264 samples at 44.1kHz)

| Backend | p50 Latency | Min Latency | Speedup |
|---------|-------------|-------------|---------|
| Baseline (PyTorch FP32) | 377ms | - | 1.0x |
| **Inductor BF16** | **108ms** | 106ms | **3.5x** |
| **TensorRT FP16** | **103ms** | 101ms | **3.6x** |

> Inductor BF16 requires no extra dependencies. TensorRT FP16 requires `torch-tensorrt`.

## Optimizations

1. **Conv1d to Conv2d channels_last** — unlocks cuDNN NHWC fast-path algorithms
2. **Polynomial Snake activation** — SFU-free sin² approximation, fuses with Inductor
3. **Weight norm removal** — eliminates per-call weight recomputation
4. **torch.compile + CUDA graph** — kernel fusion + zero launch overhead
5. **Precision**: BF16 (Inductor) / FP16 (TensorRT)
6. **DecoderBlock precomputation** — avoids dynamic `nn.Sequential` construction
7. **Deterministic VAE bottleneck** — pre-allocated noise for CUDA graph compatibility
8. **Skip watermark path** — removes unused watermark computation during inference

## Installation

```bash
pip install -e .

# For TensorRT support (optional, fastest backend):
pip install torch-tensorrt
```

## Quick Start

```python
import torch
from dacvae import DACVAE, optimize_dacvae

# Load model
model = DACVAE().cuda().eval()

# Create sample audio (10 seconds at 44.1kHz)
audio = torch.randn(1, 1, 441000, device="cuda")

# Optimize with Inductor BF16 (no extra deps)
replay_fn, description, original_length = optimize_dacvae(model, audio, backend="inductor")

# Run optimized inference
with torch.no_grad():
    output = replay_fn()

# Or use TensorRT FP16 (fastest, requires torch-tensorrt)
model2 = DACVAE().cuda().eval()
replay_fn, description, original_length = optimize_dacvae(model2, audio, backend="tensorrt")
```

## Benchmark

```bash
# Benchmark all backends (generates random audio)
python benchmark.py --duration 10

# Benchmark with a WAV file
python benchmark.py --audio your_audio.wav

# Benchmark specific backend
python benchmark.py --backend inductor
python benchmark.py --backend tensorrt
```

## How It Works

The model is **memory-bandwidth bound**: ~210GB of data moves through the GPU per forward pass. On H100 PCIe (2039 GB/s bandwidth), the theoretical minimum is ~103ms for FP16 — which is exactly what TensorRT achieves.

### Architecture

```
Audio [B, 1, T] → Encoder (Conv1d, 512x downsample) → VAE Bottleneck → Decoder (ConvTranspose1d, 512x upsample) → Audio [B, 1, T]
```

- **76.6M parameters** (encoder 40%, decoder 60%)
- **Encoder**: 4 blocks with dilated convolutions + Snake activation
- **Decoder**: 4 blocks with transposed convolutions + residual units
- **Bottleneck**: VAE with 1x1 convolution projections (negligible cost)

## Acknowledgments

Based on [DACVAE](https://github.com/facebookresearch/dacvae) by Meta Research.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
