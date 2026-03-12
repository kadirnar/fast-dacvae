#!/usr/bin/env python3
"""Benchmark: DACVAE baseline vs optimized inference (Inductor BF16 / TensorRT FP16)."""
import argparse
import gc
import time

import torch
import torch.nn.functional as F

from dacvae import DACVAE, optimize_dacvae


def generate_audio(duration_sec: float = 10.0, sample_rate: int = 44100):
    """Generate random audio tensor for benchmarking."""
    n_samples = int(duration_sec * sample_rate)
    return torch.randn(1, 1, n_samples, device="cuda")


def load_audio_wav(path: str, target_sr: int = 44100):
    """Load a WAV file and resample to target sample rate."""
    import wave

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        data = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
        if wf.getnchannels() > 1:
            data = data.view(-1, wf.getnchannels()).mean(dim=1)
    if sr != target_sr:
        data = F.interpolate(
            data[None, None],
            size=int(len(data) * target_sr / sr),
            mode="linear",
            align_corners=False,
        ).squeeze()
    return data[None, None].cuda()


def bench(fn, warmup: int = 10, timed: int = 50):
    """Benchmark a function using CUDA events."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
        for i in range(timed):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()

        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        n = len(times)
        return {
            "min": times[0],
            "p10": times[n // 10],
            "p50": times[n // 2],
            "p90": times[n * 9 // 10],
            "mean": sum(times) / n,
        }


def bench_baseline(model, x):
    """Benchmark unoptimized PyTorch baseline."""
    model = model.cuda().eval()
    with torch.no_grad():
        fn = lambda: model(x)
        return bench(fn)


def bench_optimized(x, backend="inductor"):
    """Benchmark optimized model with given backend."""
    model = DACVAE().cuda().eval()
    replay, desc, _ = optimize_dacvae(model, x, backend=backend)
    result = bench(replay, warmup=20, timed=100)
    result["desc"] = desc
    return result


def main():
    parser = argparse.ArgumentParser(description="DACVAE Inference Benchmark")
    parser.add_argument("--audio", type=str, default=None, help="Path to WAV file")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Audio duration in seconds (if no WAV file)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="all",
        choices=["all", "inductor", "tensorrt", "baseline"],
        help="Backend to benchmark",
    )
    args = parser.parse_args()

    # Load or generate audio
    if args.audio:
        x = load_audio_wav(args.audio)
        duration = x.shape[-1] / 44100
    else:
        x = generate_audio(args.duration)
        duration = args.duration

    print(f"Audio: {x.shape[-1]:,} samples ({duration:.1f}s)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = []

    # Baseline
    if args.backend in ("all", "baseline"):
        print("=== Baseline (PyTorch, FP32) ===")
        model = DACVAE().cuda().eval()
        r = bench_baseline(model, x)
        print(f"  p50={r['p50']:.1f}ms  min={r['min']:.1f}ms  p90={r['p90']:.1f}ms")
        results.append(("Baseline (FP32)", r))
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)

    # Inductor BF16
    if args.backend in ("all", "inductor"):
        print("=== Inductor BF16 (torch.compile + CUDA graph) ===")
        r = bench_optimized(x, backend="inductor")
        print(f"  p50={r['p50']:.1f}ms  min={r['min']:.1f}ms  p90={r['p90']:.1f}ms")
        results.append(("Inductor BF16", r))
        torch.cuda.empty_cache()
        gc.collect()
        torch._dynamo.reset()
        time.sleep(3)

    # TensorRT FP16
    if args.backend in ("all", "tensorrt"):
        try:
            import torch_tensorrt  # noqa: F401

            print("=== TensorRT FP16 (torch_tensorrt + CUDA graph) ===")
            r = bench_optimized(x, backend="tensorrt")
            print(f"  p50={r['p50']:.1f}ms  min={r['min']:.1f}ms  p90={r['p90']:.1f}ms")
            results.append(("TensorRT FP16", r))
        except ImportError:
            print("=== TensorRT FP16: SKIPPED (torch_tensorrt not installed) ===")
            print("  Install with: pip install torch-tensorrt")

    # Summary
    if len(results) > 1:
        baseline_p50 = next((r["p50"] for name, r in results if "Baseline" in name), None)

        print(f"\n{'=' * 70}")
        print(f"DACVAE Inference Benchmark — {x.shape[-1]:,} samples ({duration:.1f}s)")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"{'=' * 70}")
        print(f"{'Backend':<25} {'p50':>8} {'min':>8} {'p90':>8} {'Speedup':>8}")
        print(f"{'-' * 70}")
        for name, r in sorted(results, key=lambda x: x[1]["p50"]):
            speedup = f"{baseline_p50 / r['p50']:.2f}x" if baseline_p50 else "-"
            print(f"  {name:<23} {r['p50']:>7.1f}ms {r['min']:>7.1f}ms {r['p90']:>7.1f}ms {speedup:>7}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
