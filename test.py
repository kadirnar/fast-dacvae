import torch
import torch.nn.functional as F
from dacvae import DACVAE
from dacvae.optimize import optimize_dacvae

model = DACVAE().cuda().eval()

# Optimize — returns replay function (compiled + CUDA graph)
dummy = torch.randn(1, 1, 441000, device="cuda")
replay_fn, description, original_length = optimize_dacvae(model, dummy, backend="inductor")

print(f"Config: {description}\n")

# Benchmark using replay (compiled + CUDA graph)
with torch.no_grad():
    for _ in range(20):
        replay_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(100):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        output = replay_fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

times.sort()
n = len(times)
print(f"p50={times[n//2]:.2f}ms  min={times[0]:.2f}ms  p90={times[n*9//10]:.2f}ms")
print(f"Realtime factor: {441000/44100/(times[n//2]/1000):,.0f}x")
