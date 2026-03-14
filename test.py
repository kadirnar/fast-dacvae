import torch
import torch.nn.functional as F
from dacvae import DACVAE, optimize_dacvae

model = DACVAE().cuda().eval()

# Optimize model (in-place: bf16, conv2d, poly snake, etc.)
dummy = torch.randn(1, 1, 441000, device="cuda")
_, description, _ = optimize_dacvae(model, dummy, backend="inductor")

hop = model.hop_length
dtype = next(model.parameters()).dtype

# 5 audio samples on CPU
audios = [torch.randn(1, 1, 441000) for _ in range(5)]

# Warmup
with torch.no_grad():
    x = dummy.to(dtype)
    if x.shape[-1] % hop:
        x = F.pad(x, (0, hop - x.shape[-1] % hop), mode="reflect")
    for _ in range(3):
        model(x)
torch.cuda.synchronize()

# Extract closure internals from optimized forward
fwd = model.forward
_closure = dict(zip(fwd.__code__.co_freevars, (c.cell_contents for c in fwd.__closure__)))
_run_module = _closure['_run_module']
_sn = _closure['_sn']

def profiled_forward(audio_data):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    events[0].record()
    x = audio_data.unsqueeze(2).to(memory_format=torch.channels_last)

    events[1].record()
    for layer in model.encoder.block:
        x = _run_module(layer, x)

    events[2].record()
    combined = model.quantizer.in_proj(x)
    mean, scale = combined.chunk(2, dim=1)
    stdev = F.softplus(scale) + 1e-4
    z_q = _sn[0] * stdev + mean
    emb = model.quantizer.out_proj(z_q)

    events[3].record()
    for layer in model.decoder.model:
        emb = _run_module(layer, emb)
    output = emb.squeeze(2)

    events[4].record()
    torch.cuda.synchronize()

    return output, {
        "prep": events[0].elapsed_time(events[1]),
        "encoder": events[1].elapsed_time(events[2]),
        "quantizer": events[2].elapsed_time(events[3]),
        "decoder": events[3].elapsed_time(events[4]),
        "total": events[0].elapsed_time(events[4]),
    }

# Benchmark
print(f"Config: {description}\n")
header = f"{'Run':<5} {'Transfer':>10} {'Prep':>8} {'Encoder':>10} {'Quantizer':>10} {'Decoder':>10} {'Total':>10}"
print(header)
print("-" * len(header))

all_times = []
with torch.no_grad():
    for i, audio in enumerate(audios):
        ts = torch.cuda.Event(enable_timing=True)
        te = torch.cuda.Event(enable_timing=True)

        ts.record()
        x = audio.to(device="cuda", dtype=dtype)
        if x.shape[-1] % hop:
            x = F.pad(x, (0, hop - x.shape[-1] % hop), mode="reflect")
        te.record()
        torch.cuda.synchronize()
        transfer = ts.elapsed_time(te)

        output, t = profiled_forward(x)
        all_times.append({"transfer": transfer, **t})
        print(f"{i+1:<5} {transfer:>8.2f}ms {t['prep']:>6.2f}ms {t['encoder']:>8.2f}ms {t['quantizer']:>8.2f}ms {t['decoder']:>8.2f}ms {transfer+t['total']:>8.2f}ms")

print("-" * len(header))
avg = {k: sum(d[k] for d in all_times) / len(all_times) for k in all_times[0]}
print(f"{'Avg':<5} {avg['transfer']:>8.2f}ms {avg['prep']:>6.2f}ms {avg['encoder']:>8.2f}ms {avg['quantizer']:>8.2f}ms {avg['decoder']:>8.2f}ms {avg['transfer']+avg['total']:>8.2f}ms")
