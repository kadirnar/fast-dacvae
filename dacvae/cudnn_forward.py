"""
DACVAE inference using cuDNN v9 Runtime Fusion Engine.

Fuses conv+bias+snake into single cuDNN kernels, eliminating intermediate
memory writes. Combined with Triton snake for standalone activations.

Usage:
    from dacvae.cudnn_forward import optimize_cudnn
    replay = optimize_cudnn(model, x_sample)
    output = replay()  # CUDA graph replay
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cudnn

from dacvae.triton_snake import snake_forward


def nhwc_strides(n, c, h, w):
    return [c * h * w, 1, c * w, c]


class CudnnConvSnakeGraph:
    """cuDNN fused Conv2d + bias + snake epilogue."""

    def __init__(self, C_in, C_out, T, K_w, dilation, pad):
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        self._gx = g.tensor(name='x', dim=[1, C_in, 1, T],
                            stride=nhwc_strides(1, C_in, 1, T),
                            data_type=cudnn.data_type.BFLOAT16)
        self._gw = g.tensor(name='w', dim=[C_out, C_in, 1, K_w],
                            stride=nhwc_strides(C_out, C_in, 1, K_w),
                            data_type=cudnn.data_type.BFLOAT16)
        self._gb = g.tensor(name='b', dim=[1, C_out, 1, 1],
                            stride=[C_out, 1, 1, 1],
                            data_type=cudnn.data_type.BFLOAT16)
        self._ga = g.tensor(name='a', dim=[1, C_out, 1, 1],
                            stride=[C_out, 1, 1, 1],
                            data_type=cudnn.data_type.FLOAT)
        self._gia = g.tensor(name='ia', dim=[1, C_out, 1, 1],
                             stride=[C_out, 1, 1, 1],
                             data_type=cudnn.data_type.FLOAT)

        co = g.conv_fprop(image=self._gx, weight=self._gw,
                          pre_padding=[0, pad], post_padding=[0, pad],
                          stride=[1, 1], dilation=[1, dilation],
                          compute_data_type=cudnn.data_type.FLOAT, name='conv')
        bi = g.bias(name='bias', input=co, bias=self._gb)
        ax = g.mul(bi, self._ga, name='ax')
        sn = g.sin(ax, name='sin')
        s2 = g.mul(sn, sn, name='sin2')
        sc = g.mul(s2, self._gia, name='sc')
        self._go = g.add(bi, sc, name='out')
        self._go.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        ws_size = g.get_workspace_size()
        self._ws = torch.empty(max(ws_size, 1), device='cuda', dtype=torch.uint8)
        self._graph = g

    def __call__(self, x, w, b, alpha, inv_alpha, output):
        self._graph.execute(
            {self._gx: x, self._gw: w, self._gb: b,
             self._ga: alpha, self._gia: inv_alpha, self._go: output},
            self._ws
        )


class CudnnConvResGraph:
    """cuDNN fused Conv2d(k=1) + bias + residual_add."""

    def __init__(self, C_in, C_out, T):
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        self._gx = g.tensor(name='x', dim=[1, C_in, 1, T],
                            stride=nhwc_strides(1, C_in, 1, T),
                            data_type=cudnn.data_type.BFLOAT16)
        self._gw = g.tensor(name='w', dim=[C_out, C_in, 1, 1],
                            stride=nhwc_strides(C_out, C_in, 1, 1),
                            data_type=cudnn.data_type.BFLOAT16)
        self._gb = g.tensor(name='b', dim=[1, C_out, 1, 1],
                            stride=[C_out, 1, 1, 1],
                            data_type=cudnn.data_type.BFLOAT16)
        self._gr = g.tensor(name='res', dim=[1, C_out, 1, T],
                            stride=nhwc_strides(1, C_out, 1, T),
                            data_type=cudnn.data_type.BFLOAT16)

        co = g.conv_fprop(image=self._gx, weight=self._gw,
                          pre_padding=[0, 0], post_padding=[0, 0],
                          stride=[1, 1], dilation=[1, 1],
                          compute_data_type=cudnn.data_type.FLOAT, name='conv')
        bi = g.bias(name='bias', input=co, bias=self._gb)
        self._go = g.add(bi, self._gr, name='add_res')
        self._go.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        ws_size = g.get_workspace_size()
        self._ws = torch.empty(max(ws_size, 1), device='cuda', dtype=torch.uint8)
        self._graph = g

    def __call__(self, x, w, b, residual, output):
        self._graph.execute(
            {self._gx: x, self._gw: w, self._gb: b,
             self._gr: residual, self._go: output},
            self._ws
        )


class CudnnConvGraph:
    """cuDNN Conv2d + bias (no fusion, for convs without snake)."""

    def __init__(self, C_in, C_out, T, K_w, stride_w, pad, dilation=1,
                 transposed=False):
        if transposed:
            T_out = (T - 1) * stride_w - 2 * pad + K_w
        else:
            T_out = (T + 2 * pad - dilation * (K_w - 1) - 1) // stride_w + 1

        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        if transposed:
            # ConvTranspose uses dgrad
            self._gx = g.tensor(name='x', dim=[1, C_in, 1, T],
                                stride=nhwc_strides(1, C_in, 1, T),
                                data_type=cudnn.data_type.BFLOAT16)
            self._gw = g.tensor(name='w', dim=[C_in, C_out, 1, K_w],
                                stride=nhwc_strides(C_in, C_out, 1, K_w),
                                data_type=cudnn.data_type.BFLOAT16)
            self._go = g.tensor(name='out', dim=[1, C_out, 1, T_out],
                                stride=nhwc_strides(1, C_out, 1, T_out),
                                data_type=cudnn.data_type.BFLOAT16,
                                is_pass_by_value=False)
            self._go.set_output(True)

            g.conv_dgrad(image=self._go, weight=self._gw, loss=self._gx,
                         pre_padding=[0, pad], post_padding=[0, pad],
                         stride=[1, stride_w], dilation=[1, dilation],
                         compute_data_type=cudnn.data_type.FLOAT, name='conv_t')
        else:
            self._gx = g.tensor(name='x', dim=[1, C_in, 1, T],
                                stride=nhwc_strides(1, C_in, 1, T),
                                data_type=cudnn.data_type.BFLOAT16)
            self._gw = g.tensor(name='w', dim=[C_out, C_in, 1, K_w],
                                stride=nhwc_strides(C_out, C_in, 1, K_w),
                                data_type=cudnn.data_type.BFLOAT16)

            co = g.conv_fprop(image=self._gx, weight=self._gw,
                              pre_padding=[0, pad], post_padding=[0, pad],
                              stride=[1, stride_w], dilation=[1, dilation],
                              compute_data_type=cudnn.data_type.FLOAT, name='conv')
            self._gb = g.tensor(name='b', dim=[1, C_out, 1, 1],
                                stride=[C_out, 1, 1, 1],
                                data_type=cudnn.data_type.BFLOAT16)
            self._go = g.bias(name='bias', input=co, bias=self._gb)
            self._go.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

        self._transposed = transposed
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        ws_size = g.get_workspace_size()
        self._ws = torch.empty(max(ws_size, 1), device='cuda', dtype=torch.uint8)
        self._graph = g
        self.T_out = T_out

    def __call__(self, x, w, output, bias=None):
        if self._transposed:
            self._graph.execute(
                {self._gx: x, self._gw: w, self._go: output},
                self._ws
            )
        else:
            self._graph.execute(
                {self._gx: x, self._gw: w, self._gb: bias, self._go: output},
                self._ws
            )


def _extract_conv_params(mod):
    """Extract weight, bias from a Conv2d module."""
    w = mod.weight.data.to(memory_format=torch.channels_last)
    b = mod.bias.data.reshape(1, mod.out_channels, 1, 1) if mod.bias is not None else None
    return w, b


def _extract_snake_params(mod):
    """Extract alpha, inv_alpha from a Snake1d module."""
    if hasattr(mod, '_a4'):
        a = mod._a4.float().squeeze()
    else:
        a = mod.alpha.data.squeeze().float()
    ia = (1.0 / (a + 1e-9)).contiguous()
    return a.contiguous(), ia


def optimize_cudnn(model, x_sample):
    """Build cuDNN-fused forward pass and capture as CUDA graph.

    Replaces torch.compile with cuDNN v9 runtime fusion for conv+snake pairs
    and Triton kernels for standalone snake activations.

    Args:
        model: DACVAE model (on CUDA, eval mode)
        x_sample: Sample audio [1, 1, T] on CUDA

    Returns:
        replay: Function that replays the CUDA graph
    """
    from dacvae.optimize import (
        _fix_decoder_blocks, _strip_weight_norm, _convert_conv1d_to_conv2d,
        _make_deterministic_vae,
    )

    torch.backends.cudnn.benchmark = True

    # Apply standard optimizations
    _fix_decoder_blocks(model)
    _strip_weight_norm(model)
    _convert_conv1d_to_conv2d(model)
    _make_deterministic_vae(model)
    model = model.to(torch.bfloat16).eval()

    # Pre-pad input
    hop = model.hop_length
    T = x_sample.shape[-1]
    T_padded = T + (hop - T % hop) if T % hop else T
    x_typed = x_sample.clone().to(torch.bfloat16)
    if T % hop:
        x_typed = F.pad(x_typed, (0, hop - T % hop), mode="reflect")

    # === BUILD FUSED RESUNIT FUNCTION ===
    # For each ResidualUnit, build cuDNN fused graphs and Triton snake params

    fused_units = {}  # key: id(resunit) -> fused params
    fused_count = 0

    def _build_resunit(resunit, T_len):
        """Build fused ops for one ResidualUnit."""
        nonlocal fused_count
        C = resunit.block[1].in_channels if isinstance(resunit.block[1], nn.Conv2d) else None
        if C is None:
            return

        snake0 = resunit.block[0]
        conv7 = resunit.block[1]
        snake2 = resunit.block[2]
        conv1 = resunit.block[3]

        # Snake params
        a0, ia0 = _extract_snake_params(snake0)
        a2, ia2 = _extract_snake_params(snake2)

        # Conv params
        w7, b7 = _extract_conv_params(conv7)
        w1, b1 = _extract_conv_params(conv1)

        K_w = conv7.kernel_size[1]
        D = conv7.dilation[1]
        pad = D * (K_w - 1) // 2

        # Build cuDNN fused conv7+snake2
        try:
            g_cs = CudnnConvSnakeGraph(C, C, T_len, K_w, D, pad)
        except Exception as e:
            print(f"  WARN: conv_snake build failed C={C} T={T_len}: {e}")
            g_cs = None

        # Build cuDNN fused conv1+residual
        try:
            g_cr = CudnnConvResGraph(C, C, T_len)
        except Exception as e:
            print(f"  WARN: conv_res build failed: {e}")
            g_cr = None

        # Pre-allocate output buffers
        buf_cs = torch.empty(1, C, 1, T_len, device='cuda',
                             dtype=torch.bfloat16).to(memory_format=torch.channels_last)
        buf_cr = torch.empty(1, C, 1, T_len, device='cuda',
                             dtype=torch.bfloat16).to(memory_format=torch.channels_last)
        buf_sn = torch.empty(1, C, 1, T_len, device='cuda',
                             dtype=torch.bfloat16).to(memory_format=torch.channels_last)

        fused_units[id(resunit)] = {
            'a0': a0, 'ia0': ia0,       # snake0 params
            'w7': w7, 'b7': b7,          # conv7 params
            'a2': a2, 'ia2': ia2,        # snake2 params
            'w1': w1, 'b1': b1,          # conv1 params
            'g_cs': g_cs,                 # fused conv7+snake2 graph
            'g_cr': g_cr,                 # fused conv1+residual graph
            'buf_cs': buf_cs,             # output buffer for conv_snake
            'buf_cr': buf_cr,             # output buffer for conv_res
            'buf_sn': buf_sn,             # output buffer for snake
            'conv7': conv7,               # fallback
            'conv1': conv1,               # fallback
        }
        fused_count += 1

    # === TRACE AND BUILD ===
    print("Building cuDNN fused forward...")

    def _run_and_trace(mod, x):
        """Run module and build fused graphs along the way."""
        ctype = type(mod).__name__
        if ctype == 'ResidualUnit':
            _build_resunit(mod, x.shape[3])
            y = x
            for sub in mod.block:
                y = _run_and_trace(sub, y)
            if not mod.true_skip:
                return y + x
            return y + mod.shortcut(x, y)
        elif ctype == 'EncoderBlock':
            # Iterate block children to trace ResUnits and Snakes
            for sub in mod.block:
                x = _run_and_trace(sub, x)
            return x
        elif ctype == 'DecoderBlock' and hasattr(mod, '_precomputed_fwd'):
            for sub in mod._precomputed_fwd:
                x = _run_and_trace(sub, x)
            return x
        elif isinstance(mod, nn.Sequential):
            for sub in mod:
                x = _run_and_trace(sub, x)
            return x
        elif isinstance(mod, nn.Tanh):
            return torch.tanh(x)
        elif isinstance(mod, nn.Identity):
            return x
        else:
            return mod(x)

    # Extract standalone snake params
    snake_params = {}  # id(snake_mod) -> (alpha, inv_alpha, buf)

    def _register_snake(mod, shape):
        a, ia = _extract_snake_params(mod)
        buf = torch.empty(shape, device='cuda', dtype=torch.bfloat16).to(
            memory_format=torch.channels_last)
        snake_params[id(mod)] = (a, ia, buf)

    with torch.no_grad():
        z = x_typed.unsqueeze(2).to(memory_format=torch.channels_last)

        for i, layer in enumerate(model.encoder.block):
            ctype = type(layer).__name__
            if ctype == 'EncoderBlock':
                for j, sub in enumerate(layer.block):
                    if type(sub).__name__ == 'Snake1d':
                        _register_snake(sub, z.shape)
            elif type(layer).__name__ == 'Snake1d':
                _register_snake(layer, z.shape)
            z = _run_and_trace(layer, z)

        # Bottleneck
        combined = model.quantizer.in_proj(z)
        mean, scale = combined.chunk(2, dim=1)
        static_noise = torch.randn_like(mean)
        z_q = static_noise * (F.softplus(scale) + 1e-4) + mean
        emb = model.quantizer.out_proj(z_q)

        for i, layer in enumerate(model.decoder.model):
            ctype = type(layer).__name__
            if ctype == 'DecoderBlock' and hasattr(layer, '_precomputed_fwd'):
                for j, sub in enumerate(layer._precomputed_fwd):
                    if type(sub).__name__ == 'Snake1d':
                        _register_snake(sub, emb.shape)
                    elif type(sub).__name__ == 'ResidualUnit':
                        # Snake params inside ResUnit
                        for k, blk in enumerate(sub.block):
                            if type(blk).__name__ == 'Snake1d':
                                _register_snake(blk, emb.shape)
                    emb = _run_and_trace(sub, emb)
            else:
                emb = _run_and_trace(layer, emb)

    print(f"  Built {fused_count} fused ResUnits, {len(snake_params)} standalone snakes")

    # === FUSED FORWARD ===
    def _run_fused(mod, x):
        ctype = type(mod).__name__
        if ctype == 'ResidualUnit' and id(mod) in fused_units:
            fu = fused_units[id(mod)]
            residual = x

            # Snake0 (Triton)
            snake_forward(x, fu['a0'], fu['ia0'], fu['buf_sn'])

            # Conv7 + Snake2 (cuDNN fused)
            if fu['g_cs'] is not None:
                fu['g_cs'](fu['buf_sn'], fu['w7'], fu['b7'],
                           fu['a2'], fu['ia2'], fu['buf_cs'])
            else:
                fu['buf_cs'].copy_(fu['conv7'](fu['buf_sn']))
                snake_forward(fu['buf_cs'], fu['a2'], fu['ia2'], fu['buf_cs'])

            # Conv1 + residual (cuDNN fused)
            if fu['g_cr'] is not None:
                fu['g_cr'](fu['buf_cs'], fu['w1'], fu['b1'], residual, fu['buf_cr'])
            else:
                y = fu['conv1'](fu['buf_cs'])
                fu['buf_cr'].copy_(y + residual)

            return fu['buf_cr']
        elif ctype == 'ResidualUnit':
            y = x
            for sub in mod.block:
                y = _run_fused(sub, y)
            if not mod.true_skip:
                return y + x
            return y + mod.shortcut(x, y)
        elif ctype == 'DecoderBlock' and hasattr(mod, '_precomputed_fwd'):
            # Explicitly iterate _precomputed_fwd to use fused dispatch
            for sub in mod._precomputed_fwd:
                x = _run_fused(sub, x)
            return x
        elif ctype == 'Snake1d' and id(mod) in snake_params:
            a, ia, buf = snake_params[id(mod)]
            return snake_forward(x, a, ia, buf)
        elif isinstance(mod, nn.Sequential):
            for sub in mod:
                x = _run_fused(sub, x)
            return x
        elif isinstance(mod, nn.Tanh):
            return torch.tanh(x)
        elif isinstance(mod, nn.Identity):
            return x
        else:
            return mod(x)

    def forward_cudnn():
        x = x_typed.unsqueeze(2).to(memory_format=torch.channels_last)
        for layer in model.encoder.block:
            x = _run_fused(layer, x)
        combined = model.quantizer.in_proj(x)
        mean, scale = combined.chunk(2, dim=1)
        x = model.quantizer.out_proj(static_noise * (F.softplus(scale) + 1e-4) + mean)
        for layer in model.decoder.model:
            x = _run_fused(layer, x)
        return x.squeeze(2)

    # Validate
    with torch.no_grad():
        out = forward_cudnn()
        print(f"  Output: {out.shape}, valid: {not torch.isnan(out).any()}")

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            forward_cudnn()
        torch.cuda.synchronize()

    # CUDA graph capture
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_output = forward_cudnn()
        torch.cuda.synchronize()

        def replay():
            graph.replay()
            return static_output

        print("  CUDA graph captured!")
        return replay
    except Exception as e:
        print(f"  CUDA graph failed: {e}")

        def direct():
            with torch.no_grad():
                return forward_cudnn()
        return direct


if __name__ == "__main__":
    import sys
    import os
    # Fix profile shadow
    cwd = os.getcwd()
    for p in [cwd, '']:
        while p in sys.path:
            sys.path.remove(p)
    os.environ["HF_HOME"] = "/mnt/kadirnar/huggingface"

    from dacvae import DACVAE

    model = DACVAE().cuda().eval()
    x = torch.randn(1, 1, 4491264, device='cuda')

    replay = optimize_cudnn(model, x)

    # Benchmark
    with torch.no_grad():
        for _ in range(10):
            replay()
        torch.cuda.synchronize()

        times = []
        for _ in range(50):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

    times.sort()
    n = len(times)
    print(f"\nE2E cuDNN fused: p50={times[n//2]:.1f}ms  min={times[0]:.1f}ms  p90={times[n*9//10]:.1f}ms")
    print(f"vs compile baseline 224ms: {224/times[n//2]:.2f}x")
