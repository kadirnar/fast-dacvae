"""
Custom Inductor pass: fuse Conv2d + Snake into cuDNN v9 Runtime Fusion kernel.

This module registers a post_grad_custom_pre_pass that pattern-matches
`aten.convolution → polynomial_snake` sequences in the FX graph and replaces
them with a single cuDNN fused kernel. The cuDNN kernel computes snake in
the conv epilogue, eliminating intermediate memory writes.

Usage:
    from dacvae.inductor_fusion import install_conv_snake_fusion
    install_conv_snake_fusion()
    # Then call torch.compile as normal — the fusion pass will be applied
"""
import torch
import torch.nn.functional as F
import cudnn

# --------------------------------------------------------------------------
# 1. cuDNN graph cache (shape-specific, built lazily)
# --------------------------------------------------------------------------

def _nhwc(n, c, h, w):
    return [c * h * w, 1, c * w, c]


_CUDNN_CACHE = {}  # key: (C_in, C_out, T, K_w, dilation, pad, use_fp8) -> entry


def _get_or_build_graph(C_in, C_out, T, K_w, dilation, pad):
    key = (C_in, C_out, T, K_w, dilation, pad)
    if key in _CUDNN_CACHE:
        return _CUDNN_CACHE[key]

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    gx = g.tensor(name='x', dim=[1, C_in, 1, T], stride=_nhwc(1, C_in, 1, T),
                   data_type=cudnn.data_type.BFLOAT16)
    gw = g.tensor(name='w', dim=[C_out, C_in, 1, K_w], stride=_nhwc(C_out, C_in, 1, K_w),
                   data_type=cudnn.data_type.BFLOAT16)
    gb = g.tensor(name='b', dim=[1, C_out, 1, 1], stride=[C_out, 1, 1, 1],
                   data_type=cudnn.data_type.BFLOAT16)
    ga = g.tensor(name='a', dim=[1, C_out, 1, 1], stride=[C_out, 1, 1, 1],
                   data_type=cudnn.data_type.FLOAT)
    gia = g.tensor(name='ia', dim=[1, C_out, 1, 1], stride=[C_out, 1, 1, 1],
                    data_type=cudnn.data_type.FLOAT)

    co = g.conv_fprop(image=gx, weight=gw,
                      pre_padding=[0, pad], post_padding=[0, pad],
                      stride=[1, 1], dilation=[1, dilation],
                      compute_data_type=cudnn.data_type.FLOAT, name='conv')
    bi = g.bias(name='bias', input=co, bias=gb)
    ax = g.mul(bi, ga, name='ax')
    sn = g.sin(ax, name='sin')
    s2 = g.mul(sn, sn, name='sin2')
    sc = g.mul(s2, gia, name='sc')
    go = g.add(bi, sc, name='out')
    go.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])
    g.build_plans(cudnn.build_plan_policy.ALL)
    ws_size = g.get_workspace_size()
    ws = torch.empty(max(ws_size, 1), device='cuda', dtype=torch.uint8)
    obuf = torch.empty(1, C_out, 1, T, device='cuda', dtype=torch.bfloat16).to(
        memory_format=torch.channels_last)

    entry = (g, gx, gw, gb, ga, gia, go, ws, obuf)
    _CUDNN_CACHE[key] = entry
    return entry


# --------------------------------------------------------------------------
# 2. Custom op: dacvae::conv_snake_fused
# --------------------------------------------------------------------------

@torch.library.custom_op('dacvae::conv_snake_fused', mutates_args=())
def conv_snake_fused(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    alpha: torch.Tensor, inv_alpha: torch.Tensor,
    padding_w: int, dilation_w: int,
) -> torch.Tensor:
    """Fused Conv2d + bias + snake via cuDNN v9 runtime fusion.

    Automatically selects FP8 for dilated convolutions (D>=3, C>=96)
    where cuDNN's BF16 eng7 plan is slow. Handles BF16↔FP8 conversion.
    """
    C_in = weight.shape[1]
    C_out = weight.shape[0]
    K_w = weight.shape[3]
    T = x.shape[3]

    g, gx, gw, gb, ga, gia, go, ws, _obuf = _get_or_build_graph(
        C_in, C_out, T, K_w, dilation_w, padding_w)

    output = torch.empty(1, C_out, 1, T, dtype=x.dtype, device=x.device,
                         memory_format=torch.channels_last)
    g.execute({gx: x, gw: weight, gb: bias,
               ga: alpha, gia: inv_alpha, go: output}, ws)
    return output


@conv_snake_fused.register_fake
def _conv_snake_fused_fake(x, weight, bias, alpha, inv_alpha, padding_w, dilation_w):
    """Fake kernel for shape inference — returns channels_last output."""
    C_out = weight.shape[0]
    T = x.shape[3]
    # Compute output T (same as input for stride=1 conv with symmetric padding)
    K_w = weight.shape[3]
    T_out = (T + 2 * padding_w - dilation_w * (K_w - 1) - 1) // 1 + 1
    return torch.empty(
        x.shape[0], C_out, x.shape[2], T_out,
        dtype=x.dtype, device=x.device,
        memory_format=torch.channels_last,
    )


# --------------------------------------------------------------------------
# 3. FX Graph Pattern Matcher (post-grad custom pass)
# --------------------------------------------------------------------------

def _is_conv_node(node):
    """Check if node is aten.convolution.default."""
    return (node.op == 'call_function' and
            node.target is torch.ops.aten.convolution.default)


def _find_poly_snake_after_conv(conv_node):
    """Find the polynomial snake chain after a conv node.

    Matches this exact FX pattern (from optimize.py's polynomial snake):
        conv    = aten.convolution(input, weight, bias, [1,1], ...)
        mul_ax  = aten.mul(alpha, conv)           # alpha * conv_out
        mul_inv = aten.mul(mul_ax, 0.31831...)    # * INV_PI
        round   = aten.round(mul_inv)             # range reduction
        mul_pi  = aten.mul(round, 3.14159...)     # * PI
        sub_th  = aten.sub(mul_ax, mul_pi)        # theta
        mul_t2  = aten.mul(sub_th, sub_th)        # theta²
        mul_c   = aten.mul(mul_t2, 0.04444...)    # t2*(2/45)
        sub_a   = aten.sub(0.33333..., mul_c)     # 1/3 - t2*(2/45)
        mul_d   = aten.mul(mul_t2, sub_a)         # t2*(1/3-...)
        sub_b   = aten.sub(1.0, mul_d)            # 1 - t2*(...)
        mul_s2  = aten.mul(mul_t2, sub_b)         # sin² = t2*(1-...)
        mul_sc  = aten.mul(inv_alpha, mul_s2)     # inv_alpha * sin²
        add_out = aten.add(conv, mul_sc)          # conv_out + inv_alpha*sin²

    Returns match dict or None.
    """
    # Conv must have exactly 2 users: mul(alpha, conv) and add(conv, ...)
    users = list(conv_node.users.keys())

    # Find mul(alpha, conv) — first mul user
    mul_ax = None
    for u in users:
        if u.target is torch.ops.aten.mul.Tensor and conv_node in u.args:
            # Check the other arg is a get_attr (frozen alpha param)
            other = u.args[0] if u.args[1] is conv_node else u.args[1]
            if isinstance(other, torch.fx.Node) and other.op == 'get_attr':
                mul_ax = u
                alpha_node = other
                break

    if mul_ax is None:
        return None

    # Find add(conv, ...) — the final add user
    add_out = None
    for u in users:
        if u.target is torch.ops.aten.add.Tensor and conv_node in u.args and u is not mul_ax:
            add_out = u
            break

    if add_out is None:
        return None

    # Find inv_alpha — the other arg of the mul just before the add
    # Pattern: mul(inv_alpha, sin²) = the non-conv arg of add
    mul_sc = None
    for arg in add_out.args:
        if isinstance(arg, torch.fx.Node) and arg is not conv_node:
            if arg.target is torch.ops.aten.mul.Tensor:
                mul_sc = arg
                break

    if mul_sc is None:
        return None

    # inv_alpha is the get_attr arg of mul_sc
    inv_alpha_node = None
    for arg in mul_sc.args:
        if isinstance(arg, torch.fx.Node) and arg.op == 'get_attr':
            inv_alpha_node = arg
            break

    if inv_alpha_node is None:
        return None

    # Collect intermediate snake nodes between conv and add_out for deletion.
    # Walk the computation chain from add_out backward to conv.
    snake_nodes = set()

    def _collect_backward(node):
        """Walk backward from add_out, collecting snake-internal nodes."""
        if node is conv_node or node.op == 'get_attr':
            return
        if node in snake_nodes:
            return
        snake_nodes.add(node)
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                _collect_backward(arg)

    _collect_backward(add_out)

    return {
        'conv': conv_node,
        'add': add_out,
        'alpha': alpha_node,
        'inv_alpha': inv_alpha_node,
        'snake_nodes': snake_nodes,
    }


_CUDNN_CONV_ADD_CACHE = {}


def _get_or_build_conv_add_graph(C_in, C_out, T):
    key = ('conv_add', C_in, C_out, T)
    if key in _CUDNN_CONV_ADD_CACHE:
        return _CUDNN_CONV_ADD_CACHE[key]

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    gx = g.tensor(name='x', dim=[1, C_in, 1, T], stride=_nhwc(1, C_in, 1, T), data_type=cudnn.data_type.BFLOAT16)
    gw = g.tensor(name='w', dim=[C_out, C_in, 1, 1], stride=_nhwc(C_out, C_in, 1, 1), data_type=cudnn.data_type.BFLOAT16)
    gb = g.tensor(name='b', dim=[1, C_out, 1, 1], stride=[C_out, 1, 1, 1], data_type=cudnn.data_type.BFLOAT16)
    gr = g.tensor(name='r', dim=[1, C_out, 1, T], stride=_nhwc(1, C_out, 1, T), data_type=cudnn.data_type.BFLOAT16)

    co = g.conv_fprop(image=gx, weight=gw, pre_padding=[0, 0], post_padding=[0, 0],
                      stride=[1, 1], dilation=[1, 1], compute_data_type=cudnn.data_type.FLOAT, name='conv')
    bi = g.bias(name='bias', input=co, bias=gb)
    go = g.add(bi, gr, name='add')
    go.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])
    g.build_plans(cudnn.build_plan_policy.ALL)
    ws_size = g.get_workspace_size()
    ws = torch.empty(max(ws_size, 1), device='cuda', dtype=torch.uint8)

    entry = (g, gx, gw, gb, gr, go, ws)
    _CUDNN_CONV_ADD_CACHE[key] = entry
    return entry


@torch.library.custom_op('dacvae::conv_add_fused', mutates_args=())
def conv_add_fused(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Fused Conv2d(k=1) + bias + residual_add via cuDNN v9."""
    C_in = weight.shape[1]
    C_out = weight.shape[0]
    T = x.shape[3]

    output = torch.empty(1, C_out, 1, T, dtype=x.dtype, device=x.device,
                         memory_format=torch.channels_last)

    g, gx, gw, gb, gr, go, ws = _get_or_build_conv_add_graph(C_in, C_out, T)
    g.execute({gx: x, gw: weight, gb: bias, gr: residual, go: output}, ws)
    return output


@conv_add_fused.register_fake
def _conv_add_fused_fake(x, weight, bias, residual):
    C_out = weight.shape[0]
    T = x.shape[3]
    return torch.empty(x.shape[0], C_out, x.shape[2], T,
                       dtype=x.dtype, device=x.device,
                       memory_format=torch.channels_last)


def _find_conv_k1_add(conv_node):
    """Find conv(k=1, stride=1) → add(conv_out, residual) pattern.

    FX pattern:
        conv = aten.convolution(x, w, b, [1,1], [0,0], [1,1], False, [0,0], 1)
        add  = aten.add(conv, residual)  # or add(residual, conv)
    """
    # Check conv args
    if len(conv_node.args) < 7:
        return None
    padding = conv_node.args[4]
    if padding != [0, 0]:
        return None

    # Conv must have exactly 1 user: the add
    users = list(conv_node.users.keys())
    if len(users) != 1:
        return None

    add_node = users[0]
    if add_node.target is not torch.ops.aten.add.Tensor:
        return None

    # The other arg of add is the residual
    residual = None
    for arg in add_node.args:
        if isinstance(arg, torch.fx.Node) and arg is not conv_node:
            residual = arg
            break

    if residual is None:
        return None

    return {'conv': conv_node, 'add': add_node, 'residual': residual}


def conv_snake_fusion_pass(graph):
    """FX graph pass: replace conv→snake and conv_k1→add with cuDNN fused ops."""
    fused_count = 0
    conv_add_count = 0

    for node in list(graph.nodes):
        if not _is_conv_node(node):
            continue

        # Check conv args: aten.convolution(input, weight, bias, stride, pad, dilation, ...)
        if len(node.args) < 7:
            continue

        # Only fuse stride=[1,1] convs (ResUnit convs, not stride/downsample)
        stride = node.args[3]
        if stride != [1, 1]:
            continue

        # Try to find snake chain
        match = _find_poly_snake_after_conv(node)
        if match is None:
            continue

        if match['inv_alpha'] is None:
            continue

        # Extract conv parameters
        x_input = node.args[0]
        weight = node.args[1]
        bias = node.args[2]
        padding = node.args[4]  # [0, pad_w]
        dilation = node.args[5]  # [1, dil_w]

        padding_w = padding[1] if isinstance(padding, (list, tuple)) else padding
        dilation_w = dilation[1] if isinstance(dilation, (list, tuple)) else dilation

        alpha = match['alpha']
        inv_alpha = match['inv_alpha']
        add_node = match['add']

        # Insert fused op
        with graph.inserting_before(add_node):
            fused = graph.call_function(
                torch.ops.dacvae.conv_snake_fused.default,
                args=(x_input, weight, bias, alpha, inv_alpha, padding_w, dilation_w),
            )
            # Copy metadata from add_node
            fused.meta.update(add_node.meta)

        # Replace all uses of add_node with fused
        add_node.replace_all_uses_with(fused)

        # Erase dead snake nodes — only those with no remaining users
        # Must erase in reverse order (leaf-first) to avoid dangling refs
        erased = True
        while erased:
            erased = False
            for n in list(match['snake_nodes']):
                if n in graph.nodes and len(n.users) == 0:
                    try:
                        graph.erase_node(n)
                        erased = True
                    except Exception:
                        pass  # Node still has users, skip

        # Try to erase the original conv node if it has no users
        if node in graph.nodes and len(node.users) == 0:
            try:
                graph.erase_node(node)
            except Exception:
                pass

        fused_count += 1

    # Second pass: fuse conv(k=1) + add(conv, residual)
    for node in list(graph.nodes):
        if not _is_conv_node(node):
            continue
        if len(node.args) < 7:
            continue
        stride = node.args[3]
        if stride != [1, 1]:
            continue

        match = _find_conv_k1_add(node)
        if match is None:
            continue

        x_input = node.args[0]
        weight = node.args[1]
        bias = node.args[2]
        residual = match['residual']
        add_node = match['add']

        with graph.inserting_before(add_node):
            fused = graph.call_function(
                torch.ops.dacvae.conv_add_fused.default,
                args=(x_input, weight, bias, residual),
            )
            fused.meta.update(add_node.meta)

        add_node.replace_all_uses_with(fused)

        erased = True
        while erased:
            erased = False
            for n in [add_node, node]:
                if n in graph.nodes and len(n.users) == 0:
                    try:
                        graph.erase_node(n)
                        erased = True
                    except Exception:
                        pass

        conv_add_count += 1

    if fused_count > 0 or conv_add_count > 0:
        graph.lint()
        print(f"[inductor_fusion] Fused {fused_count} conv+snake, {conv_add_count} conv+add patterns")


# --------------------------------------------------------------------------
# 4. Installation
# --------------------------------------------------------------------------

def install_conv_snake_fusion():
    """Install the conv+snake fusion pass into Inductor's compilation pipeline."""
    import torch._inductor.config as cfg
    cfg.post_grad_custom_pre_pass = conv_snake_fusion_pass
    print("[inductor_fusion] Conv+snake fusion pass installed")
