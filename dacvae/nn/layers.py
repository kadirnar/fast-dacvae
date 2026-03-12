# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


def activation(act: str, **act_params):
    if act == "ELU":
        return nn.ELU(**act_params)
    elif act == "Snake":
        return Snake1d(**act_params)
    elif act == "Tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {act}")


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
    assert norm in ["none", "weight_norm"]
    if norm == "weight_norm":
        return weight_norm(module)
    else:
        return module

class NormConv1d(nn.Conv1d):
    """1D Convolution with normalization and optional causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",
        causal: bool = False,
        pad_mode: str = "none",
        **kwargs
    ):
        if pad_mode == "none":
            pad = (kernel_size - stride) * dilation // 2
        else:
            pad = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            **kwargs
        )

        apply_parametrization_norm(self, norm)

        self.causal = causal
        self.pad_mode = pad_mode

    def pad(self, x: torch.Tensor):
        if self.pad_mode == "none":
            return x

        length = x.shape[-1]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        dilation = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = (effective_kernel_size - stride)
        n_frames = (length - effective_kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        extra_padding = ideal_length - length

        if self.causal:
            pad_x = F.pad(x, (padding_total, extra_padding))
        else:
            padding_right = extra_padding // 2
            padding_left = padding_total - padding_right
            pad_x = F.pad(x, (padding_left, padding_right + extra_padding))

        return pad_x

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        return super().forward(x)

class NormConvTranspose1d(nn.ConvTranspose1d):
    """1D Transposed Convolution with normalization and optional causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",
        causal: bool = False,
        pad_mode: str = "none",
        **kwargs
    ):
        if pad_mode == "none":
            padding = (stride + 1) // 2
            output_padding = 1 if stride % 2 else 0
        else:
            padding = 0
            output_padding = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            output_padding=output_padding,
            **kwargs
        )

        self = apply_parametrization_norm(self, norm)
        self.causal = causal
        self.pad_mode = pad_mode

    def unpad(self, x: torch.Tensor):
        if self.pad_mode == "none":
            return x
        length = x.shape[-1]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]

        padding_total = kernel_size - stride
        if self.causal:
            padding_left = 0
            end = length - padding_total
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            end = length - padding_right

        x = x[..., padding_left:end]
        return x

    def forward(self, x):
        y = super().forward(x)
        return self.unpad(y)


class MsgProcessor(torch.nn.Module):
    """Apply the secret message to the encoder output."""

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        indices = 2 * torch.arange(msg.shape[-1]).to(hidden.device)
        indices = indices.repeat(msg.shape[0], 1)
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)
        msg_aux = msg_aux.sum(dim=-2)
        msg_aux = msg_aux.unsqueeze(-1).repeat(1, 1, hidden.shape[2])
        hidden = hidden + msg_aux
        return hidden
