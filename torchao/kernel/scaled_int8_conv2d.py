import torch
from torch import Tensor
from torch.utils._triton import has_triton


lib = torch.library.Library("torchao", "FRAGMENT")
lib.define(
    "scaled_int8_conv2d(Tensor X, Tensor W, Tensor batch_scale, Tensor channel_scale, int[2] stride, int[2] padding) -> Tensor"
)


@torch.library.impl(lib, "scaled_int8_conv2d", "Meta")
def _(
    X: Tensor,
    W: Tensor,
    batch_scale: Tensor,
    channel_scale: Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> Tensor:
    BATCH, _, IN_H, IN_W = X.shape
    OUT_C, _, KERNEL_H, KERNEL_W = W.shape

    # refer to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for equation
    OUT_H = (IN_H + 2 * padding[0] - KERNEL_H) // stride[0] + 1
    OUT_W = (IN_W + 2 * padding[1] - KERNEL_W) // stride[1] + 1

    out = torch.empty(
        BATCH,
        OUT_C,
        OUT_H,
        OUT_W,
        device=X.device,
        dtype=channel_scale.dtype,
        memory_format=torch.channels_last,
    )
    return out


if has_triton():
    from .scaled_int8_conv2d_triton import scaled_int8_conv2d_triton

    torch.library.impl(lib, "scaled_int8_conv2d", "CUDA")(scaled_int8_conv2d_triton)


def scaled_int8_conv2d(
    X: Tensor,
    W: Tensor,
    batch_scale: Tensor,
    channel_scale: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> None:
    assert X.dtype == W.dtype == torch.int8
    assert X.is_contiguous(memory_format=torch.channels_last)
    assert W.is_contiguous(memory_format=torch.channels_last)
    assert batch_scale.is_contiguous()
    assert channel_scale.is_contiguous()
    out_channels, in_channels, _, _ = W.shape
    assert X.shape[1] == in_channels
    assert batch_scale.shape[0] == X.shape[0]
    assert channel_scale.shape[0] == out_channels

    return torch.ops.torchao.scaled_int8_conv2d(X, W, batch_scale, channel_scale, stride, padding)
