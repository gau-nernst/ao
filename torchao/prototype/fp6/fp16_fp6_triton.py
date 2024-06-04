import torch
from torch import Tensor
import triton
import triton.language as tl
from triton import Config
from torchao.dtypes.float6_e3m2 import FLOAT6_E3M2_MAX, to_float6_e3m2


def get_cuda_autotune_config():
    return [
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ]

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


# from HQQ kernel
def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(
                            Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("c_ptr"),
                            )
                        )
    return configs


@triton.jit
def grouped_launch(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)

    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.autotune(configs=get_configs_io_bound(), key=["M", "N", "K"])
@triton.jit
def fp6_2_4_matmul_kernel(
    a_ptr,
    b_2bit_ptr,
    b_4bit_ptr,
    b_scale_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_TYPE: tl.constexpr = tl.float32,
    USE_BFLOAT16: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(K, BLOCK_K * SPLIT_K)

    pid_m, pid_n = grouped_launch(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    # offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # 2bit: load as uint8, 4 values in 1 uint8 -> divide N by 4
    # 4bit: load as uint8, 2 values in 1 uint8 -> divide N by 2
    # what if b is transposed? divide K instead
    b_2bit_offs = offs_k[:, None] * (N // 4) + (pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4))[None, :]
    b_2bit_ptrs = b_2bit_ptr + b_2bit_offs
    b_4bit_ptrs = b_4bit_ptr + (offs_k[:, None] * (N // 2) + (pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2))[None, :])
    # b_4bit_ptrs = b_4bit_ptr + b_2bit_offs

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for _ in range(0, grid_k):
        a = tl.load(a_ptrs)
        b_2bit = tl.load(b_2bit_ptrs)
        b_4bit = tl.load(b_4bit_ptrs)

        # b_2bit = b_2bit.to(tl.int32)
        # b_2bit_0 = (b_2bit & 0xc0) << 8
        # b_2bit_1 = (b_2bit & 0x30) << 10
        # b_2bit_2 = (b_2bit & 0x0c) << 12
        # b_2bit_3 = (b_2bit & 0x03) << 14
        # b_2bit = tl.interleave(tl.interleave(b_2bit_0, b_2bit_2), tl.interleave(b_2bit_1, b_2bit_3))

        # # unpack 2 4-bit into 2 16-bit
        # b_4bit = b_4bit.to(tl.int32)
        # b_4bit_0 = (b_4bit & 0xf0) << 8
        # b_4bit_1 = (b_4bit & 0x0f) << 12
        # b_4bit = tl.interleave(b_4bit_0, b_4bit_1)

        # if USE_BFLOAT16:
        #     #        sign bit     |    first exponent bit    | 2 exponent bits and 2 mantissa bits
        #     b = (b_2bit & 0x8000) | ((b_2bit & 0x4000) >> 5) | (b_4bit >> 7)
        #     b = b.to(tl.uint16).to(tl.bfloat16, bitcast=True)

        # else:
        #     #        sign bit     |    first exponent bit    | 2 exponent bits and 2 mantissa bits
        #     b = (b_2bit & 0x8000) | ((b_2bit & 0x4000) >> 2) | (b_4bit >> 4)
        #     b = b.to(tl.uint16).to(tl.float16, bitcast=True)

        # TODO: profile number of registers used
        # 4-way parallel
        # 0000 1111 2222 3333
        # ____ 0000 ____ 1111 ____ 2222 ____ 3333
        b_4bit_01, b_4bit_23 = b_4bit.reshape(BLOCK_K, BLOCK_N // 4, 2).split()
        b_4bit = ((b_4bit_01 & 0xf0) << 20) | ((b_4bit_01 & 0x0f) << 16) | ((b_4bit_23 & 0xf0) << 4) | (b_4bit_23 & 0x0f)

        # b_4bit = ((b_4bit & 0xf000) << 12) | ((b_4bit & 0x0f00) << 8) | ((b_4bit & 0x00f0) << 4) | (b_4bit & 0x000f)

        # 0011 2233
        # 00__ ____ 11__ ____ 22__ ____ 33__ ____
        # 0__0 ____ 1__1 ____ 2__2 ____ 3__3 ____
        b_2bit = ((b_2bit & 0xc0) << 24) | ((b_2bit & 0x30) << 18) | ((b_2bit & 0x0c) << 12) | ((b_2bit & 0x03) << 6)
        b_2bit = (b_2bit & 0x80808080) | ((b_2bit & 0x40404040) >> 2)

        # 0__0 0000 1__1 1111 2__2 2222 3__3 3333
        b = b_2bit | b_4bit

        b0 = ((b & 0xff000000) >> 16).to(tl.uint16)
        b1 = ((b & 0x00ff0000) >> 8).to(tl.uint16)
        b2 = (b & 0x0000ff00).to(tl.uint16)
        b3 = ((b & 0x000000ff) << 8).to(tl.uint16)
        b = tl.interleave(tl.interleave(b0, b2), tl.interleave(b1, b3))
        b = b.to(tl.float16, bitcast=True)

        acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)

        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_2bit_ptrs += BLOCK_K * SPLIT_K * (N // 4)
        b_4bit_ptrs += BLOCK_K * SPLIT_K * (N // 2)
        # b_4bit_ptrs += BLOCK_K * SPLIT_K * (N // 4)

    # rematerialize rm and rn to save registers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    b_scale = tl.load(b_scale_ptr + offs_n).to(ACC_TYPE)
    if USE_BFLOAT16:
        b_scale *= 2.0 ** (127 - 3)
    else:
        b_scale *= 2.0 ** (15 - 3)

    acc *= b_scale

    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.atomic_add(c_ptrs, acc, mask)


def fp6_2_4_matmul(A: torch.Tensor, B_2bit: torch.Tensor, B_4bit: torch.Tensor, B_scale: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    N = B_2bit.shape[1] * 4

    assert A.dtype in (torch.float16, torch.bfloat16)
    assert B_scale.dtype == A.dtype
    assert B_scale.numel() == N
    assert K % 64 == 0
    assert N % 64 == 0

    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META["SPLIT_K"])
    fp6_2_4_matmul_kernel[grid](
        A,
        B_2bit,
        B_4bit,
        B_scale,
        C,
        M, N, K,
        A.stride(0),
        A.stride(1),
        GROUP_M=8,
        USE_BFLOAT16=A.dtype is torch.bfloat16,
    )
    return C


def to_fp6_2_4(x: Tensor):
    scale = x.float().abs().amax(0) / FLOAT6_E3M2_MAX
    scale[scale == 0.0] = 1.0
    scaled_x = x / scale
    scale = scale.to(torch.half)
    x_6bit = to_float6_e3m2(scaled_x, no_bit_packing=True)

    x_2bit = (x_6bit >> 4) & 0b11
    x_4bit = x_6bit & 0b1111

    # packing
    x_2bit = (x_2bit[..., ::4] << 6) | (x_2bit[..., 1::4] << 4) | (x_2bit[..., 2::4] << 2) | x_2bit[..., 3::4]
    x_4bit = (x_4bit[..., ::2] << 4) | x_4bit[..., 1::2]
    return x_2bit, x_4bit, scale


@triton.jit
def fp6_packed_matmul_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_TYPE: tl.constexpr = tl.float32,
    USE_BFLOAT16: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(K, BLOCK_K * SPLIT_K)

    pid_m, pid_n = grouped_launch(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    # offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # 4 FP6 is packed into 3 uint8
    b_offs = offs_k[:, None] * (N // 4 * 3) + (pid_n * (BLOCK_N // 4 * 3))[None, :]
    b_ptrs = b_ptr + b_offs

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for _ in range(0, grid_k):
        a = tl.load(a_ptrs)
        
        # strided memory load, since we can't index data in shared memory
        b_byte0 = tl.load(b_ptrs + (tl.arange(0, BLOCK_N // 4)[None, :] * 3))
        b_byte1 = tl.load(b_ptrs + (tl.arange(0, BLOCK_N // 4)[None, :] * 3 + 1))
        b_byte2 = tl.load(b_ptrs + (tl.arange(0, BLOCK_N // 4)[None, :] * 3 + 2))

        b0 = b_byte0 >> 2
        b1 = ((b_byte0 & 0b11) << 4) | (b_byte1 >> 4)
        b2 = ((b_byte1 & 0b1111) << 2) | (b_byte2 >> 6)
        b3 = b_byte2 & 0b111111

        acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)

        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_K * SPLIT_K * (N // 4 * 3)

    # rematerialize rm and rn to save registers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    b_scale = tl.load(b_scale_ptr + offs_n).to(ACC_TYPE)
    if USE_BFLOAT16:
        b_scale *= 2.0 ** (127 - 3)
    else:
        b_scale *= 2.0 ** (15 - 3)

    acc *= b_scale

    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.atomic_add(c_ptrs, acc, mask)


def to_fp6_packed(x: Tensor):
    scale = x.float().abs().amax(0) / FLOAT6_E3M2_MAX
    scale[scale == 0.0] = 1.0
    scaled_x = x / scale
    scale = scale.to(torch.half)
    x_6bit = to_float6_e3m2(scaled_x, no_bit_packing=True)

    byte0 = (x_6bit[..., ::4] << 2) | (x_6bit[..., 1::4] >> 4)    # 0000 0011
    byte1 = (x_6bit[..., 1::4] << 4) | (x_6bit[..., 2::4] >> 2)   # 1111 2222
    byte2 = (x_6bit[..., 2::4] << 6) | x_6bit[..., 3::4]          # 2233 3333
    packed_x = torch.stack([byte0, byte1, byte2], dim=-1).flatten(-2)

    return packed_x, scale


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def float16_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_TYPE: tl.constexpr = tl.float32,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    A = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * N + rn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)

        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * N

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (N * idx_m)
    tl.store(c_ptr + (tl.broadcast_to(xindex, mask.shape)), acc, mask)


def float16_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    N = B.shape[1]

    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    float16_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
    )
    return C
