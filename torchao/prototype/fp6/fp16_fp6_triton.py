import torch
import triton
import triton.language as tl
from torchao.dtypes.float6_e3m2 import to_float6_e3m2, FLOAT6_E3M2_MAX


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4)
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def float16_float6_e3m2_matmul_kernel(
    a_ptr,
    b_2bit_ptr,
    b_4bit_ptr,
    c_ptr,
    scales_ptr,
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
    USE_BFLOAT16: tl.constexpr = False,
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

    # 2bit: load as uint8, 4 values in 1 uint8 -> divide N by 4
    # 4bit: load as uint8, 2 values in 1 uint8 -> divide N by 2
    # what if b is transposed? divide K instead
    B_2bit = b_2bit_ptr + rk[:, None] * (N // 4) + (pid_n * (BLOCK_N // 4)) + tl.arange(0, BLOCK_N // 4)
    B_4bit = b_4bit_ptr + rk[:, None] * (N // 2) + (pid_n * (BLOCK_N // 2)) + tl.arange(0, BLOCK_N // 2)

    scales = tl.load(
        scales_ptr + (tl.broadcast_to(rn[None, :], BLOCK_K, BLOCK_N)),
        eviction_policy="evict_last",
    )

    if USE_BFLOAT16:
        scales *= tl.full((1,), 2.0 ** (127 - 3), dtype=tl.bfloat16)
    else:
        scales *= tl.full((1,), 2.0 ** (15 - 3), dtype=tl.float16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)

        b_2bit = tl.load(B_2bit)
        b_4bit = tl.load(B_4bit)

        # unpack 4 2-bit into 4 16-bit
        b_2bit = b_2bit.to(tl.int32)
        b_2bit_0 = (b_2bit & 0xc0) << 8
        b_2bit_1 = (b_2bit & 0x30) << 10
        b_2bit_2 = (b_2bit & 0x0c) << 12
        b_2bit_3 = (b_2bit & 0x03) << 14
        b_2bit = tl.join(tl.join(b_2bit_0, b_2bit_2), tl.join(b_2bit_1, b_2bit_3)).reshape(BLOCK_K, BLOCK_N)

        # unpack 2 4-bit into 2 16-bit
        b_4bit = b_4bit.to(tl.int32)
        b_4bit_0 = (b_4bit & 0xf0) << 8
        b_4bit_1 = (b_4bit & 0x0f) << 12
        b_4bit = tl.join(b_4bit_0, b_4bit_1).reshape(BLOCK_K, BLOCK_N)

        if USE_BFLOAT16:
            #        sign bit     |    first exponent bit    | 2 exponent bits and 2 mantissa bits
            b = (b_2bit & 0x8000) | ((b_2bit & 0x4000) >> 5) | (b_4bit >> 7)
            b = b.to(tl.uint16).to(tl.bfloat16, bitcast=True)

        else:
            #        sign bit     |    first exponent bit    | 2 exponent bits and 2 mantissa bits
            b = (b_2bit & 0x8000) | ((b_2bit & 0x4000) >> 2) | (b_4bit >> 4)
            b = b.to(tl.uint16).to(tl.float16, bitcast=True)

        acc += tl.dot(a, b * scales)
        A += BLOCK_K * stride_ak

        B_2bit += BLOCK_K * (N // 4)
        B_4bit += BLOCK_K * (N // 2)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (N * idx_m)
    tl.store(c_ptr + (tl.broadcast_to(xindex, mask.shape)), acc, mask)


def float16_float6_e3m2_matmul(A: torch.Tensor, B_2bit: torch.Tensor, B_4bit: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    N = B_2bit.shape[1] * 4

    assert A.dtype in (torch.float16, torch.bfloat16)
    assert scales.dtype == A.dtype
    assert K % 64 == 0
    assert N % 64 == 0

    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    float16_float6_e3m2_matmul_kernel[grid](A, B_2bit, B_4bit, C, scales, M, N, K, A.stride(0), A.stride(1), IS_BFLOAT16=A.dtype is torch.bfloat16)
    return C


if __name__ == "__main__":
    M, N, K = 4, 1024, 1024
    dtype = torch.bfloat16
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    scales = B.float().abs().amax(0) / FLOAT6_E3M2_MAX
    scales[scales == 0.0] = 1.0
    B_scaled = B / scales
    scales = scales.to(dtype)
    B_6bit = to_float6_e3m2(B_scaled, no_bit_packing=True)

    B_2bit = (B_6bit >> 4) & 0b11
    B_4bit = B_6bit & 0b1111

    B_2bit = (B_2bit[..., ::4] << 6) | (B_2bit[..., 1::4] << 4) | (B_2bit[..., 2::4] << 2) | B_2bit[..., 3::4]
    B_4bit = (B_4bit[..., ::2] << 4) | B_4bit[..., 1::2]

    C = float16_float6_e3m2_matmul(A, B_2bit, B_4bit, scales)

    print(A @ B)
    print(C)
