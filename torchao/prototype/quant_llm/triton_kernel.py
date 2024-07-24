import itertools

import torch
from torch import Tensor
import triton
import triton.language as tl

from torchao.prototype.quant_llm.quant_llm import to_scaled_tc_fpx, _pack
from torchao.ops import quant_llm_linear
from torchao.prototype.custom_fp_utils import _f32_to_fpx_unpacked


def get_configs_io_bound():
    configs = []
    for num_stages, BLOCK_M, BLOCK_N, BLOCK_K in itertools.product([2, 3, 4, 5, 6], [16, 32], [32, 64, 128, 256], [32, 64]):
        num_warps = 2 if BLOCK_N <= 64 else 4
        config = triton.Config(
                {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
                num_warps=num_warps,
                num_stages=num_stages,
            )
        configs.append(config)
    return configs


def get_configs_compute_bound():
    configs = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=5, num_warps=2),
    ]
    return configs


@triton.autotune(
    configs=get_configs_io_bound() + get_configs_compute_bound(),
    key=['M', 'N', 'K'],
)
@triton.jit
def fp16_fp6_e3m2_linear_kernel(
    A,
    B_2bit,
    B_4bit,
    B_scale,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M : tl.constexpr = 8,
    ACC_TYPE : tl.constexpr = tl.float32,
):
    """
    A is shape (M, K)
    B is shape (N, K)
    B is packed along 2nd dim, with per-row scale
    C = A @ B.T with shape(M, N)
    """
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    # if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
    #     rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    # else:
    #     rbn = rn % N

    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)

    B_2bit = B_2bit + (tl.arange(0, BLOCK_K // 4)[:, None] + rn[None, :] * (K // 4))
    B_4bit = B_4bit + (tl.arange(0, BLOCK_K // 2)[:, None] + rn[None, :] * (K // 2))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b_2bit = tl.load(B_2bit)  # shape (BLOCK_K / 4, BLOCK_N)
        b_4bit = tl.load(B_4bit)  # shape (BLOCK_K / 2, BLOCK_N)

        # FP16 spec:     S EEEEE MMMMMMMMMM
        # FP6 E3M2 spec: S   EEE MM

        # calling position 0 is the right-most position or LSB position
        # we shift each 2bit to position 12-13
        # first 2bit shift from 6-7 to 12-13
        # second 2bit shift from 4-5 to 12-13, and so on
        b_2bit = (b_2bit[:, None, :] << (6 + tl.arange(0, 4) * 2)[None, :, None]) & 0x3000
        b_2bit = b_2bit.reshape(BLOCK_K, BLOCK_N)

        # we shift each 4bit to position 8-11
        # first 4bit shift from 4-7 to 8-11
        # second 4bit shift from 0-3 to 8-11
        b_4bit = (b_4bit[:, None, :] << (4 + tl.arange(0, 2) * 4)[None, :, None]) & 0x0F00
        b_4bit = b_4bit.reshape(BLOCK_K, BLOCK_N)

        b = b_2bit | b_4bit
        b = ((b << 2) & 0x8000) | (b & 0x1F00)  # shift sign bit to position 15
        b = b.to(tl.uint16).to(tl.float16, bitcast=True)

        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B_2bit += BLOCK_K // 4
        B_4bit += BLOCK_K // 2

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]

    scale = tl.load(B_scale + rn[None, :])
    scale *= 2 ** (15 - 3)  # difference in exponent bias
    acc = acc * scale

    mask = (idx_m < M) & (idx_n < N)
    C = C + (idx_m * stride_cm + idx_n * stride_cn)

    tl.store(C, acc, mask)


def fp16_fp6_e3m2_linear(A: Tensor, B_2bit: Tensor, B_4bit: Tensor, B_scale: Tensor):
    M, K = A.shape
    N = B_scale.shape[0]

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fp16_fp6_e3m2_linear_kernel[grid](
        A,
        B_2bit,
        B_4bit,
        B_scale,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        C.stride(0),
        C.stride(1),
    )
    return C


if __name__ == "__main__":
    M = N = K = 4096
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")

    ref = A @ B.T

    B_fp6, B_scale = to_scaled_tc_fpx(B, 3, 2)
    fp6_llm = quant_llm_linear(3, 2, A, B_fp6, B_scale)

    B_scaled = B.float() / B_scale.clip(1e-12)[:, None]
    B_6bit = _f32_to_fpx_unpacked(B_scaled, 3, 2)
    B_2bit = _pack(B_6bit & 0x30, 2)
    B_4bit = _pack(B_6bit & 0x0F, 4)

    out = fp16_fp6_e3m2_linear(A, B_2bit, B_4bit, B_scale)

    print(ref)
    print(fp6_llm)
    print(out)
