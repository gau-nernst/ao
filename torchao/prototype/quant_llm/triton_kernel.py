import triton
import triton.language as tl


@triton.jit
def fp16_fp6_matmul(
    A,
    B_2bit,
    B_4bit,
    scales,
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
    TRANSPOSE_B: tl.constexpr = False,
):
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
    # rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
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
    B_2bit = B_2bit + (rk[:, None] * (N // 4) + (pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4))[None, :])
    B_4bit = B_4bit + (rk[:, None] * (N // 2) + (pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2))[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b_2bit = tl.load(B_2bit)
        b_4bit = tl.load(B_4bit)

        # TODO: dequant b_2bit and b_4bit to b

        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B_2bit += BLOCK_K * (N // 4)
        B_4bit += BLOCK_K * (N // 2)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    C = C + (idx_m * stride_cm + idx_n * stride_cn)

    tl.store(C, acc, mask)
