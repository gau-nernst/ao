import torch
from torch import Tensor
from torch.utils._triton import has_triton
from torchao.ops import to_float6_e3m2_packed_cpu, to_float6_e3m2_unpacked_cpu, from_float6_e3m2_packed_cpu, from_float6_e3m2_unpacked_cpu


# some useful constants
FLOAT6_E3M2_MAX = 28.0
FLOAT6_E3M2_SMALLEST_SUBNORMAL = 0.0625


if has_triton():
    import triton
    from triton import language as tl

    # see _to_float6_e3m2_pt() for explanation
    @triton.jit
    def _triton_float32_to_float6_e3m2(x: tl.tensor):
        x = x.to(tl.float32)
        x = x * 2.0 ** (-127 + 3)
        bits = x.to(tl.int32, bitcast=True)

        sign = ((bits >> 31) & 0x1) << 5
        exp_and_man = (bits >> 21) & 0x1F
        result = sign | exp_and_man

        remainder = bits & 0x1F_FFFF
        do_round_up = (remainder > 0x10_0000) | ((remainder == 0x10_0000) & ((result & 1) == 1))
        result = tl.where(do_round_up, result + 1, result)
        return result.to(tl.uint8)

    @triton.jit
    def _to_float6_e3m2_triton_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        # strided memory read. there will be uncoalesced memory access
        val0 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4, mask))
        val1 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4 + 1, mask))
        val2 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4 + 2, mask))
        val3 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4 + 3, mask))

        # bit packing
        bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
        bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
        bits2 = (val2 << 6) | (val3);      # 2233 3333

        # strided memory write. there will be uncoalesced memory access
        tl.store(out_ptr + offsets * 3, bits0, mask)
        tl.store(out_ptr + offsets * 3 + 1, bits1, mask)
        tl.store(out_ptr + offsets * 3 + 2, bits2, mask)

    def _to_float6_e3m2_triton(tensor: Tensor) -> Tensor:
        out_shape = tensor.shape[:-1] + (tensor.shape[-1] // 4 * 3,)
        output = torch.empty(out_shape, device=tensor.device, dtype=torch.uint8)

        n = tensor.numel()
        grid_size = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"] * 4),)
        _to_float6_e3m2_triton_kernel[grid_size](tensor, output, n, BLOCK_SIZE=256)

        return output

else:
    _to_float6_e3m2_triton = None


# NOTE: This implementation requires FP32 denormal numbers to be handled correctly.
# On CPU, denormal numbers might be flushed to zero for performance gain (FTZ and DAZ flags).
def _to_float6_e3m2_pt(tensor: Tensor, no_bit_packing: bool = False) -> Tensor:
    tensor = tensor.float()

    # correct exponent bias. this also handles subnormal numbers correctly
    tensor = tensor * 2.0 ** (-127 + 3)
    bits = tensor.view(torch.int32)

    sign = ((bits >> 31) & 0x1) << 5
    exp_and_man = (bits >> 21) & 0x1F
    result = sign | exp_and_man

    # round to nearest even
    remainder = bits & 0x1F_FFFF  # truncated mantissa bits
    do_round_up = (remainder > 0x10_0000) | ((remainder == 0x10_0000) & ((result & 1) == 1))
    result = torch.where(do_round_up, result + 1, result)
    result = result.to(torch.uint8)

    if no_bit_packing:
        return result

    # bit packing
    val0, val1, val2, val3 = result.unflatten(-1, (-1, 4)).unbind(-1)
    bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
    bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
    bits2 = (val2 << 6) | (val3);      # 2233 3333
    return torch.stack([bits0, bits1, bits2], dim=-1).flatten(-2)


def to_float6_e3m2(tensor: Tensor, no_bit_packing: bool = False) -> Tensor:
    """Convert input tensor to FP6. This particular FP6 format has 3 exponent bits and 2 mantissa
    bits. By default, bit packing is performed: every 4 FP6 values are packed as 3 uint8 values
    (4 x 6 bits = 3 x 8 bits).

    Args:
      tensor: Input tensor. The last dimension must be divisible by 4 (unless ``no_bit_packing=False``)
      no_bit_packing: Whether to not perform bit packing. Setting this to ``True`` can be useful for
        observing the bit patterns and debugging.

    Returns:
      :class:`torch.Tensor`: FP6 tensor, stored as uint8 data. If ``no_bit_packing=False``, the last
      dimension of output tensor is 3/4 of that of input tensor.

    Note:
      This FP6 format does not represent +/-inf and NaN. Thus, make sure that input tensor does
      not have +/-inf or NaN values, and no values with magnitude >= 30 (largest number in FP6 is 28.
      All numbers >= 28 and < 30 will be rounded down to 28, while >= 30 will overflow).

      See also :func:`from_float6_e3m2`
    """
    if not no_bit_packing:
        assert tensor.shape[-1] % 4 == 0, "Last dim must be divisible by 4"

    if tensor.is_cpu:
      if no_bit_packing:
        return to_float6_e3m2_unpacked_cpu(tensor)
      
      *leading_dims, last_dim = tensor.shape
      return to_float6_e3m2_packed_cpu(tensor.view(-1, last_dim)).view(*leading_dims, -1)

    # torch.compile() cannot generate fused bit-packing triton kernel,
    # thus we write custom triton kernel for this specific case.
    if tensor.is_cuda and not no_bit_packing and _to_float6_e3m2_triton is not None:
        return _to_float6_e3m2_triton(tensor)

    else:
        return _to_float6_e3m2_pt(tensor, no_bit_packing=no_bit_packing)


# NOTE: This implementation requires FP32 denormal numbers to be handled correctly.
# On CPU, denormal numbers might be flushed to zero for performance gain (FTZ and DAZ flags).
def _pt_float6_e3m2_to_float32(tensor: Tensor) -> Tensor:
    bits = tensor.to(torch.int32)  # bit extension
    sign = bits >> 5 << 31
    exp_and_man = (bits & 0x1F) << 21
    results = sign | exp_and_man

    results = results.view(torch.float32)
    return results * 2.0 ** (127 - 3)  # exponent bias correction


def from_float6_e3m2(tensor: Tensor, no_bit_packing: bool = False, dtype: torch.dtype = torch.float32) -> Tensor:
    """Convert an FP6 tensor (created by :func:`to_float6_e3m2`) to FP32.

    Args:
      tensor: FP6 tensor, stored as uint8 data. If ``no_bit_packing=False``, the last dimension must
        be divisible by 3.
      no_bit_packing: whether the input does not have bit packing.
      dtype: returned dtype.

    Returns:
      :class:`torch.Tensor`: FP32 tensor. If ``no_bit_packing=False``, the last dimension of output
      tensor is 4/3 of that of input tensor.
    """
    assert tensor.dtype == torch.uint8
    if no_bit_packing:
        if tensor.is_cpu:
          return from_float6_e3m2_unpacked_cpu(tensor, dtype)

        return _pt_float6_e3m2_to_float32(tensor).to(dtype)

    assert tensor.shape[-1] % 3 == 0, "Last dim must be divisible by 3"
    if tensor.is_cpu:
        return from_float6_e3m2_packed_cpu(tensor, dtype)

    bits0, bits1, bits2 = tensor.unflatten(-1, (-1, 3)).unbind(-1)
    val0 = _pt_float6_e3m2_to_float32(bits0 >> 2).to(dtype)
    val1 = _pt_float6_e3m2_to_float32(((bits0 & 0x3) << 4) | (bits1 >> 4)).to(dtype)
    val2 = _pt_float6_e3m2_to_float32(((bits1 & 0xF) << 2) | (bits2 >> 6)).to(dtype)
    val3 = _pt_float6_e3m2_to_float32(bits2 & 0x3F).to(dtype)
    return torch.stack([val0, val1, val2, val3], dim=-1).flatten(-2)


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4)
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
    IS_BFLOAT16: tl.constexpr = False,
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

    if IS_BFLOAT16:
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

        if IS_BFLOAT16:
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
