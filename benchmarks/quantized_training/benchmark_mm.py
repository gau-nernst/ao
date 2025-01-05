import pandas as pd
import torch
from triton.testing import do_bench

from torchao.prototype.quantized_training.kernel import scaled_mm


def bench_f(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")


@torch.compile(mode="max-autotune", dynamic=False)
def scaled_int8_mm_inductor(A: torch.Tensor, B: torch.Tensor, scale_A: torch.Tensor, scale_B: torch.Tensor):
    return torch._int_mm(A, B) * scale_B * scale_A


torch._dynamo.config.cache_size_limit = 1000
torch._inductor.config.force_fuse_int_mm_with_mul = True


shapes = [(sz, sz, sz) for sz in [1024, 2048, 4096]]

# Llama-8B shapes
shapes += [
    # linear in attention
    (32_768, 4096, 4096),
    (4096, 4096, 32_768),
    # linear in feed-forward
    (32_768, 14_336, 4096),
    (32_768, 4096, 14_336),
    (14_336, 4096, 32_768),
]

data = []
for M, N, K in shapes:
    print(f"{M=}, {N=}, {K=}")

    A_bf16 = torch.randn(M, K).bfloat16().cuda()
    B_bf16 = torch.randn(N, K).bfloat16().cuda()
    A_i8 = torch.randint(-128, 127, size=(M, K), dtype=torch.int8).cuda()
    B_i8 = torch.randint(-128, 127, size=(N, K), dtype=torch.int8).cuda()
    A_f8 = A_bf16.to(torch.float8_e4m3fn)
    B_f8 = B_bf16.to(torch.float8_e4m3fn)
    scale_A = torch.randn(M, 1).bfloat16().cuda()
    scale_B = torch.randn(1, N).bfloat16().cuda()

    # benchmark F.linear() i.e. A @ B.T
    bf16_time = bench_f(torch.mm, A_bf16, B_bf16.T)
    cublas_i8_time = bench_f(torch._int_mm, A_i8, B_i8.T)
    inductor_scaled_i8_time = bench_f(scaled_int8_mm_inductor, A_i8, B_i8.T, scale_A, scale_B)
    triton_scaled_i8_time = bench_f(scaled_mm, A_i8, B_i8.T, scale_A, scale_B)

    # torch._scaled_mm() only supports tensor-wise FP32 scaling for Ada
    cublas_scaled_f8_time = bench_f(
        torch._scaled_mm, A_f8, B_f8.T, scale_A[0, 0].float(), scale_B[0, 0].float(), out_dtype=scale_A.dtype
    )
    triton_scaled_f8_time = bench_f(scaled_mm, A_f8, B_f8.T, scale_A, scale_B)

    # DeepSeek-v3's Fprop: (1, 128) for act, (128, 128) for weight
    scale_A = torch.randn(M, K // 128).bfloat16().cuda()
    scale_B = torch.randn(N // 128, K // 128).bfloat16().cuda()
    triton_tile_scaled_i8_time = bench_f(scaled_mm, A_i8, B_i8.T, scale_A, scale_B.T)
    triton_tile_scaled_f8_time = bench_f(scaled_mm, A_f8, B_f8.T, scale_A, scale_B.T)

    sample = [
        M,
        N,
        K,
        bf16_time / cublas_i8_time,
        bf16_time / inductor_scaled_i8_time,
        bf16_time / triton_scaled_i8_time,
        bf16_time / triton_tile_scaled_i8_time,
        bf16_time / cublas_scaled_f8_time,
        bf16_time / triton_scaled_f8_time,
        bf16_time / triton_tile_scaled_f8_time,
    ]
    data.append(sample)

df = pd.DataFrame(
    data,
    columns=[
        "M",
        "N",
        "K",
        "CuBLAS INT8 speedup",
        "Inductor scaled INT8 speedup",
        "Triton scaled INT8 speedup",
        "Triton tile-scaled INT8 speedup",
        "CuBLAS tensor-scaled FP8 speedup",
        "Triton scaled FP8 speedup",
        "Triton tile-scaled FP8 speedup",
    ],
)
print(df.to_markdown(index=False))
