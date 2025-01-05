import pandas as pd
import torch
from triton.testing import do_bench

from torchao.prototype.quantized_training.kernel import scaled_mm


def bench_f(f, *args):
    return do_bench(lambda: f(*args), return_mode="median")


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
    A_scale = torch.randn(M).bfloat16().cuda()
    B_scale = torch.randn(N).bfloat16().cuda()

    # benchmark F.linear() i.e. A @ B.T
    bf16_time = bench_f(torch.mm, A_bf16, B_bf16.T)
    i8_time = bench_f(torch._int_mm, A_i8, B_i8.T)
    i8_dequant_time = bench_f(scaled_mm, A_i8, B_i8.T, A_scale, B_scale)

    sample = [M, N, K, bf16_time / i8_time, bf16_time / i8_dequant_time]
    data.append(sample)

df = pd.DataFrame(data, columns=["M", "N", "K", "CuBLAS INT8 speedup", "Triton scaled INT8 speedup"])
print(df.to_markdown())
