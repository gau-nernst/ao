import torch
from torchao.quantization.fp6_llm import Fp6LlmLinear
from torchao.prototype.fp6.fp16_fp6_triton import float16_float6_e3m2_matmul, FLOAT6_E3M2_MAX, to_float6_e3m2
from torch.utils.benchmark import Timer
import pandas as pd


def benchmark(f, *args):
    measurement = Timer(
        stmt="f(*args)",
        globals=dict(f=f, args=args),
    ).blocked_autorange()
    return measurement.median * 1000


if __name__ == "__main__":
    M = 1
    N = 8192
    K = 8192

    A = torch.randn(M, K, device="cuda", dtype=torch.half)
    
    linear = torch.nn.Linear(K, N, device="cuda", dtype=torch.half)
    linear_fp6_llm = Fp6LlmLinear.from_float(linear)

    B = linear.weight.detach().T.contiguous()

    # for triton kernel
    scales = B.float().abs().amax(0) / FLOAT6_E3M2_MAX
    scales[scales == 0.0] = 1.0
    B_scaled = B / scales
    scales = scales.to(torch.half)
    B_6bit = to_float6_e3m2(B_scaled, no_bit_packing=True)

    B_2bit = (B_6bit >> 4) & 0b11
    B_4bit = B_6bit & 0b1111

    B_2bit = (B_2bit[..., ::4] << 6) | (B_2bit[..., 1::4] << 4) | (B_2bit[..., 2::4] << 2) | B_2bit[..., 3::4]
    B_4bit = (B_4bit[..., ::2] << 4) | B_4bit[..., 1::2]
    # end of for triton kernel

    results = []
    results.append(["Baseline (CuDNN)", benchmark(torch.matmul, A, B)])
    results.append(["FP6-LLM", benchmark(linear_fp6_llm, A)])
    results.append(["FP6-triton", benchmark(float16_float6_e3m2_matmul, A, B_2bit, B_4bit, scales)])

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
