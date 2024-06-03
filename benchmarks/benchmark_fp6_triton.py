import torch
from torchao.quantization.fp6_llm import Fp6LlmLinear
from torchao.prototype.fp6.fp16_fp6_triton import fp6_2_4_matmul, float16_matmul, to_fp6_2_4
from torch.utils.benchmark import Timer
import pandas as pd


def benchmark(f, *args):
    print(f(*args))

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
    B_2bit, B_4bit, B_scale = to_fp6_2_4(B)

    results = []
    results.append(["Baseline (CuBLAS)", benchmark(torch.matmul, A, B)])
    results.append(["FP16-triton", benchmark(float16_matmul, A, B)])
    results.append(["FP6-LLM", benchmark(linear_fp6_llm, A)])
    results.append(["FP6-triton-splitK", benchmark(fp6_2_4_matmul, A, B_2bit, B_4bit, B_scale)])

    df = pd.DataFrame(results, columns=["name", "time (ms)"])
    print(df.to_markdown(index=False))
