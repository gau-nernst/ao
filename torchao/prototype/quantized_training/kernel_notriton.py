import torch
from torch import Tensor


# This is less performant than the explicit hand-written Triton kernel, though things might
# change in the future. This does not support tile-wise scaling.
def scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    if A.dtype == torch.int8:
        # multiplying col_scale (scale_B) first is faster than the other way round.
        return torch._int_mm(A, B) * scale_B * scale_A
    else:
        # torch._scaled_mm() requires FP32 scales
        return torch._scaled_mm(
            A, B, scale_A.float(), scale_B.float(), out_dtype=scale_A.dtype
        )
