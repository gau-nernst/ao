import torch
from torchao.quantization.subclass import QuantizedLinearWeightBase
from torch import nn


class Fp8WeightOnlyQuantizedLinearWeight(QuantizedLinearWeightBase):
    @staticmethod
    def __new__(cls, int_data, q_scales, transposed, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = q_scales.dtype
        kwargs["dtype"] = dtype
        return super().__new__(cls, int_data, transposed, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, int_data, q_scales, transposed, shape, dtype=None, **kwargs):
        self.q_scales = q_scales
        super().__init__(int_data, transposed)

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        y = torch.mm(act_mat.reshape(-1, act_mat.shape[-1]), w_qtensor.int_data.to(act_mat.dtype)) * w_qtensor.q_scales
        y = y.reshape(*act_mat.shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y.to(act_mat.dtype)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.q_scales),
            self.transposed,
            self.shape,
            dtype=self.dtype,
        )

    def _change_shape(self, shape):
        return self.__class__(self.int_data, self.transposed, shape, dtype=self.dtype)

    def __tensor_flatten__(self):
        return ["int_data", "q_scales"], (self.transposed, self.shape, self.dtype)

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        int_data = tensor_data_dict["int_data"]
        q_scales = tensor_data_dict["q_scales"]
        transposed, shape, dtype = tensor_attributes
        return cls(
            int_data,
            q_scales,
            transposed,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(cls, input_float, fp8_dtype):
        dtype = input_float.dtype

        w = input_float.float()
        scales = w.abs().amax(1) / torch.finfo(fp8_dtype).max
        scales = scales.clamp(torch.finfo(fp8_dtype).eps)
        w = w / scales[:, None]

        int_data = w.T.to(fp8_dtype).contiguous()
        return cls(int_data, scales, False, input_float.shape, dtype=dtype)


def change_linear_weights_to_fp8_woqtensors(model, fp8_dtype):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight = nn.Parameter(
                Fp8WeightOnlyQuantizedLinearWeight.from_float(m.weight.data, fp8_dtype),
                requires_grad=False,
            )
