import torch
from torchao.quantization.subclass import QuantizedLinearWeightBase
from torch import nn


class Fp8E5M2WeightOnlyQuantizedLinearWeight(QuantizedLinearWeightBase):
    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        y = torch.mm(act_mat.flatten(0, -2), w_qtensor.int_data.to(act_mat.dtype))
        y = y.reshape(*act_mat.shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y.to(act_mat.dtype)

    @classmethod
    def from_float(cls, input_float):
        int_data = input_float.T.to(torch.float8_e5m2).contiguous()
        return cls(int_data, False, input_float.shape, dtype=input_float.dtype)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            self.transposed,
            self.shape,
            dtype=self.dtype,
        )

    def _change_shape(self, shape):
        return self.__class__(self.int_data, self.transposed, shape, dtype=self.dtype)

    def __tensor_flatten__(self):
        return ["int_data"], (self.transposed, self.shape, self.dtype)

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        int_data = tensor_data_dict["int_data"]
        transposed, shape, dtype = tensor_attributes
        return cls(
            int_data,
            transposed,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            strides=outer_stride,
        )


def change_linear_weights_to_fp8e5m2_woqtensors(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight = nn.Parameter(
                Fp8E5M2WeightOnlyQuantizedLinearWeight.from_float(m.weight.data),
                requires_grad=False,
            )
