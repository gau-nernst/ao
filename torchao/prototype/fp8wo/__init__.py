import torch
from torchao.quantization.subclass import QuantizedLinearWeightBase


class Fp8E5M2WeightOnlyQuantizedLinearWeight(QuantizedLinearWeightBase):
    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        y = torch.mm(act_mat.flatten(0, -2), w_qtensor.int_data.T.to(act_mat.dtype))
        y = y.reshape(*act_mat.shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y.to(act_mat.dtype)

    @classmethod
    def from_float(cls, input_float):
        return cls(input_float.to(torch.float8_e5m2), False, input_float.shape, dtype=input_float.dtype)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            self.transposed,
            self.shape,
            dtype=self.dtype,
        )

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
