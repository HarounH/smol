"""
BitTensor

collection of utiltiies for dealing with tensors representing bits
"""
from typing import Union
import torch
import numpy as np


def bittify(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    tensor = torch.as_tensor(x)
    # TODO: modify repr of tensor and maintain additional metadata.
    if tensor.dtype in [torch.int8, torch.uint8]:
        return tensor.view(torch.int8)
    elif tensor.dtype in [torch.float16, torch.bfloat16]:
        return tensor.view(torch.int16)
    elif tensor.dtype in [torch.float32, torch.int32, torch.uint32]:
        return tensor.view(torch.int32)
    elif tensor.dtype in [torch.float64, torch.int64, torch.uint64]:
        return tensor.view(torch.int64)

    raise NotImplementedError(f"unrecognized dtype: {tensor.dtype=}")

def bitt2bool(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x  # don't do anything
    bit_width = torch.iinfo(x.dtype).bits
    # Create a boolean tensor with bits from the original tensor
    bitwise_mask_size = [1] * x.ndim
    bitwise_mask = torch.arange(bit_width, dtype=x.dtype).reshape(*bitwise_mask_size, -1)
    boolean_tensor = (x[..., None] & (1 << bitwise_mask)) > 0
    return boolean_tensor


def bitt_stringify(x: torch.Tensor, element_delimiter: str = " ", line_delimiter: str = "\n") -> str:
    x = bitt2bool(x)  # bool tensor
    data = x.to(torch.int8).tolist()
    if len(data) == 0:
        return ""
    message = [""]
    for row in data:
        message.append(element_delimiter.join(map(str, row)))
    return line_delimiter.join(message)


