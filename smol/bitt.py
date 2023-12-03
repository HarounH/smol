"""
BitTensor

collection of utiltiies for dealing with tensors representing bits
"""
from typing import Union, Tuple, Callable
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



class NAND(torch.nn.Module):
    def __init__(self, data_size: Tuple[int], dtype: torch.int8) -> None:
        super().__init__()
        info = torch.iinfo(dtype)
        self.weight = torch.randint(low=info.min, high=info.max, size=data_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(torch.bitwise_and(x, self.weight))


class NOR(torch.nn.Module):
    def __init__(self, data_size: Tuple[int], dtype: torch.int8) -> None:
        super().__init__()
        info = torch.iinfo(dtype)
        self.weight = torch.randint(low=info.min, high=info.max, size=data_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(torch.bitwise_or(x, self.weight))


class XOR(torch.nn.Module):
    def __init__(self, data_size: Tuple[int], dtype: torch.int8) -> None:
        super().__init__()
        info = torch.iinfo(dtype)
        self.weight = torch.randint(low=info.min, high=info.max, size=data_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(x, self.weight)


class AND(torch.nn.Module):
    def __init__(self, data_size: Tuple[int], dtype: torch.int8) -> None:
        super().__init__()
        info = torch.iinfo(dtype)
        self.weight = torch.randint(low=info.min, high=info.max, size=data_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(x, self.weight)


class OR(torch.nn.Module):
    def __init__(self, data_size: Tuple[int], dtype: torch.int8) -> None:
        super().__init__()
        info = torch.iinfo(dtype)
        self.weight = torch.randint(low=info.min, high=info.max, size=data_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_or(x, self.weight)


class Foldl(torch.nn.Module):
    def __init__(self, f: Callable[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], dtype: torch.dtype = torch.int8) -> None:
        super().__init__()
        self.dtype = dtype
        self.f = f
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., N) of any type
        # out: (...) of self.dtype
        x_retyped = x.view(self.dtype)  # (..., N') of self.dtype
        output = x_retyped[..., 0]
        for i in range(1, x_retyped.shape[-1]):
            output = self.f(output, x_retyped[..., i])
        return output
