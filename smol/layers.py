import torch
from typing import Tuple, Callable

"""
Layers to do bit manipulation

There are a few axes along which we can think about these layers:
    - experimental (experimental_layers.py) v/s not (this file)
    - learnable v/s not
    - unary and binary
"""

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

