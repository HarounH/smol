"""
BitTensor

collection of utiltiies for dealing with tensors representing bits
"""


def bittify(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    tensor = torch.as_tensor(x)
    # TODO: modify repr of tensor and maintain additional metadata.
    if tensor.dtype in [torch.float32, torch.int32, torch.uint32]:
        return tensor.view(torch.int32)
    if tensor.dtype in [torch.float64, torch.int64, torch.uint64]:
        return tensor.view(torch.int64)

    raise NotImplementedError(f"unrecognized dtype: {tensor.dtype=}")


def bitt2bool(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32:
        WIDTH = 32
    elif x.dtype == torch.int64:
        WIDTH = 64
    else:
        raise NotImplementedError(f"unrecognized dtype: {tensor.dtype=}")

    # Create a boolean tensor with bits from the original tensor
    bitwise_mask_size = [1] * x.ndim
    bitwise_mask = torch.arange(WIDTH, dtype=x.dtype).reshape(*bitwise_mask_size, -1)
    boolean_tensor = (x[..., None] & (1 << bitwise_mask)) > 0
    return boolean_tensor