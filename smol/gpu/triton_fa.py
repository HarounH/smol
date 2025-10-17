import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel():
    ...


class KernelSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        b, h, l, d = q.shape
        o_BHLD = torch.zeros_like(q)

        # TODO: normalize in fp32?
        z_BHL = torch.zeros((b, h, l), device=q.device, dtype=q.dtype)

        # TODO:  does scale have to be on GPU?
        scale = 1.0 / math.sqrt(d)
        seq_len_k = k.size(-2)
        grid = lambda meta: (b * h, l)
        flash_attention_kernel[grid](q, k, v, o_BHLD, z_BHL, scale, l, seq_len_k, d, BLOCK_SIZE=triton.next_power_of_2(d))
        return o_BHLD
