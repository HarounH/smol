import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Z_ptr,
    scale,
    seq_len_q,
    seq_len_k,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)

    offsets_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offsets_d < dim

    q_ptr = Q_ptr + (pid_bh * seq_len_q * dim) + (pid_l * dim) + offsets_d
    k_base = K_ptr + (pid_bh * seq_len_k * dim)
    v_base = V_ptr + (pid_bh * seq_len_k * dim)
    o_ptr = O_ptr + (pid_bh * seq_len_q * dim) + (pid_l * dim) + offsets_d
    z_ptr = Z_ptr + (pid_bh * seq_len_q) + pid_l

    q = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    running_max = -float("inf")
    running_sum = 0.0

    for k_idx in range(0, seq_len_k):
        offset = k_idx * dim + offsets_d
        k = tl.load(k_base + offset, mask=mask_d, other=0.0).to(tl.float32)
        v = tl.load(v_base + offset, mask=mask_d, other=0.0).to(tl.float32)

        score = tl.sum(q * k, axis=0) * scale
        new_max = tl.maximum(score, running_max)
        exp_old = tl.exp(running_max - new_max)
        exp_new = tl.exp(score - new_max)

        running_sum = running_sum * exp_old + exp_new
        acc = acc * exp_old + exp_new * v
        running_max = new_max

    tl.store(z_ptr, running_sum, mask=pid_l < seq_len_q)
    output = acc / running_sum
    tl.store(o_ptr, output.to(O_ptr.dtype.element_ty), mask=mask_d)


class KernelFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        b, h, q_len, dim = q.shape
        o = torch.zeros_like(q)
        z = torch.zeros((b, h, q_len), device=q.device, dtype=q.dtype)
        scale = 1.0 / math.sqrt(dim)
        seq_len_k = k.size(-2)
        grid = lambda meta: (b * h, q_len)
        flash_attention_kernel[grid](
            q,
            k,
            v,
            o,
            z,
            scale,
            q_len,
            seq_len_k,
            dim,
            BLOCK_SIZE=triton.next_power_of_2(dim),
        )
        return o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for KernelFlashAttention.")
