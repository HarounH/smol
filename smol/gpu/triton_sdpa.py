import math
import torch
import triton
import triton.language as tl


@triton.jit
def sdpa_kernel(
    # Q, K, V, O are all (B, H, L, D) tensors
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

    # each program handles some one (b, h, q_i) query
    arange = tl.arange(0, BLOCK_SIZE)
    mask = arange < dim
    q_val = tl.load(
        # adding arange makes it load a BLOCK_SIZ matrix, with mask applied
        Q_ptr + (pid_bh * seq_len_q * dim) + (pid_l * dim) + arange,
        mask=arange < dim,
        other=0.0,
    )  # (D,)
    o_position = O_ptr + (pid_bh * seq_len_q * dim) + (pid_l * dim) + arange
    z_position = Z_ptr + (pid_bh * seq_len_q) + pid_l
    current_o = tl.load(
        o_position,
        mask=mask,
        other=0.0,
    )
    running_max = 0.0
    running_z = 0.0
    # TODO: load multiple K at once?
    for k_i in range(0, seq_len_k):
        k = tl.load(
            K_ptr + (pid_bh * seq_len_k * dim) + (k_i * dim) + arange,
            mask=mask,
            other=0.0,
        )  # (D)

        # TODO: causal mask
        # TODO: mult in fp32
        qk_dot = tl.sum(q_val * k, axis=0) * scale
        new_max = tl.maximum(qk_dot, running_max)
        stable_qk_dot = qk_dot - new_max
        renormalize_factor = tl.exp(running_max) * tl.exp(-new_max)
        running_max = new_max
        stable_qk_exp = tl.exp(stable_qk_dot)
        running_z = (renormalize_factor * running_z) + stable_qk_exp

        # multiply by V and add to o
        v = tl.load(
            V_ptr + (pid_bh * seq_len_k * dim) + (k_i * dim) + arange,
            mask=mask,
            other=0.0,
        )  # (D)

        current_o = tl.cast(
            (renormalize_factor * current_o) + (stable_qk_exp * v),
            current_o.type,
        )

    # divide O by Z to get final output
    tl.store(z_position, running_z, mask=pid_l < seq_len_q)
    tl.store(
        o_position,
        current_o / running_z,
        mask=mask,
    )

class KernelSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Placeholder for a custom Triton kernel implementation
        # For demonstration, we'll just call the built-in function
        b, h, l, d = q.shape
        o_BHLD = torch.zeros_like(q)

        # TODO: normalize in fp32?
        z_BHL = torch.zeros((b, h, l), device=q.device, dtype=q.dtype)

        # TODO:  does scale have to be on GPU?
        scale = 1.0 / math.sqrt(d)
        seq_len_k = k.size(-2)
        grid = lambda meta: (b * h, l)
        sdpa_kernel[grid](q, k, v, o_BHLD, z_BHL, scale, l, seq_len_k, d, BLOCK_SIZE=triton.next_power_of_2(d))
        return o_BHLD

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for KernelSDPA.")
