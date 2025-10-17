import math
import torch
import triton
import triton.language as tl


@triton.jit
def sdpa_kernel_chunked(
    # Q, K, V, O are all (B, H, L, D) tensors
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Z_ptr,
    scale,
    BATCH_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    seq_len_q: tl.constexpr,
    seq_len_k: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)

    # each program handles some one (b, h, q_i) query
    q_block_ptr = tl.make_block_ptr(
        Q_ptr, # base
        (BATCH_SIZE * NUM_HEADS, seq_len_q, dim), # shape
        (seq_len_q * dim, dim, 1), # strides
        (pid_bh, pid_l, 0),  # offsets
        (1, 1, dim), # block shape
        (0, 1, 2), # order
    )
    z_block_ptr = tl.make_block_ptr(
        Z_ptr, # base
        (BATCH_SIZE * NUM_HEADS, seq_len_q), # shape
        (seq_len_q, 1), # strides
        (pid_bh, pid_l),  # offsets
        (1, 1), # block shape
        (0, 1), # order
    )
    o_block_ptr = tl.make_block_ptr(
        O_ptr, # base
        (BATCH_SIZE * NUM_HEADS, seq_len_q, dim), # shape
        (seq_len_q * dim, dim, 1), # strides
        (pid_bh, pid_l, 0),  # offsets
        (1, 1, dim), # block shape
        (0, 1, 2), # order
    )
    running_max = 0.0
    running_z = 0.0

    q_val = tl.load(q_block_ptr)  # (1, 1, D)
    current_o = tl.load(o_block_ptr)  # (1, 1, D)

    for k_i in range(0, seq_len_k, BLOCK_SIZE_K):
        # TODO
        # load block of k
        k_block_ptr = tl.make_block_ptr(
            K_ptr, # base
            (BATCH_SIZE * NUM_HEADS, seq_len_k, dim), # shape
            (seq_len_k * dim, dim, 1), # strides
            (pid_bh, k_i, 0),  # offsets
            (1, BLOCK_SIZE_K, dim), # block shape
            (0, 1, 2), # order
        )
        k_val = tl.load(k_block_ptr)  # (1, BK, D)
        # compute qk in a stable way
        qk_dot = scale * tl.sum(q_val * k_val, axis=2)  # (1, BK)
        new_max = tl.maximum(qk_dot.max(), running_max)
        # update running max, running z
        stable_qk_dot = qk_dot - new_max
        renormalize_factor = tl.exp(running_max) * tl.exp(-new_max)
        running_max = new_max
        stable_qk_exp = tl.exp(stable_qk_dot)  # (1, BK)
        running_z = (renormalize_factor * running_z) + tl.sum(stable_qk_exp)

        # load block of v
        v_block_ptr = tl.make_block_ptr(
            V_ptr, # base
            (BATCH_SIZE * NUM_HEADS, seq_len_k, dim), # shape
            (seq_len_k * dim, dim, 1), # strides
            (pid_bh, k_i, 0),  # offsets
            (1, BLOCK_SIZE_K, dim), # block shape
            (0, 1, 2), # order
        )
        v_val = tl.load(v_block_ptr)  # (1, BK, D)
        # compute qkt * v
        new_o = tl.sum(stable_qk_exp[:, :, None] * v_val, axis=1)  # (1, D)

        # update running o
        current_o = (renormalize_factor * current_o) + new_o

    # divide O by Z to get final output
    # store Z, O
    tl.store(z_block_ptr, running_z)
    tl.store(
        o_block_ptr,
        current_o / running_z,
        boundary_check=(0, 1, 2),
    )

class KernelSDPAChunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_size: int = 1024):
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
        BLOCK_SIZE_D = triton.next_power_of_2(d)
        BLOCK_SIZE_K = block_size // BLOCK_SIZE_D
        sdpa_kernel_chunked[grid](
            q, k, v, o_BHLD, z_BHL, scale,
            b,
            h,
            l,
            seq_len_k,
            d,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
        return o_BHLD

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for KernelSDPA.")
