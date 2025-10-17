# python attn_kernel.py --batch_size 1 --q_seq_len 1 --k_seq_len 1024 --num_heads 8 --head_dim 128 --kernel sdpa --dtype bf16
import argparse
from typing import Dict, Optional

import torch
from smol.gpu.triton_sdpa import KernelSDPA
from smol.gpu.triton_sdpa_chunked import KernelSDPAChunked
from smol.gpu.triton_flash_attention import KernelFlashAttention

class CustomSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # q, k, v: [batch, heads, seq_len, head_dim]
        scale = 1.0 / torch.sqrt(torch.tensor(q.size(-1), dtype=q.dtype, device=q.device))
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for CustomSDPA.")

def get_dtype(dtype_str: str) -> torch.dtype:
    mapping: Dict[str, torch.dtype] = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str.lower() not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from fp32, fp16, bf16.")
    return mapping[dtype_str.lower()]


def run_attention(
    batch_size,
    num_heads,
    q_seq_len,
    k_seq_len,
    head_dim,
    *,
    kernel='sdpa',
    dtype=torch.bfloat16,
    seed: Optional[int] = None,
    check_correctness: bool = False,
    warmup_iters: int = 0,
):
    # Q: (batch_size, num_heads, q_seq_len, head_dim)
    # K: (batch_size, num_heads, k_seq_len, head_dim)
    # V: (batch_size, num_heads, k_seq_len, head_dim)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    Q = torch.randn(batch_size, num_heads, q_seq_len, head_dim, device='cuda', dtype=dtype)
    K = torch.randn(batch_size, num_heads, k_seq_len, head_dim, device='cuda', dtype=dtype)
    V = torch.randn(batch_size, num_heads, k_seq_len, head_dim, device='cuda', dtype=dtype)

    def compute_attention():
        if kernel == 'sdpa':
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        if kernel == "custom_sdpa":
            return CustomSDPA.apply(Q, K, V)
        if kernel == "kernel_sdpa_fwd":
            return KernelSDPA.apply(Q, K, V)
        if kernel.startswith("kernel_sdpa_chunked_fwd"):
            prefix_len = len("kernel_sdpa_chunked_fwd_")
            if  len(kernel) > prefix_len:
                chunk_size = int(kernel[prefix_len:])
            else:
                chunk_size = 256
            return KernelSDPAChunked.apply(Q, K, V, chunk_size)
        if kernel == "kernel_flash_attention_fwd":
            return KernelFlashAttention.apply(Q, K, V)
        raise ValueError(f"Unknown kernel: {kernel}")

    if warmup_iters > 0:
        for _ in range(warmup_iters):
            compute_attention()
        torch.cuda.synchronize()

    attn_output = compute_attention()

    if check_correctness:
        ref_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        # Only cast when necessary to avoid redundant copies on the fast path.
        q_ref = Q if Q.dtype == ref_dtype else Q.to(ref_dtype)
        k_ref = K if K.dtype == ref_dtype else K.to(ref_dtype)
        v_ref = V if V.dtype == ref_dtype else V.to(ref_dtype)
        with torch.no_grad():
            ref_output = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        ref_output = ref_output.to(attn_output.dtype)
        tol_low_precision = dtype in (torch.float16, torch.bfloat16)
        atol = 1e-3 if tol_low_precision else 1e-5
        rtol = 1e-3 if tol_low_precision else 1e-5
        torch.testing.assert_close(attn_output, ref_output, atol=atol, rtol=rtol)
        print("Correctness check passed!")

    print("Attention output shape:", attn_output.shape)
    print("Attention output (first element):", attn_output.flatten()[0].item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention on random Q, K, V tensors.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--num_heads', type=int, required=True, help="Number of attention heads")
    parser.add_argument('--q_seq_len', type=int, required=True, help="Q sequence length")
    parser.add_argument('--k_seq_len', type=int, required=True, help="K sequence length")
    parser.add_argument('--head_dim', type=int, required=True, help="Head dimension")
    parser.add_argument('--kernel', type=str, default='sdpa', help="Attention kernel to use (default: sdpa)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    parser.add_argument('--seed', type=int, default=1337, help="Random seed for torch and CUDA")
    parser.add_argument("-c", '--check_correctness', action='store_true', help="Compare kernel output against PyTorch SDPA")
    parser.add_argument('--warmup', type=int, default=0, help="Number of warmup iterations to run before executing the measured pass")

    args = parser.parse_args()
    run_attention(
        args.batch_size,
        args.num_heads,
        args.q_seq_len,
        args.k_seq_len,
        args.head_dim,
        kernel=args.kernel,
        dtype=get_dtype(args.dtype),
        seed=args.seed,
        check_correctness=args.check_correctness,
        warmup_iters=args.warmup,
    )
