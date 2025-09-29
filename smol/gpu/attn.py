import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# -------------------------------
# Utilities
# -------------------------------


def ensure_cuda(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Set --device cpu or install CUDA.")
    return torch.device(device)


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


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def format_num(x: float) -> str:
    if x >= 1e3:
        return f"{x/1e3:.2f}k"
    return f"{x:.0f}"


# -------------------------------
# Attention Implementations
# -------------------------------


def attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    q, k, v: [batch, heads, seq_len, head_dim]
    returns: [batch, heads, seq_len, head_dim]
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        # Mask out future positions
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, v)
    return out


def attention_sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    Wrapper around PyTorch scaled_dot_product_attention (PyTorch >= 2.0).
    Uses is_causal flag for causal masking.
    """
    # PyTorch expects shape [batch, heads, seq, dim] or [seq, batch, heads, dim]?
    # F.scaled_dot_product_attention expects [batch, heads, seq, dim] consistently when using the functional.
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)


# -------------------------------
# Benchmarking
# -------------------------------


@dataclass
class BenchmarkResult:
    name: str
    ms_per_iter: float
    iters: int
    tflops: Optional[float]


def measure_gpu_time(fn: Callable[[], torch.Tensor], iters: int, warmup: int) -> Tuple[float, torch.Tensor]:
    # Warmup
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = fn()
        cuda_synchronize()

    # Timing with CUDA events for accuracy
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    out: Optional[torch.Tensor] = None
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            out = fn()
    end_event.record()
    cuda_synchronize()

    total_ms = start_event.elapsed_time(end_event)  # milliseconds
    ms_per_iter = total_ms / max(1, iters)
    assert out is not None
    return ms_per_iter, out


def compute_attention_flops(batch: int, heads: int, seq: int, dim: int) -> float:
    """
    Rough FLOPs count for attention forward pass (no backward):
    - QK^T: (B*H) * S * S * D (multiply-add ~2* ops per element)
    - softmax: ~ S * S per (B*H)
    - (softmax)V: (B*H) * S * S * D (multiply-add)
    We'll approximate with 2*(B*H)*S*S*D for the two GEMMs and ignore softmax cost.
    Return FLOPs (not TFLOPs).
    """
    flops = 2.0 * batch * heads * (seq * seq * dim) * 2.0  # two matmuls
    return flops


def benchmark_impl(
    name: str,
    impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    iters: int,
    warmup: int,
    flops: Optional[float] = None,
) -> BenchmarkResult:
    def _run() -> torch.Tensor:
        return impl(q, k, v, causal)

    ms_per_iter, out = measure_gpu_time(_run, iters=iters, warmup=warmup)
    tflops = None
    if flops is not None and ms_per_iter > 0:
        tflops = (flops / (ms_per_iter * 1e-3)) / 1e12
    # small use to keep compiler from removing work
    if out is not None:
        _ = float(out[0, 0, 0, 0].detach().float().cpu())
    return BenchmarkResult(name=name, ms_per_iter=ms_per_iter, iters=iters, tflops=tflops)


# -------------------------------
# Main
# -------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA benchmark for softmax dot-product attention (forward only)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the naive implementation if available")
    args = parser.parse_args()

    set_determinism(args.seed)
    device = ensure_cuda(args.device)
    dtype = get_dtype(args.dtype)

    torch.backends.cudnn.benchmark = True

    # Print environment
    print("PyTorch", torch.__version__)
    if device.type == "cuda":
        print("CUDA available:", torch.cuda.is_available())
        print("Device:", torch.cuda.get_device_name(device))
        print("Compute capability:", getattr(torch.cuda.get_device_properties(device), "major", "?"), getattr(torch.cuda.get_device_properties(device), "minor", "?"))
        print("cuDNN:", torch.backends.cudnn.version())
    else:
        print("Running on CPU")

    batch, seq, heads, dim = args.batch, args.seq, args.heads, args.dim
    print(f"Config: B={batch}, H={heads}, S={seq}, D={dim}, dtype={dtype}, causal={args.causal}")

    # Create inputs
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)

    # Optional compilation for naive attention
    naive_impl = attention_naive
    if args.compile and hasattr(torch, "compile"):
        try:
            naive_impl = torch.compile(attention_naive, mode="max-autotune")  # type: ignore[attr-defined]
            print("Enabled torch.compile for naive implementation")
        except Exception as e:
            print(f"torch.compile failed to enable: {e}")

    flops = compute_attention_flops(batch, heads, seq, dim)

    # Warm up CUDA context once with a simple op to avoid context creation in timings
    _ = (q @ k.transpose(-1, -2))
    cuda_synchronize()

    results = []

    # Benchmark naive
    results.append(
        benchmark_impl(
            name="naive",
            impl=naive_impl,
            q=q,
            k=k,
            v=v,
            causal=args.causal,
            iters=args.iters,
            warmup=args.warmup,
            flops=flops,
        )
    )

    # Benchmark PyTorch fused scaled_dot_product_attention if available
    has_sdp = hasattr(F, "scaled_dot_product_attention")
    if has_sdp:
        results.append(
            benchmark_impl(
                name="scaled_dot_product_attention",
                impl=attention_sdp,
                q=q,
                k=k,
                v=v,
                causal=args.causal,
                iters=args.iters,
                warmup=args.warmup,
                flops=flops,
            )
        )

        # Correctness check vs naive at float32 reference
        with torch.no_grad():
            q32, k32, v32 = q.float(), k.float(), v.float()
            ref = attention_naive(q32, k32, v32, causal=args.causal)
            test = attention_sdp(q32, k32, v32, causal=args.causal)
            max_abs = (ref - test).abs().max().item()
            print(f"Max abs diff (fp32, naive vs sdp): {max_abs:.3e}")
    else:
        print("scaled_dot_product_attention not available in this PyTorch version")

    # Pretty print results
    print("\nResults (forward):")
    print("name\tms/iter\tTFLOP/s")
    for r in results:
        tflops_str = f"{r.tflops:.2f}" if r.tflops is not None else "-"
        print(f"{r.name}\t{r.ms_per_iter:.3f}\t{tflops_str}")


if __name__ == "__main__":
    main()


