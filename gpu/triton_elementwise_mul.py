import argparse
import os
import time

import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_mul_kernel(
    a_ptr, b_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a * b
    tl.store(out_ptr + offsets, c, mask=mask)


def run_triton_mul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, block_size: int = 1024):
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    elementwise_mul_kernel[grid](
        a, b, out, n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Elementwise multiply using Triton kernel")
    parser.add_argument("--size", type=int, default=1_000_000, help="Number of elements in the vectors")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Tensor dtype")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--block", type=int, default=1024, help="Kernel BLOCK_SIZE")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--no-check", action="store_true", help="Skip correctness check vs torch")
    return parser.parse_args()


def str_to_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA device requested but not available")

    generator = torch.Generator(device="cuda" if args.device == "cuda" else "cpu").manual_seed(args.seed)
    dtype = str_to_dtype(args.dtype)

    # NVTX range: data init
    torch.cuda.nvtx.range_push("data_init")
    a = torch.randn(args.size, device=args.device, dtype=dtype, generator=generator)
    b = torch.randn(args.size, device=args.device, dtype=dtype, generator=generator)
    out = torch.empty_like(a)
    torch.cuda.nvtx.range_pop()

    # Warmup and correctness check
    if args.device == "cuda":
        torch.cuda.synchronize()

    if not args.no_check:
        torch.cuda.nvtx.range_push("torch_reference")
        ref = a * b
        torch.cuda.nvtx.range_pop()
        if args.device == "cuda":
            torch.cuda.synchronize()
        # Allow slightly looser tolerance for low-precision types
        atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-6
        rtol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        max_abs_diff = (ref - (a * b)).abs().max().item()
        # Actual check of Triton vs ref will occur after first launch; here we just assert PyTorch op works
        del ref
        if args.device == "cuda":
            torch.cuda.synchronize()

    # Warmup launches
    torch.cuda.nvtx.range_push("warmup_triton")
    for _ in range(args.warmup):
        run_triton_mul(a, b, out, args.block)
    if args.device == "cuda":
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Correctness vs torch after warmup
    if not args.no_check:
        torch.cuda.nvtx.range_push("correctness_check")
        ref = a * b
        if args.device == "cuda":
            torch.cuda.synchronize()
        atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-6
        rtol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        try:
            torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
            ok = True
        except AssertionError as e:
            ok = False
            print("[ERROR] Triton output mismatch:", e)
        finally:
            del ref
        torch.cuda.nvtx.range_pop()
        if not ok:
            raise SystemExit(1)

    # Benchmark Triton
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push("bench_triton")
    start_event.record()
    for _ in range(args.iters):
        run_triton_mul(a, b, out, args.block)
    end_event.record()
    if args.device == "cuda":
        torch.cuda.synchronize()
    triton_ms = start_event.elapsed_time(end_event) / max(args.iters, 1)
    torch.cuda.nvtx.range_pop()

    # Benchmark PyTorch as baseline
    start_event_ref = torch.cuda.Event(enable_timing=True)
    end_event_ref = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push("bench_torch")
    start_event_ref.record()
    for _ in range(args.iters):
        out_ref = a * b
    end_event_ref.record()
    if args.device == "cuda":
        torch.cuda.synchronize()
    torch_ms = start_event_ref.elapsed_time(end_event_ref) / max(args.iters, 1)
    torch.cuda.nvtx.range_pop()

    # Throughput (GB/s) estimate: 3 reads/writes per element for multiply: a, b, out
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_bytes = (bytes_per_elem * args.size * 3)
    gb = total_bytes / (1024 ** 3)
    triton_gbs = gb / (triton_ms / 1e3)
    torch_gbs = gb / (torch_ms / 1e3)

    print(f"size={args.size} dtype={args.dtype} block={args.block} device={args.device}")
    print(f"Triton: {triton_ms:.4f} ms/iter, ~{triton_gbs:.2f} GB/s")
    print(f"PyTorch: {torch_ms:.4f} ms/iter, ~{torch_gbs:.2f} GB/s")


if __name__ == "__main__":
    # Helpful defaults for Triton debugging
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "0")
    os.environ.setdefault("TRITON_KERNEL_DUMP", "0")
    main()

