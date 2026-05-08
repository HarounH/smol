from contextlib import nullcontext
from pathlib import Path

import torch

from smol.diffusion.text.core.config import RunConfig
from smol.diffusion.text.core.model import TextDiffusionModel


class NullProfiler:
    def step(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def progress_write(message: str, progress_bar=None) -> None:
    if progress_bar is not None:
        progress_bar.write(message)
    else:
        print(message)


def create_profiler(config: RunConfig, profiling_dir: Path):
    if not config.profiling.enabled:
        return NullProfiler()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.profiling.wait_steps,
            warmup=config.profiling.warmup_steps,
            active=config.profiling.active_steps,
            repeat=config.profiling.repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiling_dir)),
        record_shapes=config.profiling.record_shapes,
        profile_memory=config.profiling.profile_memory,
        with_stack=config.profiling.with_stack,
    )


def scale_gradients(model: TextDiffusionModel, scale: float) -> None:
    if scale == 1.0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def resolve_autocast_context(config: RunConfig, device: torch.device):
    if config.mixed_precision is None:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"mixed_precision={config.mixed_precision!r} is only supported on cuda devices")
    if config.mixed_precision == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("mixed_precision='bfloat16' was requested, but this CUDA device does not support bf16")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if config.mixed_precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(
        f"unsupported mixed_precision mode {config.mixed_precision!r}; expected None, 'bfloat16', or 'float16'"
    )


def gpu_memory_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "memory/gpu_allocated_mb": None,
            "memory/gpu_reserved_mb": None,
            "memory/gpu_max_allocated_mb": None,
            "memory/gpu_max_reserved_mb": None,
        }
    allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    max_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    return {
        "memory/gpu_allocated_mb": allocated_mb,
        "memory/gpu_reserved_mb": reserved_mb,
        "memory/gpu_max_allocated_mb": max_allocated_mb,
        "memory/gpu_max_reserved_mb": max_reserved_mb,
    }
