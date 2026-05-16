import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from smol.diffusion.text.core.config import RunConfig
from smol.diffusion.text.data import ResumableTextDataLoader
from smol.diffusion.text.core.model import TextDiffusionModel
from smol.diffusion.text.core.optimizer import AdamW


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def prepare_run_dirs(config: RunConfig) -> dict[str, Path]:
    run_stamp = config.resume_date or now_timestamp()
    run_root = Path(config.output_root) / config.experiment_name / run_stamp
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    profiling_dir = run_root / "profiling"
    for path in [checkpoints_dir, logs_dir, profiling_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "run_stamp": Path(run_stamp),
        "checkpoints": checkpoints_dir,
        "logs": logs_dir,
        "profiling": profiling_dir,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def global_batch_size(config: RunConfig) -> int:
    return (
        config.data.batch_size * config.grad_accum_steps * config.data.sequence_length
    )


def find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    checkpoint_paths = sorted(checkpoints_dir.glob("*.pt"))
    if not checkpoint_paths:
        return None

    latest_checkpoint: Path | None = None
    latest_step = -1
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        step = int(checkpoint.get("step", -1))
        if step >= latest_step:
            latest_step = step
            latest_checkpoint = checkpoint_path
    return latest_checkpoint


def save_checkpoint(
    checkpoint_path: Path,
    run_config: RunConfig,
    model: TextDiffusionModel,
    optimizer: AdamW,
    dataloader: ResumableTextDataLoader,
    step: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dataloader": dataloader.state_dict(),
            "step": step,
            "run_config": asdict(run_config),
            "model_config": model.config,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: TextDiffusionModel,
    optimizer: AdamW,
    dataloader: ResumableTextDataLoader,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    dataloader.load_state_dict(checkpoint["dataloader"])
    return int(checkpoint["step"])
