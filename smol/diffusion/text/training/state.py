from dataclasses import dataclass
from datetime import datetime

import torch

from smol.diffusion.text.core.config import RunConfig
from smol.diffusion.text.core.model import TextDiffusionModel
from smol.diffusion.text.training.runtime import gpu_memory_stats, scale_gradients


@dataclass
class MicroBatchResult:
    batch: dict
    clean_input_ids: torch.Tensor
    token_mask: torch.Tensor
    sequence_lengths: list[list[int]] | None
    timesteps: torch.Tensor
    loss_value: float
    stats: dict[str, torch.Tensor]
    internals_metrics: dict[str, float]
    data_time_s: float
    forward_time_s: float
    backward_time_s: float


@dataclass
class PreviewState:
    batch: dict
    clean_input_ids: torch.Tensor
    token_mask: torch.Tensor
    sequence_lengths: list[list[int]] | None
    timesteps: torch.Tensor
    stats: dict[str, torch.Tensor]


@dataclass
class OptimizerStepReport:
    step: int
    metrics: dict
    preview_state: PreviewState


def epoch_progress(batch: dict) -> tuple[int, str]:
    source_steps_per_epoch = batch.get("source_steps_per_epoch")
    source_batches_consumed = batch.get("source_batches_consumed_in_epoch")
    if not isinstance(source_steps_per_epoch, int) or source_steps_per_epoch <= 0:
        return batch["epoch"], "unknown"
    if not isinstance(source_batches_consumed, int):
        return batch["epoch"], "unknown"
    progress = (
        100.0
        * min(source_batches_consumed, source_steps_per_epoch)
        / source_steps_per_epoch
    )
    return batch["epoch"], f"{progress:.1f}%"


class TrainAccumulator:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.reset()

    def reset(self) -> None:
        self.num_batches = 0
        self.data_time_s = 0.0
        self.forward_time_s = 0.0
        self.backward_time_s = 0.0
        self.loss = 0.0
        self.mask_rate = 0.0
        self.padding_fraction = 0.0
        self.timestep_sum = 0.0
        self.timestep_count = 0
        self.timestep_min: int | None = None
        self.timestep_max: int | None = None
        self.last_result: MicroBatchResult | None = None

    def add(self, result: MicroBatchResult) -> None:
        self.num_batches += 1
        self.data_time_s += result.data_time_s
        self.forward_time_s += result.forward_time_s
        self.backward_time_s += result.backward_time_s
        self.loss += result.loss_value
        mask_choice = result.stats.get("mask_choice")
        if mask_choice is None:
            mask_choice = result.stats["replace_mask"]
        self.mask_rate += mask_choice.float().mean().item()
        self.padding_fraction += float(result.batch["padding_fraction"])
        self.timestep_sum += result.timesteps.float().sum().item()
        self.timestep_count += result.timesteps.numel()
        batch_timestep_min = int(result.timesteps.min().item())
        batch_timestep_max = int(result.timesteps.max().item())
        self.timestep_min = (
            batch_timestep_min
            if self.timestep_min is None
            else min(self.timestep_min, batch_timestep_min)
        )
        self.timestep_max = (
            batch_timestep_max
            if self.timestep_max is None
            else max(self.timestep_max, batch_timestep_max)
        )
        self.last_result = result

    def has_pending(self) -> bool:
        return self.num_batches > 0

    def ready(self, grad_accum_steps: int) -> bool:
        return self.num_batches >= grad_accum_steps

    def scale_partial_gradients(
        self, model: TextDiffusionModel, grad_accum_steps: int
    ) -> None:
        if self.num_batches > 0:
            scale_gradients(model, grad_accum_steps / self.num_batches)

    def build_report(
        self,
        *,
        step: int,
        lr: float,
        config: RunConfig,
        tokenizer,
        device: torch.device,
    ) -> OptimizerStepReport:
        if self.last_result is None or self.num_batches <= 0:
            raise RuntimeError(
                "cannot build optimizer step report without accumulated batches"
            )
        last = self.last_result
        epoch, epoch_progress_text = epoch_progress(last.batch)
        epoch_progress_pct = (
            None
            if epoch_progress_text == "unknown"
            else float(epoch_progress_text[:-1])
        )
        metrics = {
            "step": step,
            "epoch": epoch,
            "epoch_progress": epoch_progress_text,
            "epoch_progress_pct": epoch_progress_pct,
            "loss": self.loss / self.num_batches,
            "mask_rate": self.mask_rate / self.num_batches,
            "corruption_rate": self.mask_rate / self.num_batches,
            "padding_fraction": self.padding_fraction / self.num_batches,
            "diffusion_timestep_mean": self.timestep_sum / max(self.timestep_count, 1),
            "diffusion_timestep_min": self.timestep_min,
            "diffusion_timestep_max": self.timestep_max,
            "timing/data_loading_s": self.data_time_s,
            "timing/forward_s": self.forward_time_s,
            "timing/backward_s": self.backward_time_s,
            "optimization/lr": lr,
            "optimization/lr_schedule": config.lr_schedule,
            "optimization/grad_accum_steps": self.num_batches,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **gpu_memory_stats(device),
            **last.internals_metrics,
        }
        return OptimizerStepReport(
            step=step,
            metrics=metrics,
            preview_state=PreviewState(
                batch=last.batch,
                clean_input_ids=last.clean_input_ids,
                token_mask=last.token_mask,
                sequence_lengths=last.sequence_lengths,
                timesteps=last.timesteps,
                stats=last.stats,
            ),
        )
