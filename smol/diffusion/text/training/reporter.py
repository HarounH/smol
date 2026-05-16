from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from smol.diffusion.text.core.config import RunConfig
from smol.diffusion.text.data import ResumableTextDataLoader
from smol.diffusion.text.core.model import TextDiffusionConfig, TextDiffusionModel
from smol.diffusion.text.core.optimizer import AdamW
from smol.diffusion.text.training.runtime import (
    progress_write,
    resolve_autocast_context,
)
from smol.diffusion.text.sample import sample as sample_text_diffusion
from smol.diffusion.text.training.run import append_jsonl, save_checkpoint
from smol.diffusion.text.training.state import OptimizerStepReport, PreviewState

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(
    run_config: RunConfig,
    model_config: TextDiffusionConfig,
    device: torch.device,
    run_dir: Path,
):
    if not run_config.logging.wandb_enabled:
        return None
    if wandb is None:
        raise ImportError(
            "wandb logging was requested, but the 'wandb' package is not installed"
        )
    return wandb.init(
        project=run_config.logging.wandb_project,
        entity=run_config.logging.wandb_entity,
        name=run_config.logging.wandb_name,
        tags=run_config.logging.wandb_tags,
        config={
            **asdict(run_config),
            "device": str(device),
            "resolved_model_config": asdict(model_config),
            "run_dir": str(run_dir),
        },
    )


def compact_text(tokenizer, token_ids: list[int]) -> str:
    return repr(tokenizer.decode(token_ids, skip_special_tokens=False))


def truncate_cell(text: str, width: int) -> str:
    if len(text) <= width:
        return text.ljust(width)
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def set_train_progress_postfix(
    progress_bar,
    *,
    step: int,
    loss: float,
    padding_fraction: float,
    mask_rate: float,
    timestep_mean: float,
    timestep_min: int | None,
    timestep_max: int | None,
    epoch: int,
    epoch_progress: str,
    lr: float,
    accum_batches: int,
    data_time_s: float,
    forward_time_s: float,
    backward_time_s: float,
    gpu_mem: str | None = None,
) -> None:
    parts = [
        f"l={loss:.4f}",
        f"p={padding_fraction * 100:.1f}%",
        f"m={mask_rate * 100:.1f}%",
        f"t={timestep_mean:.1f}[{timestep_min}-{timestep_max}]",
        f"s={step}",
        f"lr={lr:.2e}",
        f"e={epoch}:{epoch_progress}",
        f"a={accum_batches}",
        f"d={data_time_s:.1f}s",
        f"f={forward_time_s:.1f}s",
        f"b={backward_time_s:.1f}s",
    ]
    if gpu_mem is not None:
        parts.append(f"g={gpu_mem}")
    progress_bar.set_postfix_str(", ".join(parts), refresh=False)


def print_mask_preview(
    *,
    logger,
    tokenizer,
    train_step: int,
    clean_input_ids: torch.Tensor,
    masked_input_ids: torch.Tensor,
    predicted_input_ids: torch.Tensor,
    rollout_input_ids: torch.Tensor,
    mask_choice: torch.Tensor,
    timesteps: torch.Tensor,
    sample_index: int,
    max_tokens: int,
    progress_bar=None,
) -> None:
    clean_ids = clean_input_ids[sample_index].detach().cpu().tolist()[:max_tokens]
    masked_ids = masked_input_ids[sample_index].detach().cpu().tolist()[:max_tokens]
    predicted_ids = (
        predicted_input_ids[sample_index].detach().cpu().tolist()[:max_tokens]
    )
    rollout_ids = rollout_input_ids[sample_index].detach().cpu().tolist()[:max_tokens]
    mask = mask_choice[sample_index].detach().cpu().tolist()[:max_tokens]
    timestep = int(timesteps[sample_index].item())

    masked_count = sum(bool(changed) for changed in mask)
    masked_positions = [str(index) for index, changed in enumerate(mask) if changed]
    positions_preview = ",".join(masked_positions[:12])
    if len(masked_positions) > 12:
        positions_preview += ",..."

    clean_text = compact_text(tokenizer, clean_ids)
    masked_text = compact_text(tokenizer, masked_ids)
    predicted_text = compact_text(tokenizer, predicted_ids)
    rollout_text = compact_text(tokenizer, rollout_ids)
    masked_count_text = f"{masked_count}/{len(mask)} [{positions_preview}]"
    timestamp = datetime.now().isoformat(timespec="seconds")
    label_width = 8
    value_width = 72
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"

    progress_write(border, progress_bar)
    progress_write(
        f"| {'time':<{label_width}} | {truncate_cell(timestamp, value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'step':<{label_width}} | {truncate_cell(str(train_step), value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'timestep':<{label_width}} | {truncate_cell(str(timestep), value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'sample':<{label_width}} | {truncate_cell(str(sample_index), value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'clean':<{label_width}} | {truncate_cell(clean_text, value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'masked':<{label_width}} | {truncate_cell(masked_text, value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'fwd':<{label_width}} | {truncate_cell(predicted_text, value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'rollout':<{label_width}} | {truncate_cell(rollout_text, value_width)} |",
        progress_bar,
    )
    progress_write(
        f"| {'mask_cnt':<{label_width}} | {truncate_cell(masked_count_text, value_width)} |",
        progress_bar,
    )
    progress_write(border, progress_bar)
    logger.info(
        "mask_preview",
        train_step=train_step,
        timestep=timestep,
        sample_index=sample_index,
        clean_text=clean_text,
        masked_text=masked_text,
        predicted_text=predicted_text,
        rollout_text=rollout_text,
        masked_count=masked_count,
        sequence_length=len(mask),
        masked_positions=masked_positions,
    )


def masked_input_ids_from_stats(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    masked_input_ids = stats.get("masked_input_ids")
    if masked_input_ids is None:
        masked_input_ids = stats.get("masked_tokens")
    if masked_input_ids is None:
        masked_input_ids = stats["noisy_input_ids"]
    return masked_input_ids


def mask_choice_from_stats(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mask_choice = stats.get("mask_choice")
    if mask_choice is None:
        mask_choice = stats["replace_mask"]
    return mask_choice


class TrainReporter:
    def __init__(
        self,
        *,
        config: RunConfig,
        run_dirs: dict[str, Path],
        logger,
        wandb_run,
        progress_bar,
        tokenizer,
        device: torch.device,
    ):
        self.config = config
        self.run_dirs = run_dirs
        self.logger = logger
        self.wandb_run = wandb_run
        self.progress_bar = progress_bar
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_path = run_dirs["logs"] / "train_metrics.jsonl"

    def on_step(
        self, report: OptimizerStepReport, *, advance_progress: bool = True
    ) -> None:
        metrics = report.metrics
        append_jsonl(self.metrics_path, metrics)
        if (
            self.config.logging.log_every > 0
            and report.step % self.config.logging.log_every == 0
        ):
            self.logger.info("train_step", **metrics)
        self._log_wandb_metrics(metrics)
        if advance_progress:
            self.progress_bar.update(1)
        self._update_progress(metrics)

    def on_partial_step(self, report: OptimizerStepReport) -> None:
        self.logger.info(
            "partial_grad_accumulation_step",
            step=report.step,
            grad_accum_steps=report.metrics["optimization/grad_accum_steps"],
        )

    def maybe_preview(
        self, *, step: int, model: TextDiffusionModel, preview_state: PreviewState
    ) -> None:
        if self.config.logging.preview_corruption_every <= 0:
            return
        if step % self.config.logging.preview_corruption_every != 0:
            return

        stats = preview_state.stats
        predicted_input_ids = stats["logits"].argmax(dim=-1)
        num_samples = min(
            self.config.logging.preview_num_samples,
            preview_state.clean_input_ids.size(0),
        )
        masked_input_ids = masked_input_ids_from_stats(stats)
        mask_choice = mask_choice_from_stats(stats)
        preview_masked_input_ids = masked_input_ids[:num_samples]
        preview_timesteps = preview_state.timesteps[:num_samples]
        was_training = model.training
        model.eval()
        try:
            with resolve_autocast_context(self.config, self.device):
                rollout_input_ids = sample_text_diffusion(
                    model,
                    preview_masked_input_ids.clone(),
                    mask_choice[:num_samples],
                    self.device,
                    start_timesteps=preview_timesteps,
                    token_mask=preview_state.token_mask[:num_samples],
                    sequence_lengths=(
                        None
                        if preview_state.sequence_lengths is None
                        else preview_state.sequence_lengths[:num_samples]
                    ),
                )
        finally:
            if was_training:
                model.train()

        for sample_index in range(num_samples):
            print_mask_preview(
                logger=self.logger.bind(component="preview"),
                tokenizer=self.tokenizer,
                train_step=step,
                clean_input_ids=preview_state.clean_input_ids,
                masked_input_ids=masked_input_ids,
                predicted_input_ids=predicted_input_ids,
                rollout_input_ids=rollout_input_ids,
                mask_choice=mask_choice,
                timesteps=preview_state.timesteps,
                sample_index=sample_index,
                max_tokens=self.config.logging.preview_tokens,
                progress_bar=self.progress_bar,
            )
        self._log_wandb_preview(
            step, preview_state, predicted_input_ids, rollout_input_ids, num_samples
        )

    def maybe_checkpoint(
        self,
        *,
        step: int,
        model: TextDiffusionModel,
        optimizer: AdamW,
        dataloader: ResumableTextDataLoader,
    ) -> None:
        if (
            self.config.checkpoint_every_steps <= 0
            or step % self.config.checkpoint_every_steps != 0
        ):
            return
        checkpoint_path = self.run_dirs["checkpoints"] / f"step_{step:08d}.pt"
        save_checkpoint(
            checkpoint_path, self.config, model, optimizer, dataloader, step
        )
        progress_write(f"saved checkpoint to {checkpoint_path}", self.progress_bar)
        self.logger.info(
            "checkpoint_saved", checkpoint_path=str(checkpoint_path), step=step
        )
        if self.wandb_run is not None:
            wandb.log(
                {"checkpoint/step": step, "checkpoint/path": str(checkpoint_path)},
                step=step,
            )

    def finish(
        self, *, final_checkpoint_path: Path, step: int, exit_reason: str
    ) -> None:
        if self.wandb_run is None:
            return
        wandb.log(
            {
                "checkpoint/step": step,
                "checkpoint/path": str(final_checkpoint_path),
                "training/exit_reason": exit_reason,
            },
            step=step,
        )
        wandb.finish()

    def _update_progress(self, metrics: dict) -> None:
        gpu_mem_postfix = (
            "cpu"
            if metrics["memory/gpu_allocated_mb"] is None
            else f"{metrics['memory/gpu_allocated_mb']:.0f}MB"
        )
        set_train_progress_postfix(
            self.progress_bar,
            step=metrics["step"],
            loss=metrics["loss"],
            padding_fraction=metrics["padding_fraction"],
            mask_rate=metrics.get("mask_rate", metrics.get("corruption_rate", 0.0)),
            timestep_mean=metrics["diffusion_timestep_mean"],
            timestep_min=metrics["diffusion_timestep_min"],
            timestep_max=metrics["diffusion_timestep_max"],
            epoch=metrics["epoch"],
            epoch_progress=metrics["epoch_progress"],
            lr=metrics["optimization/lr"],
            accum_batches=metrics["optimization/grad_accum_steps"],
            data_time_s=metrics["timing/data_loading_s"],
            forward_time_s=metrics["timing/forward_s"],
            backward_time_s=metrics["timing/backward_s"],
            gpu_mem=gpu_mem_postfix,
        )

    def _log_wandb_metrics(self, metrics: dict) -> None:
        if self.wandb_run is None:
            return
        train_prefixed = {
            f"train/{name}": value
            for name, value in metrics.items()
            if name.startswith("internals/")
        }
        wandb.log(
            {
                "train/loss": metrics["loss"],
                "train/mask_rate": metrics.get(
                    "mask_rate", metrics.get("corruption_rate", 0.0)
                ),
                "train/corruption_rate": metrics.get(
                    "corruption_rate", metrics.get("mask_rate", 0.0)
                ),
                "train/padding_fraction": metrics["padding_fraction"],
                "train/epoch": metrics["epoch"],
                "train/epoch_progress_pct": metrics["epoch_progress_pct"],
                "train/step": metrics["step"],
                "train/diffusion_timestep_mean": metrics["diffusion_timestep_mean"],
                "train/diffusion_timestep_min": metrics["diffusion_timestep_min"],
                "train/diffusion_timestep_max": metrics["diffusion_timestep_max"],
                "optimization/lr": metrics["optimization/lr"],
                "optimization/grad_accum_steps": metrics[
                    "optimization/grad_accum_steps"
                ],
                "timing/data_loading_s": metrics["timing/data_loading_s"],
                "timing/forward_s": metrics["timing/forward_s"],
                "timing/backward_s": metrics["timing/backward_s"],
                "memory/gpu_allocated_mb": metrics["memory/gpu_allocated_mb"],
                "memory/gpu_reserved_mb": metrics["memory/gpu_reserved_mb"],
                "memory/gpu_max_allocated_mb": metrics["memory/gpu_max_allocated_mb"],
                "memory/gpu_max_reserved_mb": metrics["memory/gpu_max_reserved_mb"],
                **train_prefixed,
            },
            step=metrics["step"],
        )

    def _log_wandb_preview(
        self,
        step: int,
        preview_state: PreviewState,
        predicted_input_ids: torch.Tensor,
        rollout_input_ids: torch.Tensor,
        num_samples: int,
    ) -> None:
        if self.wandb_run is None:
            return
        preview_rows = []
        stats = preview_state.stats
        for sample_index in range(num_samples):
            clean_ids = (
                preview_state.clean_input_ids[sample_index]
                .detach()
                .cpu()
                .tolist()[: self.config.logging.preview_tokens]
            )
            masked_input_ids = masked_input_ids_from_stats(stats)
            masked_ids = (
                masked_input_ids[sample_index]
                .detach()
                .cpu()
                .tolist()[: self.config.logging.preview_tokens]
            )
            pred_ids = (
                predicted_input_ids[sample_index]
                .detach()
                .cpu()
                .tolist()[: self.config.logging.preview_tokens]
            )
            rollout_ids = (
                rollout_input_ids[sample_index]
                .detach()
                .cpu()
                .tolist()[: self.config.logging.preview_tokens]
            )
            preview_rows.append(
                [
                    step,
                    int(preview_state.timesteps[sample_index].item()),
                    sample_index,
                    self.tokenizer.decode(clean_ids, skip_special_tokens=False),
                    self.tokenizer.decode(masked_ids, skip_special_tokens=False),
                    self.tokenizer.decode(pred_ids, skip_special_tokens=False),
                    self.tokenizer.decode(rollout_ids, skip_special_tokens=False),
                ]
            )
        wandb.log(
            {
                "preview/samples": wandb.Table(
                    columns=[
                        "train_step",
                        "diffusion_timestep",
                        "sample_index",
                        "clean",
                        "masked",
                        "fwd_cleaned",
                        "rollout_cleaned",
                    ],
                    data=preview_rows,
                )
            },
            step=step,
        )
