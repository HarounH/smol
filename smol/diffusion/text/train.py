import argparse
import importlib
import logging
import time
from dataclasses import asdict, replace
from datetime import datetime

import torch
from tqdm.auto import tqdm

from smol.diffusion.text.core.config import RunConfig
from smol.diffusion.text.data import ResumableTextDataLoader
from smol.diffusion.text.training.logging import (
    get_logger,
    init_logging,
    shutdown_logging,
)
from smol.diffusion.text.core.model import TextDiffusionConfig, TextDiffusionModel
from smol.diffusion.text.training.internals import ModelInternalsLogger
from smol.diffusion.text.core.optimizer import AdamW
from smol.diffusion.text.training.runtime import (
    create_profiler,
    progress_write,
    resolve_autocast_context,
)
from smol.diffusion.text.training.schedules import learning_rate
from smol.diffusion.text.training.reporter import TrainReporter, init_wandb
from smol.diffusion.text.training.run import (
    find_latest_checkpoint,
    global_batch_size,
    load_checkpoint,
    prepare_run_dirs,
    save_checkpoint,
    write_json,
)
from smol.diffusion.text.training.state import MicroBatchResult, TrainAccumulator

DEFAULT_CONFIG_FN = "smol.diffusion.text.experiments.scale_300m_103:make_config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the text diffusion model from a config factory."
    )
    parser.add_argument(
        "config_fn",
        nargs="?",
        default=DEFAULT_CONFIG_FN,
        help="Function reference in the form 'package.module:function_name' that returns a RunConfig.",
    )
    return parser.parse_args()


def load_run_config(function_path: str) -> RunConfig:
    module_name, separator, function_name = function_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "config function path must use the form 'package.module:function_name'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    config = factory()
    if not isinstance(config, RunConfig):
        raise TypeError(
            f"{function_path} returned {type(config).__name__}, expected RunConfig"
        )
    return config


def _run_micro_batch(
    *,
    batch: dict,
    model: TextDiffusionModel,
    model_config: TextDiffusionConfig,
    config: RunConfig,
    device: torch.device,
    internals_logger: ModelInternalsLogger,
    data_time_s: float,
) -> MicroBatchResult:
    clean_input_ids = batch["input_ids"].to(device)
    token_mask = batch["token_mask"].to(device)
    sequence_lengths = batch.get("sequence_lengths")
    timesteps = torch.randint(
        low=0,
        high=model_config.num_diffusion_steps + 1,
        size=(clean_input_ids.size(0),),
        device=device,
    )

    internals_logger.begin_step()
    forward_start = time.perf_counter()
    with resolve_autocast_context(config, device):
        loss, stats = model.loss(
            clean_input_ids,
            timesteps,
            token_mask=token_mask,
            sequence_lengths=sequence_lengths,
        )
    forward_time_s = time.perf_counter() - forward_start

    backward_start = time.perf_counter()
    loss_value = loss.item()
    (loss / config.grad_accum_steps).backward()
    backward_time_s = time.perf_counter() - backward_start

    return MicroBatchResult(
        batch=batch,
        clean_input_ids=clean_input_ids,
        token_mask=token_mask,
        sequence_lengths=sequence_lengths,
        timesteps=timesteps,
        loss_value=loss_value,
        stats=stats,
        internals_metrics=internals_logger.finalize_step(),
        data_time_s=data_time_s,
        forward_time_s=forward_time_s,
        backward_time_s=backward_time_s,
    )


def _optimizer_step(
    *, optimizer: AdamW, profiler, config: RunConfig, step: int
) -> float:
    current_lr = learning_rate(config, step)
    optimizer.lr = current_lr
    optimizer.step()
    profiler.step()
    optimizer.zero_grad(set_to_none=True)
    return current_lr


def main(run_config: RunConfig | None = None) -> None:
    print("loading run config...", flush=True)
    config = run_config or load_run_config(parse_args().config_fn)
    if config.grad_accum_steps <= 0:
        raise ValueError(
            f"grad_accum_steps must be positive, got {config.grad_accum_steps}"
        )
    torch.manual_seed(config.seed)

    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    config.data.base_seed = config.seed
    print(f"preparing run dirs for {config.experiment_name}...", flush=True)
    run_dirs = prepare_run_dirs(config)
    init_logging(
        run_dirs["logs"] / "events.jsonl",
        level=logging.INFO,
        experiment_name=config.experiment_name,
        run_dir=str(run_dirs["run_root"]),
    )
    logger = get_logger("train")
    try:
        print("initializing dataloader/tokenizer...", flush=True)
        dataloader = ResumableTextDataLoader(config.data)
        model_config = replace(config.model)
        print(f"building model on {device}...", flush=True)
        model = TextDiffusionModel(model_config, dataloader.tokenizer).to(device)
        print("initializing optimizer/loggers...", flush=True)
        internals_logger = ModelInternalsLogger(model)
        optimizer = AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        wandb_run = init_wandb(config, model_config, device, run_dirs["run_root"])
        print("checking for checkpoints...", flush=True)
        latest_checkpoint = find_latest_checkpoint(run_dirs["checkpoints"])
        resumed_from: str | None = None
        start_step = 0
        if latest_checkpoint is not None:
            start_step = load_checkpoint(
                latest_checkpoint, model, optimizer, dataloader, device
            )
            resumed_from = str(latest_checkpoint)
            logger.info(
                "checkpoint_resumed",
                checkpoint_path=str(latest_checkpoint),
                step=start_step,
            )
            print(
                f"resumed from checkpoint {latest_checkpoint} at step={start_step}",
                flush=True,
            )

        write_json(run_dirs["logs"] / "run_config.json", asdict(config))
        resolved_global_batch_size = global_batch_size(config)
        write_json(
            run_dirs["run_root"] / "run_summary.json",
            {
                "experiment_name": config.experiment_name,
                "launched_at": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dirs["run_root"]),
                "device": str(device),
                "mixed_precision": config.mixed_precision,
                "global_batch_size": resolved_global_batch_size,
                "resumed_from": resumed_from,
                "resume_date": config.resume_date,
                "config": asdict(config),
                "resolved_model_config": asdict(model_config),
            },
        )

        print(
            f"training on {device} | vocab_size={model.vocab_size} "
            f"| batch_size={config.data.batch_size} | grad_accum_steps={config.grad_accum_steps} "
            f"| mixed_precision={config.mixed_precision or 'none'} "
            f"| global_batch_size={resolved_global_batch_size} "
            f"| sequence_length={config.data.sequence_length} "
            f"| lr_schedule={config.lr_schedule} "
            f"| lr={config.lr:.2e} "
            f"| run_dir={run_dirs['run_root']}",
            flush=True,
        )
        logger.info(
            "training_started",
            device=str(device),
            vocab_size=model.vocab_size,
            batch_size=config.data.batch_size,
            grad_accum_steps=config.grad_accum_steps,
            mixed_precision=config.mixed_precision,
            global_batch_size=resolved_global_batch_size,
            sequence_length=config.data.sequence_length,
            lr=config.lr,
            lr_schedule=config.lr_schedule,
            warmup_steps=config.warmup_steps,
            lr_stable_ratio=config.lr_stable_ratio,
            min_lr_ratio=config.min_lr_ratio,
            resumed_from=resumed_from,
        )

        step = start_step
        model.train()
        tokenizer = dataloader.tokenizer
        data_wait_start = time.perf_counter()
        progress_total = (
            None if config.max_steps <= 0 else max(config.max_steps - start_step, 0)
        )
        exit_reason = "epochs_exhausted"
        exit_details: dict[str, int | str | None] = {
            "configured_epochs": config.epochs,
            "configured_max_steps": config.max_steps,
            "final_epoch": None,
            "final_source_batches_consumed_in_epoch": None,
            "final_dataloader_global_step": None,
        }
        progress_bar_format = "{desc}: {n_fmt}/{total_fmt} {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        with tqdm(
            total=progress_total,
            initial=0,
            desc="train",
            dynamic_ncols=False,
            bar_format=progress_bar_format,
        ) as progress_bar:
            with create_profiler(config, run_dirs["profiling"]) as profiler:
                optimizer.zero_grad(set_to_none=True)
                accumulator = TrainAccumulator(model.vocab_size)
                reporter = TrainReporter(
                    config=config,
                    run_dirs=run_dirs,
                    logger=logger,
                    wandb_run=wandb_run,
                    progress_bar=progress_bar,
                    tokenizer=tokenizer,
                    device=device,
                )

                warned_flex_compile = False
                for batch in dataloader.iter_epochs(config.epochs):
                    if (
                        not warned_flex_compile
                        and batch.get("sequence_lengths") is not None
                    ):
                        progress_write(
                            "dense packing enabled: using flash-attn varlen attention for document segments",
                            progress_bar,
                        )
                        warned_flex_compile = True
                    accumulator.add(
                        _run_micro_batch(
                            batch=batch,
                            model=model,
                            model_config=model_config,
                            config=config,
                            device=device,
                            internals_logger=internals_logger,
                            data_time_s=time.perf_counter() - data_wait_start,
                        )
                    )

                    if not accumulator.ready(config.grad_accum_steps):
                        data_wait_start = time.perf_counter()
                        continue

                    current_lr = _optimizer_step(
                        optimizer=optimizer, profiler=profiler, config=config, step=step
                    )
                    step += 1
                    report = accumulator.build_report(
                        step=step,
                        lr=current_lr,
                        config=config,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    reporter.on_step(report)
                    reporter.maybe_preview(
                        step=step, model=model, preview_state=report.preview_state
                    )
                    reporter.maybe_checkpoint(
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        dataloader=dataloader,
                    )

                    if config.max_steps > 0 and step >= config.max_steps:
                        exit_reason = "max_steps_reached"
                        exit_details.update(
                            {
                                "configured_max_steps": config.max_steps,
                                "final_epoch": report.preview_state.batch["epoch"],
                                "final_source_batches_consumed_in_epoch": report.preview_state.batch[
                                    "source_batches_consumed_in_epoch"
                                ],
                                "final_dataloader_global_step": report.preview_state.batch[
                                    "global_step"
                                ],
                            }
                        )
                        break

                    accumulator.reset()
                    data_wait_start = time.perf_counter()

                if accumulator.has_pending() and (
                    config.max_steps <= 0 or step < config.max_steps
                ):
                    accumulator.scale_partial_gradients(model, config.grad_accum_steps)
                    current_lr = _optimizer_step(
                        optimizer=optimizer, profiler=profiler, config=config, step=step
                    )
                    step += 1
                    report = accumulator.build_report(
                        step=step,
                        lr=current_lr,
                        config=config,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    reporter.on_step(report, advance_progress=False)
                    reporter.on_partial_step(report)

        if exit_reason != "max_steps_reached":
            exit_details.update(
                {
                    "final_epoch": dataloader.epoch,
                    "final_source_batches_consumed_in_epoch": dataloader.source_batches_consumed_in_epoch,
                    "final_dataloader_global_step": dataloader.global_step,
                }
            )
            if config.max_steps > 0 and step >= config.max_steps:
                exit_reason = "max_steps_reached"
            else:
                exit_reason = "epochs_exhausted"

        final_checkpoint_path = run_dirs["checkpoints"] / "final.pt"
        save_checkpoint(
            final_checkpoint_path, config, model, optimizer, dataloader, step
        )
        progress_write(f"saved checkpoint to {final_checkpoint_path}")
        progress_write(f"training finished: reason={exit_reason} step={step}")
        logger.info(
            "training_finished",
            final_checkpoint_path=str(final_checkpoint_path),
            step=step,
            reason=exit_reason,
            **exit_details,
        )
        if "reporter" in locals():
            reporter.finish(
                final_checkpoint_path=final_checkpoint_path,
                step=step,
                exit_reason=exit_reason,
            )
    except Exception:
        logger.exception("training_failed")
        raise
    finally:
        if "internals_logger" in locals():
            internals_logger.close()
        shutdown_logging()


if __name__ == "__main__":
    main()
