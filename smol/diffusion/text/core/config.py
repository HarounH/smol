from dataclasses import dataclass, field

from smol.diffusion.text.data import TextDataConfig
from smol.diffusion.text.core.model import TextDiffusionConfig


@dataclass
class LoggingConfig:
    log_every: int = 10
    preview_corruption_every: int = 0
    preview_tokens: int = 32
    preview_num_samples: int = 5
    wandb_enabled: bool = False
    wandb_project: str = "smol-text-diffusion"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: list[str] | None = None


@dataclass
class ProfilingConfig:
    enabled: bool = False
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False


@dataclass
class RunConfig:
    data: TextDataConfig = field(default_factory=TextDataConfig)
    model: TextDiffusionConfig = field(default_factory=TextDiffusionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    experiment_name: str = "default"
    output_root: str = "runs"
    resume_date: str | None = None
    seed: int = 1337
    epochs: int = 1
    mixed_precision: str | None = None
    grad_accum_steps: int = 1
    max_steps: int = 0
    checkpoint_every_steps: int = 0
    lr: float = 3e-4
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    lr_stable_ratio: float = 0.4
    min_lr_ratio: float = 0.025
    weight_decay: float = 0.01
    device: str | None = None
