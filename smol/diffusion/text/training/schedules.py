from smol.diffusion.text.core.config import RunConfig


def learning_rate(config: RunConfig, completed_steps: int) -> float:
    if config.lr_schedule == "constant":
        return config.lr
    if config.lr_schedule != "wsd":
        raise ValueError(f"unsupported lr_schedule={config.lr_schedule!r}; expected 'constant' or 'wsd'")
    if config.max_steps <= 0:
        raise ValueError("lr_schedule='wsd' requires max_steps > 0")
    if config.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {config.warmup_steps}")
    if not 0.0 <= config.lr_stable_ratio <= 1.0:
        raise ValueError(f"lr_stable_ratio must be in [0, 1], got {config.lr_stable_ratio}")
    if not 0.0 <= config.min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {config.min_lr_ratio}")

    max_lr = config.lr
    min_lr = config.lr * config.min_lr_ratio
    if config.warmup_steps > 0 and completed_steps < config.warmup_steps:
        return max_lr * float(completed_steps + 1) / float(config.warmup_steps)

    stable_steps = int(config.max_steps * config.lr_stable_ratio)
    stable_end = config.warmup_steps + stable_steps
    if completed_steps < stable_end:
        return max_lr

    decay_steps = max(config.max_steps - stable_end, 1)
    decay_index = min(max(completed_steps - stable_end, 0), decay_steps)
    progress = min(max(float(decay_index) / float(decay_steps), 0.0), 1.0)
    return max_lr - progress * (max_lr - min_lr)
