"""Text diffusion package.

The implementation is grouped into subpackages (`core`, `data`, `training`,
`analysis`). Legacy module paths are registered here so older notebooks and
scripts that import modules such as `smol.diffusion.text.model` keep working.
"""

from importlib import import_module
import sys

_LEGACY_MODULE_ALIASES = {
    "analysis_utils": "smol.diffusion.text.analysis",
    "config": "smol.diffusion.text.core.config",
    "logging_utils": "smol.diffusion.text.training.logging",
    "model": "smol.diffusion.text.core.model",
    "model_internals": "smol.diffusion.text.training.internals",
    "optimizer": "smol.diffusion.text.core.optimizer",
    "runtime": "smol.diffusion.text.training.runtime",
    "schedules": "smol.diffusion.text.training.schedules",
    "tokenizer": "smol.diffusion.text.core.tokenizer",
    "training_reporter": "smol.diffusion.text.training.reporter",
    "training_run": "smol.diffusion.text.training.run",
    "training_state": "smol.diffusion.text.training.state",
}

for legacy_name, target_name in _LEGACY_MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{legacy_name}"] = import_module(target_name)
