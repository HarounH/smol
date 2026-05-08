from typing import Any

import torch


class AdamW:
    """
    Minimal AdamW optimizer used to avoid importing torch.optim in environments
    where the bundled TorchDynamo install is broken.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = [param for param in params if param.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = self.betas
        self.step_count += 1

        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("sparse gradients are not supported by this AdamW implementation")

            state = self.state.setdefault(
                id(param),
                {
                    "exp_avg": torch.zeros_like(param),
                    "exp_avg_sq": torch.zeros_like(param),
                },
            )
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if self.weight_decay != 0.0:
                param.mul_(1.0 - self.lr * self.weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**self.step_count
            bias_correction2 = 1.0 - beta2**self.step_count
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)

    def state_dict(self) -> dict[str, Any]:
        serialized_state = {
            param_id: {
                "exp_avg": state["exp_avg"].detach().cpu(),
                "exp_avg_sq": state["exp_avg_sq"].detach().cpu(),
            }
            for param_id, state in self.state.items()
        }
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_count": self.step_count,
            "state": serialized_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lr = float(state_dict["lr"])
        self.betas = tuple(state_dict["betas"])
        self.eps = float(state_dict["eps"])
        self.weight_decay = float(state_dict["weight_decay"])
        self.step_count = int(state_dict["step_count"])

        serialized_state = state_dict.get("state", {})
        self.state = {}
        for param, (param_id, param_state) in zip(self.params, serialized_state.items()):
            self.state[id(param)] = {
                "exp_avg": param_state["exp_avg"].to(device=param.device, dtype=param.dtype),
                "exp_avg_sq": param_state["exp_avg_sq"].to(device=param.device, dtype=param.dtype),
            }
