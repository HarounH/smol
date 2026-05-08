import torch
import torch.nn.functional as F

from smol.diffusion.text.core.model import TextDiffusionModel


def tensor_summary_metrics(name: str, tensor: torch.Tensor | None) -> dict[str, float]:
    if tensor is None:
        return {}
    values = tensor.detach()
    if values.numel() == 0:
        return {}
    values = values.float()
    abs_values = values.abs()
    return {
        f"{name}/norm": torch.linalg.vector_norm(values).item(),
        f"{name}/abs_max": abs_values.max().item(),
        f"{name}/abs_mean": abs_values.mean().item(),
    }


class ModelInternalsLogger:
    def __init__(self, model: TextDiffusionModel):
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._metrics: dict[str, float] = {}
        self._register_hooks()

    def _record_tensor(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or not torch.is_tensor(tensor):
            return
        self._metrics.update(tensor_summary_metrics(name, tensor))
        if tensor.requires_grad:
            tensor.register_hook(lambda grad, metric_name=name: self._record_grad(metric_name, grad))

    def _record_grad(self, name: str, grad: torch.Tensor | None) -> None:
        self._metrics.update(tensor_summary_metrics(f"{name}/grad", grad))

    def _register_hooks(self) -> None:
        encoder_layers = getattr(self.model.encoder, "layers", [])
        for layer_index, layer in enumerate(encoder_layers):
            layer_prefix = f"internals/layer_{layer_index}"

            def layer_pre_hook(module, inputs, prefix=layer_prefix):
                if inputs:
                    self._record_tensor(f"{prefix}/residual_in", inputs[0])

            def layer_forward_hook(module, inputs, output, prefix=layer_prefix):
                self._record_tensor(f"{prefix}/residual_out", output)

            def attn_pre_hook(module, inputs, prefix=layer_prefix):
                if not inputs:
                    return
                q_input = inputs[0]
                if not torch.is_tensor(q_input):
                    return
                self._record_tensor(f"{prefix}/q_input", q_input)
                if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                    q_weight, k_weight, v_weight = module.in_proj_weight.chunk(3, dim=0)
                    if module.in_proj_bias is None:
                        q_bias = k_bias = v_bias = None
                    else:
                        q_bias, k_bias, v_bias = module.in_proj_bias.chunk(3, dim=0)
                    self._record_tensor(f"{prefix}/q_proj", F.linear(q_input, q_weight, q_bias))
                    self._record_tensor(f"{prefix}/k_proj", F.linear(q_input, k_weight, k_bias))
                    self._record_tensor(f"{prefix}/v_proj", F.linear(q_input, v_weight, v_bias))
                    return
                if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
                    self._record_tensor(f"{prefix}/q_proj", module.q_proj(q_input))
                    self._record_tensor(f"{prefix}/k_proj", module.k_proj(q_input))
                    self._record_tensor(f"{prefix}/v_proj", module.v_proj(q_input))

            def attn_forward_hook(module, inputs, output, prefix=layer_prefix):
                attn_output = output[0] if isinstance(output, tuple) else output
                self._record_tensor(f"{prefix}/attention_out", attn_output)

            def mlp_forward_hook(module, inputs, output, prefix=layer_prefix):
                self._record_tensor(f"{prefix}/mlp_out", output)

            self._handles.append(layer.register_forward_pre_hook(layer_pre_hook))
            self._handles.append(layer.register_forward_hook(layer_forward_hook))
            self._handles.append(layer.self_attn.register_forward_pre_hook(attn_pre_hook))
            self._handles.append(layer.self_attn.register_forward_hook(attn_forward_hook))
            self._handles.append(layer.linear2.register_forward_hook(mlp_forward_hook))

    def begin_step(self) -> None:
        self._metrics = {}

    def finalize_step(self) -> dict[str, float]:
        for layer_index, layer in enumerate(getattr(self.model.encoder, "layers", [])):
            layer_prefix = f"internals/layer_{layer_index}"
            if hasattr(layer.self_attn, "in_proj_weight") and layer.self_attn.in_proj_weight is not None:
                q_weight, k_weight, v_weight = layer.self_attn.in_proj_weight.chunk(3, dim=0)
                grad_tensor = layer.self_attn.in_proj_weight.grad
                if grad_tensor is not None:
                    q_grad, k_grad, v_grad = grad_tensor.chunk(3, dim=0)
                else:
                    q_grad = k_grad = v_grad = None
            else:
                q_weight = layer.self_attn.q_proj.weight
                k_weight = layer.self_attn.k_proj.weight
                v_weight = layer.self_attn.v_proj.weight
                q_grad = layer.self_attn.q_proj.weight.grad
                k_grad = layer.self_attn.k_proj.weight.grad
                v_grad = layer.self_attn.v_proj.weight.grad
            self._metrics.update(tensor_summary_metrics(f"{layer_prefix}/q_weight", q_weight))
            self._metrics.update(tensor_summary_metrics(f"{layer_prefix}/k_weight", k_weight))
            self._metrics.update(tensor_summary_metrics(f"{layer_prefix}/v_weight", v_weight))
            if q_grad is not None:
                self._metrics.update(tensor_summary_metrics(f"{layer_prefix}/q_weight/grad", q_grad))
                self._metrics.update(tensor_summary_metrics(f"{layer_prefix}/k_weight/grad", k_grad))
                self._metrics.update(tensor_summary_metrics(f"{layer_prefix}/v_weight/grad", v_grad))
        return dict(self._metrics)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
