"""
python -m smol.diffusion.text.sample \
--checkpoint-path runs/scale-300m-fineweb-1bt-less-no-drop-gpt2/20260507-073216/checkpoints/final.pt  \
--input-text "i want to go to the <mask><mask><mask><mask> already! It is already <mask><mask><mask>! Lets go"
"""

import argparse
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from smol.diffusion.text.core.model import TextDiffusionConfig, TextDiffusionModel
from smol.diffusion.text.core.tokenizer import init_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a text diffusion checkpoint with iterative mask decoding."
    )
    parser.add_argument(
        "--checkpoint-path", default="checkpoints/text_diffusion_model.pt"
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--input-text",
        default=None,
        help="Optional prompt used to seed the initial token sequence before denoising.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device", default=None, help="Override device, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--append-eos", action="store_true", help="Append EOS token to the input text"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeat-penalty", type=float, default=0.15)
    parser.add_argument("--repeat-window", type=int, default=128)
    parser.add_argument("--cap-start-ratio", type=float, default=0.08)
    parser.add_argument("--cap-end-ratio", type=float, default=0.5)
    parser.add_argument("--max-decode-per-step", type=int, default=32)
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[TextDiffusionModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_config = checkpoint.get("run_config", {})
    tokenizer_name = run_config.get("data", {}).get("tokenizer_name", "char")
    tokenizer = init_tokenizer(tokenizer_name)
    model_config = checkpoint["model_config"]
    if isinstance(model_config, dict):
        model_config = TextDiffusionConfig.from_dict(model_config)

    model = TextDiffusionModel(model_config, tokenizer).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def decode_samples(tokenizer_name: str, token_ids: torch.Tensor) -> list[str]:
    tokenizer = init_tokenizer(tokenizer_name)
    return tokenizer.batch_decode(token_ids.tolist(), skip_special_tokens=False)


@torch.no_grad()
def initialize_token_ids(
    tokenizer_name: str,
    num_samples: int,
    sequence_length: int,
    input_text: str | None,
    device: torch.device,
    append_eos: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = init_tokenizer(tokenizer_name)
    token_ids = torch.full(
        (num_samples, sequence_length),
        tokenizer.mask_token_id,
        device=device,
        dtype=torch.long,
    )
    editable_mask = torch.ones_like(token_ids, dtype=torch.bool)
    if input_text is None:
        return token_ids, editable_mask

    prompt_token_ids = tokenizer.encode(input_text)[:sequence_length]
    if not prompt_token_ids:
        return token_ids, editable_mask
    if append_eos:
        assert (
            tokenizer.eos_token_id is not None
            and len(prompt_token_ids) < sequence_length
            and prompt_token_ids[-1] != tokenizer.eos_token_id
        ), f"Cannot append EOS token to prompt - {tokenizer.eos_token_id=}, {prompt_token_ids[-1]=}, {sequence_length=}, {len(prompt_token_ids)=}"
        prompt_token_ids.append(tokenizer.eos_token_id)
    prompt = torch.tensor(prompt_token_ids, device=device, dtype=token_ids.dtype)
    token_ids[:, : prompt.numel()] = prompt.unsqueeze(0).expand(num_samples, -1)
    observed_mask = prompt != tokenizer.mask_token_id
    editable_mask[:, : prompt.numel()] = ~observed_mask.unsqueeze(0)
    return token_ids, editable_mask


@torch.no_grad()
def sample(
    model: TextDiffusionModel,
    token_ids: torch.Tensor,
    editable_mask: torch.Tensor,
    device: torch.device,
    start_timesteps: torch.Tensor | int | None = None,
    token_mask: torch.Tensor | None = None,
    sequence_lengths: Sequence[Sequence[int]] | None = None,
    temperature: float = 0.8,
    confidence_threshold: float = 0.9,
    top_k: int = 8,
    repeat_penalty: float = 0.15,
    repeat_window: int = 128,
    cap_start_ratio: float = 0.08,
    cap_end_ratio: float = 0.5,
    max_decode_per_step: int = 32,
) -> torch.Tensor:
    """Iteratively fill editable mask slots using NanoLLaDA-style confidence decoding.

    `editable_mask` marks positions that may be replaced. The function assumes
    those positions are unknown/masked slots, not random-token noise.
    """
    if token_ids.size(0) == 0:
        return token_ids
    if editable_mask.shape != token_ids.shape:
        raise ValueError(
            f"editable_mask shape {tuple(editable_mask.shape)} does not match token_ids {tuple(token_ids.shape)}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    unresolved_mask = editable_mask.clone()
    if token_mask is not None:
        unresolved_mask &= token_mask
    initial_unresolved = unresolved_mask.sum(dim=1).clamp_min(1)
    start_t = _resolve_start_timesteps(model, token_ids, start_timesteps, device)

    while unresolved_mask.any():
        active_sample_mask = unresolved_mask.any(dim=1)
        active_token_ids = token_ids[active_sample_mask]
        active_unresolved = unresolved_mask[active_sample_mask]
        active_token_mask = (
            None if token_mask is None else token_mask[active_sample_mask]
        )
        active_sequence_lengths = select_sequence_lengths(
            sequence_lengths, active_sample_mask
        )
        active_initial = initial_unresolved[active_sample_mask]
        active_start_t = start_t[active_sample_mask]
        remaining = active_unresolved.sum(dim=1)
        timesteps = _timesteps_from_remaining(
            active_start_t, remaining, active_initial, model.config.num_diffusion_steps
        )

        logits = model(
            active_token_ids,
            timesteps,
            token_mask=active_token_mask,
            sequence_lengths=active_sequence_lengths,
        )
        logits[..., model.mask_token_id] = -float("inf")
        logits = _apply_repeat_penalty(
            logits,
            active_token_ids,
            mask_token_id=model.mask_token_id,
            vocab_size=model.vocab_size,
            repeat_penalty=repeat_penalty,
            repeat_window=repeat_window,
        )

        k = model.vocab_size if top_k <= 0 else min(top_k, model.vocab_size)
        probs = F.softmax(logits / temperature, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
        confidences = top_k_probs.sum(dim=-1)
        sampled_tokens = _sample_top_k(top_k_probs, top_k_indices)
        decode_mask = _select_decode_mask(
            unresolved_mask=active_unresolved,
            confidences=confidences,
            initial_unresolved=active_initial,
            confidence_threshold=confidence_threshold,
            cap_start_ratio=cap_start_ratio,
            cap_end_ratio=cap_end_ratio,
            max_decode_per_step=max_decode_per_step,
        )

        updated_active = torch.where(decode_mask, sampled_tokens, active_token_ids)
        active_indices = torch.nonzero(active_sample_mask, as_tuple=False).flatten()
        token_ids[active_indices] = updated_active
        unresolved_mask[active_indices] = active_unresolved & ~decode_mask

    return token_ids


def _resolve_start_timesteps(
    model: TextDiffusionModel,
    token_ids: torch.Tensor,
    start_timesteps: torch.Tensor | int | None,
    device: torch.device,
) -> torch.Tensor:
    num_samples = token_ids.size(0)
    if start_timesteps is None:
        return torch.full(
            (num_samples,),
            model.config.num_diffusion_steps,
            device=device,
            dtype=torch.long,
        )
    if isinstance(start_timesteps, int):
        return torch.full(
            (num_samples,), start_timesteps, device=device, dtype=torch.long
        ).clamp_min(1)
    resolved = start_timesteps.to(device=device, dtype=torch.long).clamp_min(1)
    if resolved.shape != (num_samples,):
        raise ValueError(
            f"start_timesteps must have shape ({num_samples},), got {tuple(resolved.shape)}"
        )
    return resolved


def _timesteps_from_remaining(
    start_timesteps: torch.Tensor,
    remaining: torch.Tensor,
    initial: torch.Tensor,
    max_timestep: int,
) -> torch.Tensor:
    ratio = remaining.float() / initial.float().clamp_min(1.0)
    timesteps = torch.ceil(start_timesteps.float() * ratio).long()
    return timesteps.clamp(min=1, max=max_timestep)


def _apply_repeat_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    repeat_penalty: float,
    repeat_window: int,
) -> torch.Tensor:
    if repeat_penalty <= 0.0:
        return logits
    adjusted = logits.clone()
    for sample_index in range(token_ids.size(0)):
        finalized = token_ids[sample_index][token_ids[sample_index] != mask_token_id]
        if repeat_window > 0:
            finalized = finalized[-repeat_window:]
        if finalized.numel() == 0:
            continue
        counts = torch.bincount(finalized, minlength=vocab_size).to(
            dtype=adjusted.dtype, device=adjusted.device
        )
        adjusted[sample_index] = adjusted[sample_index] - repeat_penalty * counts.view(
            1, -1
        )
    return adjusted


def _sample_top_k(
    top_k_probs: torch.Tensor, top_k_indices: torch.Tensor
) -> torch.Tensor:
    batch_size, sequence_length, k = top_k_probs.shape
    normalized = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled_k = torch.multinomial(normalized.reshape(-1, k), 1).view(
        batch_size, sequence_length
    )
    return torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)


def _select_decode_mask(
    *,
    unresolved_mask: torch.Tensor,
    confidences: torch.Tensor,
    initial_unresolved: torch.Tensor,
    confidence_threshold: float,
    cap_start_ratio: float,
    cap_end_ratio: float,
    max_decode_per_step: int,
) -> torch.Tensor:
    decode_mask = torch.zeros_like(unresolved_mask)
    min_cap = min(cap_start_ratio, cap_end_ratio)
    for sample_index in range(unresolved_mask.size(0)):
        sample_unresolved = unresolved_mask[sample_index]
        remaining = int(sample_unresolved.sum().item())
        if remaining == 0:
            continue
        progress = 1.0 - (
            remaining / max(int(initial_unresolved[sample_index].item()), 1)
        )
        cap_ratio = cap_start_ratio + (cap_end_ratio - cap_start_ratio) * progress
        cap_ratio = min(max(cap_ratio, min_cap), 1.0)
        decode_budget = max(1, int(round(remaining * cap_ratio)))
        if max_decode_per_step > 0:
            decode_budget = min(decode_budget, max_decode_per_step)

        candidate_mask = (
            confidences[sample_index] >= confidence_threshold
        ) & sample_unresolved
        decode_count = int(candidate_mask.sum().item())
        if decode_count == 0:
            masked_confidences = torch.where(
                sample_unresolved,
                confidences[sample_index],
                torch.tensor(-float("inf"), device=confidences.device),
            )
            decode_mask[sample_index, masked_confidences.argmax()] = True
        elif decode_count > decode_budget:
            candidate_confidences = torch.where(
                candidate_mask,
                confidences[sample_index],
                torch.tensor(-float("inf"), device=confidences.device),
            )
            chosen = torch.topk(candidate_confidences.view(-1), k=decode_budget).indices
            decode_mask[sample_index].view(-1)[chosen] = True
        else:
            decode_mask[sample_index] = candidate_mask
    return decode_mask


def select_sequence_lengths(
    sequence_lengths: Sequence[Sequence[int]] | None,
    active_sample_mask: torch.Tensor,
) -> list[list[int]] | None:
    if sequence_lengths is None:
        return None
    active_indices = (
        torch.nonzero(active_sample_mask, as_tuple=False).flatten().cpu().tolist()
    )
    return [list(sequence_lengths[index]) for index in active_indices]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_checkpoint(args.checkpoint_path, device)
    run_config = checkpoint.get("run_config", {})
    tokenizer_name = run_config.get("data", {}).get("tokenizer_name", "char")
    sequence_length = min(args.sequence_length, model.config.max_sequence_length)

    initial_token_ids, editable_mask = initialize_token_ids(
        tokenizer_name=tokenizer_name,
        num_samples=args.num_samples,
        sequence_length=sequence_length,
        input_text=args.input_text,
        device=device,
        append_eos=args.append_eos,
    )
    samples = sample(
        model,
        initial_token_ids.clone(),
        editable_mask,
        device,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
        repeat_window=args.repeat_window,
        cap_start_ratio=args.cap_start_ratio,
        cap_end_ratio=args.cap_end_ratio,
        max_decode_per_step=args.max_decode_per_step,
    )
    initial_decoded = decode_samples(tokenizer_name, initial_token_ids.cpu())
    decoded = decode_samples(tokenizer_name, samples.cpu())

    for idx, (initial_text, text) in enumerate(
        zip(initial_decoded, decoded, strict=True), start=1
    ):
        print(f"[sample {idx}]")
        print(f"input: {initial_text}")
        print(text)
        print()


if __name__ == "__main__":
    main()
