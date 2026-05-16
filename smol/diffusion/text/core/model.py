from dataclasses import dataclass, fields
from contextlib import contextmanager, nullcontext
import warnings
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from smol.diffusion.text.core.tokenizer import CharTokenizer, HuggingFaceTokenizer

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


@dataclass
class TextDiffusionConfig:
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_sequence_length: int = 128
    num_diffusion_steps: int = 100
    mask_probability: float = 0.8
    clean_token_loss_weight: float = 0.1

    @classmethod
    def from_dict(cls, payload: dict) -> "TextDiffusionConfig":
        field_names = {field.name for field in fields(cls)}
        return cls(
            **{key: value for key, value in payload.items() if key in field_names}
        )


class TextDiffusionSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads} for multi-head attention"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be even for rotary position embeddings"
            )
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _apply_rotary_embedding(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        half_dim = tensor.size(-1) // 2
        first_half, second_half = tensor[..., :half_dim], tensor[..., half_dim:]
        rotated_first = first_half * cos + second_half * sin
        rotated_second = first_half * (-sin) + second_half * cos
        return torch.cat([rotated_first, rotated_second], dim=-1)

    @contextmanager
    def _sdp_kernel_context(
        self, query: torch.Tensor, *, has_attention_mask: bool = False
    ):
        if not query.is_cuda:
            with nullcontext():
                yield
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated.*",
                category=FutureWarning,
            )
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=has_attention_mask,
            ):
                yield

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        token_mask: torch.Tensor | None = None,
        sequence_lengths: Sequence[Sequence[int]] | None = None,
    ) -> torch.Tensor:
        query = self._reshape_heads(self.q_proj(hidden_states))
        key = self._reshape_heads(self.k_proj(hidden_states))
        value = self._reshape_heads(self.v_proj(hidden_states))
        cos, sin = cos_sin
        cos = cos.to(dtype=query.dtype)
        sin = sin.to(dtype=query.dtype)
        query = self._apply_rotary_embedding(query, cos, sin)
        key = self._apply_rotary_embedding(key, cos, sin)

        if sequence_lengths is not None:
            attention_output = self._flash_varlen_attention(
                query, key, value, sequence_lengths
            )
        else:
            attention_mask = None
            if token_mask is not None:
                attention_mask = token_mask[:, None, None, :].to(
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
            with self._sdp_kernel_context(
                query, has_attention_mask=attention_mask is not None
            ):
                attention_output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)
        )
        return self.out_proj(attention_output)

    def _flash_varlen_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sequence_lengths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        if flash_attn_varlen_func is None:
            raise ImportError(
                "dense_packing=True requires flash-attn. Install it with "
                "`conda run --no-capture-output -n smol python -m pip install flash-attn --no-build-isolation`."
            )
        if not query.is_cuda:
            raise RuntimeError("flash-attn varlen attention requires CUDA tensors")

        batch_size, _, sequence_length, _ = query.shape
        flat_query = (
            query.transpose(1, 2).contiguous().view(-1, self.num_heads, self.head_dim)
        )
        flat_key = (
            key.transpose(1, 2).contiguous().view(-1, self.num_heads, self.head_dim)
        )
        flat_value = (
            value.transpose(1, 2).contiguous().view(-1, self.num_heads, self.head_dim)
        )
        lengths = [
            int(length) for row_lengths in sequence_lengths for length in row_lengths
        ]
        cu_seqlens = torch.empty(
            len(lengths) + 1, dtype=torch.int32, device=query.device
        )
        cu_seqlens[0] = 0
        cu_seqlens[1:] = torch.tensor(
            lengths, dtype=torch.int32, device=query.device
        ).cumsum(dim=0)
        max_seqlen = max(lengths)
        flat_output = flash_attn_varlen_func(
            flat_query,
            flat_key,
            flat_value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        )
        return flat_output.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)


class TextDiffusionEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = TextDiffusionSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        token_mask: torch.Tensor | None = None,
        sequence_lengths: Sequence[Sequence[int]] | None = None,
    ) -> torch.Tensor:
        attn_input = self.norm1(hidden_states)
        hidden_states = hidden_states + self.dropout1(
            self.self_attn(
                attn_input,
                cos_sin,
                token_mask=token_mask,
                sequence_lengths=sequence_lengths,
            )
        )

        mlp_input = self.norm2(hidden_states)
        mlp_output = self.linear2(
            self.dropout2(F.relu(self.linear1(mlp_input)).square())
        )
        hidden_states = hidden_states + self.dropout2(mlp_output)
        return hidden_states


class TextDiffusionEncoder(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TextDiffusionEncoderLayer(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        token_mask: torch.Tensor | None = None,
        sequence_lengths: Sequence[Sequence[int]] | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos_sin,
                token_mask=token_mask,
                sequence_lengths=sequence_lengths,
            )
        return hidden_states


class TextDiffusionModel(nn.Module):
    """
    A minimal discrete text diffusion model.

    Tokens are masked at a timestep-dependent rate, and the network predicts the
    original token IDs from the masked sequence.
    """

    def __init__(
        self,
        config: TextDiffusionConfig,
        tokenizer: CharTokenizer | HuggingFaceTokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.mask_token_id = tokenizer.mask_token_id
        self.register_buffer(
            "valid_token_ids",
            torch.tensor(tokenizer.valid_token_ids, dtype=torch.long),
            persistent=False,
        )

        self.token_embedding = nn.Embedding(self.vocab_size, config.hidden_size)
        rotary_cos, rotary_sin = self._precompute_rotary_embeddings(
            config.max_sequence_length,
            config.hidden_size // config.num_heads,
        )
        self.register_buffer("rotary_cos", rotary_cos, persistent=False)
        self.register_buffer("rotary_sin", rotary_sin, persistent=False)

        self.encoder = TextDiffusionEncoder(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, self.vocab_size)

    def _precompute_rotary_embeddings(
        self,
        sequence_length: int,
        head_dim: int,
        base: int = 10000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be even for rotary position embeddings"
            )
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = torch.arange(sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        return frequencies.cos()[None, None, :, :], frequencies.sin()[None, None, :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        sequence_lengths: Sequence[Sequence[int]] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length {self.config.max_sequence_length}"
            )
        if token_mask is not None and token_mask.shape != input_ids.shape:
            raise ValueError(
                f"token_mask shape {tuple(token_mask.shape)} does not match input_ids shape {tuple(input_ids.shape)}"
            )
        if sequence_lengths is not None:
            self._validate_sequence_lengths(
                sequence_lengths, batch_size, sequence_length
            )

        if timesteps.shape != (batch_size,):
            raise ValueError(
                f"timesteps must have shape ({batch_size},), got {tuple(timesteps.shape)}"
            )

        hidden_states = self.token_embedding(input_ids)
        cos_sin = (
            self.rotary_cos[:, :, :sequence_length, :].to(
                device=input_ids.device,
                dtype=hidden_states.dtype,
            ),
            self.rotary_sin[:, :, :sequence_length, :].to(
                device=input_ids.device,
                dtype=hidden_states.dtype,
            ),
        )
        hidden_states = self.encoder(
            hidden_states,
            cos_sin,
            token_mask=token_mask,
            sequence_lengths=sequence_lengths,
        )
        hidden_states = self.output_norm(hidden_states)
        return self.output_projection(hidden_states)

    def corruption_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / float(self.config.num_diffusion_steps)

    def corrupt_tokens(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_probability = self.corruption_probability(timesteps).unsqueeze(1)

        mask_choice = (
            torch.rand(
                input_ids.shape,
                device=input_ids.device,
                generator=generator,
            )
            < mask_probability
        )
        if token_mask is not None:
            mask_choice = mask_choice & token_mask
        mask_tokens = torch.full_like(input_ids, self.mask_token_id)
        masked_tokens = torch.where(mask_choice, mask_tokens, input_ids)
        return masked_tokens, mask_choice

    def loss(
        self,
        clean_input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        sequence_lengths: Sequence[Sequence[int]] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masked_tokens, mask_choice = self.corrupt_tokens(
            clean_input_ids,
            timesteps,
            token_mask=token_mask,
            generator=generator,
        )
        logits = self(
            masked_tokens,
            timesteps,
            token_mask=token_mask,
            sequence_lengths=sequence_lengths,
        )
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = clean_input_ids.reshape(-1)
        flat_mask_mask = mask_choice.reshape(-1)
        flat_token_mask = (
            torch.ones_like(token_mask, dtype=torch.bool)
            if token_mask is None
            else token_mask.reshape(-1)
        )
        per_token_loss = nn.functional.cross_entropy(
            flat_logits,
            flat_targets,
            reduction="none",
        )
        token_weights = torch.where(
            flat_mask_mask,
            torch.ones_like(per_token_loss),
            torch.full_like(per_token_loss, self.config.clean_token_loss_weight),
        )
        token_weights = torch.where(
            flat_token_mask, token_weights, torch.zeros_like(token_weights)
        )
        loss = (per_token_loss * token_weights).sum() / token_weights.sum().clamp_min(
            1e-8
        )
        stats = {
            "masked_input_ids": masked_tokens,
            "masked_tokens": masked_tokens,
            "mask_choice": mask_choice,
            "replace_mask": mask_choice,
            "logits": logits,
            "per_token_loss": per_token_loss.reshape_as(clean_input_ids),
            "target_input_ids": clean_input_ids,
            "loss_token_mask": flat_token_mask.reshape_as(clean_input_ids),
        }
        return loss, stats

    def _validate_sequence_lengths(
        self,
        sequence_lengths: Sequence[Sequence[int]],
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if len(sequence_lengths) != batch_size:
            raise ValueError(
                f"sequence_lengths must have {batch_size} rows, got {len(sequence_lengths)}"
            )
        for batch_index, row_lengths in enumerate(sequence_lengths):
            for length in row_lengths:
                if int(length) <= 0:
                    raise ValueError(f"sequence lengths must be positive, got {length}")
            row_total = sum(int(length) for length in row_lengths)
            if row_total != sequence_length:
                raise ValueError(
                    f"sequence lengths for row {batch_index} sum to {row_total}, expected {sequence_length}"
                )
