import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from smol.diffusion.text.core.tokenizer import init_tokenizer


@dataclass
class TextDataConfig:
    dataset_name: str = "Salesforce/wikitext"
    dataset_config_name: str | None = "wikitext-2-raw-v1"
    dataset_data_files: str | list[str] | dict[str, str | list[str]] | None = None
    split: str = "train"
    batch_size: int = 8
    sequence_length: int = 128
    tokenizer_name: str = "char"
    source_batch_size: int = 0
    shuffle: bool = True
    drop_last: bool = False
    base_seed: int = 1337
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    debug_num_samples: int = -1
    dense_packing: bool = False


def collate_text_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [example["text"] for example in batch]
    return {
        "text": texts,
        "num_items": len(texts),
    }


class DebuggingDataset(Dataset):
    def __init__(self, num_samples: int, sequence_length: int):
        self.num_samples = num_samples
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int | list[int]) -> dict[str, Any]:
        if isinstance(index, int):
            text = f"Sample {index}: " + "The quick brown fox jumps over the lazy dog. " * 10
            return {"text": text[: self.sequence_length]}
        else:
            return {"text": [self[i]["text"] for i in index]}


class ResumableTextDataLoader:
    def __init__(self, config: TextDataConfig):
        self.config = config
        if config.debug_num_samples > 0:
            self.dataset = DebuggingDataset(num_samples=config.debug_num_samples, sequence_length=config.sequence_length)
        else:
            load_kwargs: dict[str, Any] = {"split": config.split}
            if config.dataset_config_name is not None:
                load_kwargs["name"] = config.dataset_config_name
            if config.dataset_data_files is not None:
                load_kwargs["data_files"] = config.dataset_data_files
            self.dataset: Dataset = load_dataset(config.dataset_name, **load_kwargs)
        self.tokenizer = init_tokenizer(config.tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id

        self.epoch = 0
        self.source_batches_consumed_in_epoch = 0
        self.batches_yielded_in_epoch = 0
        self.global_step = 0
        self.sequence_buffer: list[list[int]] = []
        self.token_buffer: list[int] = []
        self.document_id_buffer: list[int] = []
        self.next_document_id = 0

    @property
    def dataset_size(self) -> int:
        return len(self.dataset)

    @property
    def source_batch_size(self) -> int:
        if self.config.source_batch_size > 0:
            return self.config.source_batch_size
        return self.config.batch_size

    @property
    def tokens_per_batch(self) -> int:
        return self.config.batch_size * self.config.sequence_length

    @property
    def source_steps_per_epoch(self) -> int:
        if self.config.drop_last:
            return self.dataset_size // self.source_batch_size
        return math.ceil(self.dataset_size / self.source_batch_size)

    @property
    def steps_per_epoch(self) -> int | None:
        return None

    @property
    def num_epochs_completed(self) -> int:
        return self.epoch

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "epoch": self.epoch,
            "source_batches_consumed_in_epoch": self.source_batches_consumed_in_epoch,
            "batches_yielded_in_epoch": self.batches_yielded_in_epoch,
            "global_step": self.global_step,
            "sequence_buffer": self.sequence_buffer,
            "token_buffer": self.token_buffer,
            "document_id_buffer": self.document_id_buffer,
            "next_document_id": self.next_document_id,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self.source_batches_consumed_in_epoch = int(state.get("source_batches_consumed_in_epoch", 0))
        self.batches_yielded_in_epoch = int(state["batches_yielded_in_epoch"])
        self.global_step = int(state["global_step"])
        self.sequence_buffer = [
            [int(token_id) for token_id in token_ids]
            for token_ids in state.get("sequence_buffer", [])
        ]
        self.token_buffer = [int(token_id) for token_id in state.get("token_buffer", [])]
        self.document_id_buffer = [int(document_id) for document_id in state.get("document_id_buffer", [])]
        self.next_document_id = int(state.get("next_document_id", 0))

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state)

    def _epoch_generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.config.base_seed + self.epoch)
        return generator

    def _build_dataloader(self) -> DataLoader:
        loader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": self.source_batch_size,
            "shuffle": self.config.shuffle,
            "drop_last": self.config.drop_last,
            "num_workers": self.config.num_workers,
            "collate_fn": collate_text_batch,
            "pin_memory": self.config.pin_memory,
            "generator": self._epoch_generator(),
        }

        if self.config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            loader_kwargs["persistent_workers"] = self.config.persistent_workers

        return DataLoader(**loader_kwargs)

    def _tokenize_texts(self, texts: list[str]) -> list[list[int]]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        sequences: list[list[int]] = []
        for token_ids in encoded:
            for start in range(0, len(token_ids), self.config.sequence_length):
                sequence = token_ids[start : start + self.config.sequence_length]
                if not sequence:
                    continue
                if len(sequence) < self.config.sequence_length:
                    sequence = [
                        *sequence,
                        *([self.tokenizer.pad_token_id] * (self.config.sequence_length - len(sequence))),
                    ]
                sequences.append(sequence)
        return sequences

    def _tokenize_documents(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

    def _extend_dense_pack_buffer(self, texts: list[str]) -> None:
        for token_ids in self._tokenize_documents(texts):
            if not token_ids:
                continue
            document_id = self.next_document_id
            self.next_document_id += 1
            self.token_buffer.extend(token_ids)
            self.document_id_buffer.extend([document_id] * len(token_ids))

    def _next_dense_batch(self) -> dict[str, torch.Tensor] | None:
        if len(self.token_buffer) < self.tokens_per_batch:
            return None

        batch_token_ids = self.token_buffer[: self.tokens_per_batch]
        batch_document_ids = self.document_id_buffer[: self.tokens_per_batch]
        sequence_lengths = self._sequence_lengths_from_document_ids(batch_document_ids)
        del self.token_buffer[: self.tokens_per_batch]
        del self.document_id_buffer[: self.tokens_per_batch]

        input_ids = torch.tensor(batch_token_ids, dtype=torch.long).view(
            self.config.batch_size,
            self.config.sequence_length,
        )
        token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "sequence_lengths": sequence_lengths,
        }

    def _sequence_lengths_from_document_ids(self, document_ids: list[int]) -> list[list[int]]:
        sequence_lengths: list[list[int]] = []
        for row_start in range(0, len(document_ids), self.config.sequence_length):
            row_document_ids = document_ids[row_start : row_start + self.config.sequence_length]
            row_lengths: list[int] = []
            current_document_id: int | None = None
            current_length = 0
            for document_id in row_document_ids:
                if current_document_id is None or document_id == current_document_id:
                    current_document_id = document_id
                    current_length += 1
                    continue
                row_lengths.append(current_length)
                current_document_id = document_id
                current_length = 1
            if current_length > 0:
                row_lengths.append(current_length)
            sequence_lengths.append(row_lengths)
        return sequence_lengths

    def _filter_text(self, text: str) -> bool:
        return len(text) > 0 and " = = = " not in text

    def iter_epoch(self):
        dataloader = self._build_dataloader()
        start_source_batch = self.source_batches_consumed_in_epoch

        for source_batch_index, batch in enumerate(dataloader):
            if source_batch_index < start_source_batch:
                continue
            texts = batch["text"]
            texts = [text for text in texts if self._filter_text(text)]
            if self.config.dense_packing:
                self._extend_dense_pack_buffer(texts)
            else:
                self.sequence_buffer.extend(self._tokenize_texts(texts))
            self.source_batches_consumed_in_epoch = source_batch_index + 1

            while (
                len(self.token_buffer) >= self.tokens_per_batch
                if self.config.dense_packing
                else len(self.sequence_buffer) >= self.config.batch_size
            ):
                if self.config.dense_packing:
                    tensor_batch = self._next_dense_batch()
                    if tensor_batch is None:
                        break
                    input_ids = tensor_batch["input_ids"]
                    token_mask = tensor_batch["token_mask"]
                    sequence_lengths = tensor_batch["sequence_lengths"]
                    buffered_sequences = len(self.token_buffer) // self.config.sequence_length
                else:
                    batch_sequences = self.sequence_buffer[: self.config.batch_size]
                    del self.sequence_buffer[: self.config.batch_size]

                    input_ids = torch.tensor(batch_sequences, dtype=torch.long)
                    token_mask = input_ids != self.tokenizer.pad_token_id
                    sequence_lengths = None
                    buffered_sequences = len(self.sequence_buffer)
                padding_tokens = int((~token_mask).sum().item())
                non_padding_tokens = int(token_mask.sum().item())
                sequence_preview = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                batch_index = self.batches_yielded_in_epoch
                global_step = self.global_step

                self.batches_yielded_in_epoch += 1
                self.global_step += 1

                output_batch = {
                    "input_ids": input_ids,
                    "token_mask": token_mask,
                    "num_items": self.config.batch_size,
                    "sequence_length": self.config.sequence_length,
                    "tokens_in_batch": self.tokens_per_batch,
                    "non_padding_tokens": non_padding_tokens,
                    "padding_tokens": padding_tokens,
                    "padding_fraction": padding_tokens / self.tokens_per_batch,
                    "text": [sequence_preview],
                    "epoch": self.epoch,
                    "batch_index": batch_index,
                    "global_step": global_step,
                    "steps_per_epoch": self.steps_per_epoch,
                    "source_steps_per_epoch": self.source_steps_per_epoch,
                    "dataset_size": self.dataset_size,
                    "num_epochs_completed": self.num_epochs_completed,
                    "buffered_sequences": buffered_sequences,
                    "source_batches_consumed_in_epoch": self.source_batches_consumed_in_epoch,
                }
                if sequence_lengths is not None:
                    output_batch["sequence_lengths"] = sequence_lengths
                yield output_batch

        self.epoch += 1
        self.source_batches_consumed_in_epoch = 0
        self.batches_yielded_in_epoch = 0
        self.sequence_buffer = []
        self.token_buffer = []
        self.document_id_buffer = []

    def iter_epochs(self, num_epochs: int):
        target_epoch = self.epoch + num_epochs
        while self.epoch < target_epoch:
            yield from self.iter_epoch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterate over WikiText with a resumable PyTorch dataloader.")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config-name", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-data-files", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--tokenizer-name", default="char")
    parser.add_argument("--source-batch-size", type=int, default=0, help="Number of raw documents to tokenize at once. 0 reuses --batch-size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N yielded batches. 0 disables it.")
    parser.add_argument("--checkpoint-path", default="checkpoints/text_dataloader.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--dense-packing", action="store_true")
    parser.add_argument("--print-text-chars", type=int, default=120)
    parser.add_argument("--no-append-eos-token", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TextDataConfig(
        debug_num_samples=100,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_data_files=args.dataset_data_files,
        split=args.split,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        tokenizer_name=args.tokenizer_name,
        source_batch_size=args.source_batch_size,
        shuffle=not args.no_shuffle,
        drop_last=args.drop_last,
        base_seed=args.seed,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        dense_packing=args.dense_packing,
    )

    loader = ResumableTextDataLoader(config)

    if args.resume and Path(args.checkpoint_path).exists():
        loader.load_checkpoint(args.checkpoint_path)
        print(f"Resumed from {args.checkpoint_path}")

    print(
        "Loader setup:",
        {
            "dataset": config.dataset_name,
            "dataset_config": config.dataset_config_name,
            "split": config.split,
            "dataset_size": loader.dataset_size,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "tokenizer_name": config.tokenizer_name,
            "source_batch_size": loader.source_batch_size,
            "steps_per_epoch": loader.steps_per_epoch,
            "source_steps_per_epoch": loader.source_steps_per_epoch,
            "shuffle": config.shuffle,
            "seed": config.base_seed,
            "num_workers": config.num_workers,
            "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else None,
            "dense_packing": config.dense_packing,
            "checkpoint_path": args.checkpoint_path,
            "starting_epoch": loader.epoch,
            "starting_global_step": loader.global_step,
        },
    )

    for batch in loader.iter_epochs(args.max_epochs):
        preview = batch["text"][0][: args.print_text_chars].replace("\n", "\\n")
        print(
            {
                "epoch": batch["epoch"],
                "batch_index": batch["batch_index"],
                "global_step": batch["global_step"],
                "num_epochs_completed": batch["num_epochs_completed"],
                "steps_per_epoch": batch["steps_per_epoch"],
                "source_steps_per_epoch": batch["source_steps_per_epoch"],
                "num_items": batch["num_items"],
                "sequence_length": batch["sequence_length"],
                "tokens_in_batch": batch["tokens_in_batch"],
                "non_padding_tokens": batch["non_padding_tokens"],
                "padding_tokens": batch["padding_tokens"],
                "padding_fraction": batch["padding_fraction"],
                "buffered_sequences": batch["buffered_sequences"],
                "preview": preview,
            }
        )

        if args.save_every > 0 and loader.global_step % args.save_every == 0:
            loader.save_checkpoint(args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")

    loader.save_checkpoint(args.checkpoint_path)
    print(
        "Finished iteration:",
        {
            "epochs_completed": loader.num_epochs_completed,
            "global_steps": loader.global_step,
            "checkpoint_path": args.checkpoint_path,
        },
    )


if __name__ == "__main__":
    main()
