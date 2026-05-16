from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = "char"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    mask_token: str = "<mask>"
    special_token_reserve_size: int = 8


def init_tokenizer(
    name: str = "char",
    *,
    pad_token: str = "<pad>",
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
    unk_token: str = "<unk>",
    mask_token: str = "<mask>",
    special_token_reserve_size: int = 8,
) -> "CharTokenizer | HuggingFaceTokenizer":
    config = TokenizerConfig(
        name=name,
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        mask_token=mask_token,
        special_token_reserve_size=special_token_reserve_size,
    )
    if name == "char":
        return CharTokenizer(config)
    if name in {"gpt2", "openai-community/gpt2"}:
        return HuggingFaceTokenizer("openai-community/gpt2", config)
    if name.startswith("hf:"):
        return HuggingFaceTokenizer(name.removeprefix("hf:"), config)
    raise ValueError(
        f"unsupported tokenizer {name!r}; use 'char', 'gpt2', or 'hf:<repo-id>'"
    )


class HuggingFaceTokenizer:
    """Small adapter exposing the tokenizer API used by this training code."""

    def __init__(self, pretrained_name: str, config: TokenizerConfig):
        self.config = config
        self.pretrained_name = pretrained_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": config.pad_token,
                "bos_token": config.bos_token,
                "eos_token": config.eos_token,
                "unk_token": config.unk_token,
                "mask_token": config.mask_token,
            }
        )

        self.pad_token = self.tokenizer.pad_token
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.unk_token = self.tokenizer.unk_token
        self.mask_token = self.tokenizer.mask_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = len(self.tokenizer)
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        self.num_special_tokens = len(self.special_token_ids)
        self.valid_token_ids = [
            token_id
            for token_id in range(self.vocab_size)
            if token_id not in self.special_token_ids
        ]

    @classmethod
    def from_name(cls, name: str = "gpt2") -> "HuggingFaceTokenizer":
        tokenizer = init_tokenizer(name=name)
        if not isinstance(tokenizer, cls):
            raise ValueError(f"tokenizer {name!r} is not a Hugging Face tokenizer")
        return tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_special_tokens:
            return [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, batch_token_ids: list[list[int]], skip_special_tokens: bool = False
    ) -> list[str]:
        return self.tokenizer.batch_decode(
            batch_token_ids, skip_special_tokens=skip_special_tokens
        )

    def __call__(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ) -> dict[str, list[list[int]]]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
        )
        input_ids = encoded["input_ids"]
        if add_special_tokens:
            input_ids = [
                [self.bos_token_id, *token_ids, self.eos_token_id]
                for token_ids in input_ids
            ]
            encoded["input_ids"] = input_ids
            if return_attention_mask:
                encoded["attention_mask"] = [
                    [1] * len(token_ids) for token_ids in input_ids
                ]
            if return_token_type_ids:
                encoded["token_type_ids"] = [
                    [0] * len(token_ids) for token_ids in input_ids
                ]
        return dict(encoded)


class CharTokenizer:
    """
    Minimal character-level tokenizer with a small special-token API.

    The base vocabulary covers Latin-1 characters so token IDs stay fixed
    across runs without requiring a separate vocab-building pass. A reserved
    block of special-token IDs is kept in front of the character vocabulary
    so new control tokens can be added later without shifting character IDs.
    """

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        self.named_special_tokens = [
            self.config.pad_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.unk_token,
            self.config.mask_token,
        ]
        if self.config.special_token_reserve_size < len(self.named_special_tokens):
            raise ValueError(
                "special_token_reserve_size must be at least the number of named special tokens"
            )
        self.reserved_special_tokens = [
            f"<special_{index:03d}>"
            for index in range(
                len(self.named_special_tokens), self.config.special_token_reserve_size
            )
        ]
        self.special_tokens = [
            *self.named_special_tokens,
            *self.reserved_special_tokens,
        ]
        self._special_token_to_id = {
            token: token_id for token_id, token in enumerate(self.special_tokens)
        }
        self._special_tokens_by_length = sorted(
            self.special_tokens, key=len, reverse=True
        )
        self.pad_token = self.config.pad_token
        self.bos_token = self.config.bos_token
        self.eos_token = self.config.eos_token
        self.unk_token = self.config.unk_token
        self.mask_token = self.config.mask_token
        self.num_named_special_tokens = len(self.named_special_tokens)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 4
        self.num_special_tokens = len(self.special_tokens)
        self.special_token_ids = set(range(len(self.special_tokens)))
        self.valid_special_token_ids = set(range(self.num_named_special_tokens))
        self.special_token_reserve_size = self.config.special_token_reserve_size

        self._char_to_id = {chr(i): i + len(self.special_tokens) for i in range(256)}
        self._id_to_char = {
            token_id: char for char, token_id in self._char_to_id.items()
        }
        self.vocab_size = len(self.special_tokens) + len(self._char_to_id)
        self.valid_token_ids = [
            *range(len(self.special_tokens), self.vocab_size),
        ]

    @classmethod
    def from_name(cls, name: str = "char") -> "CharTokenizer":
        tokenizer = init_tokenizer(name=name)
        if not isinstance(tokenizer, cls):
            raise ValueError(f"tokenizer {name!r} is not a character tokenizer")
        return tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids: list[int] = []
        index = 0
        while index < len(text):
            special_token = self._match_special_token(text, index)
            if special_token is not None:
                token_ids.append(self._special_token_to_id[special_token])
                index += len(special_token)
                continue
            token_ids.append(self._char_to_id.get(text[index], self.unk_token_id))
            index += 1
        if add_special_tokens:
            return [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id in self._id_to_char:
                pieces.append(self._id_to_char[token_id])
                continue
            if skip_special_tokens and token_id in self.special_token_ids:
                continue
            pieces.append(self._special_token_for_id(token_id))
        return "".join(pieces)

    def batch_decode(
        self, batch_token_ids: list[list[int]], skip_special_tokens: bool = False
    ) -> list[str]:
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]

    def __call__(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ) -> dict[str, list[list[int]]]:
        encoded = [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]
        output: dict[str, list[list[int]]] = {"input_ids": encoded}
        if return_attention_mask:
            output["attention_mask"] = [[1] * len(token_ids) for token_ids in encoded]
        if return_token_type_ids:
            output["token_type_ids"] = [[0] * len(token_ids) for token_ids in encoded]
        return output

    def _special_token_for_id(self, token_id: int) -> str:
        if token_id == self.pad_token_id:
            return self.pad_token
        if token_id == self.bos_token_id:
            return self.bos_token
        if token_id == self.eos_token_id:
            return self.eos_token
        if token_id == self.unk_token_id:
            return self.unk_token
        if token_id == self.mask_token_id:
            return self.mask_token
        if 0 <= token_id < len(self.special_tokens):
            return self.special_tokens[token_id]
        return self.unk_token

    def _match_special_token(self, text: str, index: int) -> str | None:
        for token in self._special_tokens_by_length:
            if text.startswith(token, index):
                return token
        return None


if __name__ == "__main__":
    # text = "replace this <mask> your string"
    # text = " <mask>"
    text = "disconnected , lights appearing <eos> definitely working"
    tokenizer = CharTokenizer()
    # token_ids = tokenizer.encode(text, add_special_tokens=False)
    # decoded_text = [tokenizer.decode([token_id]) for token_id in token_ids]

    # print(f"text: {text}")
    # print(f"token_ids: {token_ids}")
    # print(f"decoded_text: {decoded_text}")

    token_ids = tokenizer(
        [text, text],
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    for txt, toks in zip([text, text], token_ids):
        print(f"{txt}")
        print(f"{toks}")
        print(f"{tokenizer.batch_decode([toks])}")
        print(f"{len(txt)} chars, {len(toks)} tokens")
