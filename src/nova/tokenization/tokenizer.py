"""Byte-level BPE tokenizer wrapper with chat special tokens.

Uses HuggingFace `tokenizers`. Byte-level BPE means every possible byte string can
be encoded (no <unk>). Special tokens are added at training time and used by the
chat template.

The tokenizer.json snapshot is checksummed and stored alongside run checkpoints,
and re-loaded at inference to guarantee identical encoding.
"""
from __future__ import annotations

import hashlib
import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Reserved special tokens (order matters: pad must be id 0 for loss masking if used)
SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]
PAD_ID = 0  # <|pad|> is first and therefore id 0


class NovaTokenizer:
    """Small, deterministic wrapper around a byte-level BPE tokenizer."""

    def __init__(self, tokenizer: Tokenizer, special_ids: dict):
        self.tk = tokenizer
        self.special_ids = special_ids  # name -> id
        self.vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    # ---------------------------------------------------------------- build
    @classmethod
    def train_from_iter(cls, texts, vocab_size: int = 8000,
                        min_freq: int = 2) -> "NovaTokenizer":
        """Train byte-level BPE on an iterable of raw texts."""
        bt = models.BPE(unk_token=None)  # byte-level => no unk
        tok = Tokenizer(bt)
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(texts, trainer)
        ids = {name: tok.token_to_id(name) for name in SPECIAL_TOKENS}
        return cls(tok, ids)

    @classmethod
    def from_file(cls, path: str) -> "NovaTokenizer":
        tok = Tokenizer.from_file(path)
        ids = {name: tok.token_to_id(name) for name in SPECIAL_TOKENS}
        missing = [n for n, i in ids.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer {path} lacks special tokens: {missing}")
        return cls(tok, ids)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tk.save(path)

    @staticmethod
    def file_fingerprint(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ----------------------------------------------------------------- api
    def encode(self, text: str, add_special: bool = False):
        ids = self.tk.encode(text).ids
        if add_special:
            ids = [self.special_ids["<|bos|>"]] + ids + [self.special_ids["<|eos|>"]]
        return ids

    def decode(self, ids) -> str:
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=False)

    def decode_clean(self, ids) -> str:
        """Decode but strip chat special tokens (normal assistant output)."""
        specials = set(self.special_ids.values())
        return self.tk.decode([int(i) for i in ids if int(i) not in specials],
                              skip_special_tokens=False)

    @property
    def eos_id(self) -> int:
        return self.special_ids["<|eos|>"]

    @property
    def bos_id(self) -> int:
        return self.special_ids["<|bos|>"]

    @property
    def pad_id(self) -> int:
        return PAD_ID
