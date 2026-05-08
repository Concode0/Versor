# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Tokenizer utilities for raw-text SLM experiments.

The default path uses a WordPiece-style subword tokenizer from the optional
``tokenizers`` package. A small regex tokenizer is kept as a fallback so local
shape tests do not depend on the compiled tokenizer package being available.
"""

import re
from collections import Counter
from typing import Iterable, List, Optional

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


class SubwordTokenizer:
    """Small trainable tokenizer with fixed special token ids.

    Special ids:
        [PAD] = 0, [UNK] = 1, [BOS] = 2, [EOS] = 3.
    """

    pad_token = "[PAD]"
    unk_token = "[UNK]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"

    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    def __init__(
        self,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        lowercase: bool = True,
        mode: str = "subword",
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.mode = mode
        self.backend = None
        self.vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def train(self, texts: Iterable[str]) -> "SubwordTokenizer":
        """Train tokenizer state from a text iterator."""
        texts = list(texts)
        if self.mode == "subword" and self._train_wordpiece(texts):
            return self
        self._train_regex_vocab(texts)
        return self

    def _train_wordpiece(self, texts: List[str]) -> bool:
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import WordPiece as WordPieceDecoder
            from tokenizers.models import WordPiece
            from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
            from tokenizers.pre_tokenizers import BertPreTokenizer
            from tokenizers.trainers import WordPieceTrainer
        except ImportError:
            return False

        tokenizer = Tokenizer(WordPiece(unk_token=self.unk_token))
        if self.lowercase:
            tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.decoder = WordPieceDecoder(prefix="##")

        trainer = WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        special_ids = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
        if special_ids != list(range(len(SPECIAL_TOKENS))):
            raise RuntimeError(f"Unexpected tokenizer special ids: {dict(zip(SPECIAL_TOKENS, special_ids))}")
        self.backend = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        return True

    def _train_regex_vocab(self, texts: List[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(self._split_regex(text))

        keep = max(self.vocab_size - len(SPECIAL_TOKENS), 0)
        for token, count in counter.most_common(keep):
            if count >= self.min_frequency and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids."""
        if self.backend is not None:
            ids = self.backend.encode(text).ids
        else:
            ids = [self.vocab.get(token, self.unk_id) for token in self._split_regex(text)]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
            if add_special_tokens and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text."""
        if self.backend is not None:
            return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

        tokens = []
        special = set(SPECIAL_TOKENS)
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)
        text = " ".join(tokens)
        text = re.sub(r"\s+([^\w\s])", r"\1", text)
        return text.strip()

    def to_state(self) -> dict:
        """Serializable tokenizer state for SLM checkpoints."""
        backend_json = self.backend.to_str() if self.backend is not None else None
        return {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "lowercase": self.lowercase,
            "mode": self.mode,
            "vocab": self.vocab,
            "backend_json": backend_json,
        }

    @classmethod
    def from_state(cls, state: dict) -> "SubwordTokenizer":
        """Restore tokenizer state from a checkpoint payload."""
        tokenizer = cls(
            vocab_size=state.get("vocab_size", 8192),
            min_frequency=state.get("min_frequency", 2),
            lowercase=state.get("lowercase", True),
            mode=state.get("mode", "subword"),
        )
        tokenizer.vocab = dict(state.get("vocab", tokenizer.vocab))
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.vocab.items()}
        backend_json = state.get("backend_json")
        if backend_json is not None:
            try:
                from tokenizers import Tokenizer

                tokenizer.backend = Tokenizer.from_str(backend_json)
            except ImportError:
                tokenizer.backend = None
        return tokenizer

    def _split_regex(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def __len__(self) -> int:
        return len(self.vocab)
