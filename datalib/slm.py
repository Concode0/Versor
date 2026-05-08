"""Data utilities for raw-text SLM training."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

import torch
from torch.utils.data import DataLoader, Dataset

from log import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100


class TokenizerLike(Protocol):
    pad_id: int

    def encode(self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token ids."""


@dataclass
class TextCorpusConfig:
    """HuggingFace text corpus loading options."""

    name: str = "HuggingFaceTB/cosmopedia"
    config: Optional[str] = "stories"
    split: str = "train"
    text_field: Optional[str] = "text"
    sample_size: Optional[int] = 2048
    streaming: bool = True
    shuffle: bool = True
    seed: int = 0
    shuffle_buffer: int = 10_000
    max_chars_per_sample: int = 12_000
    eval_fraction: float = 0.05

    @classmethod
    def from_mapping(cls, values: dict) -> "TextCorpusConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{key: values.get(key, getattr(cls, key)) for key in keys})


@dataclass
class CausalLoaderConfig:
    """Causal LM dataloader options."""

    batch_size: int = 8
    max_seq_len: int = 128
    chunk_long_texts: bool = False
    stride: Optional[int] = None
    shuffle_train: bool = True
    num_workers: int = 0
    pin_memory: bool = False


def extract_text(row: dict, text_field: Optional[str] = None) -> str:
    """Extract a text field from a HuggingFace dataset row."""
    if text_field is not None:
        return str(row.get(text_field, ""))
    for candidate in ("text", "content", "article", "story", "generated_text"):
        if candidate in row and row[candidate]:
            return str(row[candidate])
    return ""


def load_text_samples(config: TextCorpusConfig) -> List[str]:
    """Load a sampled raw-text corpus from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The SLM data pipeline requires the optional 'datasets' dependency.")

    kwargs = {"path": config.name, "split": config.split, "streaming": config.streaming}
    if config.config:
        kwargs["name"] = config.config

    logger.info(
        "Loading dataset=%s config=%s split=%s streaming=%s sample_size=%s",
        config.name,
        config.config,
        config.split,
        config.streaming,
        config.sample_size,
    )
    dataset = load_dataset(**kwargs)

    if config.streaming:
        if config.shuffle:
            dataset = dataset.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer)
        iterator = iter(dataset)
    else:
        if config.shuffle:
            dataset = dataset.shuffle(seed=config.seed)
        if config.sample_size is not None:
            dataset = dataset.select(range(min(int(config.sample_size), len(dataset))))
        iterator = iter(dataset)

    texts = []
    for row in iterator:
        text = extract_text(row, config.text_field).strip()
        if text:
            texts.append(text[: config.max_chars_per_sample])
        if config.sample_size is not None and len(texts) >= int(config.sample_size):
            break

    if not texts:
        raise RuntimeError("No text samples were loaded. Check dataset config and text_field.")
    logger.info("Loaded %d text samples for SLM training.", len(texts))
    return texts


def split_train_eval(texts: List[str], eval_fraction: float) -> tuple:
    """Split sampled texts into train/eval partitions."""
    if len(texts) < 2 or eval_fraction <= 0:
        return texts, []
    eval_size = max(1, int(len(texts) * eval_fraction))
    return texts[:-eval_size], texts[-eval_size:]


class CausalTextDataset(Dataset):
    """Fixed-length causal LM examples from tokenized text."""

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: TokenizerLike,
        max_seq_len: int,
        chunk_long_texts: bool = False,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_id
        self.examples = []

        stride = stride or max_seq_len
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) < 2:
                continue
            if chunk_long_texts:
                starts = range(0, max(len(ids) - 1, 1), stride)
                for start in starts:
                    chunk = ids[start : start + max_seq_len + 1]
                    if len(chunk) >= 2:
                        self.examples.append(self._make_example(chunk))
            else:
                self.examples.append(self._make_example(ids[: max_seq_len + 1]))

        if not self.examples:
            raise RuntimeError("Tokenizer produced no causal LM examples.")

    def _make_example(self, ids: List[int]) -> dict:
        input_ids = ids[:-1][: self.max_seq_len]
        labels = ids[1:][: self.max_seq_len]
        valid_len = len(input_ids)
        pad_len = self.max_seq_len - valid_len

        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        attention_mask = [1] * valid_len + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_causal_lm_loaders(
    train_texts: Iterable[str],
    eval_texts: Iterable[str],
    tokenizer: TokenizerLike,
    config: CausalLoaderConfig,
) -> tuple:
    """Build train/eval dataloaders for causal LM training."""
    train_dataset = CausalTextDataset(
        train_texts,
        tokenizer,
        max_seq_len=config.max_seq_len,
        chunk_long_texts=config.chunk_long_texts,
        stride=config.stride,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    eval_loader = None
    eval_texts = list(eval_texts)
    if eval_texts:
        eval_dataset = CausalTextDataset(eval_texts, tokenizer, max_seq_len=config.max_seq_len)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    return train_loader, eval_loader
