# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Real HuggingFace datasets for Geometric Latent Reasoning (GLR) probes.

Three dataset classes for the three LQA probes:
1. CLUTRRDataset -- CLUTRR/v1 compositional chain reasoning
2. HANSDataset -- SNLI (train) + HANS (eval) asymmetric entailment
3. BoolQNegDataset -- google/boolq with rule-based negation augmentation

All follow the pattern:
  1. Check for cached .pt embeddings in data_root/lqa/
  2. If missing: download from HuggingFace, encode with frozen sentence-transformers, cache
  3. Return (embeddings, labels)
"""

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from log import get_logger

logger = get_logger(__name__)

# Helpers


def _get_encoder(encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a frozen sentence-transformer encoder."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for LQA datasets. Install via: uv pip install sentence-transformers"
        )
    model = SentenceTransformer(encoder_name)
    return model


def _encode_texts(texts: list[str], encoder, batch_size: int = 256) -> torch.Tensor:
    """Encode a list of texts into embeddings [N, encoder_dim]."""
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    return embeddings.cpu()


def _cache_path(data_root: str, name: str) -> Path:
    return Path(data_root) / "lqa" / f"{name}.pt"


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_hf_dataset(path: str, name: str = None, split: str = "train", trust_remote_code: bool = False):
    """Load a HuggingFace dataset."""
    from datasets import load_dataset

    kwargs = {"path": path, "split": split, "trust_remote_code": trust_remote_code}
    if name is not None:
        kwargs["name"] = name
    return load_dataset(**kwargs)


# CLUTRR Dataset -- Compositional Chain Reasoning (Real)

# CLUTRR relation types (18 total in CLUTRR/v1)
CLUTRR_RELATIONS = [
    "father",
    "mother",
    "son",
    "daughter",
    "brother",
    "sister",
    "grandfather",
    "grandmother",
    "grandson",
    "granddaughter",
    "uncle",
    "aunt",
    "nephew",
    "niece",
    "father-in-law",
    "mother-in-law",
    "son-in-law",
    "daughter-in-law",
]


def _split_story_sentences(story: str) -> list[str]:
    """Split a CLUTRR story into individual sentences."""
    # Split on sentence boundaries: '. ' or '.\n' or end-of-string period
    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
    return [s.strip() for s in sentences if s.strip()]


def _clutrr_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length chain samples with padding."""
    emb_list = [sample["sentence_embeddings"] for sample in batch]
    padded_embs = pad_sequence(emb_list, batch_first=True, padding_value=0.0)

    lengths = torch.tensor([sample["chain_length"] for sample in batch])
    labels = torch.stack([sample["label"] for sample in batch])

    return {
        "sentence_embeddings": padded_embs,  # [B, L_max, encoder_dim]
        "chain_length": lengths,  # [B]
        "label": labels,  # [B]
    }


class CLUTRRDataset(Dataset):
    """CLUTRR compositional chain reasoning dataset (real HuggingFace data).

    Uses CLUTRR/v1 from HuggingFace: stories with multi-hop kinship reasoning.
    Each sample: per-sentence embeddings [L, 384] + final relation label.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_samples: int = None,
    ):
        cache_name = f"clutrr_hf_{split}" + (f"_{n_samples}" if n_samples else "")
        cache = _cache_path(data_root, cache_name)
        if cache.exists():
            logger.info("Loading cached CLUTRR %s from %s", split, cache)
            cached = torch.load(cache, weights_only=False)
            self.sentence_embeddings = cached["sentence_embeddings"]
            self.chain_lengths = cached["chain_lengths"]
            self.labels = cached["labels"]
        else:
            logger.info("Downloading CLUTRR %s from HuggingFace...", split)
            hf_split = "train" if split == "train" else "test"
            ds = _load_hf_dataset("CLUTRR/v1", "gen_train234_test2to10", hf_split, trust_remote_code=True)

            if n_samples is not None and n_samples < len(ds):
                ds = ds.select(range(n_samples))

            # Build relation label mapping from data
            relation_to_idx = {r: i for i, r in enumerate(CLUTRR_RELATIONS)}

            all_sentences = []
            sentence_offsets = []
            labels = []
            chain_lengths = []

            for row in ds:
                story = row.get("story", row.get("clean_story", ""))
                target = row.get("target", row.get("target_text", ""))

                # Split story into sentences
                sents = _split_story_sentences(story)
                if not sents:
                    continue

                # Map target relation to index
                target_lower = target.lower().strip() if isinstance(target, str) else ""
                label_idx = relation_to_idx.get(target_lower, -1)
                if label_idx == -1:
                    # Try partial match
                    for r, idx in relation_to_idx.items():
                        if r in target_lower or target_lower in r:
                            label_idx = idx
                            break
                if label_idx == -1:
                    label_idx = 0  # fallback

                start = len(all_sentences)
                all_sentences.extend(sents)
                sentence_offsets.append((start, len(all_sentences)))
                labels.append(label_idx)
                chain_lengths.append(len(sents))

            logger.info("Encoding %d sentences with %s...", len(all_sentences), encoder_name)
            encoder = _get_encoder(encoder_name)
            all_embs = _encode_texts(all_sentences, encoder)

            self.sentence_embeddings = []
            for s, e in sentence_offsets:
                self.sentence_embeddings.append(all_embs[s:e])

            self.chain_lengths = torch.tensor(chain_lengths, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.long)

            _ensure_dir(cache)
            torch.save(
                {
                    "sentence_embeddings": self.sentence_embeddings,
                    "chain_lengths": self.chain_lengths,
                    "labels": self.labels,
                },
                cache,
            )
            logger.info("Cached CLUTRR %s to %s (%d samples)", split, cache, len(self.labels))

        self.num_relations = len(CLUTRR_RELATIONS)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sentence_embeddings": self.sentence_embeddings[idx],
            "chain_length": self.chain_lengths[idx],
            "label": self.labels[idx],
        }


# HANS Dataset -- Asymmetric Entailment (SNLI train + HANS eval)


class HANSDataset(Dataset):
    """SNLI (train) + HANS (eval) for entailment asymmetry testing.

    Train: stanfordnlp/snli -- 550k premise/hypothesis pairs, 3-way labels.
    Eval: jhu-cogsci/hans -- 30k adversarial NLI examples.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_samples: int = None,
    ):
        cache_name = f"hans_bin_{split}" + (f"_{n_samples}" if n_samples else "")
        cache = _cache_path(data_root, cache_name)
        if cache.exists():
            logger.info("Loading cached HANS %s from %s", split, cache)
            cached = torch.load(cache, weights_only=False)
            self.premise_emb = cached["premise_emb"]
            self.hypothesis_emb = cached["hypothesis_emb"]
            self.labels = cached["labels"]
        else:
            if split == "train":
                logger.info("Downloading SNLI train from HuggingFace...")
                ds = _load_hf_dataset("stanfordnlp/snli", split="train")
                # Filter invalid labels (-1)
                ds = ds.filter(lambda x: x["label"] != -1)
                if n_samples is not None and n_samples < len(ds):
                    ds = ds.select(range(n_samples))

                premises = ds["premise"]
                hypotheses = ds["hypothesis"]
                # Binary: 0=entailment -> 1.0, {1=neutral, 2=contradiction} -> 0.0
                labels = [1.0 if l == 0 else 0.0 for l in ds["label"]]
            else:
                logger.info("Downloading HANS validation from HuggingFace...")
                ds = _load_hf_dataset("jhu-cogsci/hans", split="validation", trust_remote_code=True)
                if n_samples is not None and n_samples < len(ds):
                    ds = ds.select(range(n_samples))

                premises = ds["premise"]
                hypotheses = ds["hypothesis"]
                # HANS: 0=entailment -> 1.0, 1=non-entailment -> 0.0
                labels = [1.0 if l == 0 else 0.0 for l in ds["label"]]

            logger.info("Encoding %d premise/hypothesis pairs...", len(premises))
            encoder = _get_encoder(encoder_name)
            self.premise_emb = _encode_texts(list(premises), encoder)
            self.hypothesis_emb = _encode_texts(list(hypotheses), encoder)
            self.labels = torch.tensor(labels, dtype=torch.float32)

            _ensure_dir(cache)
            torch.save(
                {
                    "premise_emb": self.premise_emb,
                    "hypothesis_emb": self.hypothesis_emb,
                    "labels": self.labels,
                },
                cache,
            )
            logger.info("Cached HANS %s to %s (%d samples)", split, cache, len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "premise_emb": self.premise_emb[idx],
            "hypothesis_emb": self.hypothesis_emb[idx],
            "label": self.labels[idx],
        }


# BoolQ-Neg Dataset -- Negation Sensitivity (Real)

_NEGATION_PREFIXES = [
    ("Is", "Isn't"),
    ("Can", "Can't"),
    ("Does", "Doesn't"),
    ("Do", "Don't"),
    ("Was", "Wasn't"),
    ("Were", "Weren't"),
    ("Has", "Hasn't"),
    ("Have", "Haven't"),
    ("Will", "Won't"),
    ("Are", "Aren't"),
    ("Did", "Didn't"),
    ("Could", "Couldn't"),
    ("Would", "Wouldn't"),
    ("Should", "Shouldn't"),
]


def _negate_question(q: str) -> str:
    """Apply rule-based negation to a question."""
    for pos, neg in _NEGATION_PREFIXES:
        if q.startswith(pos + " "):
            return neg + q[len(pos) :]
        if q.startswith(neg + " "):
            return pos + q[len(neg) :]
    parts = q.split(" ", 2)
    if len(parts) >= 2:
        return parts[0] + " not " + " ".join(parts[1:])
    return "not " + q


class BoolQNegDataset(Dataset):
    """BoolQ with negation augmentation for negation sensitivity testing.

    Uses google/boolq from HuggingFace. Each sample produces 2 entries:
    original question + negated question (answer flipped).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_samples: int = None,
    ):
        cache_name = f"boolqneg_hf_{split}" + (f"_{n_samples}" if n_samples else "")
        cache = _cache_path(data_root, cache_name)
        if cache.exists():
            logger.info("Loading cached BoolQ-Neg %s from %s", split, cache)
            cached = torch.load(cache, weights_only=False)
            self.passage_emb = cached["passage_emb"]
            self.question_emb = cached["question_emb"]
            self.is_negated = cached["is_negated"]
            self.answers = cached["answers"]
        else:
            hf_split = "train" if split == "train" else "validation"
            logger.info("Downloading BoolQ %s from HuggingFace...", hf_split)
            ds = _load_hf_dataset("google/boolq", split=hf_split)

            if n_samples is not None and n_samples < len(ds):
                ds = ds.select(range(n_samples))

            passages = []
            questions = []
            is_negated = []
            answers = []

            for row in ds:
                passage = row["passage"]
                question = row["question"]
                answer = row["answer"]

                # Original
                passages.append(passage)
                questions.append(question)
                is_negated.append(False)
                answers.append(int(answer))

                # Negated version
                neg_q = _negate_question(question)
                passages.append(passage)
                questions.append(neg_q)
                is_negated.append(True)
                answers.append(int(not answer))

            logger.info("Encoding %d passage/question pairs...", len(passages))
            encoder = _get_encoder(encoder_name)
            self.passage_emb = _encode_texts(passages, encoder)
            self.question_emb = _encode_texts(questions, encoder)
            self.is_negated = torch.tensor(is_negated, dtype=torch.bool)
            self.answers = torch.tensor(answers, dtype=torch.float32)

            _ensure_dir(cache)
            torch.save(
                {
                    "passage_emb": self.passage_emb,
                    "question_emb": self.question_emb,
                    "is_negated": self.is_negated,
                    "answers": self.answers,
                },
                cache,
            )
            logger.info("Cached BoolQ-Neg %s to %s (%d samples)", split, cache, len(self.answers))

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return {
            "passage_emb": self.passage_emb[idx],
            "question_emb": self.question_emb[idx],
            "is_negated": self.is_negated[idx],
            "answer": self.answers[idx],
        }


# Loader helper


def get_lqa_loaders(
    data_root: str = "data",
    probe: str = "chain",
    batch_size: int = 64,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_train: int = None,
    n_test: int = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Get train/test DataLoaders for a specific LQA probe.

    Args:
        data_root: Root data directory.
        probe: "chain", "entailment", or "negation".
        batch_size: Batch size.
        encoder_name: Sentence transformer model name.
        n_train: Max training samples (None = use full dataset).
        n_test: Max test samples (None = use full dataset).
        num_workers: DataLoader workers.
        pin_memory: Pin memory for CUDA.

    Returns:
        (train_loader, test_loader)
    """
    if probe == "chain":
        train_ds = CLUTRRDataset(data_root, "train", encoder_name, n_train)
        test_ds = CLUTRRDataset(data_root, "test", encoder_name, n_test)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_clutrr_collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_clutrr_collate_fn,
        )
    elif probe == "entailment":
        train_ds = HANSDataset(data_root, "train", encoder_name, n_train)
        test_ds = HANSDataset(data_root, "test", encoder_name, n_test)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    elif probe == "negation":
        train_ds = BoolQNegDataset(data_root, "train", encoder_name, n_train)
        test_ds = BoolQNegDataset(data_root, "test", encoder_name, n_test)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        raise ValueError(f"Unknown probe: {probe}. Use 'chain', 'entailment', or 'negation'.")

    return train_loader, test_loader
