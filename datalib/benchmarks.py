# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""BIG-Bench Hard (BBH) data loading with curriculum learning support."""

import re
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# ---------------------------------------------------------------------------
# Task difficulty tiers for curriculum learning
# ---------------------------------------------------------------------------

TASK_TIERS = {
    1: [  # Binary (2 choices) — basic pattern matching
        'boolean_expressions',
        'navigate',
        'sports_understanding',
        'web_of_lies',
        'causal_judgement',
        'formal_fallacies',
    ],
    2: [  # Simple MC (2-4 choices) — moderate reasoning
        'disambiguation_qa',
        'hyperbaton',
        'snarks',
        'ruin_names',
        'logical_deduction_three_objects',
        'tracking_shuffled_objects_three_objects',
        'temporal_sequences',
    ],
    3: [  # Complex MC (5+ choices) — multi-step reasoning
        'date_understanding',
        'movie_recommendation',
        'penguins_in_a_table',
        'salient_translation_error_detection',
        'logical_deduction_five_objects',
        'tracking_shuffled_objects_five_objects',
        'reasoning_about_colored_objects',
        'geometric_shapes',
    ],
}

ALL_CURRICULUM_TASKS = [t for tier in sorted(TASK_TIERS) for t in TASK_TIERS[tier]]


def get_tier_for_task(task_name: str) -> int:
    for tier, tasks in TASK_TIERS.items():
        if task_name in tasks:
            return tier
    return 3


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

BINARY_ANSWERS = {
    'True': 1, 'False': 0,
    'true': 1, 'false': 0,
    'Yes': 1, 'No': 0,
    'yes': 1, 'no': 0,
    'Valid': 1, 'Invalid': 0,
    'valid': 1, 'invalid': 0,
}

MC_PATTERN = re.compile(r'^\(([A-Z])\)$')


def _parse_answer(target: str, task_name: str) -> tuple:
    """Parse a BBH target string into (label_index, num_choices).

    Returns:
        (label_index, num_choices) tuple, or (None, None) if unparseable.
    """
    target = target.strip()

    if target in BINARY_ANSWERS:
        return BINARY_ANSWERS[target], 2

    mc_match = MC_PATTERN.match(target)
    if mc_match:
        letter = mc_match.group(1)
        idx = ord(letter) - ord('A')
        return idx, None  # num_choices determined by scanning all examples

    # Unparseable (free-text answer) — skip gracefully
    return None, None


# ---------------------------------------------------------------------------
# Single-task dataset
# ---------------------------------------------------------------------------

class BBHDataset(Dataset):
    """BIG-Bench Hard dataset for a single task.

    Loads from the lukaemon/bbh HuggingFace dataset, tokenizes with a
    provided tokenizer, and maps answers to class indices.
    Examples with unparseable answers are silently skipped.
    """

    def __init__(self, task_name: str, tokenizer, max_len: int = 512,
                 split: str = 'test', num_choices: int = None):
        from datasets import load_dataset
        ds = load_dataset("lukaemon/bbh", task_name, trust_remote_code=True)

        if split in ds:
            raw = ds[split]
        else:
            raw = ds[list(ds.keys())[0]]

        # Parse answers, skip unparseable
        parsed = []
        texts = []
        max_choice = 0
        for example in raw:
            label, nc = _parse_answer(example['target'], task_name)
            if label is None:
                continue
            if nc is not None:
                max_choice = max(max_choice, nc)
            else:
                max_choice = max(max_choice, label + 1)
            parsed.append(label)
            texts.append(example['input'])

        self.num_choices = num_choices or max_choice
        self.labels = parsed

        encodings = tokenizer(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Multi-task curriculum dataset
# ---------------------------------------------------------------------------

class BBHCurriculumDataset(Dataset):
    """Multi-task BBH dataset with per-example curriculum metadata.

    Each example carries its task_id, tier, and num_valid_choices so the
    training loop can mask invalid logits and the curriculum sampler can
    select examples by difficulty tier.
    """

    def __init__(self, task_names, tokenizer, max_len: int = 512):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_num_valid = []
        all_task_ids = []
        all_tiers = []

        self.task_names = []
        self.task_num_choices = {}
        max_choices = 0

        for task_id, task_name in enumerate(task_names):
            try:
                ds = BBHDataset(task_name, tokenizer, max_len)
            except Exception as e:
                print(f"  Warning: skipping task {task_name}: {e}")
                continue

            if len(ds) == 0:
                print(f"  Warning: no parseable examples for {task_name}")
                continue

            nc = ds.num_choices
            self.task_names.append(task_name)
            self.task_num_choices[task_name] = nc
            max_choices = max(max_choices, nc)
            tier = get_tier_for_task(task_name)

            for i in range(len(ds)):
                all_input_ids.append(ds.input_ids[i])
                all_attention_masks.append(ds.attention_mask[i])
                all_labels.append(ds.labels[i])
                all_num_valid.append(nc)
                all_task_ids.append(task_id)
                all_tiers.append(tier)

        self.input_ids = torch.stack(all_input_ids)
        self.attention_mask = torch.stack(all_attention_masks)
        self.labels = all_labels
        self.num_valid_choices = all_num_valid
        self.task_ids = all_task_ids
        self.tiers = all_tiers
        self.max_choices = max_choices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'num_valid_choices': torch.tensor(self.num_valid_choices[idx],
                                              dtype=torch.long),
            'task_id': torch.tensor(self.task_ids[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Curriculum sampler
# ---------------------------------------------------------------------------

class CurriculumSampler(Sampler):
    """Samples only from examples whose tier is in the active set."""

    def __init__(self, tiers: list, active_tier_ids: set):
        active = set(active_tier_ids)
        self.indices = [i for i, t in enumerate(tiers) if t in active]

    def __iter__(self):
        perm = torch.randperm(len(self.indices))
        return iter([self.indices[i] for i in perm])

    def __len__(self):
        return len(self.indices)


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def get_bbh_loaders(
    task_name: str,
    tokenizer,
    batch_size: int = 16,
    max_len: int = 512,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    num_choices: int = None,
) -> dict:
    """Create train/val DataLoaders for a single BBH task."""
    dataset = BBHDataset(task_name, tokenizer, max_len, num_choices=num_choices)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator,
    )

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        'num_choices': dataset.num_choices,
    }


def get_curriculum_loaders(
    task_names: list,
    tokenizer,
    max_len: int = 512,
    train_ratio: float = 0.8,
) -> dict:
    """Load all tasks into a single curriculum dataset with train/val split.

    Returns a dict with dataset objects and tier metadata.  The experiment
    script builds DataLoaders on the fly with CurriculumSampler.
    """
    dataset = BBHCurriculumDataset(task_names, tokenizer, max_len)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator,
    )

    # Map tiers through the subset indices
    train_tiers = [dataset.tiers[i] for i in train_ds.indices]
    val_tiers = [dataset.tiers[i] for i in val_ds.indices]

    return {
        'full_dataset': dataset,
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'train_tiers': train_tiers,
        'val_tiers': val_tiers,
        'max_choices': dataset.max_choices,
        'task_names': dataset.task_names,
        'task_num_choices': dataset.task_num_choices,
    }
