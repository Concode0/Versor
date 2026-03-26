# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""ARC dataset loaders for the Geometric Turing Machine.

Key design choices:
  - Few-shot format: each example = (demo_pairs, test_input, test_output).
    The model sees K demo (input,output) pairs to infer the rule, then
    applies it to a test input to produce the test output.
  - 2D grid preservation: grids are padded to (H_max, W_max) and kept as
    2D tensors so GridCodec can directly read row/col without re-parsing.

Provides:
  - ToyARCDataset: procedurally generated ARC-like tasks (few-shot)
  - ARCDataset: original ARC-AGI JSON tasks (few-shot)
  - collate_arc: custom collation for variable-size few-shot ARC examples
  - get_arc_loaders: factory function for train/val DataLoaders
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


class ToyARCDataset(Dataset):
    """Procedurally generated ARC-like tasks in few-shot format.

    Each item is a task: K demo pairs + 1 test pair, all sharing
    the same transformation rule.

    Task types:
        0 - color_swap: swap two colors
        1 - rotate_90: rotate grid 90° clockwise
        2 - reflect_h: reflect horizontally
        3 - fill_rect: fill a rectangular region with a color
        4 - pattern_tile: tile a small pattern across the grid
        5 - border: add a colored border
    """

    TASK_TYPES = ['color_swap', 'rotate_90', 'reflect_h', 'fill_rect',
                  'pattern_tile', 'border']

    def __init__(self, n_examples: int = 5000, max_grid_size: int = 10,
                 min_grid_size: int = 3, task_types: list = None,
                 num_demos: int = 3, seed: int = 42):
        super().__init__()
        self.n_examples = n_examples
        self.max_grid_size = max_grid_size
        self.min_grid_size = min_grid_size
        self.task_types = task_types or list(range(len(self.TASK_TYPES)))
        self.num_demos = num_demos
        self.seed = seed
        self.examples = self._generate_all()

    def _generate_all(self):
        rng = random.Random(self.seed)
        examples = []
        for _ in range(self.n_examples):
            task_type = rng.choice(self.task_types)
            # All pairs in one task share the same rule parameters
            rule_params = self._sample_rule_params(task_type, rng)

            demo_pairs = []
            for _ in range(self.num_demos):
                h = rng.randint(self.min_grid_size, self.max_grid_size)
                w = rng.randint(self.min_grid_size, self.max_grid_size)
                inp, out = self._generate_one(task_type, h, w, rng, rule_params)
                demo_pairs.append({'input': inp, 'output': out})

            # Test pair (same rule, different grid)
            h = rng.randint(self.min_grid_size, self.max_grid_size)
            w = rng.randint(self.min_grid_size, self.max_grid_size)
            test_in, test_out = self._generate_one(task_type, h, w, rng, rule_params)

            examples.append({
                'demo_pairs': demo_pairs,
                'test_input': test_in,
                'test_output': test_out,
                'task_type': task_type,
            })
        return examples

    def _sample_rule_params(self, task_type, rng):
        """Sample rule-specific parameters (shared across all pairs in a task)."""
        if task_type == 0:  # color_swap
            c1, c2 = rng.sample(range(10), 2)
            return {'c1': c1, 'c2': c2}
        elif task_type == 3:  # fill_rect — color is shared, position varies
            return {'color': rng.randint(0, 9)}
        elif task_type == 4:  # pattern_tile — pattern is shared across demos
            ph = rng.randint(1, 3)
            pw = rng.randint(1, 3)
            pattern = [[rng.randint(0, 9) for _ in range(pw)] for _ in range(ph)]
            return {'pattern': pattern}
        elif task_type == 5:  # border
            return {'color': rng.randint(1, 9)}
        return {}

    def _generate_one(self, task_type, h, w, rng, rule_params):
        """Generate a single (input, output) grid pair using shared rule params."""
        grid = [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]
        inp = torch.tensor(grid, dtype=torch.long)

        if task_type == 0:  # color_swap
            c1, c2 = rule_params['c1'], rule_params['c2']
            out = inp.clone()
            out[inp == c1] = c2
            out[inp == c2] = c1
        elif task_type == 1:  # rotate_90 clockwise
            out = inp.rot90(-1, [0, 1])
        elif task_type == 2:  # reflect_h
            out = inp.flip(1)
        elif task_type == 3:  # fill_rect
            out = inp.clone()
            r1 = rng.randint(0, max(0, h - 2))
            r2 = rng.randint(r1 + 1, h)
            c1_r = rng.randint(0, max(0, w - 2))
            c2_r = rng.randint(c1_r + 1, w)
            out[r1:r2, c1_r:c2_r] = rule_params['color']
        elif task_type == 4:  # pattern_tile (shared pattern from rule_params)
            pattern_data = rule_params['pattern']
            ph = min(len(pattern_data), h)
            pw = min(len(pattern_data[0]), w)
            pattern = torch.tensor(
                [row[:pw] for row in pattern_data[:ph]],
                dtype=torch.long,
            )
            out = pattern.repeat(
                (h + ph - 1) // ph, (w + pw - 1) // pw
            )[:h, :w]
            inp = out.clone()
            n_corrupt = max(1, h * w // 5)
            for _ in range(n_corrupt):
                ri, ci = rng.randint(0, h - 1), rng.randint(0, w - 1)
                inp[ri, ci] = rng.randint(0, 9)
        elif task_type == 5:  # border
            color = rule_params['color']
            out = inp.clone()
            out[0, :] = color
            out[-1, :] = color
            out[:, 0] = color
            out[:, -1] = color
        else:
            out = inp.clone()

        return inp, out

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.examples[idx]


class ARCDataset(Dataset):
    """Original ARC-AGI JSON dataset in few-shot format.

    Each item is one ARC task: the 'train' pairs are demos, and each
    'test' pair becomes a separate example (with all train pairs as demos).

    Expects directory structure:
        data_dir/training/*.json
        data_dir/evaluation/*.json
    """

    def __init__(self, data_dir: str, split: str = 'training'):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.examples = self._load_all()

    def _load_all(self):
        examples = []
        task_dir = os.path.join(self.data_dir, self.split)
        if not os.path.isdir(task_dir):
            return examples

        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith('.json'):
                continue
            task_id = fname.replace('.json', '')
            fpath = os.path.join(task_dir, fname)
            with open(fpath, 'r') as f:
                task_data = json.load(f)

            # Demo pairs from 'train'
            demo_pairs = []
            for pair in task_data.get('train', []):
                demo_pairs.append({
                    'input': torch.tensor(pair['input'], dtype=torch.long),
                    'output': torch.tensor(pair['output'], dtype=torch.long),
                })

            # Each test pair becomes a separate example with shared demos
            for pair in task_data.get('test', []):
                examples.append({
                    'demo_pairs': demo_pairs,
                    'test_input': torch.tensor(pair['input'], dtype=torch.long),
                    'test_output': torch.tensor(pair['output'], dtype=torch.long),
                    'task_id': task_id,
                })

            # If no test pairs, use last train pair as test
            if not task_data.get('test') and demo_pairs:
                last = demo_pairs[-1]
                examples.append({
                    'demo_pairs': demo_pairs[:-1] if len(demo_pairs) > 1 else demo_pairs,
                    'test_input': last['input'],
                    'test_output': last['output'],
                    'task_id': task_id,
                })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _pad_grid_2d(grid, H_max, W_max, pad_value=0):
    """Pad a 2D grid [H, W] to [H_max, W_max]."""
    H, W = grid.shape
    padded = torch.full((H_max, W_max), pad_value, dtype=grid.dtype)
    padded[:H, :W] = grid
    return padded


def collate_arc(batch):
    """Custom collation for few-shot ARC examples.

    Each batch item has:
        demo_pairs: list of K dicts with 'input' [H,W] and 'output' [H,W]
        test_input: [H,W]
        test_output: [H,W]

    Returns dict with:
        'demo_inputs':  [B, K, H_max, W_max] padded demo inputs
        'demo_outputs': [B, K, H_max, W_max] padded demo outputs
        'demo_masks':   [B, K, H_max, W_max] bool (True=valid input cell)
        'demo_output_masks': [B, K, H_max, W_max] bool (True=valid output cell)
        'test_inputs':  [B, H_max, W_max] padded test input
        'test_outputs': [B, H_max, W_max] padded test output (-1 = pad)
        'test_masks':   [B, H_max, W_max] bool (True=valid cell)
        'num_demos':    [B] int — actual number of demo pairs per example
        'test_sizes':   list of (H, W) tuples for test outputs
        'input_sizes':  list of (H, W) tuples for test inputs
    """
    B = len(batch)

    # Find max K (demo pairs), max H, max W across all grids
    max_K = max(len(item['demo_pairs']) for item in batch)
    all_grids = []
    for item in batch:
        for dp in item['demo_pairs']:
            all_grids.append(dp['input'])
            all_grids.append(dp['output'])
        all_grids.append(item['test_input'])
        all_grids.append(item['test_output'])

    H_max = max(g.shape[0] for g in all_grids)
    W_max = max(g.shape[1] for g in all_grids)

    # Allocate tensors
    demo_inputs = torch.zeros(B, max_K, H_max, W_max, dtype=torch.long)
    demo_outputs = torch.full((B, max_K, H_max, W_max), -1, dtype=torch.long)
    demo_masks = torch.zeros(B, max_K, H_max, W_max, dtype=torch.bool)
    demo_output_masks = torch.zeros(B, max_K, H_max, W_max, dtype=torch.bool)
    test_inputs = torch.zeros(B, H_max, W_max, dtype=torch.long)
    test_outputs = torch.full((B, H_max, W_max), -1, dtype=torch.long)
    test_masks = torch.zeros(B, H_max, W_max, dtype=torch.bool)
    num_demos = torch.zeros(B, dtype=torch.long)
    test_sizes = []
    input_sizes = []

    for i, item in enumerate(batch):
        # Demo pairs (input and output may have different dimensions)
        K = len(item['demo_pairs'])
        num_demos[i] = K
        for k, dp in enumerate(item['demo_pairs']):
            di = dp['input']
            do = dp['output']
            dH, dW = di.shape
            demo_inputs[i, k, :dH, :dW] = di
            demo_masks[i, k, :dH, :dW] = True
            doH, doW = do.shape
            demo_outputs[i, k, :doH, :doW] = do
            demo_output_masks[i, k, :doH, :doW] = True

        # Test pair
        ti = item['test_input']
        to = item['test_output']
        tH, tW = ti.shape
        toH, toW = to.shape
        test_inputs[i, :tH, :tW] = ti
        test_outputs[i, :toH, :toW] = to
        test_masks[i, :tH, :tW] = True
        test_sizes.append((toH, toW))
        input_sizes.append((tH, tW))

    return {
        'demo_inputs': demo_inputs,
        'demo_outputs': demo_outputs,
        'demo_masks': demo_masks,
        'demo_output_masks': demo_output_masks,
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'test_masks': test_masks,
        'num_demos': num_demos,
        'test_sizes': test_sizes,
        'input_sizes': input_sizes,
    }


def get_arc_loaders(data_dir: str = 'data/arc', batch_size: int = 8,
                    include_toy: bool = True, toy_n_examples: int = 5000,
                    toy_max_grid_size: int = 10, num_workers: int = 0,
                    num_demos: int = 3, seed: int = 42,
                    pin_memory: bool = False,
                    epoch_samples: int = 0):
    """Create ARC train/val DataLoaders.

    Args:
        data_dir: Path to ARC JSON directory.
        batch_size: Batch size.
        include_toy: If True, augment training with ToyARC examples.
        toy_n_examples: Number of synthetic examples.
        toy_max_grid_size: Max grid dimension for synthetic examples.
        num_workers: DataLoader workers.
        num_demos: Number of demo pairs per task (for ToyARC).
        seed: Random seed.
        pin_memory: Pin memory for CUDA async transfers.
        epoch_samples: Samples per epoch (0 = use full dataset with shuffle).

    Returns:
        dict with 'train', 'val' DataLoaders and 'num_colors' (10).
    """
    datasets = []

    # Original ARC training data
    arc_train = ARCDataset(data_dir, split='training')
    if len(arc_train) > 0:
        datasets.append(arc_train)

    # Synthetic ToyARC data
    if include_toy:
        toy = ToyARCDataset(
            n_examples=toy_n_examples,
            max_grid_size=toy_max_grid_size,
            num_demos=num_demos,
            seed=seed,
        )
        datasets.append(toy)

    if datasets:
        train_dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        train_dataset = ToyARCDataset(
            n_examples=toy_n_examples,
            max_grid_size=toy_max_grid_size,
            num_demos=num_demos,
            seed=seed,
        )

    # Validation: ARC evaluation set, fallback to small ToyARC
    val_dataset = ARCDataset(data_dir, split='evaluation')
    if len(val_dataset) == 0:
        val_dataset = ToyARCDataset(
            n_examples=min(500, toy_n_examples // 10),
            max_grid_size=toy_max_grid_size,
            num_demos=num_demos,
            seed=seed + 1,
        )

    persistent = num_workers > 0

    if epoch_samples > 0:
        sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=epoch_samples,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_arc, num_workers=num_workers, drop_last=True,
            pin_memory=pin_memory, persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_arc, num_workers=num_workers, drop_last=True,
            pin_memory=pin_memory, persistent_workers=persistent,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_arc, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent,
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'num_colors': 10,
    }
