#!/usr/bin/env python
# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Standalone GTM analysis script — run on a pretrained checkpoint.

Usage:
    uv run python scripts/analyze_gtm.py --checkpoint gtm_arc_best.pt
    uv run python scripts/analyze_gtm.py --checkpoint gtm_arc_best.pt --device cuda
    uv run python scripts/analyze_gtm.py --checkpoint gtm_arc_best.pt --n-batches 5
"""

import argparse
import sys
import torch

sys.path.insert(0, '.')

from models.gtm.analysis import GTMAnalyzer
from datalib.arc import get_arc_loaders


def main():
    parser = argparse.ArgumentParser(description='GTM Explainability Analysis')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--data-dir', default='data/arc', help='ARC data directory')
    parser.add_argument('--n-batches', type=int, default=1,
                        help='Number of batches to analyze')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', default=None, help='Save report to file')
    args = parser.parse_args()

    # Load model
    print(f'Loading checkpoint: {args.checkpoint}')
    analyzer = GTMAnalyzer.from_checkpoint(args.checkpoint, device=args.device)

    # Print static analysis (no data needed)
    print()
    print(analyzer.format_instruction_report())
    temp_info = analyzer.analyze_temperature()
    print('=== Gumbel Temperature ===')
    for i, t in enumerate(temp_info['temperatures']):
        sharp = '*' if temp_info['is_sharp'][i] else ''
        print(f'  Step {i}: tau={t:.4f} {sharp}')
    print()

    # Load validation data
    loaders = get_arc_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        include_toy=True,
        toy_n_examples=500,
        num_demos=3,
        seed=123,
    )
    val_loader = loaders['val']

    # Dynamic analysis
    full_text = []
    total_cell_acc = 0
    total_grid_correct = 0
    total_grid_count = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.n_batches:
            break

        print(f'--- Batch {batch_idx} ---')
        report_text = analyzer.full_report(batch)
        print(report_text)
        full_text.append(f'=== Batch {batch_idx} ===\n{report_text}')

        report = analyzer.analyze(batch)
        total_cell_acc += report['cell_accuracy']
        total_grid_correct += report['grid_correct'].sum().item()
        total_grid_count += report['grid_correct'].shape[0]

    # Summary
    n = min(args.n_batches, len(val_loader))
    if n > 0:
        print(f'\n=== Overall ({n} batches) ===')
        print(f'  Avg cell accuracy: {total_cell_acc / n:.4f}')
        print(f'  Grid correct: {total_grid_correct}/{total_grid_count}')

    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n\n'.join(full_text))
        print(f'\nReport saved to {args.output}')


if __name__ == '__main__':
    main()
