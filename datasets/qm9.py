# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

import torch
import numpy as np
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T

class VersorQM9(QM9):
    """QM9 Wrapper. Normalizes targets so gradients don't explode.
    """
    
    def __init__(self, root, target_idx=7, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.target_idx = target_idx
        
        if hasattr(self, '_data'):
            y = self._data.y
        else:
            y = self.data.y
            
        self.mean = y[:, target_idx].mean().item()
        self.std = y[:, target_idx].std().item()

    def get(self, idx):
        data = super().get(idx)
        # Center positions
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data

def get_qm9_loader(root, target='U0', batch_size=32, split='train', max_samples=None):
    """Loads QM9. Splits it. Batches it."""
    # 0:mu, 7:u0
    target_map = {'mu': 0, 'U0': 7}
    target_idx = target_map.get(target, 7)
    
    dataset = VersorQM9(root=root, target_idx=target_idx)
    
    N = len(dataset)
    indices = torch.randperm(N)
    
    if max_samples is not None and max_samples < N:
        indices = indices[:max_samples]
        N = max_samples
        
    train_size = int(0.8 * N)
    test_size = int(0.1 * N)
    
    if split == 'train':
        ds = dataset[indices[:train_size]]
    elif split == 'val':
        ds = dataset[indices[train_size:train_size+test_size]]
    else:
        ds = dataset[indices[train_size+test_size:]]
        
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train')), dataset.mean, dataset.std