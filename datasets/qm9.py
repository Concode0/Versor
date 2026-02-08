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
    """QM9 Wrapper. Normalizes targets to ensure gradient stability.
    """
    
    def __init__(self, root, target_idx=7, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.target_idx = target_idx

    def get(self, idx):
        data = super().get(idx)
        # Center positions
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data

def get_qm9_loaders(root, target='U0', batch_size=32, max_samples=None):
    """Loads QM9. Splits it deterministically. Batches it.
    
    Returns:
        train_loader, val_loader, test_loader, train_mean, train_std
    """
    # 0:mu, 7:u0
    target_map = {'mu': 0, 'U0': 7}
    target_idx = target_map.get(target, 7)
    
    dataset = VersorQM9(root=root, target_idx=target_idx)
    
    N = len(dataset)
    # Deterministic split
    g = torch.Generator()
    g.manual_seed(42)
    indices = torch.randperm(N, generator=g)
    
    if max_samples is not None and max_samples < N:
        indices = indices[:max_samples]
        N = max_samples
        
    train_size = int(0.8 * N)
    test_size = int(0.1 * N)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+test_size]
    test_idx = indices[train_size+test_size:]
    
    # Compute stats ONLY on training set
    # Accessing internal data structure for efficiency
    if hasattr(dataset, '_data'):
        y = dataset._data.y
    else:
        y = dataset.data.y
        
    train_y = y[train_idx, target_idx]
    mean = train_y.mean().item()
    std = train_y.std().item()
    
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader, test_loader, mean, std