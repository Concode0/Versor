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
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform


class PureRadiusGraph(BaseTransform):
    """Pure-PyTorch radius graph (no torch-cluster dependency)."""

    def __init__(self, r):
        self.r = r

    def forward(self, data):
        pos = data.pos
        dist = torch.cdist(pos, pos)  # [N, N]
        mask = (dist < self.r) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]
        data.edge_index = edge_index
        return data


class VersorMD17(MD17):
    """MD17 Wrapper for Versor. Molecular dynamics with energy and force prediction.

    Normalizes positions and provides access to energy and force targets.
    All 8 molecules supported: aspirin, benzene, ethanol, malonaldehyde,
    naphthalene, salicylic acid, toluene, uracil.
    """

    def __init__(self, root, molecule='aspirin', radius=5.0, transform=None, pre_transform=None):
        """Initialize MD17 dataset for a specific molecule.

        Args:
            root (str): Root directory for data storage
            molecule (str): Molecule name (aspirin, benzene, ethanol, etc.)
            radius (float): Cutoff radius for edge construction (in Angstrom)
            transform: Optional transform
            pre_transform: Optional pre-transform
        """
        # Use runtime transform for radius graph so edges are always computed
        # fresh and there's no dependency on torch-cluster.
        radius_graph = PureRadiusGraph(r=radius)
        if transform is None:
            transform = radius_graph
        super().__init__(root, name=molecule, transform=transform, pre_transform=pre_transform)
        self.molecule = molecule
        self.radius = radius

    def get(self, idx):
        """Get a single data point with centered positions.

        Returns:
            Data object with:
                - z: atomic numbers [N_atoms]
                - pos: atom positions [N_atoms, 3] (centered)
                - energy: total energy (scalar)
                - force: per-atom forces [N_atoms, 3]
        """
        data = super().get(idx)
        # Center positions (important for rotation invariance)
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data

def get_md17_loaders(root, molecule='aspirin', batch_size=32, max_samples=None, radius=5.0):
    """Loads MD17 dataset with deterministic train/val/test split.

    Split: 80% train, 10% val, 10% test

    Args:
        root (str): Root directory for data
        molecule (str): Molecule name
        batch_size (int): Batch size
        max_samples (int): Limit dataset size (for faster training)
        radius (float): Cutoff radius for edge construction (Angstrom)

    Returns:
        train_loader, val_loader, test_loader, energy_mean, energy_std, force_mean, force_std
    """
    dataset = VersorMD17(root=root, molecule=molecule, radius=radius)

    N = len(dataset)

    # Deterministic split
    g = torch.Generator()
    g.manual_seed(42)
    indices = torch.randperm(N, generator=g)

    if max_samples is not None and max_samples < N:
        indices = indices[:max_samples]
        N = max_samples

    train_size = int(0.8 * N)
    val_size = int(0.1 * N)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    # Compute normalization stats ONLY on training set
    # Access internal data structure
    if hasattr(dataset, '_data'):
        energy = dataset._data.energy
        force = dataset._data.force
    else:
        energy = dataset.data.energy
        force = dataset.data.force

    # Energy stats
    train_energy = energy[train_idx]
    energy_mean = train_energy.mean().item()
    energy_std = train_energy.std().item()

    # Force stats (flatten all force components from training set)
    train_force = force[train_idx].reshape(-1)  # Flatten all atoms and dimensions
    force_mean = train_force.mean().item()
    force_std = train_force.std().item()

    # Create dataloaders
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False)

    print(f">>> MD17 {molecule}: {N} samples ({train_size}/{val_size}/{len(test_idx)} train/val/test)")
    print(f">>> Energy: mean={energy_mean:.2f}, std={energy_std:.2f}")
    print(f">>> Force: mean={force_mean:.4f}, std={force_std:.4f}")

    return train_loader, val_loader, test_loader, energy_mean, energy_std, force_mean, force_std
