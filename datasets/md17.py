# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""MD17 dataset loader.

Loading priority:
    1. PyTorch Geometric (auto-downloads from official mirror)
    2. Raw .npz files in ``root/`` (manual download from sgdml.org)
    3. Clear error with install instructions (no synthetic fallback —
       fake molecular dynamics data is meaningless for energy/force training)

Install PyG:
    uv sync --extra graph

Manual download (from http://www.sgdml.org/datasets/md17):
    curl -O http://www.sgdml.org/datasets/md17/md17_aspirin.npz
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as _TorchDataLoader

# ---- Optional PyTorch Geometric import ----
_HAS_PYG = False
try:
    from torch_geometric.datasets import MD17 as _PyGMD17
    from torch_geometric.loader import DataLoader as _PyGDataLoader
    from torch_geometric.transforms import BaseTransform as _BaseTransform
    _HAS_PYG = True
except ImportError:
    _BaseTransform = object   # sentinel for class inheritance


# =====================================================================
# Radius graph (shared by both paths)
# =====================================================================

class PureRadiusGraph(_BaseTransform if _HAS_PYG else object):
    """Pure-PyTorch radius graph (no torch-cluster dependency)."""

    def __init__(self, r: float):
        self.r = r

    def forward(self, data):
        pos = data.pos
        dist = torch.cdist(pos, pos)          # [N, N]
        mask = (dist < self.r) & (dist > 0)
        data.edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        return data

    def __call__(self, data):
        # Called as a transform (both PyG and non-PyG paths)
        return self.forward(data)


# =====================================================================
# PyG-backed loader (primary path)
# =====================================================================

if _HAS_PYG:
    class VersorMD17(_PyGMD17):
        """MD17 Wrapper for Versor using PyTorch Geometric.

        Auto-downloads from official mirrors. Normalizes positions.

        All 8 molecules supported: aspirin, benzene, ethanol, malonaldehyde,
        naphthalene, salicylic_acid, toluene, uracil.
        """

        def __init__(self, root: str, molecule: str = 'aspirin',
                     radius: float = 5.0, transform=None, pre_transform=None):
            radius_graph = PureRadiusGraph(r=radius)
            if transform is None:
                transform = radius_graph
            super().__init__(root, name=molecule, transform=transform,
                             pre_transform=pre_transform)
            self.molecule = molecule
            self.radius = radius

        def get(self, idx):
            data = super().get(idx)
            data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
            return data

    def _get_pyg_normalization_stats(dataset, train_idx):
        if hasattr(dataset, '_data'):
            energy = dataset._data.energy
            force = dataset._data.force
        else:
            energy = dataset.data.energy
            force = dataset.data.force

        train_energy = energy[train_idx]
        energy_mean = train_energy.mean().item()
        energy_std = max(train_energy.std().item(), 1e-6)

        train_force = force[train_idx].reshape(-1)
        force_mean = train_force.mean().item()
        force_std = max(train_force.std().item(), 1e-6)
        return energy_mean, energy_std, force_mean, force_std


# =====================================================================
# NPZ-backed loader (fallback when PyG is not installed)
# =====================================================================

class _GraphData:
    """Minimal PyG-Data-compatible object produced by the NPZ path."""

    __slots__ = ('pos', 'z', 'energy', 'force', 'edge_index', 'num_nodes', 'batch')

    def __init__(self, pos: torch.Tensor, z: torch.Tensor,
                 energy: torch.Tensor, force: torch.Tensor,
                 edge_index: torch.Tensor):
        self.pos = pos
        self.z = z
        self.energy = energy
        self.force = force
        self.edge_index = edge_index
        self.num_nodes = pos.shape[0]
        self.batch = None   # filled by collate


def _collate_graphs(batch):
    """Collate _GraphData objects into a single batch (mimics PyG DataLoader)."""
    batch_idx = []
    pos_list, z_list, energy_list, force_list, edge_list = [], [], [], [], []
    node_offset = 0

    for i, data in enumerate(batch):
        n = data.num_nodes
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        pos_list.append(data.pos)
        z_list.append(data.z)
        energy_list.append(data.energy)
        force_list.append(data.force)
        edge_list.append(data.edge_index + node_offset)
        node_offset += n

    out = _GraphData(
        pos=torch.cat(pos_list, dim=0),
        z=torch.cat(z_list, dim=0),
        energy=torch.stack(energy_list),
        force=torch.cat(force_list, dim=0),
        edge_index=torch.cat(edge_list, dim=1) if edge_list else torch.zeros(2, 0, dtype=torch.long),
    )
    out.batch = torch.cat(batch_idx, dim=0)
    return out


# Canonical NPZ filenames (original MD17 + revised MD17)
_NPZ_PATTERNS = {
    'aspirin':         ['aspirin_dft.npz', 'md17_aspirin.npz'],
    'benzene':         ['benzene_dft.npz', 'md17_benzene2017.npz', 'md17_benzene.npz'],
    'ethanol':         ['ethanol_dft.npz', 'md17_ethanol.npz'],
    'malonaldehyde':   ['malonaldehyde_dft.npz', 'md17_malonaldehyde.npz'],
    'naphthalene':     ['naphthalene_dft.npz', 'md17_naphthalene.npz'],
    'salicylic_acid':  ['salicylic_acid_dft.npz', 'md17_salicylicacid.npz'],
    'toluene':         ['toluene_dft.npz', 'md17_toluene.npz'],
    'uracil':          ['uracil_dft.npz', 'md17_uracil.npz'],
}


class VersorMD17NPZ(Dataset):
    """NPZ-based MD17 loader (no PyTorch Geometric required).

    Expects manually downloaded .npz files in ``root/raw/`` or ``root/``.
    Download from http://www.sgdml.org/datasets/md17

    Produces _GraphData objects compatible with PyG-based task code.
    """

    def __init__(self, root: str, molecule: str = 'aspirin',
                 radius: float = 5.0):
        self.root = root
        self.molecule = molecule
        self.radius = radius
        self._radius_graph = PureRadiusGraph(r=radius)

        npz_path = self._find_npz()
        if npz_path is None:
            raise FileNotFoundError(
                f"MD17 NPZ file for '{molecule}' not found in {root}.\n"
                f"Options:\n"
                f"  1. Install PyG:  uv sync --extra graph\n"
                f"  2. Download NPZ from http://www.sgdml.org/datasets/md17/md17_{molecule}.npz"
            )

        data = np.load(npz_path)
        R = torch.tensor(data['R'], dtype=torch.float32)   # [N, natoms, 3]
        E = torch.tensor(data['E'], dtype=torch.float32).squeeze(-1)   # [N]
        F = torch.tensor(data['F'], dtype=torch.float32)   # [N, natoms, 3]
        z = torch.tensor(data['z'], dtype=torch.long)      # [natoms]

        self._R = R
        self._E = E
        self._F = F
        self._z = z
        self._edge_index_cache: Optional[torch.Tensor] = None
        print(f">>> MD17 NPZ ({molecule}): {len(R)} conformations, "
              f"{z.shape[0]} atoms")

    def _find_npz(self) -> Optional[str]:
        candidates = _NPZ_PATTERNS.get(self.molecule, [f'md17_{self.molecule}.npz'])
        search_dirs = [
            self.root,
            os.path.join(self.root, 'raw'),
            os.path.join(self.root, 'MD17', self.molecule),
        ]
        for d in search_dirs:
            for name in candidates:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    return path
        return None

    def _get_edge_index(self, pos: torch.Tensor) -> torch.Tensor:
        """Build radius graph for a single conformation."""
        dist = torch.cdist(pos, pos)
        mask = (dist < self.radius) & (dist > 0)
        return mask.nonzero(as_tuple=False).t().contiguous()

    @property
    def energy(self) -> torch.Tensor:
        return self._E

    @property
    def force(self) -> torch.Tensor:
        return self._F

    def __len__(self) -> int:
        return len(self._R)

    def __getitem__(self, idx: int) -> _GraphData:
        pos = self._R[idx]                         # [natoms, 3]
        pos = pos - pos.mean(dim=0, keepdim=True)  # center
        edge_index = self._get_edge_index(pos)

        return _GraphData(
            pos=pos,
            z=self._z,
            energy=self._E[idx],
            force=self._F[idx],
            edge_index=edge_index,
        )


# =====================================================================
# Unified loader function
# =====================================================================

def get_md17_loaders(root: str, molecule: str = 'aspirin',
                     batch_size: int = 32,
                     max_samples: Optional[int] = None,
                     radius: float = 5.0):
    """Load MD17 dataset with deterministic 80/10/10 train/val/test split.

    Automatically selects the best available loading strategy:
      1. PyTorch Geometric (auto-downloads, preferred)
      2. Raw .npz files from ``root/``

    Args:
        root:        Root directory for data storage.
        molecule:    Molecule name (aspirin, benzene, ethanol, ...).
        batch_size:  Batch size.
        max_samples: Cap the total dataset size (for fast iteration).
        radius:      Radius graph cutoff in Angstrom.

    Returns:
        (train_loader, val_loader, test_loader,
         energy_mean, energy_std, force_mean, force_std)
    """
    # --- Strategy 1: PyTorch Geometric ---
    if _HAS_PYG:
        dataset = VersorMD17(root=root, molecule=molecule, radius=radius)
        N = len(dataset)

        g = torch.Generator()
        g.manual_seed(42)
        indices = torch.randperm(N, generator=g)

        if max_samples is not None and max_samples < N:
            indices = indices[:max_samples]
            N = max_samples

        train_size = int(0.8 * N)
        val_size = int(0.1 * N)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        energy_mean, energy_std, force_mean, force_std = \
            _get_pyg_normalization_stats(dataset, train_idx)

        train_loader = _PyGDataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
        val_loader   = _PyGDataLoader(dataset[val_idx],   batch_size=batch_size, shuffle=False)
        test_loader  = _PyGDataLoader(dataset[test_idx],  batch_size=batch_size, shuffle=False)

        print(f">>> MD17 ({molecule}) via PyG: "
              f"{train_size}/{val_size}/{len(test_idx)} train/val/test")
        print(f">>> Energy: mean={energy_mean:.2f}, std={energy_std:.2f} | "
              f"Force: mean={force_mean:.4f}, std={force_std:.4f}")
        return (train_loader, val_loader, test_loader,
                energy_mean, energy_std, force_mean, force_std)

    # --- Strategy 2: Raw NPZ files ---
    warnings.warn(
        "PyTorch Geometric not installed. Attempting to load MD17 from raw .npz files.\n"
        "For full functionality: uv sync --extra graph",
        ImportWarning, stacklevel=2
    )

    dataset = VersorMD17NPZ(root=root, molecule=molecule, radius=radius)
    # ^ raises FileNotFoundError if .npz also missing

    N = len(dataset)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)

    if max_samples is not None and max_samples < N:
        indices = indices[:max_samples]
        N = max_samples

    train_size = int(0.8 * N)
    val_size = int(0.1 * N)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Normalization stats from training set
    train_energy = dataset.energy[train_idx]
    energy_mean = train_energy.mean().item()
    energy_std = max(train_energy.std().item(), 1e-6)

    train_force = dataset.force[train_idx].reshape(-1)
    force_mean = train_force.mean().item()
    force_std = max(train_force.std().item(), 1e-6)

    from torch.utils.data import Subset
    train_loader = _TorchDataLoader(
        Subset(dataset, train_idx.tolist()), batch_size=batch_size,
        shuffle=True, collate_fn=_collate_graphs
    )
    val_loader = _TorchDataLoader(
        Subset(dataset, val_idx.tolist()), batch_size=batch_size,
        shuffle=False, collate_fn=_collate_graphs
    )
    test_loader = _TorchDataLoader(
        Subset(dataset, test_idx.tolist()), batch_size=batch_size,
        shuffle=False, collate_fn=_collate_graphs
    )

    print(f">>> MD17 ({molecule}) via NPZ: "
          f"{train_size}/{val_size}/{len(test_idx)} train/val/test")
    print(f">>> Energy: mean={energy_mean:.2f}, std={energy_std:.2f} | "
          f"Force: mean={force_mean:.4f}, std={force_std:.4f}")
    return (train_loader, val_loader, test_loader,
            energy_mean, energy_std, force_mean, force_std)
