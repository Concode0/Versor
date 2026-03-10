# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""MD17 / rMD17 dataset loader.

Supports both original MD17 and revised MD17 (rMD17) datasets.
rMD17 (Christensen & von Lilienfeld 2020) fixes DFT noise in the
original MD17 by recomputing energies/forces at CCSD(T) level.

Standard rMD17 protocol: 1000 train / 1000 val / rest test.
Split sizes are configurable via ``n_train`` / ``n_val`` parameters.

Loading priority:
    1. PyTorch Geometric (auto-downloads from official mirror)
    2. Raw .npz files in ``root/`` (manual download)
    3. Clear error with install instructions (no synthetic fallback -
       fake molecular dynamics data is meaningless for energy/force training)

Install PyG:
    uv sync --extra graph

Manual download (rMD17):
    Place rmd17_{molecule}.npz files under:
        data/rMD17/raw/rmd17/npz_data/rmd17_aspirin.npz
        data/rMD17/raw/rmd17/npz_data/rmd17_benzene.npz
        data/rMD17/raw/rmd17/npz_data/rmd17_ethanol.npz
        ...etc
    PyG's MD17 for revised datasets uses {root}/raw/ (not {root}/{name}/raw/).
    NOTE: archive.materialscloud.org (record 466) and sgdml.org are
    currently unavailable. Check https://zenodo.org or the rMD17 paper
    repository for an alternative mirror.
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as _TorchDataLoader

# Optional PyTorch Geometric import
_HAS_PYG = False
try:
    from torch_geometric.datasets import MD17 as _PyGMD17
    from torch_geometric.loader import DataLoader as _PyGDataLoader
    from torch_geometric.transforms import BaseTransform as _BaseTransform
    _HAS_PYG = True
except ImportError:
    _BaseTransform = object   # sentinel for class inheritance


class PureRadiusGraph(_BaseTransform if _HAS_PYG else object):
    """Pure-PyTorch radius graph (no torch-cluster dependency)."""

    def __init__(self, r: float):
        """Initialize Radius Graph transform.

        Args:
            r (float): Cutoff radius.
        """
        self.r = r

    def forward(self, data):
        """Build radius graph.

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            torch_geometric.data.Data: Data with edge_index.
        """
        pos = data.pos
        dist = torch.cdist(pos, pos)          # [N, N]
        mask = (dist < self.r) & (dist > 0)
        data.edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        return data

    def __call__(self, data):
        """Build radius graph (transform API).

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            torch_geometric.data.Data: Data with edge_index.
        """
        # Called as a transform (both PyG and non-PyG paths)
        return self.forward(data)


if _HAS_PYG:
    class VersorMD17(_PyGMD17):
        """MD17 / rMD17 Wrapper for Versor using PyTorch Geometric.

        Auto-downloads from official mirrors. Normalizes positions.
        Supports both original MD17 and revised MD17 (rMD17).

        rMD17 molecules: aspirin, azobenzene, benzene, ethanol,
        malonaldehyde, naphthalene, paracetamol, salicylic_acid,
        toluene, uracil.
        """

        def __init__(self, root: str, molecule: str = 'aspirin',
                     radius: float = 5.0, revised: bool = True,
                     transform=None, pre_transform=None):
            """Initialize MD17/rMD17 dataset.

            Args:
                root (str): Root directory.
                molecule (str): Molecule name.
                radius (float): Cutoff radius for graph.
                revised (bool): Use revised MD17 (default True).
                transform (callable, optional): Transform to apply.
                pre_transform (callable, optional): Pre-transform to apply.
            """
            radius_graph = PureRadiusGraph(r=radius)
            if transform is None:
                transform = radius_graph
            name = f"revised {molecule}" if revised else molecule
            super().__init__(root, name=name, transform=transform,
                             pre_transform=pre_transform)
            self.molecule = molecule
            self.radius = radius
            self.revised = revised

        def get(self, idx):
            """Get a single conformation.

            Args:
                idx (int): Conformation index.

            Returns:
                torch_geometric.data.Data: Centered graph data.
            """
            data = super().get(idx)
            data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
            return data

    def _get_pyg_normalization_stats(dataset, train_idx):
        """Compute normalization stats for energy and forces.

        Args:
            dataset (VersorMD17): MD17 dataset.
            train_idx (torch.Tensor): Indices for training set.

        Returns:
            Tuple[float, float, float, float]: energy_mean, energy_std, force_mean, force_std.
        """
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


class _GraphData:
    """Minimal PyG-Data-compatible object produced by the NPZ path.

    Attributes:
        pos (torch.Tensor): Atomic positions [N, 3].
        z (torch.Tensor): Atomic numbers [N].
        energy (torch.Tensor): Potential energy [].
        force (torch.Tensor): Atomic forces [N, 3].
        edge_index (torch.Tensor): Graph connectivity [2, E].
        num_nodes (int): Number of atoms N.
        batch (torch.Tensor | None): Batch indices for nodes.
    """

    __slots__ = ('pos', 'z', 'energy', 'force', 'edge_index', 'num_nodes', 'batch')

    def __init__(self, pos: torch.Tensor, z: torch.Tensor,
                 energy: torch.Tensor, force: torch.Tensor,
                 edge_index: torch.Tensor):
        """Initialize Graph Data object."""
        self.pos = pos
        self.z = z
        self.energy = energy
        self.force = force
        self.edge_index = edge_index
        self.num_nodes = pos.shape[0]
        self.batch = None   # filled by collate

    def to(self, device):
        """Move all tensors to device (mirrors PyG Data.to())."""
        self.pos = self.pos.to(device)
        self.z = self.z.to(device)
        self.energy = self.energy.to(device)
        self.force = self.force.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self


def _collate_graphs(batch):
    """Collate _GraphData objects into a single batch (mimics PyG DataLoader).

    Args:
        batch (list): List of _GraphData objects.

    Returns:
        _GraphData: Collated batch object.
    """
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


# Canonical NPZ filenames for original MD17
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

# Canonical NPZ filenames for revised MD17 (rMD17)
_RMD17_NPZ_PATTERNS = {
    'aspirin':         ['rmd17_aspirin.npz'],
    'azobenzene':      ['rmd17_azobenzene.npz'],
    'benzene':         ['rmd17_benzene.npz'],
    'ethanol':         ['rmd17_ethanol.npz'],
    'malonaldehyde':   ['rmd17_malonaldehyde.npz'],
    'naphthalene':     ['rmd17_naphthalene.npz'],
    'paracetamol':     ['rmd17_paracetamol.npz'],
    'salicylic_acid':  ['rmd17_salicylic_acid.npz'],
    'toluene':         ['rmd17_toluene.npz'],
    'uracil':          ['rmd17_uracil.npz'],
}


class VersorMD17NPZ(Dataset):
    """NPZ-based MD17/rMD17 loader (no PyTorch Geometric required).

    Expects manually downloaded .npz files in ``root/raw/`` or ``root/``.
    Download from http://www.sgdml.org/datasets/

    Produces _GraphData objects compatible with PyG-based task code.
    """

    def __init__(self, root: str, molecule: str = 'aspirin',
                 radius: float = 5.0, revised: bool = True):
        """Initialize NPZ-based MD17/rMD17 dataset.

        Args:
            root (str): Root directory.
            molecule (str): Molecule name.
            radius (float): Cutoff radius for graph.
            revised (bool): Use revised MD17 (default True).
        """
        self.root = root
        self.molecule = molecule
        self.radius = radius
        self.revised = revised
        self._radius_graph = PureRadiusGraph(r=radius)

        npz_path = self._find_npz()
        variant = "rMD17" if revised else "MD17"
        if npz_path is None:
            raise FileNotFoundError(
                f"{variant} NPZ file for '{molecule}' not found in {root}.\n"
                f"Options:\n"
                f"  1. Install PyG:  uv sync --extra graph\n"
                f"  2. Download NPZ from http://www.sgdml.org/datasets/"
            )

        data = np.load(npz_path)
        # rMD17 npz uses 'coords' key, original MD17 uses 'R'
        if 'coords' in data:
            R = torch.tensor(data['coords'], dtype=torch.float32)
        else:
            R = torch.tensor(data['R'], dtype=torch.float32)
        # rMD17 uses 'energies', original uses 'E'
        if 'energies' in data:
            E = torch.tensor(data['energies'], dtype=torch.float32).squeeze(-1)
        else:
            E = torch.tensor(data['E'], dtype=torch.float32).squeeze(-1)
        # rMD17 uses 'forces', original uses 'F'
        if 'forces' in data:
            F = torch.tensor(data['forces'], dtype=torch.float32)
        else:
            F = torch.tensor(data['F'], dtype=torch.float32)
        # rMD17 uses 'nuclear_charges', original uses 'z'
        if 'nuclear_charges' in data:
            z = torch.tensor(data['nuclear_charges'], dtype=torch.long)
        else:
            z = torch.tensor(data['z'], dtype=torch.long)

        self._R = R
        self._E = E
        self._F = F
        self._z = z
        self._edge_index_cache: Optional[torch.Tensor] = None
        print(f">>> {variant} NPZ ({molecule}): {len(R)} conformations, "
              f"{z.shape[0]} atoms")

    def _find_npz(self) -> Optional[str]:
        """Locate NPZ file for the molecule.

        Returns:
            str | None: Absolute path to the NPZ file or None if not found.
        """
        if self.revised:
            patterns = _RMD17_NPZ_PATTERNS
            default = [f'rmd17_{self.molecule}.npz']
        else:
            patterns = _NPZ_PATTERNS
            default = [f'md17_{self.molecule}.npz']
        candidates = patterns.get(self.molecule, default)
        search_dirs = [
            self.root,
            os.path.join(self.root, 'raw'),
            os.path.join(self.root, 'MD17', self.molecule),
            os.path.join(self.root, 'rMD17', self.molecule),
        ]
        for d in search_dirs:
            for name in candidates:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    return path
        return None

    def _get_edge_index(self, pos: torch.Tensor) -> torch.Tensor:
        """Build radius graph for a single conformation.

        Args:
            pos (torch.Tensor): Atomic positions [N, 3].

        Returns:
            torch.Tensor: Graph connectivity [2, E].
        """
        dist = torch.cdist(pos, pos)
        mask = (dist < self.radius) & (dist > 0)
        return mask.nonzero(as_tuple=False).t().contiguous()

    @property
    def energy(self) -> torch.Tensor:
        """Returns potential energies."""
        return self._E

    @property
    def force(self) -> torch.Tensor:
        """Returns atomic forces."""
        return self._F

    def __len__(self) -> int:
        """Returns conformation count."""
        return len(self._R)

    def __getitem__(self, idx: int) -> _GraphData:
        """Get a single conformation graph.

        Args:
            idx (int): Conformation index.

        Returns:
            _GraphData: Atom centered graph data.
        """
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


def _compute_split(N: int, n_train: Optional[int], n_val: Optional[int]):
    """Compute train/val/test split sizes.

    If n_train and n_val are given as absolute counts, use those and assign
    the rest to test. Otherwise fall back to 80/10/10 ratio split.

    Args:
        N (int): Total number of samples.
        n_train (int | None): Number of training samples (None for ratio split).
        n_val (int | None): Number of validation samples (None for ratio split).

    Returns:
        Tuple[int, int]: (train_size, val_size). Test size = N - train - val.
    """
    if n_train is not None and n_val is not None:
        train_size = min(n_train, N)
        val_size = min(n_val, N - train_size)
    else:
        train_size = int(0.8 * N)
        val_size = int(0.1 * N)
    return train_size, val_size


def get_md17_loaders(root: str, molecule: str = 'aspirin',
                     batch_size: int = 32,
                     max_samples: Optional[int] = None,
                     revised: bool = True,
                     n_train: Optional[int] = None,
                     n_val: Optional[int] = None,
                     radius: float = 5.0,
                     num_workers: int = 2,
                     pin_memory: bool = False):
    """Load MD17/rMD17 dataset with configurable train/val/test split.

    Standard rMD17 protocol: 1000 train / 1000 val / rest test.
    Set ``n_train`` and ``n_val`` to control split sizes directly,
    or leave as None for 80/10/10 ratio split (original MD17 convention).

    Automatically selects the best available loading strategy:
      1. PyTorch Geometric (auto-downloads, preferred)
      2. Raw .npz files from ``root/``

    Args:
        root (str): Root directory for data storage.
        molecule (str): Molecule name (aspirin, benzene, ethanol, ...).
        batch_size (int): Batch size.
        max_samples (int, optional): Cap the total dataset size (for fast iteration).
        revised (bool): Use revised MD17 dataset (default True).
        n_train (int, optional): Fixed number of training samples.
        n_val (int, optional): Fixed number of validation samples.
        radius (float): Radius graph cutoff in Angstrom.
        num_workers (int): Number of worker processes for loading.
        pin_memory (bool): If True, pin memory in DataLoader.

    Returns:
        tuple: (train_loader, val_loader, test_loader,
                energy_mean, energy_std, force_mean, force_std)
    """
    variant = "rMD17" if revised else "MD17"

    # Strategy 1: PyTorch Geometric
    if _HAS_PYG:
        dataset = VersorMD17(root=root, molecule=molecule, radius=radius,
                             revised=revised)
        N = len(dataset)

        g = torch.Generator()
        g.manual_seed(42)
        indices = torch.randperm(N, generator=g)

        if max_samples is not None and max_samples < N:
            indices = indices[:max_samples]
            N = max_samples

        train_size, val_size = _compute_split(N, n_train, n_val)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        energy_mean, energy_std, force_mean, force_std = \
            _get_pyg_normalization_stats(dataset, train_idx)

        train_loader = _PyGDataLoader(dataset[train_idx], batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=pin_memory)
        val_loader   = _PyGDataLoader(dataset[val_idx],   batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers,
                                       pin_memory=pin_memory)
        test_loader  = _PyGDataLoader(dataset[test_idx],  batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers,
                                       pin_memory=pin_memory)

        print(f">>> {variant} ({molecule}) via PyG: "
              f"{train_size}/{val_size}/{len(test_idx)} train/val/test")
        print(f">>> Energy: mean={energy_mean:.2f}, std={energy_std:.2f} | "
              f"Force: mean={force_mean:.4f}, std={force_std:.4f}")
        return (train_loader, val_loader, test_loader,
                energy_mean, energy_std, force_mean, force_std)

    # Strategy 2: Raw NPZ files
    warnings.warn(
        f"PyTorch Geometric not installed. Attempting to load {variant} from raw .npz files.\n"
        "For full functionality: uv sync --extra graph",
        ImportWarning, stacklevel=2
    )

    dataset = VersorMD17NPZ(root=root, molecule=molecule, radius=radius,
                            revised=revised)
    # ^ raises FileNotFoundError if .npz also missing

    N = len(dataset)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)

    if max_samples is not None and max_samples < N:
        indices = indices[:max_samples]
        N = max_samples

    train_size, val_size = _compute_split(N, n_train, n_val)
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
        shuffle=True, collate_fn=_collate_graphs,
        num_workers=num_workers, pin_memory=pin_memory)
    val_loader = _TorchDataLoader(
        Subset(dataset, val_idx.tolist()), batch_size=batch_size,
        shuffle=False, collate_fn=_collate_graphs,
        num_workers=num_workers, pin_memory=pin_memory)
    test_loader = _TorchDataLoader(
        Subset(dataset, test_idx.tolist()), batch_size=batch_size,
        shuffle=False, collate_fn=_collate_graphs,
        num_workers=num_workers, pin_memory=pin_memory)

    print(f">>> {variant} ({molecule}) via NPZ: "
          f"{train_size}/{val_size}/{len(test_idx)} train/val/test")
    print(f">>> Energy: mean={energy_mean:.2f}, std={energy_std:.2f} | "
          f"Force: mean={force_mean:.4f}, std={force_std:.4f}")
    return (train_loader, val_loader, test_loader,
            energy_mean, energy_std, force_mean, force_std)
