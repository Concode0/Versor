# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Atom type mapping (common atoms in protein-ligand complexes)
ATOM_TYPES = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
AA_TYPES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_TYPES)}


def _build_radius_graph(pos, cutoff=5.0):
    """Build edges between atoms within cutoff distance.

    Args:
        pos: Atom positions [N, 3]
        cutoff: Distance cutoff in Angstrom

    Returns:
        edge_index: [2, E] tensor of edges
    """
    dist = torch.cdist(pos, pos)
    mask = (dist < cutoff) & (dist > 0)
    edge_index = mask.nonzero(as_tuple=False).t()
    return edge_index


class PDBBindDataset(Dataset):
    """PDBbind dataset for protein-ligand binding affinity prediction.

    Loads pre-processed protein-ligand complexes with binding affinity labels.
    Supports both refined and general sets.

    Each sample contains:
        - Protein pocket atoms (positions, types, amino acid types)
        - Ligand atoms (positions, types)
        - Binding affinity (pKd/pKi)
    """

    def __init__(self, root, version='refined', pocket_cutoff=10.0,
                 max_protein_atoms=1000, max_ligand_atoms=100, split='train'):
        """Initialize PDBbind dataset.

        Args:
            root: Root directory containing PDBbind data
            version: 'refined' (~5k) or 'general' (~20k)
            pocket_cutoff: Angstrom cutoff for binding pocket extraction
            max_protein_atoms: Max protein atoms to keep
            max_ligand_atoms: Max ligand atoms to keep
            split: 'train', 'val', or 'test'
        """
        self.root = root
        self.version = version
        self.pocket_cutoff = pocket_cutoff
        self.max_protein_atoms = max_protein_atoms
        self.max_ligand_atoms = max_ligand_atoms
        self.split = split
        self.data_list = []

        self._load_data()

    def _load_data(self):
        """Load pre-processed data or generate synthetic samples."""
        data_path = os.path.join(self.root, f'{self.version}_{self.split}.pt')

        if os.path.exists(data_path):
            self.data_list = torch.load(data_path, weights_only=False)
            return

        # Generate synthetic data for development/testing
        print(f">>> PDBbind data not found at {data_path}, generating synthetic data")
        self._generate_synthetic_data()

    def _generate_synthetic_data(self, num_samples=500):
        """Generate synthetic protein-ligand data for development.

        Creates plausible mock data with realistic distributions.
        """
        rng = np.random.RandomState(42 if self.split == 'train' else 123)

        for _ in range(num_samples):
            n_prot = rng.randint(50, min(200, self.max_protein_atoms))
            n_lig = rng.randint(10, min(40, self.max_ligand_atoms))

            # Protein pocket centered at origin
            protein_pos = torch.tensor(rng.randn(n_prot, 3) * 5.0, dtype=torch.float32)
            protein_z = torch.tensor(rng.choice([6, 7, 8, 16], size=n_prot), dtype=torch.long)
            protein_aa = torch.tensor(rng.randint(0, 20, size=n_prot), dtype=torch.long)

            # Ligand near pocket center
            ligand_center = torch.tensor(rng.randn(3) * 2.0, dtype=torch.float32)
            ligand_pos = ligand_center + torch.tensor(rng.randn(n_lig, 3) * 1.5, dtype=torch.float32)
            ligand_z = torch.tensor(rng.choice([6, 7, 8, 9, 16], size=n_lig), dtype=torch.long)

            # Affinity correlated with distance (closer = stronger binding)
            mean_dist = torch.cdist(protein_pos, ligand_pos).min(dim=0)[0].mean().item()
            affinity = 8.0 - mean_dist * 0.5 + rng.randn() * 0.5

            protein_edge_index = _build_radius_graph(protein_pos, cutoff=5.0)
            ligand_edge_index = _build_radius_graph(ligand_pos, cutoff=4.0)

            self.data_list.append({
                'protein_pos': protein_pos,
                'protein_z': protein_z,
                'protein_aa': protein_aa,
                'ligand_pos': ligand_pos,
                'ligand_z': ligand_z,
                'affinity': torch.tensor(affinity, dtype=torch.float32),
                'protein_edge_index': protein_edge_index,
                'ligand_edge_index': ligand_edge_index,
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_pdbbind(batch):
    """Custom collate for variable-size protein-ligand graphs.

    Pads and batches protein and ligand data separately.

    Args:
        batch: List of sample dicts

    Returns:
        Batched dict with padded tensors and batch indices
    """
    protein_pos_list = []
    protein_z_list = []
    protein_aa_list = []
    ligand_pos_list = []
    ligand_z_list = []
    affinity_list = []
    protein_batch_list = []
    ligand_batch_list = []
    protein_edge_list = []
    ligand_edge_list = []

    prot_offset = 0
    lig_offset = 0

    for i, sample in enumerate(batch):
        n_prot = sample['protein_pos'].size(0)
        n_lig = sample['ligand_pos'].size(0)

        protein_pos_list.append(sample['protein_pos'])
        protein_z_list.append(sample['protein_z'])
        protein_aa_list.append(sample['protein_aa'])
        ligand_pos_list.append(sample['ligand_pos'])
        ligand_z_list.append(sample['ligand_z'])
        affinity_list.append(sample['affinity'])

        protein_batch_list.append(torch.full((n_prot,), i, dtype=torch.long))
        ligand_batch_list.append(torch.full((n_lig,), i, dtype=torch.long))

        if sample['protein_edge_index'].numel() > 0:
            protein_edge_list.append(sample['protein_edge_index'] + prot_offset)
        if sample['ligand_edge_index'].numel() > 0:
            ligand_edge_list.append(sample['ligand_edge_index'] + lig_offset)

        prot_offset += n_prot
        lig_offset += n_lig

    return {
        'protein_pos': torch.cat(protein_pos_list, dim=0),
        'protein_z': torch.cat(protein_z_list, dim=0),
        'protein_aa': torch.cat(protein_aa_list, dim=0),
        'ligand_pos': torch.cat(ligand_pos_list, dim=0),
        'ligand_z': torch.cat(ligand_z_list, dim=0),
        'affinity': torch.stack(affinity_list),
        'protein_batch': torch.cat(protein_batch_list, dim=0),
        'ligand_batch': torch.cat(ligand_batch_list, dim=0),
        'protein_edge_index': torch.cat(protein_edge_list, dim=1) if protein_edge_list else torch.zeros(2, 0, dtype=torch.long),
        'ligand_edge_index': torch.cat(ligand_edge_list, dim=1) if ligand_edge_list else torch.zeros(2, 0, dtype=torch.long),
    }


def get_pdbbind_loaders(root, version='refined', batch_size=8, max_samples=None,
                        pocket_cutoff=10.0, max_protein_atoms=1000, max_ligand_atoms=100):
    """Load PDBbind dataset with train/val/test splits.

    Args:
        root: Data root directory
        version: 'refined' or 'general'
        batch_size: Batch size
        max_samples: Limit samples per split
        pocket_cutoff: Pocket extraction cutoff
        max_protein_atoms: Max protein atoms
        max_ligand_atoms: Max ligand atoms

    Returns:
        train_loader, val_loader, test_loader, affinity_mean, affinity_std
    """
    datasets = {}
    for split in ['train', 'val', 'test']:
        ds = PDBBindDataset(
            root=root, version=version, pocket_cutoff=pocket_cutoff,
            max_protein_atoms=max_protein_atoms, max_ligand_atoms=max_ligand_atoms,
            split=split
        )
        if max_samples and len(ds) > max_samples:
            ds.data_list = ds.data_list[:max_samples]
        datasets[split] = ds

    # Compute normalization stats from training set
    train_affinities = torch.stack([s['affinity'] for s in datasets['train'].data_list])
    aff_mean = train_affinities.mean().item()
    aff_std = train_affinities.std().item()

    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_pdbbind)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, collate_fn=collate_pdbbind)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_pdbbind)

    n_train = len(datasets['train'])
    n_val = len(datasets['val'])
    n_test = len(datasets['test'])
    print(f">>> PDBbind {version}: {n_train}/{n_val}/{n_test} train/val/test")
    print(f">>> Affinity: mean={aff_mean:.2f}, std={aff_std:.2f}")

    return train_loader, val_loader, test_loader, aff_mean, aff_std
