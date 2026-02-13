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
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Atom type mapping (common atoms in protein-ligand complexes)
ATOM_TYPES = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
ELEMENT_TO_Z = {v: v for v in ATOM_TYPES.values()}
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


def _parse_element(atom_name, element_col=''):
    """Extract element symbol from PDB atom name or element column."""
    element_col = element_col.strip()
    if element_col and element_col in ATOM_TYPES:
        return element_col
    name = atom_name.strip()
    # First non-digit character
    for c in name:
        if c.isalpha():
            if c + name[name.index(c)+1:name.index(c)+2] in ATOM_TYPES:
                return c + name[name.index(c)+1]
            return c
    return 'C'


def _parse_pdb_pocket(pdb_path, ligand_pos, pocket_cutoff=10.0, max_atoms=1000):
    """Parse protein PDB file and extract binding pocket atoms.

    Extracts heavy atoms (non-H) within pocket_cutoff of any ligand atom.

    Args:
        pdb_path: Path to protein PDB file
        ligand_pos: Ligand atom positions [N_lig, 3] numpy array
        pocket_cutoff: Distance cutoff for pocket extraction (Angstrom)
        max_atoms: Maximum number of pocket atoms

    Returns:
        positions: [N, 3] float32 numpy array (centered on pocket centroid)
        atomic_numbers: [N] int array
        aa_indices: [N] int array (amino acid type index)
    """
    positions = []
    elements = []
    aa_types = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            # Skip water and ligand HETATM
            if line.startswith('HETATM'):
                continue

            atom_name = line[12:16]
            element = line[76:78] if len(line) >= 78 else ''
            elem = _parse_element(atom_name, element)

            # Skip hydrogens
            if elem == 'H':
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            resname = line[17:20].strip()

            positions.append([x, y, z])
            elements.append(ATOM_TYPES.get(elem, 6))
            aa_types.append(AA_TO_IDX.get(resname, 0))

    if not positions:
        return None, None, None

    positions = np.array(positions, dtype=np.float32)

    # Extract pocket: atoms within cutoff of any ligand atom
    from scipy.spatial.distance import cdist
    dists = cdist(positions, ligand_pos)
    min_dists = dists.min(axis=1)
    pocket_mask = min_dists < pocket_cutoff

    if pocket_mask.sum() == 0:
        # Fallback: take closest atoms
        closest_idx = np.argsort(min_dists)[:min(100, len(min_dists))]
        pocket_mask = np.zeros(len(positions), dtype=bool)
        pocket_mask[closest_idx] = True

    positions = positions[pocket_mask]
    elements = np.array(elements)[pocket_mask]
    aa_types = np.array(aa_types)[pocket_mask]

    # Truncate if too many atoms
    if len(positions) > max_atoms:
        # Keep atoms closest to ligand centroid
        lig_center = ligand_pos.mean(axis=0)
        dists_to_center = np.linalg.norm(positions - lig_center, axis=1)
        keep_idx = np.argsort(dists_to_center)[:max_atoms]
        positions = positions[keep_idx]
        elements = elements[keep_idx]
        aa_types = aa_types[keep_idx]

    # Center on pocket centroid
    centroid = positions.mean(axis=0)
    positions = positions - centroid

    return positions, elements, aa_types


def _parse_sdf_ligand(sdf_path, max_atoms=100):
    """Parse ligand SDF/MOL2 file.

    Reads the first molecule from an SDF file.

    Args:
        sdf_path: Path to SDF file
        max_atoms: Maximum number of ligand atoms

    Returns:
        positions: [N, 3] float32 numpy array
        atomic_numbers: [N] int array
        centroid: [3] float32 (for pocket centering)
    """
    positions = []
    elements = []

    with open(sdf_path, 'r') as f:
        lines = f.readlines()

    # SDF V2000 format: atom count is on line 3 (0-indexed)
    if len(lines) < 4:
        return None, None, None

    # Parse counts line
    counts_line = lines[3]
    try:
        n_atoms = int(counts_line[:3].strip())
    except ValueError:
        return None, None, None

    for i in range(4, min(4 + n_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            elem = parts[3].strip()
        except (ValueError, IndexError):
            continue

        # Skip hydrogens
        if elem == 'H':
            continue

        positions.append([x, y, z])
        elements.append(ATOM_TYPES.get(elem, 6))

    if not positions:
        return None, None, None

    positions = np.array(positions, dtype=np.float32)
    centroid = positions.mean(axis=0)

    if len(positions) > max_atoms:
        positions = positions[:max_atoms]
        elements = elements[:max_atoms]

    return positions, np.array(elements), centroid


def _parse_affinity_index(index_path):
    """Parse PDBbind INDEX file for binding affinities.

    Expected format: PDB_code  resolution  year  -logKd/Ki=XX  Kd/Ki=XX  ...

    Returns:
        dict: {pdb_code: affinity_value}
    """
    affinities = {}

    with open(index_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            pdb_code = parts[0]
            # The affinity is typically the 4th column as -log(Kd/Ki)
            try:
                aff_str = parts[3]
                # Handle formats like "Kd=1.5nM" or just "6.5"
                if '=' in aff_str:
                    # e.g. "-logKd/Ki=6.50"
                    val = float(aff_str.split('=')[1])
                else:
                    val = float(aff_str)
                affinities[pdb_code] = val
            except (ValueError, IndexError):
                continue

    return affinities


def _process_pdbbind_raw(root, version='refined', pocket_cutoff=10.0,
                         max_protein_atoms=1000, max_ligand_atoms=100):
    """Process raw PDBbind data into tensor format.

    Expected directory structure:
        root/
          refined-set/ or v2020-other-PL/
            XXXX/           (PDB code)
              XXXX_protein.pdb
              XXXX_ligand.sdf
          index/
            INDEX_refined_data.2020

    Returns:
        list of sample dicts
    """
    # Find the data directory
    if version == 'refined':
        data_dir = os.path.join(root, 'refined-set')
        index_file = os.path.join(root, 'index', 'INDEX_refined_data.2020')
    else:
        data_dir = os.path.join(root, 'v2020-other-PL')
        index_file = os.path.join(root, 'index', 'INDEX_general_PL_data.2020')

    if not os.path.isdir(data_dir):
        return None

    # Try multiple index file patterns
    if not os.path.exists(index_file):
        candidates = glob.glob(os.path.join(root, 'index', 'INDEX_*_data*'))
        if candidates:
            index_file = candidates[0]
        else:
            # Try reading from data_dir
            candidates = glob.glob(os.path.join(root, 'INDEX*'))
            if candidates:
                index_file = candidates[0]
            else:
                print(f">>> Warning: No affinity index file found in {root}")
                return None

    affinities = _parse_affinity_index(index_file)
    if not affinities:
        print(f">>> Warning: No affinities parsed from {index_file}")
        return None

    data_list = []
    pdb_dirs = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 4])

    print(f">>> Processing {len(pdb_dirs)} PDBbind complexes...")

    for pdb_code in pdb_dirs:
        if pdb_code not in affinities:
            continue

        pdb_dir = os.path.join(data_dir, pdb_code)
        protein_pdb = os.path.join(pdb_dir, f'{pdb_code}_protein.pdb')
        ligand_sdf = os.path.join(pdb_dir, f'{pdb_code}_ligand.sdf')

        if not (os.path.exists(protein_pdb) and os.path.exists(ligand_sdf)):
            continue

        try:
            # Parse ligand first (need positions for pocket extraction)
            lig_pos, lig_z, lig_centroid = _parse_sdf_ligand(ligand_sdf, max_ligand_atoms)
            if lig_pos is None or len(lig_pos) < 3:
                continue

            # Parse protein pocket around ligand
            prot_pos, prot_z, prot_aa = _parse_pdb_pocket(
                protein_pdb, lig_pos, pocket_cutoff, max_protein_atoms
            )
            if prot_pos is None or len(prot_pos) < 5:
                continue

            # Center ligand on same centroid as protein pocket
            pocket_centroid = prot_pos.mean(axis=0)
            # protein already centered in _parse_pdb_pocket
            lig_pos_centered = lig_pos - (lig_centroid - np.zeros(3))
            # Re-center ligand relative to pocket
            lig_pos_centered = lig_pos - (lig_pos.mean(axis=0) - prot_pos.mean(axis=0))
            # Since protein is already centered, center ligand relative to pocket centroid
            lig_pos_centered = lig_pos - lig_pos.mean(axis=0)

            prot_pos_t = torch.tensor(prot_pos, dtype=torch.float32)
            prot_z_t = torch.tensor(prot_z, dtype=torch.long)
            prot_aa_t = torch.tensor(prot_aa, dtype=torch.long)
            lig_pos_t = torch.tensor(lig_pos_centered, dtype=torch.float32)
            lig_z_t = torch.tensor(lig_z, dtype=torch.long)

            protein_edges = _build_radius_graph(prot_pos_t, cutoff=5.0)
            ligand_edges = _build_radius_graph(lig_pos_t, cutoff=4.0)

            data_list.append({
                'protein_pos': prot_pos_t,
                'protein_z': prot_z_t,
                'protein_aa': prot_aa_t,
                'ligand_pos': lig_pos_t,
                'ligand_z': lig_z_t,
                'affinity': torch.tensor(affinities[pdb_code], dtype=torch.float32),
                'protein_edge_index': protein_edges,
                'ligand_edge_index': ligand_edges,
            })
        except Exception as e:
            print(f">>> Warning: Failed to process {pdb_code}: {e}")
            continue

    print(f">>> Successfully processed {len(data_list)}/{len(pdb_dirs)} complexes")
    return data_list if data_list else None


class PDBBindDataset(Dataset):
    """PDBbind dataset for protein-ligand binding affinity prediction.

    Loads protein-ligand complexes with binding affinity labels.
    Supports both refined and general sets.

    Data loading priority:
        1. Pre-processed .pt cache
        2. Raw PDB/SDF files (auto-processed and cached)
        3. Synthetic data (fallback for development)

    Each sample contains:
        - Protein pocket atoms (positions, types, amino acid types)
        - Ligand atoms (positions, types)
        - Binding affinity (pKd/pKi)
    """

    def __init__(self, root, version='refined', pocket_cutoff=10.0,
                 max_protein_atoms=1000, max_ligand_atoms=100, split='train'):
        self.root = root
        self.version = version
        self.pocket_cutoff = pocket_cutoff
        self.max_protein_atoms = max_protein_atoms
        self.max_ligand_atoms = max_ligand_atoms
        self.split = split
        self.data_list = []

        self._load_data()

    def _load_data(self):
        """Load data: cached .pt > raw PDB/SDF > synthetic fallback."""
        cache_path = os.path.join(self.root, f'{self.version}_{self.split}.pt')

        # 1. Try cached .pt
        if os.path.exists(cache_path):
            self.data_list = torch.load(cache_path, weights_only=False)
            print(f">>> PDBbind: loaded {len(self.data_list)} samples from {cache_path}")
            return

        # 2. Try processing raw data
        all_cache = os.path.join(self.root, f'{self.version}_all.pt')
        if os.path.exists(all_cache):
            all_data = torch.load(all_cache, weights_only=False)
        else:
            all_data = _process_pdbbind_raw(
                self.root, self.version, self.pocket_cutoff,
                self.max_protein_atoms, self.max_ligand_atoms
            )
            if all_data is not None:
                # Cache the full processed dataset
                os.makedirs(self.root, exist_ok=True)
                torch.save(all_data, all_cache)
                print(f">>> Cached {len(all_data)} processed complexes to {all_cache}")

        if all_data is not None and len(all_data) > 0:
            # Deterministic split: 80/10/10
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(all_data))
            n = len(all_data)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)

            if self.split == 'train':
                split_idx = indices[:train_end]
            elif self.split == 'val':
                split_idx = indices[train_end:val_end]
            else:
                split_idx = indices[val_end:]

            self.data_list = [all_data[i] for i in split_idx]
            # Cache split
            torch.save(self.data_list, cache_path)
            print(f">>> PDBbind {self.split}: {len(self.data_list)} samples (cached)")
            return

        # 3. Fallback: synthetic data
        print(f">>> PDBbind raw data not found, generating synthetic data")
        self._generate_synthetic_data()

    def _generate_synthetic_data(self, num_samples=500):
        """Generate synthetic protein-ligand data for development."""
        rng = np.random.RandomState(42 if self.split == 'train' else 123)

        for _ in range(num_samples):
            n_prot = rng.randint(50, min(200, self.max_protein_atoms))
            n_lig = rng.randint(10, min(40, self.max_ligand_atoms))

            protein_pos = torch.tensor(rng.randn(n_prot, 3) * 5.0, dtype=torch.float32)
            protein_z = torch.tensor(rng.choice([6, 7, 8, 16], size=n_prot), dtype=torch.long)
            protein_aa = torch.tensor(rng.randint(0, 20, size=n_prot), dtype=torch.long)

            ligand_center = torch.tensor(rng.randn(3) * 2.0, dtype=torch.float32)
            ligand_pos = ligand_center + torch.tensor(rng.randn(n_lig, 3) * 1.5, dtype=torch.float32)
            ligand_z = torch.tensor(rng.choice([6, 7, 8, 9, 16], size=n_lig), dtype=torch.long)

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
    """Custom collate for variable-size protein-ligand graphs."""
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
    """Load PDBbind dataset with train/val/test splits."""
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

    train_affinities = torch.stack([s['affinity'] for s in datasets['train'].data_list])
    aff_mean = train_affinities.mean().item()
    aff_std = train_affinities.std().item()

    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_pdbbind)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, collate_fn=collate_pdbbind)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_pdbbind)

    print(f">>> PDBbind {version}: {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])} train/val/test")
    print(f">>> Affinity: mean={aff_mean:.2f}, std={aff_std:.2f}")

    return train_loader, val_loader, test_loader, aff_mean, aff_std
