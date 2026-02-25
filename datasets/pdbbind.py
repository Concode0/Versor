# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""PDBbind dataset loader.

Protein-ligand binding affinity prediction.

Parsing strategy (best available library used automatically):
    Ligand:  RDKit > custom SDF parser
    Protein: Biopython > custom PDB parser

Spatial operations:
    Pocket extraction: scipy.spatial.KDTree > numpy cdist fallback

Real data requires manual download (free registration at https://www.pdbbind-plus.org.cn/download).

Synthetic fallback is always available for development.

Install optional libraries:
    pip install rdkit-pypi       # or: conda install -c conda-forge rdkit
    pip install biopython
    pip install scipy
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ---- Optional library detection ----
_HAS_RDKIT = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _HAS_RDKIT = True
except ImportError:
    pass

_HAS_BIOPYTHON = False
try:
    from Bio.PDB import PDBParser as _BioPDBParser, NeighborSearch as _BioNS
    _HAS_BIOPYTHON = True
except ImportError:
    pass

_HAS_SCIPY = False
try:
    from scipy.spatial import KDTree as _KDTree
    _HAS_SCIPY = True
except ImportError:
    pass


# =====================================================================
# Atom / residue type tables
# =====================================================================

ATOM_TYPES = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
}
AA_TYPES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL',
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_TYPES)}


# =====================================================================
# Shared utilities
# =====================================================================

def _build_radius_graph(pos: torch.Tensor, cutoff: float = 5.0) -> torch.Tensor:
    """Build edges between atoms within cutoff distance (pure PyTorch)."""
    dist = torch.cdist(pos, pos)
    mask = (dist < cutoff) & (dist > 0)
    return mask.nonzero(as_tuple=False).t()


def _pocket_mask(protein_pos: np.ndarray,
                 ligand_pos:  np.ndarray,
                 pocket_cutoff: float = 10.0) -> np.ndarray:
    """Return boolean mask selecting protein atoms within ``pocket_cutoff`` of any ligand atom.

    Uses scipy KDTree if available (faster); falls back to numpy cdist.
    """
    if _HAS_SCIPY:
        tree = _KDTree(protein_pos)
        hits = tree.query_ball_point(ligand_pos, r=pocket_cutoff)
        idx = set()
        for h in hits:
            idx.update(h)
        mask = np.zeros(len(protein_pos), dtype=bool)
        mask[list(idx)] = True
        return mask

    # Fallback: vectorised numpy
    diff = protein_pos[:, None, :] - ligand_pos[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))    # [N_prot, N_lig]
    return dists.min(axis=1) < pocket_cutoff


def _parse_element(atom_name: str, element_col: str = '') -> str:
    """Extract element symbol from PDB atom name or element column."""
    element_col = element_col.strip()
    if element_col and element_col in ATOM_TYPES:
        return element_col
    name = atom_name.strip()
    for c in name:
        if c.isalpha():
            two = c + name[name.index(c) + 1:name.index(c) + 2]
            if two in ATOM_TYPES:
                return two
            return c
    return 'C'


# =====================================================================
# Ligand parsing (RDKit > custom SDF)
# =====================================================================

def _parse_sdf_ligand_rdkit(sdf_path: str,
                             max_atoms: int = 100
                             ) -> Tuple[Optional[np.ndarray],
                                        Optional[np.ndarray],
                                        Optional[np.ndarray]]:
    """Parse ligand using RDKit (supports SDF, MOL, MOL2 with Chem.MolFromMolFile).

    Returns:
        positions:       [N, 3] float32 (3D conformer)
        atomic_numbers:  [N] int
        centroid:        [3] float32
    """
    mol = Chem.MolFromMolFile(sdf_path, sanitize=True, removeHs=True)
    if mol is None:
        return None, None, None

    conf = mol.GetConformer() if mol.GetNumConformers() else None
    if conf is None:
        # No 3D conformer - generate one
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0:
            return None, None, None
        conf = mol.GetConformer()

    positions = []
    atomic_nums = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue   # skip hydrogens
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        atomic_nums.append(atom.GetAtomicNum())

    if not positions:
        return None, None, None

    positions = np.array(positions, dtype=np.float32)
    if len(positions) > max_atoms:
        positions = positions[:max_atoms]
        atomic_nums = atomic_nums[:max_atoms]

    centroid = positions.mean(axis=0)
    return positions, np.array(atomic_nums, dtype=np.int64), centroid


def _parse_sdf_ligand_custom(sdf_path: str,
                              max_atoms: int = 100
                              ) -> Tuple[Optional[np.ndarray],
                                         Optional[np.ndarray],
                                         Optional[np.ndarray]]:
    """Parse SDF V2000 file with pure Python (no RDKit)."""
    with open(sdf_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 4:
        return None, None, None

    try:
        n_atoms = int(lines[3][:3].strip())
    except ValueError:
        return None, None, None

    positions = []
    elements = []
    for i in range(4, min(4 + n_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            elem = parts[3].strip()
        except (ValueError, IndexError):
            continue
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

    return positions, np.array(elements, dtype=np.int64), centroid


def _parse_sdf_ligand(sdf_path: str,
                       max_atoms: int = 100
                       ) -> Tuple[Optional[np.ndarray],
                                   Optional[np.ndarray],
                                   Optional[np.ndarray]]:
    """Parse ligand file - uses RDKit if available, otherwise custom parser."""
    if _HAS_RDKIT:
        try:
            return _parse_sdf_ligand_rdkit(sdf_path, max_atoms)
        except Exception as e:
            warnings.warn(f"RDKit ligand parsing failed ({e}); falling back to custom parser.")

    return _parse_sdf_ligand_custom(sdf_path, max_atoms)


# =====================================================================
# Protein PDB parsing (Biopython > custom)
# =====================================================================

def _parse_pdb_pocket_biopython(pdb_path: str,
                                  ligand_pos: np.ndarray,
                                  pocket_cutoff: float = 10.0,
                                  max_atoms: int = 1000
                                  ) -> Tuple[Optional[np.ndarray],
                                              Optional[np.ndarray],
                                              Optional[np.ndarray]]:
    """Parse protein pocket using Biopython.

    Handles insertion codes, alternate locations, and SEQRES records
    correctly.  Returns heavy atoms only.
    """
    parser = _BioPDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', pdb_path)
    except Exception as e:
        warnings.warn(f"Biopython PDB parsing failed ({e}).")
        return None, None, None

    positions, elements, aa_types = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue   # skip HETATM and water
                resname = residue.get_resname().strip()
                for atom in residue:
                    if atom.element == 'H' or atom.get_id().startswith('H'):
                        continue
                    elem = atom.element.strip() if atom.element else 'C'
                    positions.append(atom.get_vector().get_array())
                    elements.append(ATOM_TYPES.get(elem, 6))
                    aa_types.append(AA_TO_IDX.get(resname, 0))
        break   # first model only

    if not positions:
        return None, None, None

    positions = np.array(positions, dtype=np.float32)
    elements  = np.array(elements,  dtype=np.int64)
    aa_types  = np.array(aa_types,  dtype=np.int64)

    # Pocket extraction
    mask = _pocket_mask(positions, ligand_pos, pocket_cutoff)
    if mask.sum() == 0:
        closest = np.argsort(
            np.linalg.norm(positions - ligand_pos.mean(axis=0), axis=1)
        )[:100]
        mask = np.zeros(len(positions), dtype=bool)
        mask[closest] = True

    positions, elements, aa_types = positions[mask], elements[mask], aa_types[mask]

    if len(positions) > max_atoms:
        center = ligand_pos.mean(axis=0)
        order = np.argsort(np.linalg.norm(positions - center, axis=1))[:max_atoms]
        positions, elements, aa_types = positions[order], elements[order], aa_types[order]

    centroid = positions.mean(axis=0)
    return positions - centroid, elements, aa_types


def _parse_pdb_pocket_custom(pdb_path: str,
                               ligand_pos: np.ndarray,
                               pocket_cutoff: float = 10.0,
                               max_atoms: int = 1000
                               ) -> Tuple[Optional[np.ndarray],
                                           Optional[np.ndarray],
                                           Optional[np.ndarray]]:
    """Parse protein pocket with pure Python PDB reader (no Biopython)."""
    positions, elements, aa_types = [], [], []

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name  = line[12:16]
            element    = line[76:78] if len(line) >= 78 else ''
            elem = _parse_element(atom_name, element)
            if elem == 'H':
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            resname = line[17:20].strip()
            positions.append([x, y, z])
            elements.append(ATOM_TYPES.get(elem, 6))
            aa_types.append(AA_TO_IDX.get(resname, 0))

    if not positions:
        return None, None, None

    positions = np.array(positions, dtype=np.float32)
    elements  = np.array(elements,  dtype=np.int64)
    aa_types  = np.array(aa_types,  dtype=np.int64)

    mask = _pocket_mask(positions, ligand_pos, pocket_cutoff)
    if mask.sum() == 0:
        closest = np.argsort(
            np.linalg.norm(positions - ligand_pos.mean(axis=0), axis=1)
        )[:100]
        mask = np.zeros(len(positions), dtype=bool)
        mask[closest] = True

    positions, elements, aa_types = positions[mask], elements[mask], aa_types[mask]

    if len(positions) > max_atoms:
        center = ligand_pos.mean(axis=0)
        order = np.argsort(np.linalg.norm(positions - center, axis=1))[:max_atoms]
        positions, elements, aa_types = positions[order], elements[order], aa_types[order]

    centroid = positions.mean(axis=0)
    return positions - centroid, elements, aa_types


def _parse_pdb_pocket(pdb_path: str,
                       ligand_pos: np.ndarray,
                       pocket_cutoff: float = 10.0,
                       max_atoms: int = 1000
                       ) -> Tuple[Optional[np.ndarray],
                                   Optional[np.ndarray],
                                   Optional[np.ndarray]]:
    """Parse protein pocket - uses Biopython if available, else custom."""
    if _HAS_BIOPYTHON:
        try:
            return _parse_pdb_pocket_biopython(pdb_path, ligand_pos,
                                                pocket_cutoff, max_atoms)
        except Exception as e:
            warnings.warn(f"Biopython pocket parsing failed ({e}); "
                          "falling back to custom parser.")

    return _parse_pdb_pocket_custom(pdb_path, ligand_pos, pocket_cutoff, max_atoms)


# =====================================================================
# Affinity index reader
# =====================================================================

def _parse_affinity_index(index_path: str) -> Dict[str, float]:
    """Parse PDBbind INDEX file for binding affinities.

    Handles multiple common column layouts:
        PDB  resolution  year  -logKd/Ki=X.XX  ...
        PDB  resolution  year  X.XX  ...
    """
    affinities: Dict[str, float] = {}
    with open(index_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            pdb_code = parts[0]
            for col in parts[3:6]:
                try:
                    val = float(col.split('=')[-1])
                    affinities[pdb_code] = val
                    break
                except ValueError:
                    continue
    return affinities


# =====================================================================
# Raw data processing
# =====================================================================

def _process_pdbbind_raw(root: str, version: str = 'refined',
                          pocket_cutoff: float = 10.0,
                          max_protein_atoms: int = 1000,
                          max_ligand_atoms: int = 100) -> Optional[List[dict]]:
    """Process raw PDBbind data into tensor format.

    Expected directory structure:
        root/
          refined-set/   (or v2020-other-PL/)
            XXXX/
              XXXX_protein.pdb
              XXXX_ligand.sdf
          index/
            INDEX_refined_data.2020  (or similar)

    Reports which parser is being used for protein/ligand.

    Returns:
        List of sample dicts, or None if data not found.
    """
    data_dir = os.path.join(root, 'refined-set' if version == 'refined' else 'v2020-other-PL')
    if not os.path.isdir(data_dir):
        return None

    # Find index file
    index_file = None
    for pattern in [
        os.path.join(root, 'index', 'INDEX_refined_data.2020'),
        os.path.join(root, 'index', 'INDEX_general_PL_data.2020'),
    ]:
        if os.path.exists(pattern):
            index_file = pattern
            break
    if index_file is None:
        candidates = (glob.glob(os.path.join(root, 'index', 'INDEX_*')) +
                      glob.glob(os.path.join(root, 'INDEX*')))
        if candidates:
            index_file = sorted(candidates)[0]

    if index_file is None:
        warnings.warn(f"PDBbind: no affinity index file found in {root}")
        return None

    affinities = _parse_affinity_index(index_file)
    if not affinities:
        warnings.warn(f"PDBbind: no affinities parsed from {index_file}")
        return None

    # Report available parsers
    lig_parser = "RDKit" if _HAS_RDKIT else "custom SDF"
    prot_parser = "Biopython" if _HAS_BIOPYTHON else "custom PDB"
    dist_lib = "scipy KDTree" if _HAS_SCIPY else "numpy"
    print(f">>> PDBbind: ligand={lig_parser}, protein={prot_parser}, "
          f"distances={dist_lib}")

    pdb_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 4
    ])
    print(f">>> Processing {len(pdb_dirs)} PDBbind complexes ...")

    data_list = []
    n_failed = 0

    for pdb_code in pdb_dirs:
        if pdb_code not in affinities:
            continue

        pdb_dir     = os.path.join(data_dir, pdb_code)
        protein_pdb = os.path.join(pdb_dir, f'{pdb_code}_protein.pdb')
        ligand_sdf  = os.path.join(pdb_dir, f'{pdb_code}_ligand.sdf')

        if not (os.path.exists(protein_pdb) and os.path.exists(ligand_sdf)):
            continue

        try:
            lig_pos, lig_z, lig_centroid = _parse_sdf_ligand(
                ligand_sdf, max_ligand_atoms)
            if lig_pos is None or len(lig_pos) < 3:
                n_failed += 1
                continue

            prot_pos, prot_z, prot_aa = _parse_pdb_pocket(
                protein_pdb, lig_pos, pocket_cutoff, max_protein_atoms)
            if prot_pos is None or len(prot_pos) < 5:
                n_failed += 1
                continue

            # Center ligand relative to pocket centroid (protein already centered)
            lig_pos_c = lig_pos - lig_pos.mean(axis=0)

            prot_pos_t = torch.tensor(prot_pos, dtype=torch.float32)
            lig_pos_t  = torch.tensor(lig_pos_c, dtype=torch.float32)

            data_list.append({
                'protein_pos':        prot_pos_t,
                'protein_z':          torch.tensor(prot_z,   dtype=torch.long),
                'protein_aa':         torch.tensor(prot_aa,  dtype=torch.long),
                'ligand_pos':         lig_pos_t,
                'ligand_z':           torch.tensor(lig_z,    dtype=torch.long),
                'affinity':           torch.tensor(affinities[pdb_code], dtype=torch.float32),
                'protein_edge_index': _build_radius_graph(prot_pos_t, cutoff=5.0),
                'ligand_edge_index':  _build_radius_graph(lig_pos_t,  cutoff=4.0),
            })
        except Exception as e:
            warnings.warn(f"PDBbind: failed to process {pdb_code}: {e}")
            n_failed += 1
            continue

    print(f">>> Successfully processed {len(data_list)}/{len(pdb_dirs)} complexes "
          f"({n_failed} failed)")
    return data_list if data_list else None


# =====================================================================
# Dataset
# =====================================================================

class PDBBindDataset(Dataset):
    """PDBbind dataset for protein-ligand binding affinity prediction.

    Loading priority:
        1. Pre-processed .pt cache
        2. Raw PDB/SDF files (auto-processed and cached)
        3. Synthetic data (fallback for development)

    Each sample:
        protein_pos / protein_z / protein_aa: pocket atoms
        ligand_pos  / ligand_z:               ligand atoms
        affinity:                              pKd/pKi scalar
        protein_edge_index / ligand_edge_index: radius graphs
    """

    def __init__(self, root: str, version: str = 'refined',
                 pocket_cutoff: float = 10.0,
                 max_protein_atoms: int = 1000,
                 max_ligand_atoms: int = 100,
                 split: str = 'train'):
        self.root = root
        self.version = version
        self.pocket_cutoff = pocket_cutoff
        self.max_protein_atoms = max_protein_atoms
        self.max_ligand_atoms = max_ligand_atoms
        self.split = split
        self.data_list: List[dict] = []
        self._load_data()

    def _load_data(self):
        cache_path = os.path.join(self.root, f'{self.version}_{self.split}.pt')

        # 1. Cached .pt
        if os.path.exists(cache_path):
            self.data_list = torch.load(cache_path, weights_only=False)
            print(f">>> PDBbind: loaded {len(self.data_list)} samples from {cache_path}")
            return

        # 2. Raw PDB/SDF
        all_cache = os.path.join(self.root, f'{self.version}_all.pt')
        if os.path.exists(all_cache):
            all_data = torch.load(all_cache, weights_only=False)
        else:
            all_data = _process_pdbbind_raw(
                self.root, self.version, self.pocket_cutoff,
                self.max_protein_atoms, self.max_ligand_atoms
            )
            if all_data is not None:
                os.makedirs(self.root, exist_ok=True)
                torch.save(all_data, all_cache)
                print(f">>> PDBbind: cached {len(all_data)} complexes -> {all_cache}")

        if all_data is not None and len(all_data) > 0:
            rng = np.random.RandomState(42)
            idx = rng.permutation(len(all_data))
            n = len(all_data)
            train_end, val_end = int(0.8 * n), int(0.9 * n)

            split_idx = {
                'train': idx[:train_end],
                'val':   idx[train_end:val_end],
                'test':  idx[val_end:],
            }[self.split]

            self.data_list = [all_data[i] for i in split_idx]
            torch.save(self.data_list, cache_path)
            print(f">>> PDBbind {self.split}: {len(self.data_list)} samples (split cached)")
            return

        # 3. Synthetic fallback
        print(f">>> PDBbind: raw data not found, generating synthetic data "
              f"(download from http://www.pdbbind.org.cn)")
        self._generate_synthetic_data()

    def _generate_synthetic_data(self, num_samples: int = 500):
        rng = np.random.RandomState(42 if self.split == 'train' else 123)
        for _ in range(num_samples):
            n_prot = rng.randint(50, min(200, self.max_protein_atoms))
            n_lig  = rng.randint(10, min(40,  self.max_ligand_atoms))

            protein_pos = torch.tensor(rng.randn(n_prot, 3) * 5.0, dtype=torch.float32)
            ligand_pos  = torch.tensor(
                rng.randn(3) * 2.0 + rng.randn(n_lig, 3) * 1.5, dtype=torch.float32)

            mean_dist = torch.cdist(protein_pos, ligand_pos).min(dim=0)[0].mean().item()
            affinity = 8.0 - mean_dist * 0.5 + rng.randn() * 0.5

            self.data_list.append({
                'protein_pos':        protein_pos,
                'protein_z':          torch.tensor(rng.choice([6, 7, 8, 16], n_prot), dtype=torch.long),
                'protein_aa':         torch.tensor(rng.randint(0, 20, n_prot), dtype=torch.long),
                'ligand_pos':         ligand_pos,
                'ligand_z':           torch.tensor(rng.choice([6, 7, 8, 9, 16], n_lig), dtype=torch.long),
                'affinity':           torch.tensor(affinity, dtype=torch.float32),
                'protein_edge_index': _build_radius_graph(protein_pos, cutoff=5.0),
                'ligand_edge_index':  _build_radius_graph(ligand_pos,  cutoff=4.0),
            })

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]


# =====================================================================
# Collate and loader factory
# =====================================================================

def collate_pdbbind(batch: list) -> dict:
    """Custom collate for variable-size protein-ligand graphs."""
    prot_pos, prot_z, prot_aa = [], [], []
    lig_pos, lig_z = [], []
    affinities = []
    prot_batch, lig_batch = [], []
    prot_edges, lig_edges = [], []
    prot_off = lig_off = 0

    for i, s in enumerate(batch):
        np_ = s['protein_pos'].size(0)
        nl_ = s['ligand_pos'].size(0)

        prot_pos.append(s['protein_pos'])
        prot_z.append(s['protein_z'])
        prot_aa.append(s['protein_aa'])
        lig_pos.append(s['ligand_pos'])
        lig_z.append(s['ligand_z'])
        affinities.append(s['affinity'])

        prot_batch.append(torch.full((np_,), i, dtype=torch.long))
        lig_batch.append(torch.full((nl_,), i, dtype=torch.long))

        if s['protein_edge_index'].numel() > 0:
            prot_edges.append(s['protein_edge_index'] + prot_off)
        if s['ligand_edge_index'].numel() > 0:
            lig_edges.append(s['ligand_edge_index'] + lig_off)

        prot_off += np_
        lig_off  += nl_

    return {
        'protein_pos':        torch.cat(prot_pos),
        'protein_z':          torch.cat(prot_z),
        'protein_aa':         torch.cat(prot_aa),
        'ligand_pos':         torch.cat(lig_pos),
        'ligand_z':           torch.cat(lig_z),
        'affinity':           torch.stack(affinities),
        'protein_batch':      torch.cat(prot_batch),
        'ligand_batch':       torch.cat(lig_batch),
        'protein_edge_index': torch.cat(prot_edges, dim=1) if prot_edges
                              else torch.zeros(2, 0, dtype=torch.long),
        'ligand_edge_index':  torch.cat(lig_edges,  dim=1) if lig_edges
                              else torch.zeros(2, 0, dtype=torch.long),
    }


def get_pdbbind_loaders(root: str, version: str = 'refined',
                         batch_size: int = 8,
                         max_samples: Optional[int] = None,
                         pocket_cutoff: float = 10.0,
                         max_protein_atoms: int = 1000,
                         max_ligand_atoms: int = 100,
                         num_workers: int = 2,
                         pin_memory: bool = False):
    """Load PDBbind dataset with train/val/test splits.

    Uses real data when available, synthetic fallback otherwise.

    Returns:
        train_loader, val_loader, test_loader, aff_mean, aff_std
    """
    datasets: Dict[str, PDBBindDataset] = {}
    for split in ['train', 'val', 'test']:
        ds = PDBBindDataset(
            root=root, version=version, pocket_cutoff=pocket_cutoff,
            max_protein_atoms=max_protein_atoms,
            max_ligand_atoms=max_ligand_atoms, split=split
        )
        if max_samples and len(ds) > max_samples:
            ds.data_list = ds.data_list[:max_samples]
        datasets[split] = ds

    train_aff = torch.stack([s['affinity'] for s in datasets['train'].data_list])
    aff_mean = train_aff.mean().item()
    aff_std  = max(train_aff.std().item(), 1e-6)

    train_loader = DataLoader(datasets['train'], batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_pdbbind,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(datasets['val'],   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_pdbbind,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(datasets['test'],  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_pdbbind,
                              num_workers=num_workers, pin_memory=pin_memory)

    print(f">>> PDBbind {version}: "
          f"{len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])} "
          f"train/val/test")
    print(f">>> Affinity: mean={aff_mean:.2f}, std={aff_std:.2f}")

    return train_loader, val_loader, test_loader, aff_mean, aff_std
