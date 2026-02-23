# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""ABC CAD dataset loader.

Point cloud autoencoding and primitive reconstruction from CAD meshes.

Loading priority:
    1. Cached .pt files  (fastest)
    2. Mesh files via trimesh  (OBJ / OFF / PLY / STL / …, rich format support)
    3. Mesh files via custom parsers  (OBJ / OFF / PLY — pure Python)
    4. Synthetic fallback  (geometric primitives, always available)

Optional normal estimation:
    open3d:  fast KD-tree-based PCA (preferred)
    fallback: per-sample torch.linalg.eigh on KNN covariance (slow for large N)

Install libraries:
    pip install trimesh           # better mesh loading
    pip install open3d            # fast normal estimation
    Download from https://deep-geometry.github.io/abc-dataset/
"""

from __future__ import annotations

import os
import glob
import struct
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ---- Optional library detection ----
_HAS_TRIMESH = False
try:
    import trimesh as _trimesh
    _HAS_TRIMESH = True
except ImportError:
    pass

_HAS_OPEN3D = False
try:
    import open3d as _o3d
    _HAS_OPEN3D = True
except ImportError:
    pass


# =====================================================================
# Mesh loading — trimesh path (preferred)
# =====================================================================

def _load_mesh_trimesh(path: str
                       ) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Load any mesh format supported by trimesh.

    Trimesh handles OBJ, OFF, PLY, STL, GLB, GLTF, 3MF, DAE, and more.
    Falls back automatically for degenerate / empty meshes.

    Returns:
        (vertices [N, 3], faces [M, 3]) or (None, None) on failure.
    """
    try:
        mesh = _trimesh.load(path, force='mesh', process=False)
        if not isinstance(mesh, _trimesh.Trimesh):
            # Scene or other type — try to extract first mesh
            if hasattr(mesh, 'geometry'):
                geoms = list(mesh.geometry.values())
                if geoms:
                    mesh = geoms[0]
                else:
                    return None, None
            else:
                return None, None
        if len(mesh.vertices) < 10:
            return None, None
        return np.array(mesh.vertices, dtype=np.float32), mesh.faces.tolist()
    except Exception:
        return None, None


def _sample_trimesh(path: str, num_points: int) -> Optional[np.ndarray]:
    """Sample a point cloud from a mesh surface via trimesh.

    Uses area-weighted face sampling with barycentric coordinates —
    same algorithm as the manual path but implemented in C++ by trimesh.

    Returns:
        [num_points, 3] float32 or None on failure.
    """
    try:
        mesh = _trimesh.load(path, force='mesh', process=False)
        if not isinstance(mesh, _trimesh.Trimesh):
            if hasattr(mesh, 'geometry'):
                geoms = list(mesh.geometry.values())
                if geoms:
                    mesh = geoms[0]
                else:
                    return None
            else:
                return None
        if len(mesh.vertices) < 10:
            return None
        pts, _ = _trimesh.sample.sample_surface(mesh, num_points)
        return pts.astype(np.float32)
    except Exception:
        return None


# =====================================================================
# Mesh loading — pure Python parsers (OBJ / OFF / PLY)
# =====================================================================

def _load_obj(path: str) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Parse Wavefront OBJ file."""
    vertices, faces = [], []
    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'v' and len(parts) >= 4:
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        pass
                elif parts[0] == 'f' and len(parts) >= 4:
                    try:
                        face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                        faces.append(face)
                    except (ValueError, IndexError):
                        pass
    except OSError:
        return None, None

    if not vertices:
        return None, None
    return np.array(vertices, dtype=np.float32), faces


def _load_off(path: str) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Parse Object File Format (OFF) file."""
    try:
        with open(path, 'r', errors='replace') as f:
            header = f.readline().strip()
            if header == 'OFF':
                counts = f.readline().strip().split()
            elif header.startswith('OFF'):
                counts = header[3:].strip().split()
            else:
                return None, None

            n_verts, n_faces = int(counts[0]), int(counts[1])
            vertices = []
            for _ in range(n_verts):
                parts = f.readline().split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

            faces = []
            for _ in range(n_faces):
                parts = f.readline().split()
                n = int(parts[0])
                if n >= 3:
                    faces.append([int(parts[i + 1]) for i in range(3)])
    except (OSError, ValueError, IndexError):
        return None, None

    if not vertices:
        return None, None
    return np.array(vertices, dtype=np.float32), faces


def _load_ply(path: str) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Parse Stanford PLY file (ASCII and binary little-endian).

    Reads vertex x/y/z and (optionally) face vertex_indices.
    Does not use any external library — struct module only.
    """
    try:
        with open(path, 'rb') as f:
            # --- Parse header ---
            header_lines = []
            while True:
                raw = f.readline()
                line = raw.decode('ascii', errors='replace').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
                if len(header_lines) > 200:
                    return None, None   # malformed

            format_line = next((l for l in header_lines if l.startswith('format')), None)
            if format_line is None:
                return None, None

            is_binary_le = 'binary_little_endian' in format_line
            is_binary_be = 'binary_big_endian' in format_line
            is_ascii = 'ascii' in format_line

            # Count vertices and faces
            n_verts = n_faces = 0
            in_vertex = in_face = False
            vertex_props: List[str] = []
            face_list_type = None

            for line in header_lines:
                if line.startswith('element vertex'):
                    n_verts = int(line.split()[-1])
                    in_vertex, in_face = True, False
                elif line.startswith('element face'):
                    n_faces = int(line.split()[-1])
                    in_vertex, in_face = False, True
                elif line.startswith('element'):
                    in_vertex = in_face = False
                elif line.startswith('property') and in_vertex:
                    vertex_props.append(line.split()[-1])
                elif line.startswith('property list') and in_face:
                    parts = line.split()
                    face_list_type = parts[2]   # e.g. 'uchar' or 'uint'

            if n_verts == 0:
                return None, None

            # Map PLY scalar types to struct format chars
            _ply_fmt = {
                'float': 'f', 'float32': 'f',
                'double': 'd', 'float64': 'd',
                'int': 'i', 'int32': 'i',
                'uint': 'I', 'uint32': 'I',
                'short': 'h', 'int16': 'h',
                'uchar': 'B', 'uint8': 'B',
                'char': 'b', 'int8': 'b',
            }
            _ply_size = {'f': 4, 'd': 8, 'i': 4, 'I': 4, 'h': 2, 'H': 2,
                         'B': 1, 'b': 1}

            # --- Read vertices ---
            x_idx = vertex_props.index('x') if 'x' in vertex_props else 0
            y_idx = vertex_props.index('y') if 'y' in vertex_props else 1
            z_idx = vertex_props.index('z') if 'z' in vertex_props else 2

            prop_types = []
            for line in header_lines:
                if line.startswith('property') and not line.startswith('property list'):
                    parts = line.split()
                    if len(parts) >= 2:
                        prop_types.append(_ply_fmt.get(parts[1], 'f'))

            n_props = len(prop_types)
            if n_props < 3:
                prop_types = ['f', 'f', 'f']

            vertices = np.zeros((n_verts, 3), dtype=np.float32)

            if is_ascii:
                for i in range(n_verts):
                    vals = f.readline().decode('ascii', errors='replace').split()
                    if len(vals) < 3:
                        return None, None
                    vertices[i] = [float(vals[x_idx]),
                                   float(vals[y_idx]),
                                   float(vals[z_idx])]
            elif is_binary_le or is_binary_be:
                endian = '<' if is_binary_le else '>'
                row_fmt = endian + ''.join(prop_types)
                row_size = struct.calcsize(row_fmt)
                for i in range(n_verts):
                    raw = f.read(row_size)
                    if len(raw) < row_size:
                        return None, None
                    vals = struct.unpack(row_fmt, raw)
                    vertices[i] = [vals[x_idx], vals[y_idx], vals[z_idx]]
            else:
                return None, None

            # --- Read faces ---
            faces: List[List[int]] = []
            cnt_fmt = _ply_fmt.get(face_list_type or 'uchar', 'B')
            cnt_size = _ply_size[cnt_fmt]
            idx_fmt = 'i'   # most PLY files use int32 face indices

            if is_ascii:
                for _ in range(n_faces):
                    vals = f.readline().decode('ascii', errors='replace').split()
                    if not vals:
                        continue
                    n = int(vals[0])
                    if n >= 3:
                        faces.append([int(vals[k + 1]) for k in range(3)])
            elif is_binary_le or is_binary_be:
                for _ in range(n_faces):
                    cnt_raw = f.read(cnt_size)
                    if len(cnt_raw) < cnt_size:
                        break
                    n = struct.unpack(endian + cnt_fmt, cnt_raw)[0]
                    idx_raw = f.read(4 * n)   # int32 per index
                    if len(idx_raw) < 4 * n:
                        break
                    idxs = struct.unpack(endian + 'i' * n, idx_raw)
                    if n >= 3:
                        faces.append(list(idxs[:3]))

    except (OSError, struct.error, UnicodeDecodeError):
        return None, None

    return vertices, faces


def _load_mesh_custom(path: str) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Dispatch to custom pure-Python parsers (OBJ / OFF / PLY)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.obj':
        return _load_obj(path)
    elif ext == '.off':
        return _load_off(path)
    elif ext == '.ply':
        return _load_ply(path)
    return None, None


# =====================================================================
# Point sampling from mesh
# =====================================================================

def _sample_from_mesh(vertices: np.ndarray, faces: list,
                       num_points: int) -> np.ndarray:
    """Sample points uniformly from mesh surface via area-weighted face sampling."""
    if not faces:
        idx = np.random.choice(len(vertices), num_points,
                               replace=len(vertices) < num_points)
        return vertices[idx]

    face_arr = np.array(faces, dtype=np.int64)
    v0 = vertices[face_arr[:, 0]]
    v1 = vertices[face_arr[:, 1]]
    v2 = vertices[face_arr[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.maximum(areas, 1e-12)
    probs = areas / areas.sum()

    sampled = np.random.choice(len(faces), num_points, p=probs)
    r1 = np.sqrt(np.random.rand(num_points))
    r2 = np.random.rand(num_points)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2

    return (w0[:, None] * v0[sampled] +
            w1[:, None] * v1[sampled] +
            w2[:, None] * v2[sampled]).astype(np.float32)


# =====================================================================
# Normal estimation
# =====================================================================

def _estimate_normals_open3d(points: torch.Tensor) -> torch.Tensor:
    """Estimate normals via open3d (fast KD-tree-based PCA)."""
    pcd = _o3d.geometry.PointCloud()
    pcd.points = _o3d.utility.Vector3dVector(points.numpy())
    pcd.estimate_normals(
        search_param=_o3d.geometry.KDTreeSearchParamKNN(knn=10)
    )
    pcd.orient_normals_towards_camera_location(points.mean(dim=0).numpy())
    return torch.tensor(np.asarray(pcd.normals), dtype=torch.float32)


def _estimate_normals_torch(points: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Estimate normals via local KNN covariance (pure PyTorch, O(N·k·d²))."""
    dist = torch.cdist(points, points)
    _, knn_idx = dist.topk(k + 1, largest=False)
    knn_idx = knn_idx[:, 1:]   # remove self

    N = points.size(0)
    normals = torch.zeros_like(points)
    for i in range(N):
        neighbors = points[knn_idx[i]]
        centered  = neighbors - neighbors.mean(dim=0, keepdim=True)
        cov = centered.t() @ centered / k
        _, vecs = torch.linalg.eigh(cov)
        normals[i] = vecs[:, 0]

    # Orient outward from centroid
    center = points.mean(dim=0)
    flip = (normals * (points - center)).sum(dim=-1) < 0
    normals[flip] *= -1
    return normals


def _estimate_normals(points: torch.Tensor) -> torch.Tensor:
    """Estimate surface normals — uses open3d if available, else torch KNN."""
    if _HAS_OPEN3D:
        try:
            return _estimate_normals_open3d(points)
        except Exception:
            pass   # fall through
    return _estimate_normals_torch(points)


# =====================================================================
# Dataset
# =====================================================================

class ABCDataset(Dataset):
    """ABC Dataset for CAD parametric model reconstruction.

    Loading priority:
        1. Cached .pt files
        2. Mesh files via trimesh (OBJ/OFF/PLY/STL/…)
        3. Mesh files via custom pure-Python parsers (OBJ/OFF/PLY)
        4. Synthetic geometric primitives

    Two task modes:
        reconstruction: Reconstruct input point cloud (self-supervised)
        primitive:      Predict primitive parameters (sphere/cylinder/box/…)
    """

    def __init__(self, root: str, task: str = 'reconstruction',
                 num_points: int = 2048,
                 augment: bool = False, split: str = 'train'):
        self.root = root
        self.task = task
        self.num_points = num_points
        self.augment = augment and (split == 'train')
        self.split = split
        self.data_list: List[dict] = []
        self._load_data()

    def _load_data(self):
        cache_path = os.path.join(self.root, f'abc_{self.task}_{self.split}.pt')

        # 1. Cached .pt
        if os.path.exists(cache_path):
            self.data_list = torch.load(cache_path, weights_only=False)
            print(f">>> ABC: loaded {len(self.data_list)} samples from {cache_path}")
            return

        # 2. Load from mesh files
        if not _HAS_TRIMESH:
            print(">>> ABC: trimesh not installed (pip install trimesh); "
                  "using custom OBJ/OFF/PLY parsers.")
        if not _HAS_OPEN3D:
            print(">>> ABC: open3d not installed (pip install open3d); "
                  "using torch KNN for normal estimation.")

        if self._load_from_meshes():
            os.makedirs(self.root, exist_ok=True)
            torch.save(self.data_list, cache_path)
            print(f">>> ABC: cached {len(self.data_list)} samples → {cache_path}")
            return

        # 3. Synthetic fallback
        print(f">>> ABC: no mesh data found at {self.root}, "
              "generating synthetic data (download from https://deep-geometry.github.io/abc-dataset/)")
        self._generate_synthetic_data()

    def _load_from_meshes(self) -> bool:
        """Load point clouds from mesh files in root directory.

        Tries trimesh first for each file, then falls back to custom parsers.
        Skips files silently on failure but logs aggregate failure count.

        Expected directory structure (any of):
            root/{split}/*.{obj,off,ply}
            root/meshes/*.{obj,off,ply}
            root/*.{obj,off,ply}
        """
        extensions = ['*.obj', '*.off', '*.ply']
        mesh_dir = None
        for subdir in [self.split, 'meshes', '']:
            candidate = os.path.join(self.root, subdir) if subdir else self.root
            all_files = []
            for ext in extensions:
                all_files.extend(glob.glob(os.path.join(candidate, ext)))
            if all_files:
                mesh_dir = candidate
                break

        if mesh_dir is None:
            return False

        mesh_files = sorted(
            f for ext in extensions
            for f in glob.glob(os.path.join(mesh_dir, ext))
        )
        if not mesh_files:
            return False

        # Deterministic split
        rng = np.random.RandomState(42)
        idx_all = rng.permutation(len(mesh_files))
        n = len(mesh_files)
        train_end, val_end = int(0.8 * n), int(0.9 * n)
        split_slice = {
            'train': idx_all[:train_end],
            'val':   idx_all[train_end:val_end],
            'test':  idx_all[val_end:],
        }[self.split]

        print(f">>> ABC: processing {len(split_slice)} mesh files ({self.split}) …")
        n_ok = n_fail = 0

        for file_idx in split_slice:
            path = mesh_files[file_idx]
            try:
                points = self._load_and_sample(path)
                if points is None:
                    n_fail += 1
                    continue

                points = points - points.mean(dim=0, keepdim=True)
                scale = points.abs().max() + 1e-6
                points = points / scale
                normals = _estimate_normals(points)

                sample: dict = {
                    'points': points,
                    'normals': normals,
                    'target_points': points.clone(),
                }
                if self.task == 'primitive':
                    sample['primitive_type'] = 0
                    sample['primitive_params'] = torch.zeros(13)

                self.data_list.append(sample)
                n_ok += 1
            except Exception as e:
                warnings.warn(f"ABC: failed to load {os.path.basename(path)}: {e}")
                n_fail += 1

        if n_fail > 0:
            print(f">>> ABC: {n_ok} loaded, {n_fail} failed out of {len(split_slice)}")

        return n_ok > 0

    def _load_and_sample(self, path: str) -> Optional[torch.Tensor]:
        """Load a mesh file and sample a point cloud.

        Tries trimesh first (handles many formats and degenerate cases),
        then falls back to custom parsers (OBJ / OFF / PLY).
        """
        # --- trimesh path ---
        if _HAS_TRIMESH:
            pts = _sample_trimesh(path, self.num_points)
            if pts is not None:
                return torch.tensor(pts, dtype=torch.float32)

        # --- custom parser path ---
        vertices, faces = _load_mesh_custom(path)
        if vertices is None or len(vertices) < 10:
            return None

        pts = _sample_from_mesh(vertices, faces, self.num_points)
        return torch.tensor(pts, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------

    def _generate_synthetic_data(self, num_samples: int = 500):
        """Generate synthetic CAD-like point clouds from geometric primitives."""
        rng = np.random.RandomState(42 if self.split == 'train' else 123)
        primitive_types = ['sphere', 'cylinder', 'box', 'torus', 'combined']

        for _ in range(num_samples):
            prim_type = rng.choice(primitive_types)
            pts = self._generate_primitive(prim_type, rng)

            if len(pts) > self.num_points:
                pts = pts[rng.choice(len(pts), self.num_points, replace=False)]
            elif len(pts) < self.num_points:
                pts = pts[rng.choice(len(pts), self.num_points, replace=True)]

            points = torch.tensor(pts, dtype=torch.float32)
            normals = _estimate_normals(points)

            sample: dict = {
                'points': points,
                'normals': normals,
                'target_points': points.clone(),
            }
            if self.task == 'primitive':
                sample['primitive_type'] = primitive_types.index(prim_type)
                sample['primitive_params'] = self._get_primitive_params(prim_type, rng)

            self.data_list.append(sample)

    def _generate_primitive(self, prim_type: str, rng: np.random.RandomState) -> np.ndarray:
        n = self.num_points * 2

        if prim_type == 'sphere':
            r = 0.5 + rng.rand() * 0.5
            theta = rng.uniform(0, 2 * np.pi, n)
            phi   = rng.uniform(0, np.pi, n)
            pts = r * np.stack([np.sin(phi) * np.cos(theta),
                                 np.sin(phi) * np.sin(theta),
                                 np.cos(phi)], axis=1)

        elif prim_type == 'cylinder':
            r, h = 0.3 + rng.rand() * 0.3, 1.0 + rng.rand()
            theta = rng.uniform(0, 2 * np.pi, n)
            z     = rng.uniform(-h / 2, h / 2, n)
            pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

        elif prim_type == 'box':
            dims = 0.5 + rng.rand(3) * 0.5
            face = rng.randint(0, 6, n)
            u = rng.uniform(-1, 1, n)
            v = rng.uniform(-1, 1, n)
            pts = np.zeros((n, 3))
            for f in range(6):
                m = face == f
                ax = f // 2
                sign = 2 * (f % 2) - 1
                other = [i for i in range(3) if i != ax]
                pts[m, ax] = sign * dims[ax]
                pts[m, other[0]] = u[m] * dims[other[0]]
                pts[m, other[1]] = v[m] * dims[other[1]]

        elif prim_type == 'torus':
            R, r = 0.7 + rng.rand() * 0.3, 0.1 + rng.rand() * 0.2
            theta = rng.uniform(0, 2 * np.pi, n)
            phi   = rng.uniform(0, 2 * np.pi, n)
            pts = np.stack([(R + r * np.cos(phi)) * np.cos(theta),
                             (R + r * np.cos(phi)) * np.sin(theta),
                             r * np.sin(phi)], axis=1)

        else:   # combined
            a = self._generate_primitive(rng.choice(['sphere', 'cylinder', 'box']), rng)
            b = self._generate_primitive(rng.choice(['sphere', 'cylinder', 'box']), rng)
            pts = np.concatenate([a[:n // 2], b[:n // 2]])

        pts = pts - pts.mean(axis=0)
        scale = np.abs(pts).max() + 1e-6
        return (pts / scale).astype(np.float32)

    @staticmethod
    def _get_primitive_params(prim_type: str, rng: np.random.RandomState) -> torch.Tensor:
        params = torch.zeros(13)
        type_idx = ['sphere', 'cylinder', 'box', 'torus', 'combined'].index(prim_type)
        params[type_idx] = 1.0
        params[5:8]  = torch.tensor(rng.randn(3) * 0.3, dtype=torch.float32)
        params[8:11] = torch.tensor(0.5 + rng.rand(3) * 0.5, dtype=torch.float32)
        return params

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        sample = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in self.data_list[idx].items()}

        if self.augment:
            axis  = torch.randn(3)
            axis  = axis / axis.norm()
            theta = torch.rand(1) * 2 * 3.14159265
            K = torch.tensor([
                [0,        -axis[2],  axis[1]],
                [axis[2],   0,       -axis[0]],
                [-axis[1],  axis[0],  0      ],
            ])
            R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
            sample['points']        = (R @ sample['points'].t()).t()
            sample['normals']       = (R @ sample['normals'].t()).t()
            sample['target_points'] = sample['points'].clone()

        return sample


# =====================================================================
# Collate and loader factory
# =====================================================================

def collate_abc(batch: list) -> dict:
    result = {
        'points':        torch.stack([s['points']        for s in batch]),
        'normals':       torch.stack([s['normals']       for s in batch]),
        'target_points': torch.stack([s['target_points'] for s in batch]),
    }
    if 'primitive_type' in batch[0]:
        result['primitive_type']   = torch.tensor([s['primitive_type'] for s in batch])
        result['primitive_params'] = torch.stack([s['primitive_params'] for s in batch])
    return result


def get_abc_loaders(root: str, task: str = 'reconstruction',
                    num_points: int = 2048, augment: bool = True,
                    batch_size: int = 32,
                    max_samples: Optional[int] = None,
                    num_workers: int = 2):
    """Load ABC dataset with train/val/test splits.

    Uses real mesh data when available, synthetic fallback otherwise.

    Returns:
        train_loader, val_loader, test_loader
    """
    datasets: Dict[str, ABCDataset] = {}
    for split in ['train', 'val', 'test']:
        ds = ABCDataset(root=root, task=task, num_points=num_points,
                        augment=augment, split=split)
        if max_samples and len(ds) > max_samples:
            ds.data_list = ds.data_list[:max_samples]
        datasets[split] = ds

    train_loader = DataLoader(datasets['train'], batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_abc,
                              num_workers=num_workers)
    val_loader   = DataLoader(datasets['val'],   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_abc,
                              num_workers=num_workers)
    test_loader  = DataLoader(datasets['test'],  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_abc,
                              num_workers=num_workers)

    print(f">>> ABC Dataset ({task}): {num_points} pts/sample, "
          f"loader={'trimesh' if _HAS_TRIMESH else 'custom'}, "
          f"normals={'open3d' if _HAS_OPEN3D else 'torch KNN'}")
    print(f">>> {len(datasets['train'])}/{len(datasets['val'])}/"
          f"{len(datasets['test'])} train/val/test")

    return train_loader, val_loader, test_loader
