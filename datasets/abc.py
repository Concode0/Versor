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


class ABCDataset(Dataset):
    """ABC Dataset for CAD parametric model reconstruction.

    Loads point cloud representations of CAD models for autoencoding
    and parametric reconstruction tasks.

    Supports two modes:
        - reconstruction: Reconstruct input point cloud (self-supervised)
        - primitive: Predict primitive parameters (spheres, planes, cylinders)
    """

    def __init__(self, root, task='reconstruction', num_points=2048,
                 augment=False, split='train'):
        """Initialize ABC dataset.

        Args:
            root: Root directory for data
            task: 'reconstruction' or 'primitive'
            num_points: Number of points per sample
            augment: Apply random SO(3) rotation + translation
            split: 'train', 'val', or 'test'
        """
        self.root = root
        self.task = task
        self.num_points = num_points
        self.augment = augment and (split == 'train')
        self.split = split
        self.data_list = []

        self._load_data()

    def _load_data(self):
        """Load data: cached .pt > raw mesh files > synthetic fallback."""
        cache_path = os.path.join(self.root, f'abc_{self.task}_{self.split}.pt')

        # 1. Try cached .pt
        if os.path.exists(cache_path):
            self.data_list = torch.load(cache_path, weights_only=False)
            print(f">>> ABC: loaded {len(self.data_list)} samples from {cache_path}")
            return

        # 2. Try loading raw mesh files
        loaded = self._load_from_meshes()
        if loaded:
            os.makedirs(self.root, exist_ok=True)
            torch.save(self.data_list, cache_path)
            print(f">>> ABC: cached {len(self.data_list)} samples to {cache_path}")
            return

        # 3. Fallback: synthetic data
        print(f">>> ABC data not found at {cache_path}, generating synthetic data")
        self._generate_synthetic_data()

    def _load_from_meshes(self):
        """Load point clouds from OBJ/OFF mesh files in root directory.

        Expected directory structure:
            root/
              train/ or meshes/
                *.obj or *.off or *.ply

        Samples point clouds from mesh surfaces via random face sampling.

        Returns:
            True if data was loaded successfully, False otherwise.
        """
        import glob as glob_mod

        # Find mesh directory
        mesh_dir = None
        for subdir in [self.split, 'meshes', 'train' if self.split == 'train' else self.split, '']:
            candidate = os.path.join(self.root, subdir) if subdir else self.root
            mesh_files = (glob_mod.glob(os.path.join(candidate, '*.obj')) +
                         glob_mod.glob(os.path.join(candidate, '*.off')) +
                         glob_mod.glob(os.path.join(candidate, '*.ply')))
            if mesh_files:
                mesh_dir = candidate
                break

        if mesh_dir is None:
            return False

        mesh_files = sorted(
            glob_mod.glob(os.path.join(mesh_dir, '*.obj')) +
            glob_mod.glob(os.path.join(mesh_dir, '*.off')) +
            glob_mod.glob(os.path.join(mesh_dir, '*.ply'))
        )

        if not mesh_files:
            return False

        # Split files deterministically
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(mesh_files))
        n = len(mesh_files)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if self.split == 'train':
            file_indices = indices[:train_end]
        elif self.split == 'val':
            file_indices = indices[train_end:val_end]
        else:
            file_indices = indices[val_end:]

        print(f">>> Processing {len(file_indices)} ABC mesh files ({self.split})...")

        for idx in file_indices:
            mesh_path = mesh_files[idx]
            try:
                vertices, faces = self._load_mesh(mesh_path)
                if vertices is None or len(vertices) < 10:
                    continue

                # Sample point cloud from mesh surface
                points = self._sample_from_mesh(vertices, faces, self.num_points)
                points = torch.tensor(points, dtype=torch.float32)

                # Center and normalize
                points = points - points.mean(dim=0, keepdim=True)
                scale = points.abs().max() + 1e-6
                points = points / scale

                normals = self._estimate_normals(points)

                sample = {
                    'points': points,
                    'normals': normals,
                    'target_points': points.clone(),
                }

                if self.task == 'primitive':
                    sample['primitive_type'] = 0
                    sample['primitive_params'] = torch.zeros(13)

                self.data_list.append(sample)
            except Exception as e:
                continue

        return len(self.data_list) > 0

    @staticmethod
    def _load_mesh(path):
        """Load mesh vertices and faces from OBJ or OFF file."""
        ext = os.path.splitext(path)[1].lower()

        if ext == '.obj':
            return ABCDataset._load_obj(path)
        elif ext == '.off':
            return ABCDataset._load_off(path)
        else:
            return None, None

    @staticmethod
    def _load_obj(path):
        """Parse OBJ file for vertices and faces."""
        vertices = []
        faces = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v' and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    # Handle vertex/texture/normal indices
                    face_verts = []
                    for p in parts[1:]:
                        v_idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                        face_verts.append(v_idx)
                    if len(face_verts) >= 3:
                        faces.append(face_verts[:3])
        if not vertices:
            return None, None
        return np.array(vertices, dtype=np.float32), faces

    @staticmethod
    def _load_off(path):
        """Parse OFF file for vertices and faces."""
        with open(path, 'r') as f:
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
                parts = f.readline().strip().split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

            faces = []
            for _ in range(n_faces):
                parts = f.readline().strip().split()
                n = int(parts[0])
                face = [int(parts[i+1]) for i in range(min(n, 3))]
                if len(face) == 3:
                    faces.append(face)

        if not vertices:
            return None, None
        return np.array(vertices, dtype=np.float32), faces

    @staticmethod
    def _sample_from_mesh(vertices, faces, num_points):
        """Sample points uniformly from mesh surface via face area weighting."""
        if not faces:
            # No faces: sample from vertices directly
            if len(vertices) >= num_points:
                idx = np.random.choice(len(vertices), num_points, replace=False)
            else:
                idx = np.random.choice(len(vertices), num_points, replace=True)
            return vertices[idx]

        # Compute face areas for weighted sampling
        v = vertices
        face_arr = np.array(faces)
        v0 = v[face_arr[:, 0]]
        v1 = v[face_arr[:, 1]]
        v2 = v[face_arr[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        areas = np.maximum(areas, 1e-10)
        probs = areas / areas.sum()

        # Sample faces proportional to area
        sampled_faces = np.random.choice(len(faces), num_points, p=probs)

        # Random barycentric coordinates
        r1 = np.sqrt(np.random.rand(num_points))
        r2 = np.random.rand(num_points)
        w0 = 1 - r1
        w1 = r1 * (1 - r2)
        w2 = r1 * r2

        points = (w0[:, None] * v[face_arr[sampled_faces, 0]] +
                  w1[:, None] * v[face_arr[sampled_faces, 1]] +
                  w2[:, None] * v[face_arr[sampled_faces, 2]])

        return points.astype(np.float32)

    def _generate_synthetic_data(self, num_samples=500):
        """Generate synthetic CAD-like point clouds for development.

        Creates point clouds from basic geometric primitives:
        spheres, cylinders, boxes, and their combinations.
        """
        rng = np.random.RandomState(42 if self.split == 'train' else 123)
        primitive_types = ['sphere', 'cylinder', 'box', 'torus', 'combined']

        for _ in range(num_samples):
            prim_type = rng.choice(primitive_types)
            points = self._generate_primitive(prim_type, rng)

            # Sample exactly num_points
            if points.shape[0] > self.num_points:
                indices = rng.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices]
            elif points.shape[0] < self.num_points:
                indices = rng.choice(points.shape[0], self.num_points, replace=True)
                points = points[indices]

            points = torch.tensor(points, dtype=torch.float32)

            # Compute normals (approximate via local PCA or finite diff)
            normals = self._estimate_normals(points)

            sample = {
                'points': points,          # [N, 3]
                'normals': normals,        # [N, 3]
                'target_points': points.clone(),  # For reconstruction
            }

            if self.task == 'primitive':
                # Encode primitive type and parameters
                sample['primitive_type'] = primitive_types.index(prim_type)
                sample['primitive_params'] = self._get_primitive_params(prim_type, rng)

            self.data_list.append(sample)

    def _generate_primitive(self, prim_type, rng):
        """Generate point cloud for a geometric primitive."""
        n = self.num_points * 2  # Oversample then subsample

        if prim_type == 'sphere':
            # Random sphere
            center = rng.randn(3) * 0.3
            radius = 0.5 + rng.rand() * 0.5
            theta = rng.uniform(0, 2 * np.pi, n)
            phi = rng.uniform(0, np.pi, n)
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            points = np.stack([x, y, z], axis=1)

        elif prim_type == 'cylinder':
            center = rng.randn(3) * 0.3
            radius = 0.3 + rng.rand() * 0.3
            height = 1.0 + rng.rand()
            theta = rng.uniform(0, 2 * np.pi, n)
            h = rng.uniform(-height / 2, height / 2, n)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = center[2] + h
            points = np.stack([x, y, z], axis=1)

        elif prim_type == 'box':
            center = rng.randn(3) * 0.3
            dims = 0.5 + rng.rand(3) * 0.5
            # Sample on box surface
            face = rng.randint(0, 6, n)
            u = rng.uniform(-1, 1, n)
            v = rng.uniform(-1, 1, n)
            points = np.zeros((n, 3))
            for f in range(6):
                mask = face == f
                axis = f // 2
                sign = 2 * (f % 2) - 1
                other = [i for i in range(3) if i != axis]
                points[mask, axis] = sign * dims[axis]
                points[mask, other[0]] = u[mask] * dims[other[0]]
                points[mask, other[1]] = v[mask] * dims[other[1]]
            points += center

        elif prim_type == 'torus':
            R = 0.7 + rng.rand() * 0.3  # Major radius
            r = 0.1 + rng.rand() * 0.2  # Minor radius
            theta = rng.uniform(0, 2 * np.pi, n)
            phi = rng.uniform(0, 2 * np.pi, n)
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            points = np.stack([x, y, z], axis=1)

        else:  # combined
            # Union of two primitives
            points1 = self._generate_primitive(rng.choice(['sphere', 'cylinder', 'box']), rng)
            points2 = self._generate_primitive(rng.choice(['sphere', 'cylinder', 'box']), rng)
            points = np.concatenate([points1[:n // 2], points2[:n // 2]], axis=0)

        # Center the point cloud
        points = points - points.mean(axis=0, keepdims=True)
        # Normalize to unit sphere
        scale = np.abs(points).max() + 1e-6
        points = points / scale

        return points.astype(np.float32)

    def _estimate_normals(self, points):
        """Estimate surface normals via local neighborhood PCA.

        Simple k-nearest neighbor approach for normal estimation.
        """
        k = 8
        # Compute pairwise distances
        dist = torch.cdist(points, points)
        _, knn_idx = dist.topk(k + 1, largest=False)  # +1 for self
        knn_idx = knn_idx[:, 1:]  # Remove self

        # Local covariance -> smallest eigenvector = normal
        normals = torch.zeros_like(points)
        for i in range(points.size(0)):
            neighbors = points[knn_idx[i]]
            centered = neighbors - neighbors.mean(dim=0, keepdim=True)
            cov = centered.t() @ centered / k
            # Smallest eigenvector
            _, vecs = torch.linalg.eigh(cov)
            normals[i] = vecs[:, 0]  # Smallest eigenvalue

        # Consistent orientation (point outward from center)
        center = points.mean(dim=0)
        outward = points - center
        flip = (normals * outward).sum(dim=-1) < 0
        normals[flip] *= -1

        return normals

    def _get_primitive_params(self, prim_type, rng):
        """Get parametric representation for a primitive."""
        # Fixed-size param vector: [type_onehot(5), center(3), dims(3), extra(2)] = 13
        params = torch.zeros(13)
        type_idx = ['sphere', 'cylinder', 'box', 'torus', 'combined'].index(prim_type)
        params[type_idx] = 1.0
        params[5:8] = torch.tensor(rng.randn(3) * 0.3)
        params[8:11] = torch.tensor(0.5 + rng.rand(3) * 0.5)
        return params

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx].copy()

        if self.augment:
            # Random SO(3) rotation
            theta = torch.rand(1) * 2 * 3.14159
            axis = torch.randn(3)
            axis = axis / axis.norm()

            # Rodrigues rotation
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

            sample['points'] = (R @ sample['points'].t()).t()
            sample['normals'] = (R @ sample['normals'].t()).t()
            sample['target_points'] = sample['points'].clone()

        return sample


def collate_abc(batch):
    """Collate ABC samples into batch."""
    result = {
        'points': torch.stack([s['points'] for s in batch]),
        'normals': torch.stack([s['normals'] for s in batch]),
        'target_points': torch.stack([s['target_points'] for s in batch]),
    }
    if 'primitive_type' in batch[0]:
        result['primitive_type'] = torch.tensor([s['primitive_type'] for s in batch])
        result['primitive_params'] = torch.stack([s['primitive_params'] for s in batch])
    return result


def get_abc_loaders(root, task='reconstruction', num_points=2048,
                    augment=True, batch_size=32, max_samples=None):
    """Load ABC dataset with train/val/test splits.

    Returns:
        train_loader, val_loader, test_loader
    """
    datasets = {}
    for split in ['train', 'val', 'test']:
        ds = ABCDataset(
            root=root, task=task, num_points=num_points,
            augment=augment, split=split
        )
        if max_samples and len(ds) > max_samples:
            ds.data_list = ds.data_list[:max_samples]
        datasets[split] = ds

    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_abc)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, collate_fn=collate_abc)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_abc)

    print(f">>> ABC Dataset ({task}): {num_points} points/sample")
    print(f">>> {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])} train/val/test")

    return train_loader, val_loader, test_loader
