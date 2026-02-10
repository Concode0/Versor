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
        """Load pre-processed data or generate synthetic samples."""
        data_path = os.path.join(self.root, f'abc_{self.task}_{self.split}.pt')

        if os.path.exists(data_path):
            self.data_list = torch.load(data_path, weights_only=False)
            return

        print(f">>> ABC data not found at {data_path}, generating synthetic data")
        self._generate_synthetic_data()

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
