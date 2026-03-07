# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Variable grouping for high-dimensional symbolic regression.

Groups correlated variables via spectral clustering, assigns per-group
metric signatures via MetricSearch, and constructs a mother algebra
Cl(P,Q,R) that encompasses all groups for cross-term discovery.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from core.algebra import CliffordAlgebra

logger = logging.getLogger(__name__)


@dataclass
class VariableGroup:
    """A group of correlated variables with its own metric signature.

    Attributes:
        var_indices: Indices into original X columns.
        var_names: Human-readable variable names.
        signature: (p, q, r) from MetricSearch.
        algebra: CliffordAlgebra for this group.
        svd_Vt: SVD right-singular vectors for this group (or None).
        mother_offset: Bit offset in mother algebra basis.
    """
    var_indices: list
    var_names: list
    signature: tuple
    algebra: CliffordAlgebra
    svd_Vt: np.ndarray = None
    mother_offset: int = 0


class VariableGrouper:
    """Groups correlated variables and assigns per-group metric signatures.

    For n_vars <= 6, returns a single group (no clustering needed).
    For n_vars > 6, uses spectral clustering on the absolute correlation
    matrix to identify groups of related variables, then runs MetricSearch
    on each group independently.

    Args:
        max_groups: Maximum number of variable groups.
        min_group_size: Minimum variables per group.
        device: Computation device.
    """

    def __init__(self, max_groups=4, min_group_size=2, device='cpu'):
        self.max_groups = max_groups
        self.min_group_size = min_group_size
        self.device = device

    def group(self, X, y, var_names=None):
        """Main entry: cluster variables, SVD per group, MetricSearch per group.

        Args:
            X: np.ndarray [N, k] input features.
            y: np.ndarray [N] target values.
            var_names: Optional list of variable name strings.

        Returns:
            list[VariableGroup]: One group per cluster.
        """
        n_vars = X.shape[1]
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n_vars)]

        if n_vars <= 6:
            return [self._single_group(X, y, var_names)]

        # 1. Compute |corr(xi, xj)| affinity matrix
        corr_matrix = self._correlation_matrix(X)
        affinity = np.abs(corr_matrix)
        np.fill_diagonal(affinity, 0.0)

        # 2. Spectral clustering
        n_groups = min(n_vars // 3, self.max_groups)
        n_groups = max(1, n_groups)
        labels = self._spectral_cluster(affinity, n_groups)

        # 3. Build groups
        groups = []
        for g in range(n_groups):
            indices = [i for i in range(n_vars) if labels[i] == g]
            if len(indices) < self.min_group_size:
                continue
            group = self._build_group(X, y, indices, var_names)
            groups.append(group)

        # If clustering produced no valid groups, fall back to single group
        if not groups:
            return [self._single_group(X, y, var_names)]

        # Assign mother algebra offsets
        offset = 0
        for g in groups:
            g.mother_offset = offset
            p, q, r = g.signature
            offset += p + q + r

        return groups

    def build_mother_algebra(self, groups):
        """Construct Cl(sum(p), sum(q), sum(r)) encompassing all groups.

        Caps at n=12 (algebra.py hard limit). If exceeded, reduces
        largest groups to 2 dims each via SVD.

        Args:
            groups: list[VariableGroup].

        Returns:
            (CliffordAlgebra, list[int]): Mother algebra and per-group offsets.
        """
        P = sum(g.signature[0] for g in groups)
        Q = sum(g.signature[1] for g in groups)
        R = sum(g.signature[2] for g in groups)

        if P + Q + R > 12:
            self._reduce_groups(groups, target_n=12)
            P = sum(g.signature[0] for g in groups)
            Q = sum(g.signature[1] for g in groups)
            R = sum(g.signature[2] for g in groups)

        mother = CliffordAlgebra(P, Q, R, device=self.device)
        offsets = []
        off = 0
        for g in groups:
            offsets.append(off)
            g.mother_offset = off
            off += g.signature[0] + g.signature[1] + g.signature[2]

        return mother, offsets

    def inject_to_mother(self, mv_local, group, mother_algebra):
        """Map [B, C, 2^n_local] -> [B, C, 2^N_mother] by bit-shifting.

        Each local basis blade index is shifted by group.mother_offset bits.

        Args:
            mv_local: torch.Tensor [..., 2^n_local].
            group: VariableGroup with mother_offset set.
            mother_algebra: CliffordAlgebra for the mother space.

        Returns:
            torch.Tensor [..., 2^N_mother].
        """
        local_dim = group.algebra.dim
        mother_dim = mother_algebra.dim
        offset = group.mother_offset

        batch_shape = mv_local.shape[:-1]
        result = torch.zeros(*batch_shape, mother_dim,
                             device=mv_local.device, dtype=mv_local.dtype)

        for local_idx in range(local_dim):
            # Shift local blade index bits by offset
            mother_idx = local_idx << offset
            if mother_idx < mother_dim:
                result[..., mother_idx] = mv_local[..., local_idx]

        return result

    def _single_group(self, X, y, var_names):
        """Create a single group encompassing all variables."""
        from core.search import MetricSearch

        n_vars = X.shape[1]
        indices = list(range(n_vars))

        # SVD for warm-start
        X_c = X - X.mean(axis=0)
        try:
            _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        except np.linalg.LinAlgError:
            S, Vt = None, None

        # MetricSearch
        data = torch.tensor(
            np.column_stack([X, y.reshape(-1, 1)]),
            dtype=torch.float32, device=self.device,
        )
        if data.shape[0] > 500:
            idx = torch.randperm(data.shape[0])[:500]
            data = data[idx]

        # Standardize
        mu = data.mean(0)
        std = data.std(0).clamp(min=1e-8)
        data = (data - mu) / std

        # Cap at 6 dims for MetricSearch
        if data.shape[1] > 6:
            data_c = data - data.mean(0)
            _, _, V = torch.linalg.svd(data_c, full_matrices=False)
            data = data_c @ V[:6].T

        try:
            searcher = MetricSearch(device=self.device, num_probes=4,
                                    probe_epochs=40, micro_batch_size=64)
            p, q, r = searcher.search(data)
            n = p + q + r
            if n < 2:
                p = max(p, 2 - n + p)
        except Exception:
            p, q, r = min(n_vars, 4), 0, 0

        algebra = CliffordAlgebra(p, q, r, device=self.device)
        return VariableGroup(
            var_indices=indices,
            var_names=[var_names[i] for i in indices],
            signature=(p, q, r),
            algebra=algebra,
            svd_Vt=Vt,
        )

    def _build_group(self, X, y, indices, var_names):
        """Build a VariableGroup for a subset of variable indices."""
        from core.search import MetricSearch

        X_sub = X[:, indices]
        names_sub = [var_names[i] for i in indices]

        # SVD
        X_c = X_sub - X_sub.mean(axis=0)
        try:
            _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        except np.linalg.LinAlgError:
            S, Vt = None, None

        # MetricSearch on group data
        data = torch.tensor(
            np.column_stack([X_sub, y.reshape(-1, 1)]),
            dtype=torch.float32, device=self.device,
        )
        if data.shape[0] > 500:
            idx = torch.randperm(data.shape[0])[:500]
            data = data[idx]

        mu = data.mean(0)
        std = data.std(0).clamp(min=1e-8)
        data = (data - mu) / std

        if data.shape[1] > 6:
            data_c = data - data.mean(0)
            _, _, V = torch.linalg.svd(data_c, full_matrices=False)
            data = data_c @ V[:6].T

        try:
            searcher = MetricSearch(device=self.device, num_probes=2,
                                    probe_epochs=20, micro_batch_size=64)
            p, q, r = searcher.search(data)
            n = p + q + r
            if n < 2:
                p = max(p, 2 - n + p)
        except Exception:
            p, q, r = min(len(indices), 3), 0, 0

        algebra = CliffordAlgebra(p, q, r, device=self.device)
        return VariableGroup(
            var_indices=indices,
            var_names=names_sub,
            signature=(p, q, r),
            algebra=algebra,
            svd_Vt=Vt,
        )

    def _correlation_matrix(self, X):
        """Compute Pearson correlation matrix, NaN-safe."""
        n_vars = X.shape[1]
        corr = np.eye(n_vars)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                xi = X[:, i] - X[:, i].mean()
                xj = X[:, j] - X[:, j].mean()
                denom = np.sqrt((xi ** 2).sum() * (xj ** 2).sum())
                if denom < 1e-30:
                    c = 0.0
                else:
                    c = float(np.dot(xi, xj) / denom)
                if not np.isfinite(c):
                    c = 0.0
                corr[i, j] = c
                corr[j, i] = c
        return corr

    def _spectral_cluster(self, affinity, n_clusters):
        """Simple spectral clustering using Laplacian eigenvectors + k-means."""
        n = affinity.shape[0]
        if n <= n_clusters:
            return list(range(n))

        # Normalized Laplacian
        D = np.diag(affinity.sum(axis=1) + 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return [i % n_clusters for i in range(n)]

        V = eigenvectors[:, :n_clusters]
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        V = V / norms

        labels = self._kmeans(V, n_clusters, max_iter=20)
        return labels

    def _kmeans(self, X, k, max_iter=20):
        """Simple k-means clustering."""
        n = X.shape[0]
        rng = np.random.default_rng(42)

        indices = rng.choice(n, size=min(k, n), replace=False)
        centroids = X[indices].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            for i in range(n):
                dists = np.linalg.norm(X[i] - centroids, axis=1)
                labels[i] = int(np.argmin(dists))

            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels.tolist()

    def _reduce_groups(self, groups, target_n=12):
        """Reduce group dimensions via SVD until total n <= target_n."""
        total = sum(g.signature[0] + g.signature[1] + g.signature[2] for g in groups)
        while total > target_n and len(groups) > 0:
            sizes = [g.signature[0] + g.signature[1] + g.signature[2] for g in groups]
            largest = max(range(len(groups)), key=lambda i: sizes[i])
            g = groups[largest]
            p, q, r = g.signature

            new_p = min(p, 2)
            new_q = 0
            new_r = 0
            reduction = (p + q + r) - (new_p + new_q + new_r)
            g.signature = (new_p, new_q, new_r)
            g.algebra = CliffordAlgebra(new_p, new_q, new_r, device=self.device)
            total -= reduction

            if reduction == 0:
                break
