# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Variable grouping for high-dimensional symbolic regression.

Uses GA-native analysis (commutator, coherence, spectral, symmetry) to
build a relationship graph of typed variable interactions, then clusters
via spectral clustering on the GA-derived affinity matrix.  Per-group
metric signatures are assigned via MetricSearch, and a mother algebra
Cl(P,Q,R) is constructed for cross-term discovery.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from core.algebra import CliffordAlgebra
from models.sr.utils import standardize, subsample, safe_svd

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
        internal_edges: VariableEdge list within this group.
        group_coherence: Geodesic coherence of this group's subspace.
        group_curvature: Geodesic curvature of this group's subspace.
    """

    var_indices: list
    var_names: list
    signature: tuple
    algebra: CliffordAlgebra
    svd_Vt: np.ndarray = None
    mother_offset: int = 0
    internal_edges: list = field(default_factory=list)
    group_coherence: float = 0.0
    group_curvature: float = 0.0


class VariableGrouper:
    """Groups variables using GA-native relationship analysis.

    For n_vars <= 3, computes only global geometry (coherence/curvature)
    without a full relationship graph.  For n_vars <= 6, builds the graph
    but returns a single group.  For n_vars > 6, uses spectral clustering
    on the GA-derived affinity matrix.

    Args:
        max_groups: Maximum number of variable groups.
        min_group_size: Minimum variables per group.
        device: Computation device.
        sample_size: Max data points for analysis subsampling.
        commutator_weight: Weight for commutator norm in affinity.
        coherence_weight: Weight for pairwise coherence in affinity.
        spectral_weight: Weight for bivector energy in affinity.
    """

    def __init__(
        self,
        max_groups=4,
        min_group_size=2,
        device='cpu',
        sample_size=500,
        commutator_weight=0.4,
        coherence_weight=0.3,
        spectral_weight=0.3,
    ):
        self.max_groups = max_groups
        self.min_group_size = min_group_size
        self.device = device
        self.sample_size = sample_size
        self.commutator_weight = commutator_weight
        self.coherence_weight = coherence_weight
        self.spectral_weight = spectral_weight

    def group(self, X, y, var_names=None):
        """Main entry: build relationship graph, cluster, assign signatures.

        Args:
            X: np.ndarray [N, k] input features.
            y: np.ndarray [N] target values.
            var_names: Optional list of variable name strings.

        Returns:
            (list[VariableGroup], RelationshipGraph): Groups and their
            relationship graph with typed edges.
        """
        from models.sr.relationship_graph import RelationshipGraph

        n_vars = X.shape[1]
        if var_names is None:
            var_names = [f"x{i + 1}" for i in range(n_vars)]

        # Build the relationship graph using GA analysis tools
        graph = self._build_relationship_graph(X, y, var_names)

        if n_vars <= 6:
            group = self._single_group(X, y, var_names)
            group.internal_edges = list(graph.edges)
            group.group_coherence = graph.global_coherence
            group.group_curvature = graph.global_curvature
            # Set group assignments
            for i in range(n_vars):
                graph.group_assignments[i] = 0
            return [group], graph

        # Compute affinity matrix from graph edges
        affinity = self._compute_affinity_from_graph(graph, n_vars)

        # Spectral clustering on GA-derived affinity
        n_groups = min(n_vars // 3, self.max_groups)
        n_groups = max(1, n_groups)
        labels = self._spectral_cluster(affinity, n_groups)

        # Build groups
        groups = []
        for g in range(n_groups):
            indices = [i for i in range(n_vars) if labels[i] == g]
            if len(indices) < self.min_group_size:
                continue
            group = self._build_group(X, y, indices, var_names)
            # Attach graph edges for this group
            group_var_set = set(indices)
            group.internal_edges = [e for e in graph.edges if e.var_i in group_var_set and e.var_j in group_var_set]
            groups.append(group)

        if not groups:
            group = self._single_group(X, y, var_names)
            group.internal_edges = list(graph.edges)
            group.group_coherence = graph.global_coherence
            group.group_curvature = graph.global_curvature
            for i in range(n_vars):
                graph.group_assignments[i] = 0
            return [group], graph

        # Assign mother algebra offsets and group assignments
        offset = 0
        for g_idx, g in enumerate(groups):
            g.mother_offset = offset
            p, q, r = g.signature
            offset += p + q + r
            for vi in g.var_indices:
                graph.group_assignments[vi] = g_idx

        return groups, graph

    def _build_relationship_graph(self, X, y, var_names):
        """Build a RelationshipGraph using GA analysis tools.

        Steps:
        1. Subsample data
        2. Dimension analysis → intrinsic_dim
        3. PCA reduction if n_vars > 6
        4. MetricSearch → (p, q, r) and algebra
        5. Embed as grade-1 multivectors
        6. CommutatorAnalyzer → pairwise non-commutativity
        7. GeodesicFlow → global coherence/curvature
        8. SpectralAnalyzer → primary coupling planes
        9. SymmetryDetector → null dirs, involution, reflections
        10. Classify edges, assemble graph
        """
        from core.analysis import (
            CommutatorAnalyzer,
            GeodesicFlow,
            SpectralAnalyzer,
            SymmetryDetector,
            EffectiveDimensionAnalyzer,
            MetricSearch,
        )
        from models.sr.relationship_graph import (
            RelationshipGraph,
            VariableEdge,
            VariableNode,
        )

        n_vars = X.shape[1]

        # 1. Subsample
        combined = np.column_stack([X, y.reshape(-1, 1)])
        data_t = torch.tensor(combined, dtype=torch.float32, device=self.device)
        data_t = subsample(data_t, self.sample_size)
        data_t = standardize(data_t)

        # Separate back out for per-variable analysis
        X_sub = data_t[:, :n_vars]
        n_analysis = n_vars  # dimensions used in the analysis

        # 2. Dimension analysis
        dim_analyzer = EffectiveDimensionAnalyzer(device=self.device)
        dim_result = dim_analyzer.analyze(data_t)
        intrinsic_dim = dim_result.intrinsic_dim

        # 3. PCA reduction for algebra (cap at 6 for tractable 2^n dim)
        analysis_data = data_t
        if data_t.shape[1] > 6:
            target_dim = min(intrinsic_dim, 6)
            target_dim = max(target_dim, 2)  # need at least 2 for bivectors
            analysis_data = dim_analyzer.reduce(data_t, target_dim)
            n_analysis = target_dim

        # 4. MetricSearch for global signature
        from models.sr.utils import safe_metric_search

        p, q, r = safe_metric_search(
            analysis_data,
            self.device,
            n_analysis,
        )

        algebra = CliffordAlgebra(p, q, r, device=self.device)

        # 5. Embed as grade-1 multivectors
        alg_n = algebra.n
        embed_data = analysis_data
        if analysis_data.shape[1] > alg_n:
            embed_data = analysis_data[:, :alg_n]
        elif analysis_data.shape[1] < alg_n:
            pad = torch.zeros(
                analysis_data.shape[0],
                alg_n - analysis_data.shape[1],
                device=self.device,
            )
            embed_data = torch.cat([analysis_data, pad], dim=-1)
        mv_data = algebra.embed_vector(embed_data)  # [N, dim]

        # 6. Commutator analysis → pairwise non-commutativity
        comm_analyzer = CommutatorAnalyzer(algebra)
        comm_result = comm_analyzer.analyze(mv_data)
        comm_matrix = comm_result.commutativity_matrix  # [n, n]

        # 7. Geodesic flow → global coherence and curvature
        geo_flow = GeodesicFlow(algebra, k=min(8, mv_data.shape[0] - 1))
        global_coherence = float(geo_flow.coherence(mv_data))
        global_curvature = float(geo_flow.curvature(mv_data))

        # 8. Spectral analysis → bivector field spectrum
        spectral_analyzer = SpectralAnalyzer(algebra)
        spectral_result = spectral_analyzer.analyze(mv_data)
        # Build a map of bivector energy per plane index
        bv_energy_map = self._build_bivector_energy_map(
            spectral_result,
            algebra,
        )

        # 9. Symmetry detection
        sym_detector = SymmetryDetector(algebra)
        sym_result = sym_detector.analyze(mv_data, commutator_result=comm_result)

        # 10. Build nodes
        nodes = []
        null_scores = sym_result.null_scores  # [n] tensor
        refl_syms = sym_result.reflection_symmetries  # list of dicts
        refl_map = {}
        for rs in refl_syms:
            refl_map[rs["direction"]] = rs["score"]

        for i in range(n_vars):
            ns = float(null_scores[i]) if i < len(null_scores) else 0.0
            rs = refl_map.get(i, 0.0)
            nodes.append(
                VariableNode(
                    var_idx=i,
                    var_name=var_names[i],
                    null_score=ns,
                    reflection_score=rs,
                )
            )

        # 11. Build edges — for each variable pair in original space
        edges = []
        # bv_sq_scalar for edge type classification
        bv_sq = algebra.bv_sq_scalar if hasattr(algebra, 'bv_sq_scalar') else None

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Commutator norm (only if both fit in algebra dims)
                c_norm = 0.0
                if i < comm_matrix.shape[0] and j < comm_matrix.shape[1]:
                    c_norm = float(comm_matrix[i, j])

                # Bivector energy for the (i, j) plane
                bv_e = bv_energy_map.get((i, j), 0.0)

                # Pair coherence: approximate via commutator relationship
                # High commutator norm → strong algebraic coupling → high pair coherence
                pair_coh = min(c_norm * 2.0, 1.0)

                # Classify edge type via bivector square
                edge_type = self._classify_edge_type(i, j, algebra, bv_sq)

                # Plane index: bivector basis index for e_i ^ e_j
                plane_idx = self._bivector_index(i, j)

                # Combined strength
                raw = self.commutator_weight * c_norm + self.coherence_weight * pair_coh + self.spectral_weight * bv_e
                strength = min(raw, 1.0)

                edges.append(
                    VariableEdge(
                        var_i=i,
                        var_j=j,
                        edge_type=edge_type,
                        strength=strength,
                        commutator_norm=c_norm,
                        coherence=pair_coh,
                        bivector_energy=bv_e,
                        plane_index=plane_idx,
                    )
                )

        # Sort by strength descending
        edges.sort(key=lambda e: e.strength, reverse=True)

        graph = RelationshipGraph(
            nodes=nodes,
            edges=edges,
            global_coherence=global_coherence,
            global_curvature=global_curvature,
            intrinsic_dim=intrinsic_dim,
            involution_symmetry=float(sym_result.involution_symmetry),
            continuous_symmetry_dim=sym_result.continuous_symmetry_dim,
            null_directions=sym_result.null_directions,
        )

        logger.info(
            f"Relationship graph: {n_vars} vars, {len(edges)} edges, "
            f"coherence={global_coherence:.3f}, curvature={global_curvature:.3f}, "
            f"involution={sym_result.involution_symmetry:.3f}, "
            f"top edge: {edges[0].var_i}-{edges[0].var_j} "
            f"({edges[0].edge_type}, str={edges[0].strength:.3f})"
            if edges
            else f"Relationship graph: {n_vars} vars, 0 edges"
        )

        return graph

    def _build_bivector_energy_map(self, spectral_result, algebra):
        """Map (var_i, var_j) pairs to their bivector energy contribution.

        Uses the simple components from bivector_field_spectrum to assign
        energy to specific variable-pair planes.
        """
        energy_map = {}
        n = algebra.n

        # spectral_result.bivector_spectrum is a 1-D tensor of singular values
        # spectral_result.simple_components is a list of [dim] tensors
        sv = spectral_result.bivector_spectrum
        components = spectral_result.simple_components

        if sv is None or components is None:
            return energy_map

        # For each simple component, find the dominant (i,j) plane
        for k, comp in enumerate(components):
            if k >= len(sv):
                break
            energy = float(sv[k])
            if energy < 1e-8:
                continue

            # Find the largest bivector coefficient in this component
            best_val = 0.0
            best_pair = None
            for vi in range(n):
                for vj in range(vi + 1, n):
                    bv_idx = (1 << vi) | (1 << vj)
                    if bv_idx < len(comp):
                        val = abs(float(comp[bv_idx]))
                        if val > best_val:
                            best_val = val
                            best_pair = (vi, vj)

            if best_pair is not None:
                prev = energy_map.get(best_pair, 0.0)
                energy_map[best_pair] = prev + energy

        # Normalize to [0, 1]
        if energy_map:
            max_e = max(energy_map.values())
            if max_e > 1e-12:
                for k in energy_map:
                    energy_map[k] /= max_e

        return energy_map

    def _classify_edge_type(self, var_i, var_j, algebra, bv_sq):
        """Classify edge type via bivector square B_{ij}^2.

        Uses the algebra's precomputed bv_sq_scalar to determine:
        - B^2 < -0.5 → "elliptic" (rotation)
        - B^2 > +0.5 → "hyperbolic" (boost)
        - |B^2| < 0.5 → "parabolic" (shear/translation)
        """
        from core.analysis._types import CONSTANTS

        if bv_sq is None or var_i >= algebra.n or var_j >= algebra.n:
            return "elliptic"  # default assumption

        # bv_sq_scalar is indexed by bivector basis index
        bv_idx = (1 << var_i) | (1 << var_j)
        # Find the position in the grade-2 list
        grade2_mask = algebra.grade_masks[2]
        grade2_indices = grade2_mask.nonzero(as_tuple=True)[0]
        for pos, idx in enumerate(grade2_indices):
            if int(idx) == bv_idx and pos < len(bv_sq):
                sq_val = float(bv_sq[pos])
                if sq_val < CONSTANTS.bv_sq_elliptic_bound:
                    return "elliptic"
                elif sq_val > CONSTANTS.bv_sq_hyperbolic_bound:
                    return "hyperbolic"
                else:
                    return "parabolic"

        return "elliptic"

    def _bivector_index(self, var_i, var_j):
        """Compute bivector basis index for e_i ^ e_j."""
        return (1 << var_i) | (1 << var_j)

    def _compute_affinity_from_graph(self, graph, n_vars):
        """Convert relationship graph edges to affinity matrix for clustering."""
        affinity = np.zeros((n_vars, n_vars))
        for e in graph.edges:
            if e.var_i < n_vars and e.var_j < n_vars:
                affinity[e.var_i, e.var_j] = e.strength
                affinity[e.var_j, e.var_i] = e.strength
        np.fill_diagonal(affinity, 0.0)
        return affinity

    # ------------------------------------------------------------------
    # Mother algebra (unchanged)
    # ------------------------------------------------------------------

    def build_mother_algebra(self, groups):
        """Construct Cl(sum(p), sum(q), sum(r)) encompassing all groups.

        Caps at n=12 (algebra.py hard limit). If exceeded, reduces
        largest groups to 2 dims each via SVD.

        Builds per-group basis vector maps that respect signature ordering:
        mother basis is [e1+...eP+, e1-...eQ-, e1deg...eRdeg], and each
        group's local positive/negative/null vectors are mapped to the
        correct mother slots.

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

        p_offset = 0
        q_offset = P
        r_offset = P + Q

        offsets = []
        for g in groups:
            gp, gq, gr = g.signature
            vec_map = {}
            for i in range(gp):
                vec_map[i] = p_offset + i
            for i in range(gq):
                vec_map[gp + i] = q_offset + i
            for i in range(gr):
                vec_map[gp + gq + i] = r_offset + i

            g._mother_vec_map = vec_map
            g.mother_offset = 0
            offsets.append(vec_map)

            p_offset += gp
            q_offset += gq
            r_offset += gr

        return mother, offsets

    def inject_to_mother(self, mv_local, group, mother_algebra):
        """Map [B, C, 2^n_local] -> [B, C, 2^N_mother] via basis vector map.

        Uses the per-group vector map built by build_mother_algebra to
        correctly translate blade indices respecting signature ordering.
        Local blade index bits are remapped to mother blade index bits.

        Args:
            mv_local: torch.Tensor [..., 2^n_local].
            group: VariableGroup with _mother_vec_map set.
            mother_algebra: CliffordAlgebra for the mother space.

        Returns:
            torch.Tensor [..., 2^N_mother].
        """
        local_dim = group.algebra.dim
        mother_dim = mother_algebra.dim
        vec_map = group._mother_vec_map
        n_local = group.algebra.n

        batch_shape = mv_local.shape[:-1]
        result = torch.zeros(*batch_shape, mother_dim, device=mv_local.device, dtype=mv_local.dtype)

        for local_idx in range(local_dim):
            mother_idx = 0
            for bit in range(n_local):
                if local_idx & (1 << bit):
                    mother_bit = vec_map.get(bit)
                    if mother_bit is None:
                        break
                    mother_idx |= 1 << mother_bit
            else:
                if mother_idx < mother_dim:
                    result[..., mother_idx] = mv_local[..., local_idx]

        return result

    # ------------------------------------------------------------------
    # Per-group builders (reuse MetricSearch + SVD)
    # ------------------------------------------------------------------

    def _single_group(self, X, y, var_names):
        """Create a single group encompassing all variables."""
        from core.analysis import MetricSearch

        n_vars = X.shape[1]
        indices = list(range(n_vars))

        X_c = X - X.mean(axis=0)
        S, Vt = safe_svd(X_c)

        data = torch.tensor(
            np.column_stack([X, y.reshape(-1, 1)]),
            dtype=torch.float32,
            device=self.device,
        )
        data = subsample(data, 500)
        data = standardize(data)

        if data.shape[1] > 6:
            data_c = data - data.mean(0)
            _, _, V = torch.linalg.svd(data_c, full_matrices=False)
            data = data_c @ V[:6].T

        from models.sr.utils import safe_metric_search

        p, q, r = safe_metric_search(data, self.device, n_vars)

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
        from core.analysis import MetricSearch

        X_sub = X[:, indices]
        names_sub = [var_names[i] for i in indices]

        X_c = X_sub - X_sub.mean(axis=0)
        S, Vt = safe_svd(X_c)

        data = torch.tensor(
            np.column_stack([X_sub, y.reshape(-1, 1)]),
            dtype=torch.float32,
            device=self.device,
        )
        data = subsample(data, 500)
        data = standardize(data)

        if data.shape[1] > 6:
            data_c = data - data.mean(0)
            _, _, V = torch.linalg.svd(data_c, full_matrices=False)
            data = data_c @ V[:6].T

        from models.sr.utils import safe_metric_search

        p, q, r = safe_metric_search(
            data,
            self.device,
            len(indices),
            num_probes=2,
            probe_epochs=20,
        )

        algebra = CliffordAlgebra(p, q, r, device=self.device)
        return VariableGroup(
            var_indices=indices,
            var_names=names_sub,
            signature=(p, q, r),
            algebra=algebra,
            svd_Vt=Vt,
        )

    # ------------------------------------------------------------------
    # Clustering utilities (reused, operate on GA-derived affinity)
    # ------------------------------------------------------------------

    def _spectral_cluster(self, affinity, n_clusters):
        """Simple spectral clustering using Laplacian eigenvectors + k-means."""
        n = affinity.shape[0]
        if n <= n_clusters:
            return list(range(n))

        from models.sr.numerics import safe_inv_sqrt_diag

        D = np.diag(affinity.sum(axis=1) + 1e-10)
        D_inv_sqrt = np.diag(safe_inv_sqrt_diag(np.diag(D)))
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
