# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Relationship graph for GA-native variable analysis in symbolic regression.

Encodes typed, weighted edges between variables discovered by commutator
analysis, geodesic coherence, and spectral decomposition. Replaces the
flat variable-group model with a richer structure that guides extraction
ordering and cross-term discovery.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VariableEdge:
    """A typed relationship between two variables.

    Attributes:
        var_i: Index in original X columns.
        var_j: Index in original X columns.
        edge_type: Geometric type of interaction:
            "elliptic" (rotation), "hyperbolic" (boost),
            "parabolic" (shear), or "null" (degenerate).
        strength: Combined coupling score in [0, 1].
        commutator_norm: Pairwise E[||[x_i, x_j]||] from CommutatorAnalyzer.
        coherence: Geodesic coherence of the variable pair subspace.
        bivector_energy: Energy in the e_i ^ e_j plane from SpectralAnalyzer.
        plane_index: Bivector basis index encoding this pair's plane.
    """

    var_i: int
    var_j: int
    edge_type: str
    strength: float
    commutator_norm: float = 0.0
    coherence: float = 0.0
    bivector_energy: float = 0.0
    plane_index: int = -1


@dataclass
class VariableNode:
    """Metadata for a single variable in the graph.

    Attributes:
        var_idx: Index in original X columns.
        var_name: Human-readable name.
        null_score: Near-null energy score from SymmetryDetector.
        reflection_score: Reflection symmetry score for this direction.
    """

    var_idx: int
    var_name: str
    null_score: float = 0.0
    reflection_score: float = 0.0


@dataclass
class RelationshipGraph:
    """Graph of typed variable relationships discovered by GA analysis.

    The graph carries both per-variable metadata (nodes) and pairwise
    relationship information (edges). Group assignments map each edge
    to a VariableGroup (by index). Global geometry metrics (coherence,
    curvature, symmetry) are cached here for use by the implicit solver.

    Attributes:
        nodes: Per-variable metadata.
        edges: Pairwise relationships, sorted by strength descending.
        group_assignments: Maps var_idx -> group_idx for all variables.
        global_coherence: Overall manifold coherence from GeodesicFlow.
        global_curvature: Overall manifold curvature from GeodesicFlow.
        intrinsic_dim: Effective intrinsic dimension from DimensionAnalyzer.
        involution_symmetry: Odd-grade energy fraction from SymmetryDetector.
        continuous_symmetry_dim: Lie symmetry group dimension.
        null_directions: Indices of near-null basis directions.
    """

    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    group_assignments: dict = field(default_factory=dict)
    global_coherence: float = 0.0
    global_curvature: float = 0.0
    intrinsic_dim: int = 0
    involution_symmetry: float = 0.0
    continuous_symmetry_dim: int = 0
    null_directions: list = field(default_factory=list)

    def edges_for_group(self, group_idx: int) -> list:
        """Return edges within a specific group, sorted by strength descending.

        An edge is "within" a group if both endpoints belong to that group.
        """
        group_vars = {v for v, g in self.group_assignments.items() if g == group_idx}
        return [e for e in self.edges if e.var_i in group_vars and e.var_j in group_vars]

    def cross_group_edges(self) -> list:
        """Return edges that span different groups, sorted by strength descending."""
        return [
            e
            for e in self.edges
            if (
                e.var_i in self.group_assignments
                and e.var_j in self.group_assignments
                and self.group_assignments[e.var_i] != self.group_assignments[e.var_j]
            )
        ]

    def strongest_edges(self, n: int = 10) -> list:
        """Return the n strongest edges globally."""
        return self.edges[:n]

    def edge_between(self, var_i: int, var_j: int) -> Optional["VariableEdge"]:
        """Lookup a specific edge (order-independent)."""
        for e in self.edges:
            if (e.var_i == var_i and e.var_j == var_j) or (e.var_i == var_j and e.var_j == var_i):
                return e
        return None

    def geometric_report(self) -> dict:
        """Build a geometric report dict for the implicit solver.

        Returns a dictionary with the cached geometry metrics that
        the ImplicitSolver uses for its geometric decision criteria.
        """
        return {
            "curvature": self.global_curvature,
            "coherence": self.global_coherence,
            "involution_symmetry": self.involution_symmetry,
            "continuous_symmetry_dim": self.continuous_symmetry_dim,
            "intrinsic_dim": self.intrinsic_dim,
            "null_directions": list(self.null_directions),
        }
