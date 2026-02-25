# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Strict Orthogonality Enforcement for Geometric Algebra Networks.

Provides loss terms, projection operators, and diagnostic tools that
enforce and monitor grade orthogonality in multivector representations.
Designed as an external, pluggable module for mathematical debugging.

In a Clifford algebra, different grades are algebraically orthogonal
under the Hermitian inner product. In practice, numerical operations
in neural networks can introduce parasitic cross-grade energy. This
module detects and corrects such drift.

Usage as external settings:

    from experiments.orthogonality import StrictOrthogonality, OrthogonalitySettings

    settings = OrthogonalitySettings(
        enabled=True,
        mode='loss',
        weight=0.1,
        target_grades=[0, 2],     # even subalgebra only
        coupling_warn_threshold=0.3,
    )
    ortho = StrictOrthogonality(algebra, settings)

    # In training loop:
    ortho_loss = ortho(hidden_features)
    total_loss = task_loss + ortho_loss

    # Weight annealing (ramp from 0 to weight over warmup_epochs):
    effective_w = ortho.anneal_weight(epoch, warmup_epochs=20, total_epochs=200)

    # Diagnostics:
    report = ortho.format_diagnostics(hidden_features)

    # Coupling heatmap (returns matplotlib Figure):
    fig = ortho.visualize_coupling(hidden_features)
    fig.savefig("coupling.png")
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class OrthogonalitySettings:
    """External configuration for strict orthogonality enforcement.

    Attributes:
        enabled: Master switch.
        mode: 'loss' (soft penalty) or 'project' (hard zeroing).
        weight: Penalty weight when mode='loss'.
        target_grades: Which grades are permitted (others are parasitic).
                       None means all grades are permitted (orthogonality disabled).
        tolerance: Parasitic ratio below which orthogonality is satisfied.
        monitor_interval: Epochs between full diagnostic reports.
        coupling_warn_threshold: Issue a warning when max off-diagonal coupling
                                 exceeds this threshold (0 = warn always, 1 = never).
    """
    enabled: bool = True
    mode: str = 'loss'
    weight: float = 0.1
    target_grades: Optional[List[int]] = None
    tolerance: float = 1e-4
    monitor_interval: int = 10
    coupling_warn_threshold: float = 0.3


class StrictOrthogonality(nn.Module):
    """Enforce and monitor grade orthogonality in multivector features.

    Given a Clifford algebra Cl(p,q) and a set of target grades,
    this module:
      1. Penalizes energy in parasitic (non-target) grades
      2. Monitors cross-grade coupling as a diagnostic
      3. Optionally hard-projects to the target grade subspace

    The key invariant: if the computation is algebraically confined to
    a subalgebra (e.g. the even subalgebra for complex analysis),
    then non-target grade energy indicates numerical grade leakage.

    Grade masks are stored as registered buffers and move with the module
    when calling .to(device).
    """

    def __init__(self, algebra, settings: Optional[OrthogonalitySettings] = None):
        super().__init__()
        self.algebra = algebra

        if settings is None:
            settings = OrthogonalitySettings()
        self.settings = settings

        self._n_grades = algebra.n + 1
        self._build_grade_masks()

    def _build_grade_masks(self):
        """Precompute boolean masks as a registered [n_grades, dim] buffer.

        Stored as ``grade_masks_tensor`` so masks are automatically moved
        to the correct device with .to(device).
        """
        n_grades = self._n_grades
        dim = self.algebra.dim

        # [n_grades, dim] boolean tensor: grade_masks_tensor[g, i] = (popcount(i) == g)
        masks = torch.zeros(n_grades, dim, dtype=torch.bool)
        for idx in range(dim):
            g = bin(idx).count('1')
            masks[g, idx] = True
        self.register_buffer('grade_masks_tensor', masks)

        # Build target / parasitic masks
        target_grades = self.settings.target_grades
        if target_grades is None:
            target_grades = list(range(n_grades))

        target_mask = masks[target_grades].any(dim=0)   # [dim]
        self.register_buffer('target_mask', target_mask)
        self.register_buffer('parasitic_mask', ~target_mask)

    # ------------------------------------------------------------------
    # Loss / Projection
    # ------------------------------------------------------------------

    def parasitic_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squared energy in non-target grade components.

        Args:
            x: [..., dim] multivector coefficients.

        Returns:
            Scalar loss tensor.
        """
        parasitic = x[..., self.parasitic_mask]
        if parasitic.numel() == 0:
            # No parasitic grades (target_grades covers everything) -> zero loss.
            return x.new_zeros(())
        return (parasitic ** 2).mean()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Hard projection: zero out parasitic grade components.

        Differentiable (straight-through estimator): gradients flow
        through the target-grade coefficients unchanged.

        Args:
            x: [..., dim] multivector.

        Returns:
            Projected multivector.
        """
        return x * self.target_mask.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply orthogonality enforcement.

        Returns:
            mode='loss':    scalar penalty (add to total loss externally).
            mode='project': projected multivector.
        """
        if not self.settings.enabled:
            return x if self.settings.mode == 'project' else x.new_zeros(())

        if self.settings.mode == 'project':
            return self.project(x)

        return self.settings.weight * self.parasitic_energy(x)

    # ------------------------------------------------------------------
    # Weight Annealing
    # ------------------------------------------------------------------

    def anneal_weight(self, epoch: int, warmup_epochs: int,
                      total_epochs: int) -> float:
        """Compute linearly annealed weight for progressive enforcement.

        The penalty ramps from 0 -> settings.weight over the first
        ``warmup_epochs`` epochs, then stays constant.

        Args:
            epoch: Current epoch (1-indexed).
            warmup_epochs: Number of warmup epochs.
            total_epochs: Total training epochs (unused, kept for signature).

        Returns:
            Effective weight to use this epoch.

        Example::

            w = ortho.anneal_weight(epoch, warmup_epochs=20, total_epochs=200)
            loss = task_loss + w * ortho.parasitic_energy(h)
        """
        if warmup_epochs <= 0:
            return self.settings.weight
        frac = min(epoch / warmup_epochs, 1.0)
        return self.settings.weight * frac

    # ------------------------------------------------------------------
    # Diagnostics (mathematical debugger)
    # ------------------------------------------------------------------

    def grade_energies(self, x: torch.Tensor) -> Dict[int, float]:
        """Mean squared energy per grade.

        Args:
            x: [..., dim] multivector.

        Returns:
            dict {grade: float}.
        """
        energies = {}
        for g in range(self._n_grades):
            mask = self.grade_masks_tensor[g]          # [dim] bool, on correct device
            components = x[..., mask]
            energies[g] = (components ** 2).mean().item()
        return energies

    def parasitic_ratio(self, x: torch.Tensor) -> float:
        """Fraction of total energy in parasitic grades.

        Returns:
            float in [0, 1]. 0 = perfect confinement.
        """
        energies = self.grade_energies(x)
        total = sum(energies.values()) + 1e-12
        target_grades = self.settings.target_grades
        if target_grades is None:
            return 0.0
        target = sum(energies.get(g, 0.0) for g in target_grades)
        return 1.0 - target / total

    def cross_grade_coupling(self, x: torch.Tensor) -> torch.Tensor:
        """Correlation matrix of grade energies across the batch.

        Non-diagonal entries indicate that grade energies co-fluctuate,
        which may signal that the network is mixing grades.

        Args:
            x: [B, ..., dim] batched multivectors (first dim is batch).

        Returns:
            [n_grades, n_grades] coupling matrix in [-1, 1].
        """
        B = x.shape[0]
        n_grades = self._n_grades

        # Per-sample grade energy: [n_grades, B]
        energies = []
        for g in range(n_grades):
            mask = self.grade_masks_tensor[g]          # [dim] bool, correct device
            comp = x[..., mask]
            e = (comp ** 2).reshape(B, -1).sum(dim=-1)  # [B]
            energies.append(e)
        energies = torch.stack(energies, dim=0)        # [n_grades, B]

        # Normalize per grade (z-score)
        means = energies.mean(dim=1, keepdim=True)
        stds = energies.std(dim=1, keepdim=True) + 1e-8
        normed = (energies - means) / stds             # [n_grades, B]

        coupling = (normed @ normed.t()) / B           # [n_grades, n_grades]
        return coupling

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict:
        """Full diagnostic report for mathematical debugging.

        Issues a UserWarning if cross-grade coupling exceeds
        ``settings.coupling_warn_threshold``.

        Returns:
            dict with keys:
                grade_energies:          {grade: float}
                parasitic_ratio:         float
                coupling_matrix:         Tensor [n_grades, n_grades] or None
                orthogonality_satisfied: bool
                coupling_max_off_diag:   float
        """
        energies = self.grade_energies(x)
        p_ratio = self.parasitic_ratio(x)

        coupling = None
        max_off = 0.0
        if x.dim() >= 2 and x.shape[0] > 1:
            coupling = self.cross_grade_coupling(x)
            n = coupling.shape[0]
            off_mask = ~torch.eye(n, dtype=torch.bool, device=coupling.device)
            if off_mask.any():
                max_off = coupling[off_mask].abs().max().item()

            if max_off > self.settings.coupling_warn_threshold:
                warnings.warn(
                    f"High cross-grade coupling detected: max_off_diag={max_off:.4f} "
                    f"(threshold={self.settings.coupling_warn_threshold}). "
                    "The network may be mixing grades.",
                    RuntimeWarning, stacklevel=2
                )

        return {
            'grade_energies': energies,
            'parasitic_ratio': p_ratio,
            'coupling_matrix': coupling,
            'orthogonality_satisfied': p_ratio < self.settings.tolerance,
            'coupling_max_off_diag': max_off,
        }

    def format_diagnostics(self, x: torch.Tensor) -> str:
        """Human-readable diagnostic string with ASCII grade energy bars.

        Example output::

            Grade energies (ASCII bar):
              G0 [######          ]  0.3842  <- target
              G1 [##              ]  0.0931
              G2 [########        ]  0.4980  <- target
              G3 [#               ]  0.0247
            Parasitic ratio: 1.1780%
            Orthogonality satisfied: YES (tol=0.0001)
            Coupling: max_off_diag=0.1234
        """
        d = self.diagnostics(x)
        energies = d['grade_energies']
        target_grades = set(self.settings.target_grades or range(self._n_grades))

        lines = ["  Grade energies (ASCII bar):"]
        max_e = max(energies.values()) + 1e-12
        bar_width = 16
        for g, e in energies.items():
            filled = int(bar_width * e / max_e)
            bar = '#' * filled + ' ' * (bar_width - filled)
            tag = '  <- target' if g in target_grades else ''
            lines.append(f"    G{g} [{bar}]  {e:.4f}{tag}")

        lines.append(f"  Parasitic ratio: {d['parasitic_ratio']:.4%}")
        status = "YES" if d['orthogonality_satisfied'] else "NO"
        lines.append(f"  Orthogonality satisfied: {status} "
                     f"(tol={self.settings.tolerance})")

        if d['coupling_matrix'] is not None:
            max_off = d['coupling_max_off_diag']
            lines.append(f"  Coupling: max_off_diag={max_off:.4f}")

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_coupling(self, x: torch.Tensor,
                           title: str = "Cross-Grade Coupling") -> Optional[object]:
        """Plot the cross-grade coupling matrix as a heatmap.

        Target grades are highlighted with lime-colored borders.

        Args:
            x: [B, ..., dim] batched multivectors.
            title: Plot title.

        Returns:
            matplotlib.figure.Figure, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            warnings.warn("matplotlib not available; visualize_coupling() returning None.")
            return None

        with torch.no_grad():
            coupling = self.cross_grade_coupling(x).detach().cpu().numpy()

        n = coupling.shape[0]
        fig, ax = plt.subplots(figsize=(max(5, n + 1), max(5, n + 1)))
        im = ax.imshow(coupling, cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                       aspect='auto', interpolation='nearest')

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = coupling[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f'G{g}' for g in range(n)])
        ax.set_yticklabels([f'G{g}' for g in range(n)])
        ax.set_xlabel('Grade')
        ax.set_ylabel('Grade')
        ax.set_title(title)

        # Highlight target grades with lime borders
        target_grades = self.settings.target_grades or list(range(n))
        for g in target_grades:
            if 0 <= g < n:
                rect = plt.Rectangle((g - 0.5, g - 0.5), 1, 1,
                                     linewidth=2, edgecolor='lime',
                                     facecolor='none')
                ax.add_patch(rect)

        plt.colorbar(im, ax=ax, label='Correlation', fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig
