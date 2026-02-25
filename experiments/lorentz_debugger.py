# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Lorentz Transformation Debugger in Spacetime Algebra Cl(3,1).

Validates Versor's geometric algebra operations against known physical
laws of special relativity. Spacetime events are grade-1 vectors,
Lorentz boosts are rotors R = exp(-phi/2 * B), and the spacetime
interval s^2 = x^2 + y^2 + z^2 - t^2 is a Lorentz invariant.

In Cl(3,1) with signature (+,+,+,-):
  - Boost bivectors (e14, e24, e34) square to +1 -> hyperbolic rotors
  - Rotation bivectors (e12, e13, e23) square to -1 -> trigonometric rotors
  - algebra.exp() handles both via Taylor series + scaling-and-squaring

Debugger checks:
  1. Interval invariance: s^2(x) == s^2(RxR~)
  2. Rotor normalization: RR~ = 1
  3. Grade confinement: energy in even grades {0, 2, 4} vs parasitic {1, 3}
  4. Causality classification: timelike / spacelike / lightlike preservation
  5. Invariant mass: m^2 = -s^2 preserved under transformation
  6. Rapidity extraction: recover phi from learned rotor (pure boost)
  7. Velocity addition: R(phi1)R(phi2) = R(phi1 + phi2)

Usage:
    uv run python -m experiments.lorentz_debugger --epochs 200
    uv run python -m experiments.lorentz_debugger --strict-ortho --boost-type combined
    uv run python -m experiments.lorentz_debugger --save-plots --output-dir my_plots
    uv run python -m experiments.lorentz_debugger --rapidity-max 2.0 --diag-interval 10
"""

from __future__ import annotations

import sys
import os
import argparse
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from core.metric import (
    signature_norm_squared, hermitian_distance, hermitian_norm,
    hermitian_grade_spectrum, signature_trace_form,
)
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam
from experiments.orthogonality import StrictOrthogonality, OrthogonalitySettings


# =====================================================================
# Boost Bivector Helpers
# =====================================================================

def _boost_bivector(algebra, axis: int) -> torch.Tensor:
    """Create a unit boost bivector in Cl(3,1).

    Boost planes: e14 (x-boost), e24 (y-boost), e34 (z-boost).
    In Cl(3,1) with basis e1,e2,e3,e4, the 4th dimension (index 3)
    has negative signature.

    Args:
        algebra: CliffordAlgebra(3, 1).
        axis: 0, 1, or 2 for x, y, z boost direction.

    Returns:
        Bivector [16] with unit coefficient in the boost plane.
    """
    bv = torch.zeros(algebra.dim, device=algebra.device)
    idx = (1 << axis) | (1 << 3)   # e_{axis+1, 4}
    bv[idx] = 1.0
    return bv


def _rotation_bivector(algebra, plane: int) -> torch.Tensor:
    """Create a unit rotation bivector in Cl(3,1).

    Rotation planes: e12 (xy), e13 (xz), e23 (yz).

    Args:
        algebra: CliffordAlgebra(3, 1).
        plane: 0 (e12), 1 (e13), 2 (e23).

    Returns:
        Bivector [16] with unit coefficient in the rotation plane.
    """
    bv = torch.zeros(algebra.dim, device=algebra.device)
    planes = [(0, 1), (0, 2), (1, 2)]
    a, b = planes[plane]
    bv[(1 << a) | (1 << b)] = 1.0
    return bv


# =====================================================================
# Dataset
# =====================================================================

class LorentzDataset(Dataset):
    """Dataset of spacetime event pairs related by known Lorentz transformations.

    Each sample: (event_pair [2, 16], true_rotor [16], rapidity).
    The event_pair contains [original_event, boosted_event].

    All events are guaranteed to be strictly timelike (s^2 < 0 in our
    signature convention s^2 = x^2 + y^2 + z^2 - t^2).

    Args:
        algebra: CliffordAlgebra(3, 1).
        num_samples: Number of samples.
        boost_type: 'pure_boost', 'pure_rotation', or 'combined'.
        rapidity_max: Maximum rapidity (boost parameter).
        seed: Random seed.
    """

    def __init__(self, algebra, num_samples: int,
                 boost_type: str = 'pure_boost',
                 rapidity_max: float = 1.5, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.algebra = algebra
        dim = algebra.dim  # 16 for Cl(3,1)

        events = torch.zeros(num_samples, 2, dim)
        rotors = torch.zeros(num_samples, dim)
        rapidities = torch.zeros(num_samples)

        for i in range(num_samples):
            # Random spacetime event: (x, y, z, t) with |t| > |x,y,z| for timelike
            spatial = rng.uniform(-2.0, 2.0, 3).astype(np.float32)
            spatial_norm = np.sqrt((spatial ** 2).sum())
            t = np.float32(spatial_norm + rng.uniform(0.5, 3.0))
            coords = np.concatenate([spatial, [t]])

            event = algebra.embed_vector(
                torch.tensor(coords, dtype=torch.float32, device=algebra.device)
            )

            # Random rapidity / angle
            phi = float(rng.uniform(-rapidity_max, rapidity_max))
            rapidities[i] = phi

            # Build the rotor
            if boost_type == 'pure_boost':
                axis = rng.randint(0, 3)
                bv = _boost_bivector(algebra, axis)
            elif boost_type == 'pure_rotation':
                plane = rng.randint(0, 3)
                bv = _rotation_bivector(algebra, plane)
            else:  # combined
                boost_axis = rng.randint(0, 3)
                rot_plane = rng.randint(0, 3)
                bv_boost = _boost_bivector(algebra, boost_axis)
                bv_rot = _rotation_bivector(algebra, rot_plane)
                alpha = np.float32(rng.uniform(0.3, 0.7))
                bv = alpha * bv_boost + (1 - alpha) * bv_rot
                bv_norm = bv.norm()
                if bv_norm > 1e-6:
                    bv = bv / bv_norm

            # Rotor: R = exp(-phi/2 * B)
            bivector = (-phi / 2.0) * bv
            rotor = algebra.exp(bivector.unsqueeze(0)).squeeze(0)

            # Apply transformation: x' = R x R~
            rotor_rev = algebra.reverse(rotor.unsqueeze(0)).squeeze(0)
            temp = algebra.geometric_product(
                rotor.unsqueeze(0), event.unsqueeze(0)
            ).squeeze(0)
            boosted = algebra.geometric_product(
                temp.unsqueeze(0), rotor_rev.unsqueeze(0)
            ).squeeze(0)

            events[i, 0] = event.cpu()
            events[i, 1] = boosted.cpu()
            rotors[i] = rotor.cpu()

        self.events = events
        self.rotors = rotors
        self.rapidities = rapidities

        print(f"  Generated {num_samples} Lorentz samples "
              f"(type={boost_type}, rapidity_max={rapidity_max})")

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        return self.events[idx], self.rotors[idx], self.rapidities[idx]


# =====================================================================
# Network
# =====================================================================

class LorentzNet(nn.Module):
    """GA network for learning Lorentz rotors in Cl(3,1).

    Architecture mirrors ZetaNet:
        1. Fourier feature embedding of 8 scalar coords (4 + 4)
        2. Lift to multivector hidden space
        3. Residual blocks of RotorLayer + CliffordLayerNorm + GeometricGELU
        4. BladeSelector + CliffordLinear to output
        5. Read off predicted rotor [B, 16]
    """

    def __init__(self, algebra, hidden_dim: int = 64,
                 num_layers: int = 6, num_freqs: int = 32):
        super().__init__()
        self.algebra = algebra
        self.hidden_dim = hidden_dim

        # Fourier feature embedding: 8 coords -> high-dimensional features
        # coord_norm: stabilize raw spacetime coordinates (can have large scale variance)
        self.coord_norm = nn.LayerNorm(8)
        self.register_buffer('freq_bands', torch.randn(8, num_freqs) * 2.0)
        input_dim = 8 + 2 * num_freqs  # raw + sin + cos

        # Lift to multivector hidden space
        self.input_lift = nn.Linear(input_dim, hidden_dim * algebra.dim)
        # Normalize immediately after lift to bound the initial hidden state
        self.input_norm = CliffordLayerNorm(algebra, hidden_dim)

        # Residual GA blocks (Pre-LN: norm is applied before rotor in forward)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'norm': CliffordLayerNorm(algebra, hidden_dim),
                'rotor': RotorLayer(algebra, hidden_dim),
                'act': GeometricGELU(algebra, channels=hidden_dim),
                'linear': CliffordLinear(algebra, hidden_dim, hidden_dim),
            }))

        # Output head
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)
        self.output_proj = CliffordLinear(algebra, hidden_dim, 1)

        self._intermediates: List[torch.Tensor] = []

    def forward(self, event_pairs: torch.Tensor):
        """Forward pass.

        Args:
            event_pairs: [B, 2, 16] pair of (original, boosted) events.

        Returns:
            rotor_pred: [B, 16] predicted rotor.
            intermediates: list of hidden states for diagnostics.
        """
        B = event_pairs.shape[0]

        # Extract 8 scalar coordinates from grade-1 components
        grade1_indices = [1 << i for i in range(self.algebra.n)]
        coords = []
        for ev_idx in range(2):
            for gi in grade1_indices:
                coords.append(event_pairs[:, ev_idx, gi])
        raw = torch.stack(coords, dim=-1)  # [B, 8]

        # Normalize raw spacetime coords before Fourier embedding
        # (time vs spatial components can differ by orders of magnitude)
        raw = self.coord_norm(raw)

        # Fourier features
        proj = raw @ self.freq_bands   # [B, num_freqs]
        features = torch.cat([raw, torch.sin(proj), torch.cos(proj)], dim=-1)

        # Lift to multivector space
        h = self.input_lift(features)
        h = h.reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)  # bound magnitude before first block

        # Residual blocks - Pre-LN: norm applied to residual stream BEFORE rotor.
        # This keeps the residual path clean (unnormalized) and prevents
        # float32 magnitude drift across deep layers.
        intermediates: List[torch.Tensor] = []
        self._intermediates = []
        for block in self.blocks:
            residual = h
            h = block['norm'](h)    # normalize first
            h = block['rotor'](h)
            h = block['act'](h)
            h = block['linear'](h)
            h = residual + h        # add back to unmodified residual stream
            intermediates.append(h)
            self._intermediates.append(h.detach())

        # Output - normalize the final residual stream, then select grades
        h = self.output_norm(h)
        h = self.blade_selector(h)
        out = self.output_proj(h)   # [B, 1, algebra.dim]
        out = out.squeeze(1)         # [B, algebra.dim]

        return out, intermediates


# =====================================================================
# Lorentz Debugger
# =====================================================================

class LorentzDebugger:
    """Physics diagnostics for Lorentz transformations in Cl(3,1).

    Validates learned rotors against known physical invariants.
    All check methods return dicts for structured access; use
    ``format_report()`` for a human-readable summary.
    """

    def __init__(self, algebra):
        self.algebra = algebra

    # ------------------------------------------------------------------
    # Check 1: Spacetime interval invariance
    # ------------------------------------------------------------------

    def interval_invariance(self, events: torch.Tensor,
                            rotors: torch.Tensor) -> dict:
        """Check that the spacetime interval s^2 is preserved.

        s^2(x) should equal s^2(RxR~) for all events. Evaluated on the
        ground-truth boosted events already stored in ``events[:, 1]``.

        Args:
            events: [B, 2, 16] (original, boosted) event pairs.
            rotors: [B, 16] rotors (unused here, kept for API symmetry).

        Returns:
            dict with interval_error (mean |s^2_orig - s^2_boost|).
        """
        original = events[:, 0]
        boosted = events[:, 1]

        s2_orig = signature_norm_squared(self.algebra, original)
        s2_boost = signature_norm_squared(self.algebra, boosted)

        err = (s2_orig - s2_boost).abs()
        return {
            'interval_error': err.mean().item(),
            'interval_max_error': err.max().item(),
            's2_orig_mean': s2_orig.mean().item(),
            's2_boost_mean': s2_boost.mean().item(),
        }

    # ------------------------------------------------------------------
    # Check 2: Rotor normalization
    # ------------------------------------------------------------------

    def rotor_normalization(self, rotors: torch.Tensor) -> dict:
        """Check RR~ = 1 (rotor is a unit element of Spin(3,1)).

        Args:
            rotors: [B, 16] rotors.

        Returns:
            dict with norm_error (mean |RR~ - 1|).
        """
        rr_rev = signature_trace_form(self.algebra, rotors, rotors)
        err = (rr_rev - 1.0).abs()
        return {
            'norm_error': err.mean().item(),
            'norm_max_error': err.max().item(),
            'rr_rev_mean': rr_rev.mean().item(),
        }

    # ------------------------------------------------------------------
    # Check 3: Grade confinement
    # ------------------------------------------------------------------

    def grade_confinement(self, rotors: torch.Tensor) -> dict:
        """Check that rotors live in the even subalgebra {0, 2, 4}.

        Args:
            rotors: [B, 16] rotors.

        Returns:
            dict with even_ratio, odd_energy, grade_spectrum.
        """
        spectrum = hermitian_grade_spectrum(self.algebra, rotors)
        even_energy = (spectrum[:, 0] + spectrum[:, 2] + spectrum[:, 4]).mean().item()
        odd_energy = (spectrum[:, 1] + spectrum[:, 3]).mean().item()
        total = even_energy + odd_energy + 1e-12
        return {
            'even_ratio': even_energy / total,
            'odd_energy': odd_energy,
            'grade_spectrum': {k: spectrum[:, k].mean().item() for k in range(5)},
        }

    # ------------------------------------------------------------------
    # Check 4: Causality classification
    # ------------------------------------------------------------------

    def causality_preservation(self, events: torch.Tensor,
                               rotors_pred: torch.Tensor,
                               lightlike_tol: float = 0.05) -> dict:
        """Check that the causal character of events is preserved.

        In Cl(3,1) with signature (+,+,+,-):
          - s^2 < 0: timelike
          - s^2 > 0: spacelike
          - s^2 ~ 0: lightlike (within lightlike_tol)

        Args:
            events: [B, 2, 16] event pairs.
            rotors_pred: [B, 16] predicted rotors.
            lightlike_tol: Absolute threshold for calling an event lightlike.

        Returns:
            dict with per-class preservation rates and overall summary.
        """
        original = events[:, 0]
        s2_orig = signature_norm_squared(self.algebra, original).squeeze(-1)

        R_rev = self.algebra.reverse(rotors_pred)
        temp = self.algebra.geometric_product(rotors_pred, original)
        transformed = self.algebra.geometric_product(temp, R_rev)
        s2_trans = signature_norm_squared(self.algebra, transformed).squeeze(-1)

        def _classify(s2, tol):
            lightlike = s2.abs() < tol
            timelike = (s2 < -tol)
            spacelike = (s2 > tol)
            return timelike, spacelike, lightlike

        tl_o, sl_o, ll_o = _classify(s2_orig, lightlike_tol)
        tl_t, sl_t, ll_t = _classify(s2_trans, lightlike_tol)

        def _rate(orig_mask, trans_mask):
            if orig_mask.sum() == 0:
                return float('nan')
            return (orig_mask & trans_mask).float().sum().item() / orig_mask.float().sum().item()

        return {
            'timelike_preservation': _rate(tl_o, tl_t),
            'spacelike_preservation': _rate(sl_o, sl_t),
            'lightlike_preservation': _rate(ll_o, ll_t),
            'n_timelike': tl_o.sum().item(),
            'n_spacelike': sl_o.sum().item(),
            'n_lightlike': ll_o.sum().item(),
            'overall_preservation': ((tl_o & tl_t) | (sl_o & sl_t) | (ll_o & ll_t)
                                     ).float().mean().item(),
        }

    # ------------------------------------------------------------------
    # Check 5: Invariant mass
    # ------------------------------------------------------------------

    def invariant_mass_check(self, events: torch.Tensor,
                             rotors_pred: torch.Tensor) -> dict:
        """Check that invariant mass m^2 = -s^2 is preserved.

        For massive particles (timelike events), m^2 > 0 should be
        identical before and after transformation.

        Args:
            events: [B, 2, 16] event pairs.
            rotors_pred: [B, 16] predicted rotors.

        Returns:
            dict with mass_error (mean |m^2_orig - m^2_transformed|) for
            timelike events only.
        """
        original = events[:, 0]
        s2_orig = signature_norm_squared(self.algebra, original).squeeze(-1)
        m2_orig = -s2_orig  # positive for timelike

        R_rev = self.algebra.reverse(rotors_pred)
        temp = self.algebra.geometric_product(rotors_pred, original)
        transformed = self.algebra.geometric_product(temp, R_rev)
        s2_trans = signature_norm_squared(self.algebra, transformed).squeeze(-1)
        m2_trans = -s2_trans

        # Only evaluate on timelike events (m^2 > 0)
        timelike = m2_orig > 0.05
        if timelike.sum() == 0:
            return {'mass_error': float('nan'), 'n_timelike': 0}

        err = (m2_orig[timelike] - m2_trans[timelike]).abs()
        return {
            'mass_error': err.mean().item(),
            'mass_max_error': err.max().item(),
            'n_timelike': timelike.sum().item(),
        }

    # ------------------------------------------------------------------
    # Check 6: Rapidity extraction
    # ------------------------------------------------------------------

    def rapidity_extraction(self, rotors_pred: torch.Tensor,
                            rapidities_true: torch.Tensor) -> dict:
        """Extract rapidity from learned rotors and compare to ground truth.

        For a pure boost rotor R = cosh(phi/2) + sinh(phi/2)*B,
        phi = 2 * atanh(|bivector_part| / |scalar_part|).

        Note: This formula is exact only for pure boost rotors. For
        combined or rotation rotors, the extracted value is a proxy.
        A warning is issued if atanh clamping occurs on many samples.

        Args:
            rotors_pred: [B, 16] predicted rotors.
            rapidities_true: [B] true rapidities.

        Returns:
            dict with rapidity_mae, rapidity_corr, and clamping stats.
        """
        scalar = rotors_pred[:, 0]  # grade-0 component

        # Grade-2 (bivector) components
        bv_mask = torch.zeros(self.algebra.dim, dtype=torch.bool,
                              device=rotors_pred.device)
        for i in range(self.algebra.dim):
            if bin(i).count('1') == 2:
                bv_mask[i] = True
        bv_norm = rotors_pred[:, bv_mask].norm(dim=-1)

        # phi = 2 * atanh(bv_norm / |scalar|)
        raw_ratio = bv_norm / (scalar.abs() + 1e-8)
        n_clamped = (raw_ratio >= 0.999).sum().item()
        if n_clamped > len(rotors_pred) * 0.05:
            warnings.warn(
                f"atanh clamping on {n_clamped}/{len(rotors_pred)} samples "
                f"({100*n_clamped/len(rotors_pred):.1f}%). "
                "Rapidity extraction may be unreliable; rotor may not be pure boost.",
                RuntimeWarning, stacklevel=2
            )

        ratio = raw_ratio.clamp(0.0, 0.999)
        phi_pred = 2.0 * torch.atanh(ratio)

        phi_true_abs = rapidities_true.abs()
        mae = (phi_pred - phi_true_abs).abs().mean().item()

        corr = 0.0
        if phi_pred.std() > 1e-6 and phi_true_abs.std() > 1e-6:
            corr = torch.corrcoef(
                torch.stack([phi_pred, phi_true_abs])
            )[0, 1].item()

        return {
            'rapidity_mae': mae,
            'rapidity_corr': corr,
            'n_clamped': n_clamped,
            'phi_pred': phi_pred.detach(),
            'phi_true': phi_true_abs,
        }

    # ------------------------------------------------------------------
    # Check 7: Velocity addition
    # ------------------------------------------------------------------

    def velocity_addition(self, algebra, n_tests: int = 50,
                          rapidity_max: float = 1.5) -> dict:
        """Verify rapidity additivity: R(phi1) R(phi2) = R(phi1 + phi2).

        This is a pure algebra test that does not depend on any learning.
        It validates the group structure of Spin(3,1) as implemented by
        CliffordAlgebra.exp() and geometric_product().

        Args:
            algebra: CliffordAlgebra.
            n_tests: Number of random tests.
            rapidity_max: Max rapidity for tests.

        Returns:
            dict with addition_error (mean) and addition_max_error.
        """
        rng = np.random.RandomState(123)
        errors = []

        for _ in range(n_tests):
            phi1 = rng.uniform(-rapidity_max, rapidity_max)
            phi2 = rng.uniform(-rapidity_max, rapidity_max)
            axis = rng.randint(0, 3)
            bv = _boost_bivector(algebra, axis)

            R1 = algebra.exp((-phi1 / 2.0 * bv).unsqueeze(0)).squeeze(0)
            R2 = algebra.exp((-phi2 / 2.0 * bv).unsqueeze(0)).squeeze(0)
            R12 = algebra.exp((-(phi1 + phi2) / 2.0 * bv).unsqueeze(0)).squeeze(0)

            R_composed = algebra.geometric_product(
                R1.unsqueeze(0), R2.unsqueeze(0)
            ).squeeze(0)

            # Compare R_composed with R12 (up to sign: R and -R represent same rotation)
            d1 = (R_composed - R12).norm().item()
            d2 = (R_composed + R12).norm().item()
            errors.append(min(d1, d2))

        return {
            'addition_error': float(np.mean(errors)),
            'addition_max_error': float(np.max(errors)),
        }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def format_report(self, events: torch.Tensor,
                      rotors_true: torch.Tensor,
                      rotors_pred: torch.Tensor,
                      rapidities: torch.Tensor) -> str:
        """Generate a human-readable diagnostic report.

        Args:
            events: [B, 2, 16] event pairs.
            rotors_true: [B, 16] ground truth rotors.
            rotors_pred: [B, 16] predicted rotors.
            rapidities: [B] true rapidities.

        Returns:
            Formatted multi-line string.
        """
        lines = ["  --- Lorentz Debugger Report ---"]

        # 1. Interval invariance (on true transformations)
        inv = self.interval_invariance(events, rotors_true)
        lines.append(f"  Interval invariance error:  "
                     f"{inv['interval_error']:.6e}  "
                     f"(max {inv['interval_max_error']:.6e})")

        # 2. Rotor normalization
        norm_true = self.rotor_normalization(rotors_true)
        norm_pred = self.rotor_normalization(rotors_pred)
        lines.append(f"  Rotor norm error (true):    "
                     f"{norm_true['norm_error']:.6e}  "
                     f"(max {norm_true['norm_max_error']:.6e})")
        lines.append(f"  Rotor norm error (pred):    "
                     f"{norm_pred['norm_error']:.6e}  "
                     f"(max {norm_pred['norm_max_error']:.6e})")

        # 3. Grade confinement
        conf = self.grade_confinement(rotors_pred)
        lines.append(f"  Even subalgebra ratio:      {conf['even_ratio']:.4%}")
        spec = conf['grade_spectrum']
        lines.append(f"  Grade spectrum: "
                     f"{{{', '.join(f'G{k}:{v:.4f}' for k, v in spec.items())}}}")

        # 4. Causality classification
        caus = self.causality_preservation(events, rotors_pred)
        lines.append(f"  Causality preservation:")
        lines.append(f"    Timelike  ({int(caus['n_timelike']):4d}): "
                     f"{caus['timelike_preservation']:.4%}")
        lines.append(f"    Spacelike ({int(caus['n_spacelike']):4d}): "
                     f"{caus['spacelike_preservation']:.4%}")
        lines.append(f"    Lightlike ({int(caus['n_lightlike']):4d}): "
                     f"{caus['lightlike_preservation']:.4%}")
        lines.append(f"    Overall:  {caus['overall_preservation']:.4%}")

        # 5. Invariant mass
        mass = self.invariant_mass_check(events, rotors_pred)
        if math.isnan(mass['mass_error']):
            lines.append(f"  Invariant mass check:       N/A (no timelike events)")
        else:
            lines.append(f"  Invariant mass m^2 error:   "
                         f"{mass['mass_error']:.6e}  "
                         f"(max {mass['mass_max_error']:.6e}, "
                         f"n={int(mass['n_timelike'])})")

        # 6. Rapidity extraction
        rap = self.rapidity_extraction(rotors_pred, rapidities)
        clamp_note = (f", {rap['n_clamped']} clamped"
                      if rap['n_clamped'] > 0 else "")
        lines.append(f"  Rapidity MAE:               "
                     f"{rap['rapidity_mae']:.6f}{clamp_note}")
        lines.append(f"  Rapidity correlation:       {rap['rapidity_corr']:.4f}")

        return '\n'.join(lines)


# =====================================================================
# Visualization
# =====================================================================

def _save_plots(history: dict, debugger: LorentzDebugger,
                events: torch.Tensor, rotors_true: torch.Tensor,
                rotors_pred: torch.Tensor, rapidities: torch.Tensor,
                algebra, ortho: StrictOrthogonality,
                last_intermediates: List[torch.Tensor],
                output_dir: str) -> None:
    """Save diagnostic plots to disk.

    Plots:
        1. training_curves.png   - Multi-metric training history.
        2. minkowski_diagram.png - t vs x scatter with light cones.
        3. grade_spectrum.png    - Bar chart of grade energies (even=green, odd=red).
        4. rapidity_comparison.png - True vs predicted |rapidity| scatter.
        5. interval_histogram.png  - Histogram of per-sample interval errors.
        6. coupling_heatmap.png    - Cross-grade coupling matrix heatmap.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Training curves (multi-metric)
    # ------------------------------------------------------------------ #
    n_panels = sum(bool(history.get(k)) for k in
                   ['losses', 'interval_errors', 'norm_errors', 'even_ratios'])
    if n_panels > 0:
        fig, axes = plt.subplots(1, max(1, n_panels), figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]
        idx = 0
        epochs = history.get('epochs', [])

        if history.get('losses'):
            ax = axes[idx]; idx += 1
            ax.semilogy(epochs, history['losses'], color='steelblue', label='Test loss')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title('Test Loss'); ax.grid(True, alpha=0.3); ax.legend()

        if history.get('interval_errors'):
            ax = axes[idx]; idx += 1
            ax.semilogy(epochs, history['interval_errors'], color='tomato',
                        label='Interval error')
            ax.set_xlabel('Epoch'); ax.set_ylabel('|s^2 error|')
            ax.set_title('Interval Invariance'); ax.grid(True, alpha=0.3); ax.legend()

        if history.get('norm_errors'):
            ax = axes[idx]; idx += 1
            ax.semilogy(epochs, history['norm_errors'], color='goldenrod',
                        label='Norm error')
            ax.set_xlabel('Epoch'); ax.set_ylabel('|RR~-1|')
            ax.set_title('Rotor Normalization'); ax.grid(True, alpha=0.3); ax.legend()

        if history.get('even_ratios'):
            ax = axes[idx]; idx += 1
            ax.plot(epochs, history['even_ratios'], color='purple',
                    label='Even ratio')
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Even ratio')
            ax.set_title('Grade Confinement'); ax.grid(True, alpha=0.3); ax.legend()

        fig.suptitle('Lorentz Debugger - Training History', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Minkowski diagram: t vs x with light cones
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 8))
    orig = events[:, 0]
    boosted = events[:, 1]
    # e1 is index 1, e4 (time) is index 8
    x_orig = orig[:, 1].numpy()
    t_orig = orig[:, 8].numpy()
    x_boost = boosted[:, 1].numpy()
    t_boost = boosted[:, 8].numpy()

    lc = np.linspace(-5, 5, 100)
    ax.plot(lc, lc, 'k--', alpha=0.3, label='Light cone')
    ax.plot(lc, -lc, 'k--', alpha=0.3)

    ax.scatter(x_orig, t_orig, c='royalblue', alpha=0.4, s=15, label='Original')
    ax.scatter(x_boost, t_boost, c='tomato', alpha=0.4, s=15, label='Boosted (true)')

    R_rev = algebra.reverse(rotors_pred)
    temp = algebra.geometric_product(rotors_pred, orig)
    pred_boosted = algebra.geometric_product(temp, R_rev)
    x_pred = pred_boosted[:, 1].detach().numpy()
    t_pred = pred_boosted[:, 8].detach().numpy()
    ax.scatter(x_pred, t_pred, c='limegreen', alpha=0.4, s=15, marker='x',
               label='Boosted (predicted)')

    ax.set_xlabel('x'); ax.set_ylabel('t')
    ax.set_title('Minkowski Diagram (x component)')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'minkowski_diagram.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. Grade energy spectrum (bar chart)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, rots, label in [(axes[0], rotors_true, 'True rotors'),
                             (axes[1], rotors_pred, 'Predicted rotors')]:
        spec = hermitian_grade_spectrum(algebra, rots).mean(dim=0).detach().numpy()
        grades = list(range(len(spec)))
        colors = ['mediumseagreen' if g % 2 == 0 else 'tomato' for g in grades]
        bars = ax.bar(grades, spec, color=colors, alpha=0.8, edgecolor='white')
        # Annotate bars
        for bar, val in zip(bars, spec):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1e-5,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Grade'); ax.set_ylabel('Mean Energy')
        ax.set_title(f'Grade Spectrum - {label}\n(green=even, red=odd)')
        ax.set_xticks(grades)
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'grade_spectrum.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. Rapidity comparison scatter + histogram
    # ------------------------------------------------------------------ #
    rap = debugger.rapidity_extraction(rotors_pred, rapidities)
    phi_true = rap['phi_true'].numpy()
    phi_pred = rap['phi_pred'].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(phi_true, phi_pred, alpha=0.4, s=15, color='steelblue')
    lim = max(phi_true.max(), phi_pred.max(), 0.1) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='y = x')
    ax.set_xlabel('True |rapidity|'); ax.set_ylabel('Predicted |rapidity|')
    ax.set_title(f'Rapidity Extraction\n'
                 f'MAE={rap["rapidity_mae"]:.4f}, corr={rap["rapidity_corr"]:.3f}')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    err_rap = (phi_pred - phi_true)
    ax.hist(err_rap, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero error')
    ax.axvline(err_rap.mean(), color='orange', linestyle=':',
               label=f'Mean = {err_rap.mean():.4f}')
    ax.set_xlabel('Rapidity error (pred - true)'); ax.set_ylabel('Count')
    ax.set_title('Rapidity Error Distribution'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'rapidity_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5. Interval error histogram
    # ------------------------------------------------------------------ #
    orig_ev = events[:, 0]
    boosted_ev = events[:, 1]
    s2_orig = signature_norm_squared(algebra, orig_ev).squeeze(-1).numpy()
    s2_boost = signature_norm_squared(algebra, boosted_ev).squeeze(-1).numpy()
    interval_errs = np.abs(s2_orig - s2_boost)

    R_rev = algebra.reverse(rotors_pred)
    temp = algebra.geometric_product(rotors_pred, orig_ev)
    pred_boosted2 = algebra.geometric_product(temp, R_rev)
    s2_pred = signature_norm_squared(algebra, pred_boosted2).squeeze(-1).detach().numpy()
    interval_errs_pred = np.abs(s2_orig - s2_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(interval_errs, bins=30, color='goldenrod', edgecolor='white', alpha=0.8,
            label=f'True rotor (mean={interval_errs.mean():.2e})')
    ax.set_xlabel('|s^2(orig) - s^2(boost)|'); ax.set_ylabel('Count')
    ax.set_title('Interval Error (True Rotor)'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(interval_errs_pred, bins=30, color='tomato', edgecolor='white', alpha=0.8,
            label=f'Pred rotor (mean={interval_errs_pred.mean():.2e})')
    ax.set_xlabel('|s^2(orig) - s^2(R_pred x R_pred~)|'); ax.set_ylabel('Count')
    ax.set_title('Interval Error (Predicted Rotor)'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'interval_histogram.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 6. Coupling heatmap (via ortho module)
    # ------------------------------------------------------------------ #
    if last_intermediates:
        h_flat = last_intermediates[-1].reshape(-1, algebra.dim)
        fig_coup = ortho.visualize_coupling(
            h_flat, title="Cross-Grade Coupling (Last Hidden Layer)"
        )
        if fig_coup is not None:
            fig_coup.savefig(os.path.join(output_dir, 'coupling_heatmap.png'),
                             dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as _plt
            _plt.close(fig_coup)

    print(f"  Plots saved to {output_dir}/")


# =====================================================================
# Evaluation
# =====================================================================

@torch.no_grad()
def _evaluate(model: LorentzNet, loader: DataLoader,
              algebra, device: str):
    """Evaluate model and return predictions for diagnostics."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_events, all_rotors_true, all_rotors_pred, all_raps = [], [], [], []
    last_intermediates: List[torch.Tensor] = []

    for event_pairs, true_rotors, raps in loader:
        event_pairs = event_pairs.to(device)
        true_rotors = true_rotors.to(device)

        pred_rotors, intermediates = model(event_pairs)

        d_pos = hermitian_distance(algebra, pred_rotors, true_rotors).mean()
        d_neg = hermitian_distance(algebra, pred_rotors, -true_rotors).mean()
        loss = torch.min(d_pos, d_neg)

        total_loss += loss.item() * event_pairs.shape[0]
        n += event_pairs.shape[0]

        all_events.append(event_pairs.cpu())
        all_rotors_true.append(true_rotors.cpu())
        all_rotors_pred.append(pred_rotors.cpu())
        all_raps.append(raps)
        last_intermediates = [h.cpu() for h in intermediates]

    avg_loss = total_loss / max(n, 1)
    events = torch.cat(all_events, dim=0)
    rotors_true = torch.cat(all_rotors_true, dim=0)
    rotors_pred = torch.cat(all_rotors_pred, dim=0)
    raps = torch.cat(all_raps, dim=0)

    return avg_loss, events, rotors_true, rotors_pred, raps, last_intermediates


# =====================================================================
# Training
# =====================================================================

def train(args):
    """Main training loop with physics diagnostics and history tracking."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Algebra: Cl(3,1), dim=16
    algebra = CliffordAlgebra(p=3, q=1, device=device)
    print(f"\n{'='*60}")
    print(f" Lorentz Transformation Debugger - Cl(3,1)")
    print(f" Signature: (+,+,+,-)")
    print(f" Boost type: {args.boost_type}")
    print(f" Strict Orthogonality: {'ON' if args.strict_ortho else 'OFF'}"
          f"{f' (weight={args.ortho_weight}, mode={args.ortho_mode})' if args.strict_ortho else ''}")
    print(f"{'='*60}\n")

    # Orthogonality: even subalgebra {0, 2, 4}
    ortho_settings = OrthogonalitySettings(
        enabled=args.strict_ortho,
        mode=args.ortho_mode,
        weight=args.ortho_weight,
        target_grades=[0, 2, 4],  # even subalgebra of Cl(3,1)
        tolerance=1e-3,
        monitor_interval=args.diag_interval,
        coupling_warn_threshold=0.3,
    )
    ortho = StrictOrthogonality(algebra, ortho_settings).to(device)

    # Debugger
    debugger = LorentzDebugger(algebra)

    # Velocity addition check (pure algebra, no learning)
    print("Verifying rapidity additivity (algebra sanity check)...")
    vel_add = debugger.velocity_addition(algebra, n_tests=100,
                                          rapidity_max=args.rapidity_max)
    print(f"  Rapidity additivity error: {vel_add['addition_error']:.6e} "
          f"(max: {vel_add['addition_max_error']:.6e})")
    if vel_add['addition_max_error'] > 1e-4:
        warnings.warn(f"Rapidity additivity max error {vel_add['addition_max_error']:.2e} "
                      "is large; check algebra.exp() implementation.",
                      RuntimeWarning)

    # Dataset
    print("\nGenerating datasets...")
    train_ds = LorentzDataset(
        algebra, num_samples=args.num_train,
        boost_type=args.boost_type, rapidity_max=args.rapidity_max,
        seed=args.seed,
    )
    test_ds = LorentzDataset(
        algebra, num_samples=args.num_test,
        boost_type=args.boost_type, rapidity_max=args.rapidity_max,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Network
    model = LorentzNet(
        algebra,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_freqs=args.num_freqs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nLorentzNet: {args.hidden_dim} hidden, {args.num_layers} layers, "
          f"{n_params:,} parameters\n")

    # Optimizer
    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history for plots
    history: Dict[str, list] = {
        'epochs': [], 'losses': [], 'interval_errors': [],
        'norm_errors': [], 'even_ratios': [],
    }

    best_test_loss = float('inf')
    last_intermediates: List[torch.Tensor] = []
    intermediates: List[torch.Tensor] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rotor = 0.0
        epoch_ortho = 0.0
        n_batches = 0

        for event_pairs, true_rotors, raps in train_loader:
            event_pairs = event_pairs.to(device)
            true_rotors = true_rotors.to(device)

            pred_rotors, intermediates = model(event_pairs)

            # Rotor loss with sign ambiguity: min(d(R, R_true), d(R, -R_true))
            d_pos = hermitian_distance(algebra, pred_rotors, true_rotors).mean()
            d_neg = hermitian_distance(algebra, pred_rotors, -true_rotors).mean()
            rotor_loss = torch.min(d_pos, d_neg)

            # Normalization loss: encourage RR~ = 1
            rr_rev = signature_trace_form(algebra, pred_rotors, pred_rotors)
            norm_loss = ((rr_rev - 1.0) ** 2).mean()

            # Orthogonality loss (with annealing)
            ortho_loss = torch.tensor(0.0, device=device)
            if args.strict_ortho and intermediates:
                eff_weight = ortho.anneal_weight(epoch,
                                                  warmup_epochs=args.ortho_warmup,
                                                  total_epochs=args.epochs)
                for h in intermediates:
                    h_flat = h.reshape(-1, algebra.dim)
                    ortho_loss = ortho_loss + eff_weight * ortho.parasitic_energy(h_flat)
                ortho_loss = ortho_loss / len(intermediates)

            loss = rotor_loss + args.norm_weight * norm_loss + ortho_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rotor += rotor_loss.item()
            epoch_ortho += ortho_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_rotor = epoch_rotor / n_batches
        avg_ortho = epoch_ortho / n_batches

        # Evaluate periodically
        if epoch % args.diag_interval == 0 or epoch == 1 or epoch == args.epochs:
            (test_loss, test_events, test_rotors_true,
             test_rotors_pred, test_raps, last_intermediates) = \
                _evaluate(model, test_loader, algebra, device)

            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} (rotor: {avg_rotor:.6f}, ortho: {avg_ortho:.6f}) | "
                  f"Test: {test_loss:.6f} | LR: {lr:.6f}")

            # Physics diagnostics
            with torch.no_grad():
                report = debugger.format_report(
                    test_events, test_rotors_true, test_rotors_pred, test_raps
                )
                print(report)

                inv = debugger.interval_invariance(test_events, test_rotors_true)
                norm_r = debugger.rotor_normalization(test_rotors_pred)
                conf = debugger.grade_confinement(test_rotors_pred)

                history['epochs'].append(epoch)
                history['losses'].append(test_loss)
                history['interval_errors'].append(inv['interval_error'])
                history['norm_errors'].append(norm_r['norm_error'])
                history['even_ratios'].append(conf['even_ratio'])

            # Orthogonality diagnostics
            if args.strict_ortho and intermediates:
                # Detach for diagnostics - no grad needed here
                last_h = intermediates[-1].detach().reshape(-1, algebra.dim)
                print(ortho.format_diagnostics(last_h))

            if test_loss < best_test_loss:
                best_test_loss = test_loss

    # ------------------------------------------------------------------ #
    # Final evaluation
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f" Final Evaluation")
    print(f"{'='*60}")

    (test_loss, test_events, test_rotors_true,
     test_rotors_pred, test_raps, last_intermediates) = \
        _evaluate(model, test_loader, algebra, device)

    print(f"  Test loss:          {test_loss:.6f}")
    print(f"  Best test loss:     {best_test_loss:.6f}")

    with torch.no_grad():
        print(debugger.format_report(
            test_events, test_rotors_true, test_rotors_pred, test_raps
        ))

    # Save plots if requested
    if args.save_plots:
        print("\nGenerating plots...")
        with torch.no_grad():
            _save_plots(
                history=history,
                debugger=debugger,
                events=test_events,
                rotors_true=test_rotors_true,
                rotors_pred=test_rotors_pred,
                rapidities=test_raps,
                algebra=algebra,
                ortho=ortho,
                last_intermediates=last_intermediates,
                output_dir=args.output_dir,
            )

    print(f"\n{'='*60}\n")
    return model, best_test_loss


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Lorentz Transformation Debugger in Cl(3,1)')

    # Data
    p.add_argument('--boost-type', choices=['pure_boost', 'pure_rotation', 'combined'],
                   default='pure_boost', help='Type of Lorentz transformation')
    p.add_argument('--rapidity-max', type=float, default=1.5,
                   help='Maximum rapidity / rotation angle')
    p.add_argument('--num-train', type=int, default=3000)
    p.add_argument('--num-test', type=int, default=500)

    # Model
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--num-freqs', type=int, default=32)

    # Training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--norm-weight', type=float, default=0.1,
                   help='Weight for rotor normalization loss')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')

    # Orthogonality
    p.add_argument('--strict-ortho', action='store_true',
                   help='Enable strict orthogonality enforcement')
    p.add_argument('--ortho-weight', type=float, default=0.1,
                   help='Orthogonality penalty weight')
    p.add_argument('--ortho-mode', choices=['loss', 'project'], default='loss',
                   help='Enforcement mode: soft loss or hard projection')
    p.add_argument('--ortho-warmup', type=int, default=20,
                   help='Epochs to linearly ramp orthogonality weight from 0')
    p.add_argument('--diag-interval', type=int, default=20,
                   help='Epochs between diagnostic reports')

    # Output / visualization
    p.add_argument('--save-plots', action='store_true',
                   help='Save diagnostic plots to --output-dir')
    p.add_argument('--output-dir', type=str, default='lorentz_plots',
                   help='Directory for saving plots')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
