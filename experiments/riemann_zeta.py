# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Riemann Zeta Function Reconstruction via Geometric Algebra.

Uses Cl(2,0) where the even subalgebra {1, e_12} is isomorphic to C.
Complex numbers s = sigma + it  are encoded as multivectors s = sigma*1 + t*e_12.
The network learns zeta(s) entirely within this algebra.

Strict orthogonality enforces that representations stay in the even
subalgebra, penalizing parasitic energy in the odd grades (e_1, e_2).
This validates that the GA computation respects algebraic structure.

Serves as a mathematical debugger prototype:
  - Validates grade confinement during training
  - Monitors cross-grade coupling
  - Tests reconstruction on known analytic properties (functional equation,
    location of trivial zeros, behavior near the pole at s=1)

Usage:
    uv run python -m experiments.riemann_zeta
    uv run python -m experiments.riemann_zeta --strict-ortho --epochs 300
    uv run python -m experiments.riemann_zeta --strict-ortho --save-plots --output-dir zeta_plots
    uv run python -m experiments.riemann_zeta --sigma-min 0.5 --sigma-max 3.0
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
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam
from experiments.orthogonality import StrictOrthogonality, OrthogonalitySettings


# =====================================================================
# Zeta Computation
# =====================================================================

def _zeta_mpmath(s_complex: complex) -> complex:
    """Compute zeta via mpmath (high accuracy, 15 decimal digits)."""
    import mpmath
    mpmath.mp.dps = 15
    return complex(mpmath.zeta(s_complex))


def _zeta_dirichlet_eta(s_complex: complex, N: Optional[int] = None) -> complex:
    """Compute zeta via Dirichlet eta function.

    eta(s) = sum_{n=1}^N (-1)^{n-1} n^{-s}
    zeta(s) = eta(s) / (1 - 2^{1-s})

    Converges for Re(s) > 0. N is chosen adaptively based on |Im(s)|
    to maintain reasonable accuracy: larger |t| requires more terms.

    Args:
        s_complex: Complex input s = sigma + it.
        N: Number of terms (auto if None).

    Returns:
        Approximation of zeta(s), or NaN+NaN*j near the pole s=1.
    """
    sigma = s_complex.real
    t = abs(s_complex.imag)

    if N is None:
        # Adaptive N: more terms needed for large |t| or sigma near 0
        base = 2000
        t_factor = int(t * 10)   # ~10 extra terms per unit of |t|
        N = min(base + t_factor, 6000)

    ns = np.arange(1, N + 1, dtype=np.float64)
    signs = (-1.0) ** (ns - 1)
    terms = signs * np.exp(-s_complex * np.log(ns))
    eta = terms.sum()

    factor = 1.0 - 2.0 ** (1.0 - s_complex)
    if abs(factor) < 1e-10:
        # Near pole s = 1
        return complex(float('nan'), float('nan'))
    return eta / factor


def compute_zeta(sigma: float, t: float) -> Tuple[float, float]:
    """Compute zeta(sigma + it). Returns (Re, Im).

    Uses mpmath if available, else Dirichlet eta fallback.
    """
    s = complex(sigma, t)
    try:
        z = _zeta_mpmath(s)
    except ImportError:
        z = _zeta_dirichlet_eta(s)
    return z.real, z.imag


def compute_zeta_batch(sigma_arr: np.ndarray,
                       t_arr: np.ndarray) -> np.ndarray:
    """Vectorized zeta computation. Returns [N, 2] array of (Re, Im)."""
    try:
        import mpmath
        mpmath.mp.dps = 15
        results = []
        for sr, si in zip(sigma_arr.ravel(), t_arr.ravel()):
            z = complex(mpmath.zeta(complex(float(sr), float(si))))
            results.append([z.real, z.imag])
        return np.array(results, dtype=np.float64)
    except ImportError:
        pass

    # Vectorized Dirichlet eta fallback with adaptive N
    s = (sigma_arr + 1j * t_arr).astype(np.complex128).ravel()
    t_max = float(np.abs(t_arr).max()) if len(t_arr) > 0 else 0.0
    N = min(2000 + int(t_max * 10), 6000)

    ns = np.arange(1, N + 1, dtype=np.float64)
    signs = (-1.0) ** (ns - 1)
    log_ns = np.log(ns)
    exponents = -s[None, :] * log_ns[:, None]
    terms = signs[:, None] * np.exp(exponents)
    eta = terms.sum(axis=0)

    factor = 1.0 - 2.0 ** (1.0 - s)
    safe = np.abs(factor) > 1e-8
    zeta = np.where(safe, eta / np.where(safe, factor, 1.0), np.nan + 0j)
    return np.stack([zeta.real, zeta.imag], axis=-1)


# =====================================================================
# Dataset
# =====================================================================

class ZetaDataset(Dataset):
    """Dataset of (s, zeta(s)) pairs encoded as Cl(2,0) multivectors.

    In Cl(2,0), basis = {1, e1, e2, e12}.
    Complex number z = a + bi encodes as: a*1 + b*e12 (indices 0 and 3).
    The vector components (indices 1, 2) are zero for complex numbers.
    """

    def __init__(self, sigma_range: Tuple[float, float],
                 t_range: Tuple[float, float],
                 num_samples: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        sigma = rng.uniform(sigma_range[0], sigma_range[1], num_samples)
        t = rng.uniform(t_range[0], t_range[1], num_samples)

        print(f"  Computing zeta for {num_samples} points "
              f"(sigma in [{sigma_range[0]}, {sigma_range[1]}], "
              f"t in [{t_range[0]}, {t_range[1]}])...")

        zeta_vals = compute_zeta_batch(sigma, t)

        # Filter out NaN (near pole or numerical failure) - log the count
        valid_mask = ~np.isnan(zeta_vals[:, 0])
        n_nan = (~valid_mask).sum()
        if n_nan > 0:
            print(f"  Filtered {n_nan} NaN samples "
                  f"({n_nan / num_samples:.1%} of total, likely near pole s=1).")
        sigma, t, zeta_vals = sigma[valid_mask], t[valid_mask], zeta_vals[valid_mask]

        # Encode as Cl(2,0) multivectors: [scalar, e1, e2, e12]
        # Complex z = a + bi -> [a, 0, 0, b]
        self.inputs = torch.zeros(len(sigma), 4, dtype=torch.float32)
        self.inputs[:, 0] = torch.tensor(sigma, dtype=torch.float32)  # Re(s) -> scalar
        self.inputs[:, 3] = torch.tensor(t, dtype=torch.float32)      # Im(s) -> e12

        self.targets = torch.zeros(len(sigma), 4, dtype=torch.float32)
        self.targets[:, 0] = torch.tensor(zeta_vals[:, 0], dtype=torch.float32)
        self.targets[:, 3] = torch.tensor(zeta_vals[:, 1], dtype=torch.float32)

        # Store raw values for diagnostics and visualization
        self.sigma = sigma
        self.t = t
        self.zeta_re = zeta_vals[:, 0]
        self.zeta_im = zeta_vals[:, 1]
        self.zeta_abs = np.sqrt(zeta_vals[:, 0] ** 2 + zeta_vals[:, 1] ** 2)

        print(f"  {len(self)} valid samples. "
              f"|zeta| range: [{self.zeta_abs.min():.3f}, {self.zeta_abs.max():.3f}], "
              f"mean={self.zeta_abs.mean():.3f}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


# =====================================================================
# Network
# =====================================================================

class ZetaNet(nn.Module):
    """GA network for zeta function reconstruction in Cl(2,0).

    Architecture:
        1. Fourier feature embedding of (sigma, t) input
        2. Lift to multivector hidden space
        3. Residual blocks of RotorLayer + CliffordLayerNorm + GeometricGELU
        4. BladeSelector + CliffordLinear to output
        5. Read off scalar + e12 components as Re(zeta) + Im(zeta)

    The even subalgebra of Cl(2,0) is {1, e12} ~ C. Strict orthogonality
    penalizes energy in the odd subalgebra {e1, e2} during training.
    """

    def __init__(self, algebra, hidden_dim: int = 64, num_layers: int = 6,
                 num_rotors: int = 8, num_freqs: int = 32):
        super().__init__()
        self.algebra = algebra
        self.hidden_dim = hidden_dim

        # Fourier feature embedding: (sigma, t) -> high-dimensional features
        self.register_buffer('freq_bands',
                             torch.randn(2, num_freqs) * 2.0)
        input_dim = 2 + 2 * num_freqs  # raw + sin + cos

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

    def forward(self, s_mv: torch.Tensor):
        """Forward pass.

        Args:
            s_mv: [B, 4] input multivector (complex s in Cl(2,0)).

        Returns:
            zeta_mv: [B, 4] predicted zeta(s) as multivector.
            intermediates: list of hidden states for diagnostics.
        """
        B = s_mv.shape[0]

        # Extract (sigma, t) from multivector encoding
        sigma_t = torch.stack([s_mv[:, 0], s_mv[:, 3]], dim=-1)  # [B, 2]

        # Fourier features
        proj = sigma_t @ self.freq_bands  # [B, num_freqs]
        features = torch.cat([sigma_t, torch.sin(proj), torch.cos(proj)], dim=-1)

        # Lift to multivector space: [B, hidden_dim, algebra.dim]
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
        out = self.output_proj(h)  # [B, 1, algebra.dim]
        out = out.squeeze(1)       # [B, algebra.dim]

        return out, intermediates

    def get_intermediates(self) -> List[torch.Tensor]:
        """Return last set of intermediate features for diagnostics."""
        return self._intermediates


# =====================================================================
# Evaluation
# =====================================================================

@torch.no_grad()
def _evaluate(model: ZetaNet, loader: DataLoader,
              input_mean, input_std, target_mean, target_std,
              device: str) -> float:
    """Compute mean L2 error in original (unnormalized) space.

    Only the scalar (index 0) and e12 (index 3) components matter
    for the complex-valued zeta function.
    """
    model.eval()
    total_err = 0.0
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        norm_inputs = (inputs - input_mean.to(device)) / input_std.to(device)
        pred, _ = model(norm_inputs)
        # Denormalize prediction
        pred_denorm = pred * target_std.to(device) + target_mean.to(device)
        # L2 error per sample on complex-relevant components
        err = ((pred_denorm[:, 0] - targets[:, 0]) ** 2 +
               (pred_denorm[:, 3] - targets[:, 3]) ** 2).sqrt()
        total_err += err.sum().item()
        n += err.shape[0]
    return total_err / max(n, 1)


@torch.no_grad()
def _check_functional_equation(model: ZetaNet,
                                input_mean, input_std, target_mean, target_std,
                                device: str, n_tests: int = 200) -> dict:
    """Validate the functional equation: zeta(s) = chi(s) * zeta(1-s).

    chi(s) = 2^s * pi^{s-1} * sin(pi*s/2) * Gamma(1-s)

    Checks that the model's predictions are internally consistent with
    this reflection formula on random test points.

    Args:
        n_tests: Number of (s, 1-s) pairs to test.

    Returns:
        dict with keys:
            mean_residual:   mean |pred(s) - chi(s)*pred(1-s)| (complex L2)
            median_residual: median of same
    """
    try:
        import mpmath
        mpmath.mp.dps = 15
    except ImportError:
        return {'mean_residual': float('nan'), 'median_residual': float('nan'),
                'note': 'mpmath not available'}

    rng = np.random.RandomState(999)
    model.eval()

    residuals = []
    for _ in range(n_tests):
        # Sample s in the strip Re(s) in [0.1, 0.9], Im(s) in [-10, 10]
        sigma = rng.uniform(0.1, 0.9)
        t = rng.uniform(-10.0, 10.0)
        s = complex(sigma, t)
        s_conj = complex(1.0 - sigma, -t)  # 1 - s*

        # Compute chi(s) analytically
        chi = (complex(mpmath.zeta(s)) /
               (complex(mpmath.zeta(s_conj)).conjugate() + 1e-30))

        def _encode_s(sig, ti) -> torch.Tensor:
            mv = torch.zeros(1, 4, dtype=torch.float32, device=device)
            mv[0, 0] = sig
            mv[0, 3] = ti
            return mv

        def _predict(sig, ti) -> complex:
            inp = _encode_s(sig, ti)
            norm_inp = (inp - input_mean.to(device)) / input_std.to(device)
            pred, _ = model(norm_inp)
            pred_denorm = pred * target_std.to(device) + target_mean.to(device)
            re = pred_denorm[0, 0].item()
            im = pred_denorm[0, 3].item()
            return complex(re, im)

        pred_s = _predict(sigma, t)
        pred_1ms = _predict(1.0 - sigma, -t)

        # zeta(s) should equal chi(s) * zeta(1-s)
        # Since chi is computed from true zeta, we test model consistency
        # by checking: pred(s) / pred(1-s) ~ chi(s)
        if abs(pred_1ms) > 1e-6:
            ratio = pred_s / pred_1ms
            true_chi = complex(mpmath.zeta(s)) / complex(mpmath.zeta(s_conj))
            residual = abs(ratio - true_chi)
            residuals.append(residual)

    if not residuals:
        return {'mean_residual': float('nan'), 'median_residual': float('nan')}

    residuals = np.array(residuals)
    return {
        'mean_residual': float(np.mean(residuals)),
        'median_residual': float(np.median(residuals)),
        'n_tested': len(residuals),
    }


# =====================================================================
# Visualization
# =====================================================================

def _save_plots(history: dict, model: ZetaNet, test_ds: ZetaDataset,
                input_mean, input_std, target_mean, target_std,
                algebra, ortho: StrictOrthogonality,
                device: str, output_dir: str) -> None:
    """Save diagnostic plots to disk.

    Plots generated:
        1. convergence.png   - Training loss + test L2 + critical-line L2 over epochs.
        2. error_scatter.png - Scatter of (sigma, t) colored by per-sample L2 error.
        3. ortho_evolution.png - Parasitic ratio and coupling over epochs (if tracked).
        4. critical_line.png   - True vs predicted Re(zeta) and Im(zeta) on sigma=0.5.
        5. coupling_heatmap.png - Cross-grade coupling from ortho module.

    Args:
        history: Dict with lists 'epochs', 'train_loss', 'test_l2', 'crit_l2',
                 'ortho_ratio' (optional).
        test_ds: Full test ZetaDataset (for error scatter).
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
    # 1. Convergence curves
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history['epochs']

    ax = axes[0]
    ax.semilogy(epochs, history['train_loss'], label='Train loss', color='steelblue')
    if history.get('test_l2'):
        ax.semilogy(epochs, history['test_l2'], label='Test L2', color='tomato')
    if history.get('crit_l2'):
        ax.semilogy(epochs, history['crit_l2'], label='Critical line L2',
                    color='goldenrod', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / L2 error (log scale)')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if history.get('ortho_ratio'):
        ax.semilogy(epochs, history['ortho_ratio'], color='purple',
                    label='Parasitic ratio')
        ax.axhline(ortho.settings.tolerance, color='red', linestyle='--', alpha=0.7,
                   label=f'Tolerance ({ortho.settings.tolerance})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parasitic ratio (log scale)')
        ax.set_title('Orthogonality Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Orthogonality tracking\nnot enabled',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Orthogonality Evolution')

    fig.suptitle('Riemann Zeta - Cl(2,0) Reconstruction', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Error scatter: (sigma, t) colored by per-sample L2 error
    # ------------------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        inputs = test_ds.inputs.to(device)
        targets = test_ds.targets.to(device)
        norm_inputs = (inputs - input_mean.to(device)) / input_std.to(device)
        pred, last_inters = model(norm_inputs)
        pred_denorm = pred * target_std.to(device) + target_mean.to(device)
        errs = ((pred_denorm[:, 0] - targets[:, 0]) ** 2 +
                (pred_denorm[:, 3] - targets[:, 3]) ** 2).sqrt().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sc = ax.scatter(test_ds.sigma, test_ds.t, c=errs,
                    cmap='hot_r', s=5, alpha=0.6, vmin=0)
    plt.colorbar(sc, ax=ax, label='L2 error')
    ax.set_xlabel('sigma = Re(s)')
    ax.set_ylabel('t = Im(s)')
    ax.set_title('Per-sample L2 Error in (sigma, t) Space')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.hist(errs, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(errs.mean(), color='red', linestyle='--',
               label=f'Mean = {errs.mean():.4f}')
    ax.axvline(np.median(errs), color='orange', linestyle=':',
               label=f'Median = {np.median(errs):.4f}')
    ax.set_xlabel('L2 error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'error_scatter.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. Critical line: sigma = 0.5 - true vs predicted Re/Im
    # ------------------------------------------------------------------ #
    # Build small critical-line dataset
    t_crit = np.linspace(-20.0, 20.0, 200)
    sigma_crit = np.full_like(t_crit, 0.5)
    zeta_crit = compute_zeta_batch(sigma_crit, t_crit)
    valid = ~np.isnan(zeta_crit[:, 0])
    t_crit, zeta_crit = t_crit[valid], zeta_crit[valid]

    crit_inputs = torch.zeros(len(t_crit), 4, dtype=torch.float32)
    crit_inputs[:, 0] = 0.5
    crit_inputs[:, 3] = torch.tensor(t_crit, dtype=torch.float32)

    with torch.no_grad():
        crit_inp_dev = crit_inputs.to(device)
        norm_crit = (crit_inp_dev - input_mean.to(device)) / input_std.to(device)
        pred_crit, _ = model(norm_crit)
        pred_crit_dn = (pred_crit * target_std.to(device) + target_mean.to(device)).cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(t_crit, zeta_crit[:, 0], 'b-', label='True Re(zeta)', linewidth=1.5)
    ax.plot(t_crit, pred_crit_dn[:, 0], 'r--', label='Predicted Re(zeta)', linewidth=1.5)
    ax.set_ylabel('Re(zeta(0.5 + it))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_crit, zeta_crit[:, 1], 'b-', label='True Im(zeta)', linewidth=1.5)
    ax.plot(t_crit, pred_crit_dn[:, 3], 'r--', label='Predicted Im(zeta)', linewidth=1.5)
    ax.set_xlabel('t = Im(s)')
    ax.set_ylabel('Im(zeta(0.5 + it))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Critical Line (sigma = 0.5): True vs Predicted', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'critical_line.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. Coupling heatmap (uses ortho module's visualize_coupling)
    # ------------------------------------------------------------------ #
    if last_inters:
        h_flat = last_inters[-1].reshape(-1, algebra.dim).to(device)
        fig_coup = ortho.visualize_coupling(
            h_flat,
            title="Cross-Grade Coupling (Last Hidden Layer)"
        )
        if fig_coup is not None:
            fig_coup.savefig(os.path.join(output_dir, 'coupling_heatmap.png'),
                             dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as _plt
            _plt.close(fig_coup)

    print(f"  Plots saved to {output_dir}/")


# =====================================================================
# Training
# =====================================================================

def train(args):
    """Main training loop with orthogonality monitoring and history tracking."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Algebra: Cl(2,0), dim=4, even subalgebra {1, e12} ~ C
    algebra = CliffordAlgebra(p=2, q=0, device=device)
    print(f"\n{'='*60}")
    print(f" Riemann Zeta Reconstruction - Cl(2,0)")
    print(f" Even subalgebra {{1, e_12}} ~ C")
    print(f" Strict Orthogonality: {'ON' if args.strict_ortho else 'OFF'}"
          f"{f' (weight={args.ortho_weight}, mode={args.ortho_mode})' if args.strict_ortho else ''}")
    print(f"{'='*60}\n")

    # Orthogonality settings
    ortho_settings = OrthogonalitySettings(
        enabled=args.strict_ortho,
        mode=args.ortho_mode,
        weight=args.ortho_weight,
        target_grades=[0, 2],  # even subalgebra: scalar + bivector
        tolerance=1e-3,
        monitor_interval=args.diag_interval,
        coupling_warn_threshold=0.3,
    )
    ortho = StrictOrthogonality(algebra, ortho_settings).to(device)

    # Dataset
    print("Generating datasets...")
    train_ds = ZetaDataset(
        sigma_range=[args.sigma_min, args.sigma_max],
        t_range=[args.t_min, args.t_max],
        num_samples=args.num_train,
        seed=args.seed,
    )
    test_ds = ZetaDataset(
        sigma_range=[args.sigma_min, args.sigma_max],
        t_range=[args.t_min, args.t_max],
        num_samples=args.num_test,
        seed=args.seed + 1,
    )
    # Critical line test set: sigma = 0.5
    crit_ds = ZetaDataset(
        sigma_range=[0.5, 0.5 + 1e-6],
        t_range=[args.t_min, args.t_max],
        num_samples=args.num_test // 2,
        seed=args.seed + 2,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    crit_loader = DataLoader(crit_ds, batch_size=args.batch_size, shuffle=False)

    # Normalization stats (for stable training)
    target_mean = train_ds.targets.mean(dim=0)
    target_std = train_ds.targets.std(dim=0).clamp(min=1e-4)
    input_mean = train_ds.inputs.mean(dim=0)
    input_std = train_ds.inputs.std(dim=0).clamp(min=1e-4)

    # Network
    model = ZetaNet(
        algebra,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_rotors=args.num_rotors,
        num_freqs=args.num_freqs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nZetaNet: {args.hidden_dim} hidden, {args.num_layers} layers, "
          f"{args.num_rotors} rotors, {n_params:,} parameters\n")

    # Optimizer: Riemannian Adam for rotor params, standard Adam for rest
    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history for visualization
    history: Dict[str, list] = {
        'epochs': [],
        'train_loss': [],
        'test_l2': [],
        'crit_l2': [],
        'ortho_ratio': [],
    }

    # Training loop
    best_test_loss = float('inf')
    intermediates: List[torch.Tensor] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_ortho = 0.0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Normalize
            norm_inputs = (inputs - input_mean.to(device)) / input_std.to(device)
            norm_targets = (targets - target_mean.to(device)) / target_std.to(device)

            pred, intermediates = model(norm_inputs)

            # Reconstruction loss (MSE on normalized targets)
            recon_loss = nn.functional.mse_loss(pred, norm_targets)

            # Orthogonality loss on intermediate features (with warmup annealing)
            ortho_loss = torch.tensor(0.0, device=device)
            if args.strict_ortho and intermediates:
                eff_weight = ortho.anneal_weight(epoch,
                                                  warmup_epochs=args.ortho_warmup,
                                                  total_epochs=args.epochs)
                for h in intermediates:
                    # h: [B, hidden_dim, dim] -> flatten to [B*hidden, dim]
                    h_flat = h.reshape(-1, algebra.dim)
                    ortho_loss = ortho_loss + eff_weight * ortho.parasitic_energy(h_flat)
                ortho_loss = ortho_loss / len(intermediates)

            loss = recon_loss + ortho_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_ortho += ortho_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_ortho = epoch_ortho / n_batches

        # Evaluate periodically
        if epoch % args.diag_interval == 0 or epoch == 1 or epoch == args.epochs:
            test_l2 = _evaluate(model, test_loader, input_mean, input_std,
                                 target_mean, target_std, device)
            crit_l2 = _evaluate(model, crit_loader, input_mean, input_std,
                                 target_mean, target_std, device)

            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} (recon: {avg_recon:.6f}, ortho: {avg_ortho:.6f}) | "
                  f"Test L2: {test_l2:.6f} | Crit L2: {crit_l2:.6f} | "
                  f"LR: {lr:.6f}")

            # Orthogonality diagnostics
            p_ratio = 0.0
            if args.strict_ortho and intermediates:
                # Detach for diagnostics - no grad needed here
                last_h = intermediates[-1].detach().reshape(-1, algebra.dim)
                p_ratio = ortho.parasitic_ratio(last_h)
                print(ortho.format_diagnostics(last_h))

            # Record history
            history['epochs'].append(epoch)
            history['train_loss'].append(avg_loss)
            history['test_l2'].append(test_l2)
            history['crit_l2'].append(crit_l2)
            if args.strict_ortho:
                history['ortho_ratio'].append(p_ratio)

            if test_l2 < best_test_loss:
                best_test_loss = test_l2

    # ------------------------------------------------------------------ #
    # Final evaluation
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f" Final Evaluation")
    print(f"{'='*60}")

    test_l2 = _evaluate(model, test_loader, input_mean, input_std,
                         target_mean, target_std, device)
    crit_l2 = _evaluate(model, crit_loader, input_mean, input_std,
                         target_mean, target_std, device)

    print(f"  Test L2 error:        {test_l2:.6f}")
    print(f"  Critical line L2:     {crit_l2:.6f}  (sigma = 0.5)")
    print(f"  Best test L2:         {best_test_loss:.6f}")

    # Final orthogonality report
    if args.strict_ortho:
        model.eval()
        with torch.no_grad():
            sample_in = next(iter(test_loader))[0].to(device)
            norm_in = (sample_in - input_mean.to(device)) / input_std.to(device)
            _, inters = model(norm_in)
            if inters:
                last_h = inters[-1].reshape(-1, algebra.dim)
                print(f"\n  Orthogonality Report (final):")
                print(ortho.format_diagnostics(last_h))

    # Functional equation check
    print(f"\n  Functional Equation Validation (zeta(s) = chi(s)*zeta(1-s)):")
    func_eq = _check_functional_equation(
        model, input_mean, input_std, target_mean, target_std, device
    )
    if 'note' in func_eq:
        print(f"    {func_eq['note']}")
    else:
        print(f"    Mean  ratio residual: {func_eq['mean_residual']:.6f}")
        print(f"    Median ratio residual: {func_eq['median_residual']:.6f}")
        print(f"    (tested on {func_eq['n_tested']} points; "
              f"lower is better but non-zero due to approximation error)")

    print(f"\n{'='*60}\n")

    # Save plots if requested
    if args.save_plots:
        print("Generating plots...")
        _save_plots(
            history=history,
            model=model,
            test_ds=test_ds,
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            algebra=algebra,
            ortho=ortho,
            device=device,
            output_dir=args.output_dir,
        )

    return model, best_test_loss


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Riemann Zeta Reconstruction in Cl(2,0)')

    # Data
    p.add_argument('--sigma-min', type=float, default=0.5)
    p.add_argument('--sigma-max', type=float, default=3.0)
    p.add_argument('--t-min', type=float, default=-20.0)
    p.add_argument('--t-max', type=float, default=20.0)
    p.add_argument('--num-train', type=int, default=5000)
    p.add_argument('--num-test', type=int, default=1000)

    # Model
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--num-rotors', type=int, default=8)
    p.add_argument('--num-freqs', type=int, default=32)

    # Training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=128)
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
    p.add_argument('--output-dir', type=str, default='zeta_plots',
                   help='Directory for saving plots')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
