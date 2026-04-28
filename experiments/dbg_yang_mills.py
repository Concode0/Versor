# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""
==============================================================================
VERSOR EXPERIMENT: MATHEMATICAL DEBUGGER
==============================================================================

This script is designed to validate topological and algebraic phenomena rather
than to achieve State-of-the-Art (SOTA) on traditional benchmarks. Its focus
is to confirm that the Clifford Algebra framework computes known identities
and physical laws correctly, and to surface regressions when they do not.

Please kindly note that as an experimental module, formal mathematical proofs
and exhaustive literature reviews may still be in progress. Contributions that
tighten the validation suite — additional check_* methods, sharper tolerances,
cross-references to the literature — are warmly welcomed.

==============================================================================

Yang-Mills SU(2) Instanton in CGA Cl(4,1) — Supervised BPST Reconstruction.

Hypothesis
  A CGA-embedded GBN should recover the BPST instanton gauge potential
  through backpropagation without breaking the geometric inductive bias.
  ``SU(2) ~= Spin(3)`` lives in the spatial bivectors ``{e12, e13, e23}``,
  and CGA rotors encode translations and dilations so the
  ``1 / (|x|^2 + rho^2)`` envelope is implicit. A single
  ``MSE(A_pred, A_exact)`` on the analytic BPST field drives training; self-
  duality, action density, topological charge, Yang-Mills residuals, gauge
  covariance, and grade-2 purity are all evaluated post-training.

Execute Command
  uv run python -m experiments.dbg_yang_mills --epochs 200
  uv run python -m experiments.dbg_yang_mills --epochs 50 --rho 1.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.algebra import CliffordAlgebra
from core.metric import hermitian_grade_spectrum, hermitian_inner_product
from experiments._lib import (
    build_visualization_metadata,
    count_parameters,
    ensure_output_dir,
    make_experiment_parser,
    print_banner,
    report_diagnostics,
    run_supervised_loop,
    save_training_curve,
    set_seed,
    setup_algebra,
    signature_metadata,
)
from functional.activation import GeometricGELU
from layers import BladeSelector, CliffordLayerNorm, CliffordLinear, RotorLayer
from layers.adapters.conformal import ConformalEmbedding
from layers.primitives.base import CliffordModule
from optimizers.riemannian import RiemannianAdam

# ---------------------------------------------------------------------------
# su(2) bivector layout — identical in Cl(3,0) and CGA Cl(4,1).
# ---------------------------------------------------------------------------

_BV_INDICES = [3, 5, 6]  # e₁₂, e₁₃, e₂₃
_BV_MAP_INDICES = torch.tensor([6, 5, 3])
_BV_MAP_SIGNS = torch.tensor([1.0, -1.0, 1.0])


class SU2BladeSelector(CliffordModule):
    """Restrict [B, C, D] multivectors to the su(2) bivector slots.

    Pass-through gates on {e₁₂, e₁₃, e₂₃}; zeros every other component.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        super().__init__(algebra)
        self.channels = channels
        self.su2_gates = nn.Parameter(torch.ones(channels, 3))
        mask = torch.zeros(algebra.dim)
        for idx in _BV_INDICES:
            mask[idx] = 1.0
        self.register_buffer("su2_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.su2_gates)
        full = torch.zeros(
            self.channels,
            self.algebra.dim,
            device=x.device,
            dtype=x.dtype,
        )
        for i, bv_idx in enumerate(_BV_INDICES):
            full[:, bv_idx] = gates[:, i]
        return x * full.unsqueeze(0)


# ---------------------------------------------------------------------------
# 't Hooft symbol + analytic BPST fields.
# ---------------------------------------------------------------------------


def _build_thooft_eta() -> torch.Tensor:
    eta = torch.zeros(3, 4, 4)
    eta[0, 1, 2] = 1.0
    eta[0, 2, 1] = -1.0
    eta[1, 2, 0] = 1.0
    eta[1, 0, 2] = -1.0
    eta[2, 0, 1] = 1.0
    eta[2, 1, 0] = -1.0
    for a in range(3):
        eta[a, a, 3] = 1.0
        eta[a, 3, a] = -1.0
    return eta


_ETA_THOOFT = _build_thooft_eta()


def bpst_gauge_potential(
    x: torch.Tensor,
    rho: float,
    algebra_dim: int,
) -> torch.Tensor:
    """A_μ^a(x) = η^a_μν x_ν / (|x|² + ρ²), embedded as bivectors.

    Returns [B, 4, algebra_dim] — 4 spacetime directions, each a su(2)
    bivector (grade-2) in the algebra.
    """
    B = x.shape[0]
    r2 = (x**2).sum(dim=-1, keepdim=True)
    denom = r2 + rho**2

    eta = _ETA_THOOFT.to(x.device, x.dtype)
    bv_idx = _BV_MAP_INDICES.to(x.device)
    bv_signs = _BV_MAP_SIGNS.to(x.device, x.dtype)

    coeff = torch.einsum("amn, bn -> bam", eta, x) / denom.unsqueeze(-1)

    A = torch.zeros(B, 4, algebra_dim, dtype=x.dtype, device=x.device)
    for a in range(3):
        mask = torch.zeros(algebra_dim, device=x.device, dtype=x.dtype)
        mask[bv_idx[a]] = 1.0
        A = A + bv_signs[a] * coeff[:, a, :].unsqueeze(-1) * mask
    return A


def bpst_field_strength(
    x: torch.Tensor,
    rho: float,
    algebra_dim: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """F_μν^a(x) = -2ρ² η^a_μν / (|x|² + ρ²)² — exact BPST field strength."""
    B = x.shape[0]
    r2 = (x**2).sum(dim=-1, keepdim=True)
    prefactor = -2.0 * rho**2 / (r2 + rho**2) ** 2

    eta = _ETA_THOOFT.to(x.device, x.dtype)
    bv_idx = _BV_MAP_INDICES.to(x.device)
    bv_signs = _BV_MAP_SIGNS.to(x.device, x.dtype)

    F_dict: Dict[Tuple[int, int], torch.Tensor] = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F = torch.zeros(B, algebra_dim, dtype=x.dtype, device=x.device)
            for a in range(3):
                e = eta[a, mu, nu].item()
                if e != 0.0:
                    F[:, bv_idx[a]] = bv_signs[a] * e * prefactor.squeeze(-1)
            F_dict[(mu, nu)] = F
    return F_dict


def bpst_action_density(x: torch.Tensor, rho: float) -> torch.Tensor:
    r2 = (x**2).sum(dim=-1)
    return 48.0 * rho**4 / (r2 + rho**2) ** 4


# ---------------------------------------------------------------------------
# Hodge dual on spacetime indices.
# ---------------------------------------------------------------------------

_HODGE_DUAL_MAP: Dict[Tuple[int, int], Tuple[Tuple[int, int], float]] = {
    (0, 1): ((2, 3), 1.0),
    (0, 2): ((1, 3), -1.0),
    (0, 3): ((1, 2), 1.0),
    (1, 2): ((0, 3), 1.0),
    (1, 3): ((0, 2), -1.0),
    (2, 3): ((0, 1), 1.0),
}


def hodge_dual_4d(
    F_dict: Dict[Tuple[int, int], torch.Tensor],
) -> Dict[Tuple[int, int], torch.Tensor]:
    return {key: sign * F_dict[src] for key, (src, sign) in _HODGE_DUAL_MAP.items() if src in F_dict}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BPSTInstantonDataset(Dataset):
    """4D spacetime samples with analytic BPST (A_μ, action density) targets."""

    def __init__(self, num_samples: int, rho: float, algebra_dim: int, sampling_radius: float = 5.0, seed: int = 42):
        rng = np.random.RandomState(seed)
        directions = rng.randn(num_samples, 4).astype(np.float32)
        directions = directions / (np.linalg.norm(directions, axis=-1, keepdims=True) + 1e-8)
        radii = rng.exponential(scale=2.0 * rho, size=(num_samples, 1)).astype(np.float32)
        radii = np.clip(radii, 0.1 * rho, sampling_radius)
        self.coords = torch.tensor(directions * radii)
        self.A_mu = bpst_gauge_potential(self.coords, rho, algebra_dim)
        self.action_density = bpst_action_density(self.coords, rho)
        self.rho = rho
        r = self.coords.norm(dim=-1)
        print(f"  BPST set: {num_samples} points, rho={rho:.2f}, r=[{r.min().item():.3f}, {r.max().item():.3f}]")

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        return self.coords[idx], self.A_mu[idx], self.action_density[idx]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class YangMillsNet(CliffordModule):
    """CGA GBN for SU(2) instantons in Cl(4,1).

    conformal_embed + Fourier → lift → residual GBN blocks → CliffordLinear(C, 4)
    → SU2BladeSelector → A_μ [B, 4, 32].
    """

    def __init__(self, algebra, hidden_dim: int = 32, num_layers: int = 4, num_freqs: int = 16):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim

        self.conformal_embed = ConformalEmbedding(algebra, euclidean_dim=3)
        self.register_buffer("freq_bands", torch.randn(4, num_freqs) * 2.0)
        input_dim = 4 + 2 * num_freqs + algebra.dim + 2

        self.input_lift = nn.Linear(input_dim, hidden_dim * algebra.dim)
        self.input_norm = CliffordLayerNorm(algebra, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": CliffordLayerNorm(algebra, hidden_dim),
                        "rotor": RotorLayer(algebra, hidden_dim),
                        "act": GeometricGELU(algebra, channels=hidden_dim),
                        "linear": CliffordLinear(algebra, hidden_dim, hidden_dim),
                    }
                )
            )

        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)
        self.output_proj = CliffordLinear(algebra, hidden_dim, 4)
        self.su2_selector = SU2BladeSelector(algebra, channels=4)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B = coords.shape[0]
        x_spatial = coords[:, :3]
        t = coords[:, 3:4]
        P = self.conformal_embed.embed(x_spatial)
        proj = coords @ self.freq_bands
        features = torch.cat(
            [
                coords,
                torch.sin(proj),
                torch.cos(proj),
                P,
                t,
                t**2,
            ],
            dim=-1,
        )

        h = self.input_lift(features)
        h = h.reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)
        for block in self.blocks:
            residual = h
            h = block["norm"](h)
            h = block["rotor"](h)
            h = block["act"](h)
            h = block["linear"](h)
            h = residual + h
        h = self.output_norm(h)
        h = self.blade_selector(h)
        A = self.output_proj(h)
        return self.su2_selector(A)


# ---------------------------------------------------------------------------
# Post-training field strength (no gradient path).
# ---------------------------------------------------------------------------


def _field_strength_nograph(
    algebra,
    A_mu: torch.Tensor,
    coords: torch.Tensor,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Build F_μν from a predicted A_μ via cached Jacobian — eval-time only.

    12 autograd calls (4 directions × 3 su(2) components); no create_graph,
    so this is not backprop-safe and must only be used in diagnostics.
    """
    B, D = A_mu.shape[0], A_mu.shape[-1]
    J = torch.zeros(B, 4, 3, 4, device=A_mu.device, dtype=A_mu.dtype)
    for nu in range(4):
        for c_idx, bv_comp in enumerate(_BV_INDICES):
            grad = torch.autograd.grad(
                A_mu[:, nu, bv_comp].sum(),
                coords,
                create_graph=False,
                retain_graph=True,
            )[0]
            J[:, nu, c_idx, :] = grad

    F_dict: Dict[Tuple[int, int], torch.Tensor] = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            abelian = torch.zeros(B, D, device=A_mu.device, dtype=A_mu.dtype)
            for c_idx, bv_comp in enumerate(_BV_INDICES):
                abelian[:, bv_comp] = J[:, nu, c_idx, mu] - J[:, mu, c_idx, nu]
            commutator = algebra.commutator(A_mu[:, mu], A_mu[:, nu])
            F_dict[(mu, nu)] = (abelian - commutator).detach()
    return F_dict


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def supervised_mse(model, loader, device) -> float:
    sse, n = 0.0, 0
    for coords, A_exact, _ in loader:
        coords = coords.to(device)
        A_exact = A_exact.to(device)
        pred = model(coords)
        sse += ((pred - A_exact) ** 2).mean().item() * coords.shape[0]
        n += coords.shape[0]
    return sse / max(n, 1)


def self_duality_and_action(algebra, model, coords, action_exact, rho):
    """Self-duality, action MSE, Sobolev on F, and topological charge density."""
    coords = coords.clone().requires_grad_(True)
    A_pred = model(coords)
    F_dict = _field_strength_nograph(algebra, A_pred, coords)

    F_dual = hodge_dual_4d(F_dict)
    sd = 0.0
    n = 0
    for key, F in F_dict.items():
        if key in F_dual:
            sd += ((F - F_dual[key]) ** 2).mean().item()
            n += 1
    sd = sd / max(n, 1)

    action_pred = torch.zeros(coords.shape[0], device=coords.device)
    for F in F_dict.values():
        action_pred = action_pred + hermitian_inner_product(
            algebra,
            F,
            F,
        ).squeeze(-1)
    action_mse = ((action_pred - action_exact) ** 2).mean().item()

    F_exact = bpst_field_strength(coords.detach(), rho, algebra.dim)
    sobolev = 0.0
    m = 0
    for key, F in F_dict.items():
        if key in F_exact:
            sobolev += ((F - F_exact[key]) ** 2).mean().item()
            m += 1
    sobolev = sobolev / max(m, 1)

    q_density = torch.zeros(coords.shape[0], device=coords.device)
    for key in F_dict:
        if key in F_dual:
            q_density = q_density + hermitian_inner_product(
                algebra,
                F_dict[key],
                F_dual[key],
            ).squeeze(-1)

    r = coords.detach().norm(dim=-1).clamp(min=0.1)
    w = r**3
    w = w / w.sum()
    q_pred = (q_density * w).sum().item()
    q_exact = (action_exact * w).sum().item()
    q_ratio = q_pred / max(abs(q_exact), 1e-12)

    return {
        "self_duality_residual": sd,
        "action_mse": action_mse,
        "sobolev_F_mse": sobolev,
        "topological_Q_ratio": q_ratio,
    }


@torch.no_grad()
def gauge_covariance_residual(algebra, F_dict) -> float:
    """|<F,F>_H − <RFR̃, RFR̃>_H| averaged over F_μν pairs."""
    device = next(iter(F_dict.values())).device
    bv = torch.zeros(1, algebra.dim, device=device)
    bvc = torch.randn(3, device=device) * 0.3
    bv[0, 3], bv[0, 5], bv[0, 6] = bvc[0], bvc[1], bvc[2]
    R = algebra.exp(-0.5 * bv)
    R_rev = algebra.reverse(R)
    total, n = 0.0, 0
    for F in F_dict.values():
        B = F.shape[0]
        tmp = algebra.geometric_product(R.expand(B, -1), F)
        F_tr = algebra.geometric_product(tmp, R_rev.expand(B, -1))
        n0 = hermitian_inner_product(algebra, F, F)
        n1 = hermitian_inner_product(algebra, F_tr, F_tr)
        total += (n0 - n1).abs().mean().item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def grade_purity_A(algebra, A_mu) -> float:
    g2, tot = 0.0, 0.0
    for mu in range(A_mu.shape[1]):
        A = A_mu[:, mu]
        g2 += (algebra.grade_projection(A, 2) ** 2).sum().item()
        tot += (A**2).sum().item()
    return g2 / max(tot, 1e-12)


@torch.no_grad()
def F_grade_spectrum(algebra, F_dict) -> Dict[str, float]:
    spec = None
    n = 0
    for F in F_dict.values():
        s = hermitian_grade_spectrum(algebra, F).mean(dim=0)
        spec = s if spec is None else spec + s
        n += 1
    spec = spec / max(n, 1)
    return {f"F_g{k}": spec[k].item() for k in range(spec.shape[0])}


def post_training_diagnostics(model, algebra, test_ds, device, rho):
    """Gather every ex-loss-term as a number, single pass over diag_coords."""
    diag = {}
    diag_coords = test_ds.coords[: min(256, len(test_ds))].to(device)
    action_exact = test_ds.action_density[: min(256, len(test_ds))].to(device)

    diag.update(
        self_duality_and_action(
            algebra,
            model,
            diag_coords,
            action_exact,
            rho,
        )
    )

    # Gather F_dict once more for no-grad diagnostics.
    coords_leaf = diag_coords.clone().requires_grad_(True)
    A_pred = model(coords_leaf)
    F_dict = _field_strength_nograph(algebra, A_pred, coords_leaf)

    diag["gauge_covariance"] = gauge_covariance_residual(algebra, F_dict)
    diag["A_grade2_purity"] = grade_purity_A(algebra, A_pred.detach())
    diag.update(F_grade_spectrum(algebra, F_dict))
    return diag


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = make_experiment_parser(
        "Yang-Mills SU(2) instanton in CGA Cl(4,1) — supervised BPST.",
        defaults={"output_dir": "ym_plots"},
    )
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--num-train", type=int, default=2048)
    p.add_argument("--num-test", type=int, default=512)
    p.add_argument("--sampling-radius", type=float, default=5.0)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-freqs", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    algebra = setup_algebra(p=4, q=1, device=device)

    train_ds = BPSTInstantonDataset(
        args.num_train,
        rho=args.rho,
        algebra_dim=algebra.dim,
        sampling_radius=args.sampling_radius,
        seed=args.seed,
    )
    test_ds = BPSTInstantonDataset(
        args.num_test,
        rho=args.rho,
        algebra_dim=algebra.dim,
        sampling_radius=args.sampling_radius,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = YangMillsNet(
        algebra,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_freqs=args.num_freqs,
    ).to(device)

    print_banner(
        "Yang-Mills — CGA Cl(4,1) supervised BPST reconstruction",
        signature=f"Cl(4, 1) — su(2) bivectors at indices {tuple(_BV_INDICES)}",
        natural_loss="MSE(A_pred, A_exact) on 4-component gauge potential",
        rho=args.rho,
        parameters=f"{count_parameters(model):,}",
    )

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    def loss_fn(_model, batch):
        coords, A_exact, _action = (b.to(device) for b in batch)
        return nn.functional.mse_loss(_model(coords), A_exact)

    def diag_fn(_model, _epoch) -> Dict[str, float]:
        return {"test_A_mse": supervised_mse(_model, test_loader, device)}

    history = run_supervised_loop(
        model,
        optimizer,
        loss_fn,
        train_loader,
        epochs=args.epochs,
        diag_interval=args.diag_interval,
        grad_clip=1.0,
        diag_fn=diag_fn,
        history_extra_keys=("test_A_mse",),
    )

    diagnostics = {
        "test_A_mse": supervised_mse(model, test_loader, device),
    }
    diagnostics.update(
        post_training_diagnostics(
            model,
            algebra,
            test_ds,
            device,
            rho=args.rho,
        )
    )
    print(
        report_diagnostics(
            diagnostics,
            title="Yang-Mills post-training physics diagnostics",
        )
    )

    ensure_output_dir(args.output_dir)
    metadata = build_visualization_metadata(
        signature_metadata(4, 1),
        rho=args.rho,
        seed=args.seed,
    )
    path = save_training_curve(
        history,
        output_dir=args.output_dir,
        experiment_name="dbg_yang_mills",
        metadata=metadata,
        plot_name="training_curve",
        args=args,
        module=__name__,
        title="Yang-Mills — supervised BPST gauge potential loss",
    )
    print(f"  curve saved to {path}")


if __name__ == "__main__":
    main()
