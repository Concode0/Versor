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

Navier-Stokes in Cl(3,0) — Supervised IC+BC Reconstruction.

Hypothesis
  A GBN trained on the analytic Taylor-Green vortex should recover the fluid
  state ``Psi = p + u + omega + h`` together with the spatial Jacobian
  ``∂u_i/∂x_j`` on initial and boundary samples through backpropagation
  without breaking the geometric inductive bias. The natural loss is the
  joint MSE on the packed multivector and on the 9-component velocity
  Jacobian; the analytic gradient is supplied as a richer supervised
  target (Yang-Mills-style — predict the rich operational field, derive
  invariants post-hoc). Incompressibility, vorticity-curl consistency,
  gauge covariance, and energy or enstrophy balance are evaluated only
  after training.

Execute Command
  uv run python -m experiments.dbg_navier_stokes --epochs 200
  uv run python -m experiments.dbg_navier_stokes --epochs 50 --re 1000
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
from layers.primitives.base import CliffordModule
from optimizers.riemannian import RiemannianAdam

# ---------------------------------------------------------------------------
# Taylor-Green vortex (analytic)
# ---------------------------------------------------------------------------


def _tgv_velocity(x, y, z, t, nu, A=1.0):
    decay = torch.exp(-3.0 * nu * t)
    u1 = A * torch.sin(x) * torch.cos(y) * torch.cos(z) * decay
    u2 = -A * torch.cos(x) * torch.sin(y) * torch.cos(z) * decay
    u3 = torch.zeros_like(x)
    return u1, u2, u3


def _tgv_pressure(x, y, z, t, nu, A=1.0):
    decay = torch.exp(-6.0 * nu * t)
    return (A * A / 16.0) * (torch.cos(2 * x) + torch.cos(2 * y)) * (torch.cos(2 * z) + 2.0) * decay


def _tgv_vorticity(x, y, z, t, nu, A=1.0):
    decay = torch.exp(-3.0 * nu * t)
    w1 = -A * torch.cos(x) * torch.sin(y) * torch.sin(z) * decay
    w2 = -A * torch.sin(x) * torch.cos(y) * torch.sin(z) * decay
    w3 = 2.0 * A * torch.sin(x) * torch.sin(y) * torch.cos(z) * decay
    return w1, w2, w3


def _tgv_velocity_grad(x, y, z, t, nu, A=1.0) -> torch.Tensor:
    """Spatial Jacobian of u from the TGV closed form.

    Returns ``[B, 9]`` flattened ``∂u_i/∂x_j`` in row-major order
    ``(∂u₁/∂x, ∂u₁/∂y, ∂u₁/∂z, ∂u₂/∂x, ..., ∂u₃/∂z)``. By construction the
    diagonal sum is zero (∇·u = 0), and the off-diagonal antisymmetric
    combinations are the vorticity components.
    """
    decay = torch.exp(-3.0 * nu * t)
    du1_dx = A * torch.cos(x) * torch.cos(y) * torch.cos(z) * decay
    du1_dy = -A * torch.sin(x) * torch.sin(y) * torch.cos(z) * decay
    du1_dz = -A * torch.sin(x) * torch.cos(y) * torch.sin(z) * decay
    du2_dx = A * torch.sin(x) * torch.sin(y) * torch.cos(z) * decay
    du2_dy = -A * torch.cos(x) * torch.cos(y) * torch.cos(z) * decay
    du2_dz = A * torch.cos(x) * torch.sin(y) * torch.sin(z) * decay
    zero = torch.zeros_like(x)
    return torch.stack(
        [du1_dx, du1_dy, du1_dz, du2_dx, du2_dy, du2_dz, zero, zero, zero],
        dim=-1,
    )


def _pack_mv(p, u1, u2, u3, w1, w2, w3):
    """Pack analytic fields into Cl(3,0) multivector slots [B, 8].

    Index layout: 0=scalar p, 1=e1, 2=e2, 4=e3 (velocity); vorticity via
    Hodge dual: ω_1→+e23 (idx 6), ω_2→-e13 (idx 5), ω_3→+e12 (idx 3).
    Helicity slot (idx 7) populated as u·ω scalar density.
    """
    B = p.shape[0]
    mv = torch.zeros(B, 8, dtype=p.dtype, device=p.device)
    mv[:, 0] = p
    mv[:, 1] = u1
    mv[:, 2] = u2
    mv[:, 4] = u3
    mv[:, 3] = w3
    mv[:, 5] = -w2
    mv[:, 6] = w1
    mv[:, 7] = u1 * w1 + u2 * w2 + u3 * w3
    return mv


# ---------------------------------------------------------------------------
# Dataset — analytic IC (t=0) + BC (periodic-face) supervision
# ---------------------------------------------------------------------------


class TaylorGreenSupervisedDataset(Dataset):
    """Analytic samples on the IC slab and the six periodic faces.

    No collocation/PDE points: the natural loss is supervised reconstruction
    on samples whose target is the closed-form TGV solution.
    """

    def __init__(self, num_ic: int, num_bc: int, re: float, t_max: float = 1.0, seed: int = 42):
        rng = np.random.RandomState(seed)
        nu = 1.0 / re
        self.nu = nu
        self.re = re

        x_i = rng.uniform(0, 2 * np.pi, num_ic).astype(np.float32)
        y_i = rng.uniform(0, 2 * np.pi, num_ic).astype(np.float32)
        z_i = rng.uniform(0, 2 * np.pi, num_ic).astype(np.float32)
        t_i = np.zeros(num_ic, dtype=np.float32)

        per_face = max(num_bc // 6, 1)
        face_pts = []
        for axis in range(3):
            for face_val in (0.0, 2.0 * np.pi):
                pts = rng.uniform(0, 2 * np.pi, (per_face, 3)).astype(np.float32)
                pts[:, axis] = face_val
                t_face = rng.uniform(0, t_max, per_face).astype(np.float32)
                face_pts.append(np.column_stack([pts, t_face]))
        face_arr = np.concatenate(face_pts, axis=0)

        x_b, y_b, z_b, t_b = (face_arr[:, k] for k in range(4))

        x = torch.tensor(np.concatenate([x_i, x_b]))
        y = torch.tensor(np.concatenate([y_i, y_b]))
        z = torch.tensor(np.concatenate([z_i, z_b]))
        t = torch.tensor(np.concatenate([t_i, t_b]))

        u1, u2, u3 = _tgv_velocity(x, y, z, t, nu)
        p = _tgv_pressure(x, y, z, t, nu)
        w1, w2, w3 = _tgv_vorticity(x, y, z, t, nu)
        targets = _pack_mv(p, u1, u2, u3, w1, w2, w3)
        target_grad = _tgv_velocity_grad(x, y, z, t, nu)

        log_re = torch.full_like(x, math.log(re))
        self.coords = torch.stack([x, y, z, t, log_re], dim=-1)
        self.targets = targets
        self.target_grad = target_grad

        print(f"  TGV supervised set: {num_ic} IC + {len(t_b)} BC, Re={re:.0f}, nu={nu:.6f}, t_max={t_max}")

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        return self.coords[idx], self.targets[idx], self.target_grad[idx]


def _eval_grid(re: float, t: float, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense (x,y,z) grid at fixed time, with analytic targets — for diagnostics."""
    nu = 1.0 / re
    lin = torch.linspace(0, 2 * math.pi, n, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    x, y, z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
    tt = torch.full_like(x, t)
    log_re = torch.full_like(x, math.log(re))
    coords = torch.stack([x, y, z, tt, log_re], dim=-1)
    u1, u2, u3 = _tgv_velocity(x, y, z, tt, nu)
    p = _tgv_pressure(x, y, z, tt, nu)
    w1, w2, w3 = _tgv_vorticity(x, y, z, tt, nu)
    return coords, _pack_mv(p, u1, u2, u3, w1, w2, w3)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class GaugeFluidNet(CliffordModule):
    """GBN for Navier-Stokes in Cl(3,0).

    Fourier features → multivector lift → residual blocks
    (norm → rotor → activation → linear) → two parallel readouts:

    * a packed Cl(3,0) multivector ``[B, 8]`` carrying ``(p, u, ω, h)``;
    * a flat 9-vector ``[B, 9]`` carrying the spatial Jacobian ``∂u_i/∂x_j``.

    The gradient head is a plain ``nn.Linear`` over the gated hidden state.
    Both heads share the entire backbone, and the joint MSE loss on the
    pair is the natural Yang-Mills-style supervision: divergence-freeness
    and vorticity-curl consistency are inside the loss because their
    constituent components are inside the target.
    """

    def __init__(
        self,
        algebra,
        hidden_dim: int = 32,
        num_layers: int = 4,
        num_spatial_freqs: int = 6,
        num_temporal_freqs: int = 8,
    ):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim

        spatial_freqs = torch.arange(1, num_spatial_freqs + 1, dtype=torch.float32)
        self.register_buffer("spatial_freqs", spatial_freqs)
        self.register_buffer("temporal_freqs", torch.randn(num_temporal_freqs) * 2.0)
        self.register_buffer("re_freqs", torch.randn(num_temporal_freqs) * 0.5)

        input_dim = 5 + 3 * 2 * num_spatial_freqs + 2 * num_temporal_freqs + 2 * num_temporal_freqs

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
        self.output_proj = CliffordLinear(algebra, hidden_dim, 1)
        self.grad_head = nn.Linear(hidden_dim * algebra.dim, 9)

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        x, y, z, t, log_re = coords.unbind(-1)
        feats = [coords]
        for s in (x, y, z):
            proj = s.unsqueeze(-1) * self.spatial_freqs
            feats.append(torch.sin(proj))
            feats.append(torch.cos(proj))
        t_proj = t.unsqueeze(-1) * self.temporal_freqs
        feats.append(torch.sin(t_proj))
        feats.append(torch.cos(t_proj))
        re_proj = log_re.unsqueeze(-1) * self.re_freqs
        feats.append(torch.sin(re_proj))
        feats.append(torch.cos(re_proj))
        return torch.cat(feats, dim=-1)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = coords.shape[0]
        h = self.input_lift(self._features(coords))
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
        mv = self.output_proj(h).squeeze(1)
        grad_u = self.grad_head(h.reshape(B, -1))
        return mv, grad_u


# ---------------------------------------------------------------------------
# Post-training diagnostics — every ex-loss-term, now a measurement
# ---------------------------------------------------------------------------


@torch.no_grad()
def supervised_l2(model, loader, device) -> Dict[str, float]:
    """Joint MSE on the (packed mv, ∂u/∂x_i) test pair."""
    sse_mv, sse_grad, n = 0.0, 0.0, 0
    for coords, targets, target_grad in loader:
        coords = coords.to(device)
        targets = targets.to(device)
        target_grad = target_grad.to(device)
        pred_mv, pred_grad = model(coords)
        sse_mv += ((pred_mv - targets) ** 2).mean().item() * coords.shape[0]
        sse_grad += ((pred_grad - target_grad) ** 2).mean().item() * coords.shape[0]
        n += coords.shape[0]
    return {
        "test_l2_mv": sse_mv / max(n, 1),
        "test_l2_grad": sse_grad / max(n, 1),
    }


def _autograd_div_curl_lap(
    model: GaugeFluidNet,
    coords: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute div(u), curl(u) and ∇²u from the trained model via autograd.

    Used post-training only — not in the gradient path.
    """
    x = coords[:, 0:1].detach().requires_grad_(True)
    y = coords[:, 1:2].detach().requires_grad_(True)
    z = coords[:, 2:3].detach().requires_grad_(True)
    t = coords[:, 3:4].detach()
    log_re = coords[:, 4:5].detach()
    leaf_coords = torch.cat([x, y, z, t, log_re], dim=-1)

    mv, _ = model(leaf_coords)
    u1, u2, u3 = mv[:, 1], mv[:, 2], mv[:, 4]

    def grad(out, inp):
        return torch.autograd.grad(out.sum(), inp, create_graph=False, retain_graph=True)[0].squeeze(-1)

    du1_dx = grad(u1, x)
    du1_dy = grad(u1, y)
    du1_dz = grad(u1, z)
    du2_dx = grad(u2, x)
    du2_dy = grad(u2, y)
    du2_dz = grad(u2, z)
    du3_dx = grad(u3, x)
    du3_dy = grad(u3, y)
    du3_dz = grad(u3, z)

    div_u = du1_dx + du2_dy + du3_dz
    curl = torch.stack([du3_dy - du2_dz, du1_dz - du3_dx, du2_dx - du1_dy], dim=-1)
    return {"mv": mv.detach(), "div_u": div_u.detach(), "curl": curl.detach()}


@torch.no_grad()
def gauge_covariance_residual(model, algebra, coords, device) -> float:
    """|<u,u>_H − <RuR̃, RuR̃>_H| for a random rotor, averaged over coords."""
    coords = coords.to(device)
    mv, _ = model(coords)
    vel = algebra.grade_projection(mv, 1)
    bv = torch.zeros(1, algebra.dim, device=device)
    bvc = torch.randn(3, device=device) * 0.5
    bv[0, 3], bv[0, 5], bv[0, 6] = bvc[0], bvc[1], bvc[2]
    R = algebra.exp(-0.5 * bv)
    R_rev = algebra.reverse(R)
    transformed = algebra.geometric_product(
        algebra.geometric_product(R.expand(vel.shape[0], -1), vel),
        R_rev.expand(vel.shape[0], -1),
    )
    n0 = hermitian_inner_product(algebra, vel, vel)
    n1 = hermitian_inner_product(algebra, transformed, transformed)
    return (n0 - n1).abs().mean().item()


@torch.no_grad()
def grade_spectrum(model, algebra, coords, device) -> Dict[str, float]:
    coords = coords.to(device)
    mv, _ = model(coords)
    spec = hermitian_grade_spectrum(algebra, mv).mean(dim=0)
    labels = ["g0_pressure", "g1_velocity", "g2_vorticity", "g3_helicity"]
    return {labels[k]: spec[k].item() for k in range(4)}


def post_training_diagnostics(
    model,
    algebra,
    coords,
    device,
) -> Dict[str, float]:
    """Single-pass gather of all demoted diagnostics."""
    coords = coords.to(device)
    out = {}

    # autograd-based: divergence and curl from the velocity head only.
    derived = _autograd_div_curl_lap(model, coords)
    mv = derived["mv"]
    out["div_residual"] = (derived["div_u"] ** 2).mean().item()

    w_pred = torch.stack([mv[:, 6], -mv[:, 5], mv[:, 3]], dim=-1)
    out["vorticity_consistency"] = ((derived["curl"] - w_pred) ** 2).mean().item()

    # Gradient-head divergence: trace of the predicted ∂u/∂x Jacobian.
    # Should also collapse toward 0 because every TGV target has zero trace.
    with torch.no_grad():
        _, pred_grad = model(coords.detach())
    trace_div = pred_grad[:, 0] + pred_grad[:, 4] + pred_grad[:, 8]
    out["div_residual_grad_head"] = (trace_div**2).mean().item()

    # algebraic / no_grad
    out["gauge_covariance"] = gauge_covariance_residual(
        model,
        algebra,
        coords.detach(),
        device,
    )
    out.update(grade_spectrum(model, algebra, coords.detach(), device))

    return out


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = make_experiment_parser(
        "Navier-Stokes in Cl(3,0) — supervised IC+BC reconstruction.",
        defaults={"output_dir": "ns_plots"},
    )
    p.add_argument("--re", type=float, default=100.0)
    p.add_argument("--t-max", type=float, default=1.0)
    p.add_argument("--num-ic", type=int, default=1024)
    p.add_argument("--num-bc", type=int, default=512)
    p.add_argument("--num-test", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    algebra = setup_algebra(p=3, q=0, device=device)

    train_ds = TaylorGreenSupervisedDataset(
        args.num_ic,
        args.num_bc,
        re=args.re,
        t_max=args.t_max,
        seed=args.seed,
    )
    test_ds = TaylorGreenSupervisedDataset(
        args.num_test // 2,
        args.num_test // 2,
        re=args.re,
        t_max=args.t_max,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GaugeFluidNet(
        algebra,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    print_banner(
        "Navier-Stokes — Cl(3,0) supervised reconstruction",
        signature="Cl(3, 0)  grades: G0=p, G1=u, G2=ω, G3=h",
        natural_loss="MSE on (packed mv, ∂u/∂x_i) at IC + BC samples",
        Re=f"{args.re:.0f}",
        parameters=f"{count_parameters(model):,}",
    )

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    def loss_fn(_model, batch):
        coords, targets, target_grad = (b.to(device) for b in batch)
        pred_mv, pred_grad = _model(coords)
        return nn.functional.mse_loss(pred_mv, targets) + nn.functional.mse_loss(pred_grad, target_grad)

    def diag_fn(_model, _epoch) -> Dict[str, float]:
        return supervised_l2(_model, test_loader, device)

    history = run_supervised_loop(
        model,
        optimizer,
        loss_fn,
        train_loader,
        epochs=args.epochs,
        diag_interval=args.diag_interval,
        grad_clip=1.0,
        diag_fn=diag_fn,
        history_extra_keys=("test_l2_mv", "test_l2_grad"),
    )

    diagnostics = supervised_l2(model, test_loader, device)
    diag_coords = test_ds.coords[: min(256, len(test_ds))]
    diagnostics.update(
        post_training_diagnostics(
            model,
            algebra,
            diag_coords,
            device,
        )
    )
    print(
        report_diagnostics(
            diagnostics,
            title="Navier-Stokes post-training physics diagnostics",
        )
    )

    ensure_output_dir(args.output_dir)
    metadata = build_visualization_metadata(
        signature_metadata(3, 0),
        re=args.re,
        seed=args.seed,
    )
    path = save_training_curve(
        history,
        output_dir=args.output_dir,
        experiment_name="dbg_navier_stokes",
        metadata=metadata,
        plot_name="training_curve",
        args=args,
        module=__name__,
        title="Navier-Stokes — supervised IC+BC loss",
    )
    print(f"  curve saved to {path}")


if __name__ == "__main__":
    main()
