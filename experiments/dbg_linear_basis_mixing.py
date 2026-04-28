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

Linear Basis Mixing Debugger.

Hypothesis
  Versor's geometric bias should learn channel mixing through plain
  backpropagation without breaking multivector grade structure. Across five
  candidate backends and three synthetic regimes, a single supervised MSE
  should fit the task while grade leakage, grade-preservation, and
  rotation-invariance remain post-training measurements instead of gradient
  targets.

Execute Command
  uv run python -m experiments.dbg_linear_basis_mixing
  uv run python -m experiments.dbg_linear_basis_mixing --epochs 20
  uv run python -m experiments.dbg_linear_basis_mixing --p 2 --q 0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.algebra import CliffordAlgebra
from experiments._lib import (
    RawDefaultsHelpFormatter,
    build_visualization_metadata,
    count_parameters,
    ensure_output_dir,
    make_experiment_parser,
    run_supervised_loop,
    save_experiment_figure,
    set_seed,
    setup_algebra,
    signature_metadata,
)
from layers import BladeSelector, CliffordLinear
from layers.primitives.base import CliffordModule

# ==============================================================================
# Synthetic regime generators
# ==============================================================================


def _random_rotor(algebra: CliffordAlgebra, scale: float = 0.7) -> torch.Tensor:
    if algebra.n < 2:
        raise RuntimeError("random rotor requires n >= 2 (bivector space empty)")
    grade2 = algebra.grade_masks_float[2]
    B = torch.randn(1, algebra.dim) * scale * grade2
    return algebra.exp(-0.5 * B)


def make_R1_pure_grade(algebra, N, C, grade=1):
    x = torch.randn(N, C, algebra.dim) * algebra.grade_masks_float[grade]
    R = _random_rotor(algebra, scale=0.5).expand(N, algebra.dim).contiguous()
    y = algebra.sandwich_product(R, x)
    return x, y


def make_R2_mixed(algebra, N, C):
    x = torch.randn(N, C, algebra.dim) * 0.5
    x_next = torch.roll(x, shifts=-1, dims=1)
    wedged = algebra.wedge(x.reshape(-1, algebra.dim), x_next.reshape(-1, algebra.dim)).reshape(N, C, algebra.dim)
    return x, 0.7 * x + 0.3 * wedged


def make_R3_rotinv(algebra, N, C):
    x = torch.randn(N, C, algebra.dim) * 0.5
    y = torch.zeros_like(x)
    y[..., 0] = x.pow(2).sum(dim=-1)
    return x, y


REGIMES: Dict[str, Callable] = {
    "R1 (pure grade-1)": lambda alg, N, C: make_R1_pure_grade(alg, N, C, grade=1),
    "R2 (mixed wedge)": lambda alg, N, C: make_R2_mixed(alg, N, C),
    "R3 (rot-invariant)": lambda alg, N, C: make_R3_rotinv(alg, N, C),
}


# ==============================================================================
# Mixing-method wrappers  (all map [B, C, D] -> [B, C, D])
# ==============================================================================


class NNLinearBaseline(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.C, self.D = channels, dim
        self.lin = nn.Linear(channels * dim, channels * dim)

    def forward(self, x):
        return self.lin(x.reshape(x.shape[0], self.C * self.D)).reshape(x.shape[0], self.C, self.D)


class ScalarCliffordLinear(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.layer = CliffordLinear(algebra, channels, channels, backend="traditional")

    def forward(self, x):
        return self.layer(x)


class RotorCliffordLinear(nn.Module):
    def __init__(self, algebra, channels, num_rotor_pairs=4):
        super().__init__()
        self.layer = CliffordLinear(
            algebra, channels, channels, backend="rotor", num_rotor_pairs=num_rotor_pairs, aggregation="mean"
        )

    def forward(self, x):
        return self.layer(x)


class GradeWiseLinear(CliffordModule):
    """Independent ``nn.Linear`` per grade slot; scatter back."""

    def __init__(self, algebra, channels):
        super().__init__(algebra)
        self.C = channels
        self.grade_slots: List[torch.Tensor] = []
        self.lins = nn.ModuleList()
        for k in range(algebra.num_grades):
            idx = algebra.grade_masks[k].nonzero(as_tuple=False).squeeze(-1)
            self.register_buffer(f"_slot_{k}", idx, persistent=False)
            self.grade_slots.append(idx)
            feat = channels * int(idx.numel())
            self.lins.append(nn.Linear(feat, feat) if feat > 0 else nn.Identity())

    def forward(self, x):
        B = x.shape[0]
        out = torch.zeros_like(x)
        for idx, lin in zip(self.grade_slots, self.lins):
            if idx.numel() == 0:
                continue
            sub = x.index_select(dim=-1, index=idx)
            mixed = lin(sub.reshape(B, self.C * int(idx.numel()))).reshape(B, self.C, int(idx.numel()))
            out.index_copy_(-1, idx, mixed)
        return out


class BladeGatedScalar(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.mix = CliffordLinear(algebra, channels, channels, backend="traditional")
        self.gate = BladeSelector(algebra, channels)

    def forward(self, x):
        return self.gate(self.mix(x))


def build_methods(algebra, channels) -> Dict[str, nn.Module]:
    methods: Dict[str, nn.Module] = {
        "nn.Linear baseline": NNLinearBaseline(channels, algebra.dim),
        "CliffordLinear scalar": ScalarCliffordLinear(algebra, channels),
        "Grade-wise Linear": GradeWiseLinear(algebra, channels),
        "Blade-gated scalar": BladeGatedScalar(algebra, channels),
    }
    if algebra.n >= 2:
        methods["CliffordLinear rotor"] = RotorCliffordLinear(algebra, channels, num_rotor_pairs=4)
    return methods


# ==============================================================================
# Leakage matrix + derived metrics
# ==============================================================================


@torch.no_grad()
def leakage_matrix(model, algebra, x, eps=1e-8) -> torch.Tensor:
    ng = algebra.num_grades
    L = torch.full((ng, ng), float("nan"))
    for j in range(ng):
        xj = algebra.grade_projection(x, j)
        denom = xj.pow(2).sum().item()
        if denom < eps:
            continue
        yj = model(xj)
        for k in range(ng):
            L[k, j] = algebra.grade_projection(yj, k).pow(2).sum().item() / denom
    return L


def grade_preservation(L: torch.Tensor) -> float:
    ng = L.shape[0]
    valid = ~torch.isnan(L).any(dim=0)
    if valid.sum() == 0:
        return float("nan")
    diag = sum(L[k, k].item() for k in range(ng) if valid[k])
    total = L[:, valid].sum().item()
    return diag / total if total > 1e-12 else float("nan")


@torch.no_grad()
def rotation_invariance_gap(model, algebra, x, trials=5) -> float:
    if algebra.n < 2:
        return float("nan")
    y0 = model(x)
    y0_e = y0.pow(2).sum().clamp(min=1e-12).item()
    gap = 0.0
    for _ in range(trials):
        R = _random_rotor(algebra, scale=0.6).expand(x.shape[0], algebra.dim).contiguous()
        x_rot = algebra.sandwich_product(R, x)
        y_rot = model(x_rot)
        y_rot_back = algebra.sandwich_product(algebra.reverse(R), y_rot)
        gap += (y_rot_back - y0).pow(2).sum().item() / y0_e
    return gap / trials


# ==============================================================================
# Lifting initializer:  nn.Linear  ->  scalar CliffordLinear
# ==============================================================================


def lift_nnlinear_to_clifford(nn_lin, algebra, channels) -> CliffordLinear:
    """Grade-0 to grade-0 block of the flattened weight matrix."""
    D = algebra.dim
    W = nn_lin.weight.data.reshape(channels, D, channels, D)
    W00 = W[:, 0, :, 0].contiguous()
    b0 = nn_lin.bias.data.reshape(channels, D)[:, 0]
    cl = CliffordLinear(algebra, channels, channels, backend="traditional")
    with torch.no_grad():
        cl.weight.copy_(W00)
        cl.bias.zero_()
        cl.bias[:, 0] = b0
    return cl


# ==============================================================================
# Single-loss training harness
# ==============================================================================


@dataclass
class RunResult:
    method: str
    params: int = 0
    train_mse: float = float("nan")
    test_mse: float = float("nan")
    grade_pres: float = float("nan")
    rot_inv_gap: float = float("nan")
    L: Optional[torch.Tensor] = None


def _mse_loss(model, batch):
    xb, yb = batch
    return F.mse_loss(model(xb), yb)


def fit_and_measure(model, algebra, x_tr, y_tr, x_te, y_te, *, epochs, batch, lr) -> Tuple[float, float]:
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = run_supervised_loop(
        model, opt, _mse_loss, loader, epochs=epochs, diag_interval=max(epochs // 5, 1), log=False, grad_clip=None
    )
    train_mse = history["train_loss"][-1]
    with torch.no_grad():
        model.eval()
        test_mse = F.mse_loss(model(x_te), y_te).item()
    return train_mse, test_mse


def run_regime(algebra, regime_fn, channels, n_train, n_test, epochs, batch, lr, device) -> List[RunResult]:
    x_tr, y_tr = regime_fn(algebra, n_train, channels)
    x_te, y_te = regime_fn(algebra, n_test, channels)
    x_tr, y_tr, x_te, y_te = [t.to(device) for t in (x_tr, y_tr, x_te, y_te)]
    results: List[RunResult] = []
    for name, model in build_methods(algebra, channels).items():
        model.to(device)
        train_mse, test_mse = fit_and_measure(
            model,
            algebra,
            x_tr,
            y_tr,
            x_te,
            y_te,
            epochs=epochs,
            batch=batch,
            lr=lr,
        )
        L = leakage_matrix(model, algebra, x_te).cpu()
        results.append(
            RunResult(
                method=name,
                params=count_parameters(model),
                train_mse=train_mse,
                test_mse=test_mse,
                grade_pres=grade_preservation(L),
                rot_inv_gap=rotation_invariance_gap(model, algebra, x_te),
                L=L,
            )
        )
    return results


def run_lifting(algebra, channels, n_train, n_test, epochs_full, epochs_short, batch, lr, device) -> Dict[str, float]:
    x_tr, y_tr = make_R2_mixed(algebra, n_train, channels)
    x_te, y_te = make_R2_mixed(algebra, n_test, channels)
    x_tr, y_tr, x_te, y_te = [t.to(device) for t in (x_tr, y_tr, x_te, y_te)]

    base = NNLinearBaseline(channels, algebra.dim).to(device)
    _, base_test = fit_and_measure(base, algebra, x_tr, y_tr, x_te, y_te, epochs=epochs_full, batch=batch, lr=lr)

    lifted = lift_nnlinear_to_clifford(base.lin, algebra, channels).to(device)
    with torch.no_grad():
        lifted_init_test = F.mse_loss(lifted(x_te), y_te).item()
    _, lifted_test = fit_and_measure(lifted, algebra, x_tr, y_tr, x_te, y_te, epochs=epochs_short, batch=batch, lr=lr)

    scratch = ScalarCliffordLinear(algebra, channels).to(device)
    _, scratch_test = fit_and_measure(scratch, algebra, x_tr, y_tr, x_te, y_te, epochs=epochs_short, batch=batch, lr=lr)
    return {
        "nn.Linear (full train) test MSE": base_test,
        "Lifted init (no fine-tune) test MSE": lifted_init_test,
        f"Lifted + {epochs_short}ep fine-tune test MSE": lifted_test,
        f"Scratch CliffordLinear ({epochs_short}ep) test MSE": scratch_test,
        "Fine-tune gap (lifted / scratch)": (lifted_test / scratch_test if scratch_test > 1e-12 else float("inf")),
    }


# ==============================================================================
# Reporting
# ==============================================================================


def _fmt(x, width=8, prec=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—".rjust(width)
    return f"{x:.{prec}f}".rjust(width)


def print_regime_table(title, results):
    print(f"\n## {title}\n")
    name_w = max(len(r.method) for r in results)
    header = f"| {'Method'.ljust(name_w)} | Params | Train MSE | Test MSE  | Grade-pres | Rot-inv gap |"
    print(header)
    print(f"|{'-' * (name_w + 2)}|--------|-----------|-----------|------------|-------------|")
    for r in results:
        print(
            f"| {r.method.ljust(name_w)} | {r.params:6d} | "
            f"{_fmt(r.train_mse, 9, 5)} | {_fmt(r.test_mse, 9, 5)} | "
            f"{_fmt(r.grade_pres, 10, 4)} | {_fmt(r.rot_inv_gap, 11, 4)} |"
        )


def print_leakage(title, results, algebra):
    print(f"\n### Leakage matrices — {title}")
    print(f"(rows = output grade 0..{algebra.n}, cols = input grade 0..{algebra.n}; `—` = undefined)\n")
    for r in results:
        if r.L is None:
            continue
        print(f"  {r.method}")
        for k in range(algebra.num_grades):
            row = "    "
            for j in range(algebra.num_grades):
                v = r.L[k, j].item()
                row += (f"{v:6.2f}" if not math.isnan(v) else "     —") + " "
            print(row)
        print()


def winner(results, key="test_mse") -> str:
    valid = [r for r in results if not math.isnan(getattr(r, key))]
    return min(valid, key=lambda r: getattr(r, key)).method if valid else "—"


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_regime_summary_plot(
    title: str,
    results: List[RunResult],
    *,
    output_dir: str,
    metadata: str,
    args: argparse.Namespace,
) -> str:
    plt = _load_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Linear Basis Mixing — {title}", fontsize=13)

    methods = [r.method for r in results]
    xs = list(range(len(results)))
    test_mse = [max(r.test_mse, 1e-12) for r in results]
    grade_pres = [0.0 if math.isnan(r.grade_pres) else r.grade_pres for r in results]
    rot_gap = [max(r.rot_inv_gap, 1e-12) if not math.isnan(r.rot_inv_gap) else 1e-12 for r in results]

    axes[0].bar(xs, test_mse, color="steelblue")
    axes[0].set_yscale("log")
    axes[0].set_title("Test MSE")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(methods, rotation=30, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(xs, grade_pres, color="darkorange")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Grade Preservation")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(methods, rotation=30, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(xs, rot_gap, color="seagreen")
    axes[2].set_yscale("log")
    axes[2].set_title("Rotation Invariance Gap")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(methods, rotation=30, ha="right")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return save_experiment_figure(
        fig,
        output_dir=output_dir,
        experiment_name="dbg_linear_basis_mixing",
        metadata=metadata,
        plot_name=f"{title}_summary",
        args=args,
        module=__name__,
        dpi=150,
    )


def save_leakage_heatmap_plot(
    title: str,
    results: List[RunResult],
    algebra: CliffordAlgebra,
    *,
    output_dir: str,
    metadata: str,
    args: argparse.Namespace,
) -> str:
    plt = _load_pyplot()
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), squeeze=False)
    fig.suptitle(f"Leakage Matrices — {title}", fontsize=13)
    axes_row = axes[0]
    im = None

    for ax, result in zip(axes_row, results):
        if result.L is None:
            ax.axis("off")
            continue
        matrix = result.L.numpy()
        masked = np.nan_to_num(matrix, nan=0.0)
        im = ax.imshow(masked, vmin=0.0, vmax=max(1.0, float(masked.max())), cmap="magma")
        ax.set_title(result.method, fontsize=10)
        ax.set_xlabel("Input grade")
        ax.set_ylabel("Output grade")
        ax.set_xticks(range(algebra.num_grades))
        ax.set_yticks(range(algebra.num_grades))
        for row in range(algebra.num_grades):
            for col in range(algebra.num_grades):
                label = "—" if math.isnan(matrix[row, col]) else f"{matrix[row, col]:.2f}"
                ax.text(col, row, label, ha="center", va="center", fontsize=7, color="white")

    if im is not None:
        fig.colorbar(im, ax=axes_row.tolist(), shrink=0.75, label="Leakage ratio")
    fig.tight_layout()
    return save_experiment_figure(
        fig,
        output_dir=output_dir,
        experiment_name="dbg_linear_basis_mixing",
        metadata=metadata,
        plot_name=f"{title}_leakage",
        args=args,
        module=__name__,
        dpi=150,
    )


def save_lifting_plot(
    stats: Dict[str, float],
    *,
    output_dir: str,
    metadata: str,
    args: argparse.Namespace,
) -> str:
    plt = _load_pyplot()
    labels = list(stats.keys())
    values = [stats[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color="mediumpurple")
    ax.set_yscale("log")
    ax.set_title("Lifting Study")
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return save_experiment_figure(
        fig,
        output_dir=output_dir,
        experiment_name="dbg_linear_basis_mixing",
        metadata=metadata,
        plot_name="lifting_study",
        args=args,
        module=__name__,
        dpi=150,
    )


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = make_experiment_parser(
        __doc__,
        include=("seed", "device", "output_dir"),
        defaults={"seed": 0, "output_dir": "linear_basis_mixing_plots"},
        formatter_class=RawDefaultsHelpFormatter,
    )
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--q", type=int, default=0)
    p.add_argument("--r", type=int, default=0)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--epochs-short", type=int, default=50, help="fine-tune epochs for lifting study")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--regimes", type=str, default="all", help='comma-separated list: R1,R2,R3 (or "all")')
    p.add_argument("--skip-lifting", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    algebra = setup_algebra(args.p, args.q, args.r, device=args.device)

    if algebra.n < 1:
        raise SystemExit("n = p + q + r must be >= 1 for any multivector to exist")

    print(
        f"# Linear Basis Mixing — Cl({args.p},{args.q},{args.r})  "
        f"n={algebra.n}  dim={algebra.dim}  channels={args.channels}"
    )
    print(
        f"# {args.n_train} train / {args.n_test} test · {args.epochs} epochs · "
        f"batch {args.batch} · seed {args.seed} · device {args.device}"
    )
    if algebra.n < 2:
        print("# NOTE: n < 2 — rotor backend and rotation-invariance checks are disabled.")

    chosen = (
        list(REGIMES)
        if args.regimes == "all"
        else [name for name in REGIMES if any(name.startswith(tag) for tag in args.regimes.split(","))]
    )
    metadata = build_visualization_metadata(
        signature_metadata(args.p, args.q, args.r),
        channels=args.channels,
        regimes=args.regimes,
        seed=args.seed,
    )
    ensure_output_dir(args.output_dir)

    all_results: Dict[str, List[RunResult]] = {}
    for title in chosen:
        print(f"\n---\n[running] {title}")
        all_results[title] = run_regime(
            algebra,
            REGIMES[title],
            args.channels,
            args.n_train,
            args.n_test,
            args.epochs,
            args.batch,
            args.lr,
            device,
        )

    for title, results in all_results.items():
        print_regime_table(title, results)
        print_leakage(title, results, algebra)
        print(f"### Winners — {title}")
        print(f"  best test MSE        : {winner(results, 'test_mse')}")
        valid_gp = [r for r in results if not math.isnan(r.grade_pres)]
        if valid_gp:
            print(f"  best grade-preserv.  : {max(valid_gp, key=lambda r: r.grade_pres).method}")
        if algebra.n >= 2:
            valid_rig = [r for r in results if not math.isnan(r.rot_inv_gap)]
            if valid_rig:
                print(f"  smallest rot-inv gap : {min(valid_rig, key=lambda r: r.rot_inv_gap).method}")
        summary_path = save_regime_summary_plot(
            title,
            results,
            output_dir=args.output_dir,
            metadata=metadata,
            args=args,
        )
        leakage_path = save_leakage_heatmap_plot(
            title,
            results,
            algebra,
            output_dir=args.output_dir,
            metadata=metadata,
            args=args,
        )
        print(f"  summary plot saved to {summary_path}")
        print(f"  leakage plot saved to {leakage_path}")

    if not args.skip_lifting:
        print("\n---\n## Lifting study — nn.Linear → CliffordLinear (R2)\n")
        stats = run_lifting(
            algebra,
            args.channels,
            args.n_train,
            args.n_test,
            args.epochs,
            args.epochs_short,
            args.batch,
            args.lr,
            device,
        )
        for k, v in stats.items():
            print(f"  {k:55s} : {v:.5f}")
        lifting_path = save_lifting_plot(
            stats,
            output_dir=args.output_dir,
            metadata=metadata,
            args=args,
        )
        print(f"  lifting plot saved to {lifting_path}")


if __name__ == "__main__":
    main()
