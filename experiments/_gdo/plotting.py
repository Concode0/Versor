"""Visualization for GDO: pre-exploration, trajectory, controller, comparison, landscape."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments._lib import ensure_output_dir

from .config import ExperimentResult
from .pre_exploration import PreExplorationResult
from .topology import CriticalPointType, LandscapeMap


def _ensure_output_dir(output_dir: str):
    ensure_output_dir(output_dir)


def plot_pre_exploration(
    pre_result: PreExplorationResult,
    title: str = "Pre-Exploration Analysis",
    output_dir: str = "gdo_plots",
):
    """2x3 dashboard: eigenvalues, local dims, grade energy, coherence, geometry, config."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    if pre_result.dim_result is not None:
        ev = pre_result.dim_result.eigenvalues.cpu().numpy()
        ax.semilogy(range(1, len(ev) + 1), ev, 'b.-')
        ax.axhline(y=ev[pre_result.dim_result.broken_stick_threshold - 1]
                    if pre_result.dim_result.broken_stick_threshold > 0
                    else ev[-1],
                    color='r', linestyle='--', alpha=0.7, label='broken-stick')
        ax.set_title(f"Eigenvalue Spectrum\n"
                     f"intrinsic_dim={pre_result.dim_result.intrinsic_dim}, "
                     f"PR={pre_result.dim_result.participation_ratio:.1f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No dimension\nanalysis", ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if pre_result.dim_result is not None and pre_result.dim_result.local_dims is not None:
        ld = pre_result.dim_result.local_dims.cpu().numpy()
        ax.hist(ld, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(pre_result.dim_result.participation_ratio, color='red',
                   linestyle='--', label=f'PR={pre_result.dim_result.participation_ratio:.1f}')
        ax.set_title("Local Dimension Distribution")
        ax.legend()
    else:
        ls = pre_result.loss_statistics
        if ls:
            vals = [ls["min"], ls["mean"], ls["max"]]
            labels = ["min", "mean", "max"]
            ax.barh(labels, vals, color=['green', 'steelblue', 'red'], alpha=0.7)
            ax.set_title("Loss Statistics")
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    gs = pre_result.geometric_scores
    if pre_result.spectral_result is not None:
        ge = pre_result.spectral_result.grade_energy.cpu().numpy()
        grades = list(range(len(ge)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ge)))
        ax.bar(grades, ge, color=colors, edgecolor='white')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Energy")
        ax.set_title("Grade Energy Spectrum")
    elif "grade_energy" in gs:
        ge = gs["grade_energy"].cpu().numpy()
        grades = list(range(len(ge)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ge)))
        ax.bar(grades, ge, color=colors, edgecolor='white')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Energy")
        ax.set_title("Grade Energy Spectrum")
    else:
        ax.text(0.5, 0.5, "No spectral\nanalysis", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    metrics_names = ["Coherence", "Curvature"]
    values = [pre_result.landscape_coherence, pre_result.landscape_curvature]
    bar_colors = []
    for v, name in zip(values, metrics_names):
        if name == "Coherence":
            bar_colors.append('green' if v > 0.5 else ('orange' if v > 0.3 else 'red'))
        else:
            bar_colors.append('green' if v < 0.3 else ('orange' if v < 0.5 else 'red'))
    ax.barh(metrics_names, values, color=bar_colors, alpha=0.7, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_title(f"Landscape Geometry\nStrategy: {pre_result.strategy_label}")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels_gs, vals_gs, colors_gs = [], [], []
    if "closure_error" in gs:
        labels_gs.append("Lie Closure\nError")
        vals_gs.append(gs["closure_error"])
        ce = gs["closure_error"]
        colors_gs.append('green' if ce < 0.1 else ('orange' if ce < 0.5 else 'red'))
    if "grade_entropy" in gs:
        labels_gs.append("Grade\nEntropy")
        vals_gs.append(gs["grade_entropy"])
        colors_gs.append('purple')
    if "coherence" in gs:
        labels_gs.append("Bivector\nCoherence")
        vals_gs.append(gs["coherence"])
        colors_gs.append('steelblue')
    if labels_gs:
        ax.barh(labels_gs, vals_gs, color=colors_gs, alpha=0.7, edgecolor='white')
        ax.set_xlim(0, max(1.0, max(vals_gs) * 1.1))
        ax.set_title("Geometric Signals")
    else:
        ax.text(0.5, 0.5, "No geometric\nscores", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis('off')
    cfg = pre_result.recommended_config
    lines = [
        f"lr: {cfg.lr}",
        f"probe_interval: {cfg.probe_interval}",
        f"topology_interval: {cfg.topology_interval}",
        f"sprint_after: {cfg.sprint_after}",
        f"lift_patience: {cfg.lift_patience}",
        f"lift_sigma: {cfg.lift_sigma}",
        f"lorentz_max_beta: {cfg.lorentz_max_beta}",
        f"commutator_threshold: {cfg.commutator_threshold}",
    ]
    ax.text(0.05, 0.95, "Recommended Config\n" + "-" * 25 + "\n" + "\n".join(lines),
            transform=ax.transAxes, va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, f"pre_exploration_{title.replace(' ', '_').lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_optimization_trajectory(
    history: Dict,
    title: str = "Optimization Trajectory",
    output_dir: str = "gdo_plots",
):
    """Loss curve, probe results, landscape map summary."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    losses = history.get("losses", [])
    modes = history.get("modes", [])
    if losses:
        ax.semilogy(losses, 'b-', linewidth=0.8)
        mode_colors = {"explore": "#cce5ff", "navigate": "#d4edda", "sprint": "#fff3cd"}
        if modes:
            prev = modes[0]
            start = 0
            for i, m in enumerate(modes + [None]):
                if m != prev or i == len(modes):
                    ax.axvspan(start, i, alpha=0.15,
                               color=mode_colors.get(prev, '#ffffff'))
                    prev = m
                    start = i
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    probe_steps = history.get("probe_steps", [])
    curvatures = history.get("curvatures", [])
    grad_norms = history.get("grad_norms", [])
    if probe_steps and curvatures:
        ax.plot(probe_steps, curvatures, 'b.-', label='Mean curvature')
        ax2 = ax.twinx()
        ax2.plot(probe_steps, grad_norms, 'r.-', alpha=0.7, label='Grad norm')
        ax2.set_ylabel("Grad Norm", color='red')
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Curvature", color='blue')
        ax.set_title("Probe Results")
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, "No probe data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    betas = history.get("betas", [])
    if probe_steps and betas:
        ax.plot(probe_steps, betas, 'g.-')
        ax.set_xlabel("Step")
        ax.set_ylabel("Lorentz beta")
        ax.set_title("Lorentz Warp Factor")
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No warp data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    lifts = history.get("lifts", [])
    if lifts:
        lift_steps = [l["step"] for l in lifts]
        lift_losses = [l["loss"] for l in lifts]
        lift_colors = ['green' if l["success"] else 'red' for l in lifts]
        ax.scatter(lift_steps, lift_losses, c=lift_colors, s=50, zorder=5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss at Lift")
        ax.set_title(f"Lift Oracle Events ({len(lifts)} total)")
        patches = [Patch(color='green', label='Success'),
                   Patch(color='red', label='Fail')]
        ax.legend(handles=patches, fontsize=8)
    else:
        if modes:
            mode_counts = {}
            for m in modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            ax.bar(mode_counts.keys(), mode_counts.values(),
                   color=['steelblue', 'seagreen', 'orange'][:len(mode_counts)])
            ax.set_title("Mode Distribution")
        else:
            ax.text(0.5, 0.5, "No lift/mode data", ha='center', va='center',
                    transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"trajectory_{title.replace(' ', '_').lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_geometric_controller(
    diagnostics: Dict,
    title: str = "Geometric Parameter Controller",
    output_dir: str = "gdo_plots",
):
    """2x2 dashboard: FIM, commutativity heatmap, group scales, grade energy."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    fim = diagnostics.get("fim_diag", {})
    if fim:
        groups = sorted(fim.keys())
        means = [fim[g].mean().item() for g in groups]
        ax.bar(groups, means, color='steelblue', edgecolor='white')
        ax.set_xlabel("Param Group")
        ax.set_ylabel("Mean FIM")
        ax.set_title("Fisher Information (per group)")
    else:
        ax.text(0.5, 0.5, "No FIM data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    gs = diagnostics.get("geometric_scores", {})
    if "comm_result" in gs:
        mat = gs["comm_result"].commutativity_matrix.cpu().numpy()
        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Commutativity Matrix")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Dimension")
        schedule = diagnostics.get("schedule", [])
        sched_colors = plt.cm.Set2(np.linspace(0, 1, max(len(schedule), 1)))
        for ci, color_group in enumerate(schedule):
            for g in color_group:
                if g < mat.shape[0]:
                    ax.axhline(y=g, color=sched_colors[ci], linewidth=2, alpha=0.5)
    else:
        hybrid = diagnostics.get("hybrid_scores", {})
        if hybrid:
            n = max(max(k) for k in hybrid.keys()) + 1 if hybrid else 1
            mat = np.zeros((n, n))
            for (i, j), v in hybrid.items():
                mat[i, j] = v
                mat[j, i] = v
            im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Hybrid Commutativity Scores")
        else:
            ax.text(0.5, 0.5, "No commutativity\ndata", ha='center', va='center',
                    transform=ax.transAxes)
    ax.grid(False)

    ax = axes[1, 0]
    scales = diagnostics.get("scales", [])
    if scales:
        x = list(range(len(scales)))
        bar_colors = []
        for s in scales:
            if s > 1.3:
                bar_colors.append('green')
            elif s < 0.5:
                bar_colors.append('red')
            elif s < 0.8:
                bar_colors.append('orange')
            else:
                bar_colors.append('steelblue')
        ax.bar(x, scales, color=bar_colors, edgecolor='white')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Param Group")
        ax.set_ylabel("Update Scale")
        ax.set_title("Group Update Scales\n(green=trust, red=caution)")
    else:
        ax.text(0.5, 0.5, "No scale data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "grade_energy" in gs:
        ge = gs["grade_energy"].cpu().numpy()
        grades = list(range(len(ge)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ge)))
        ax.bar(grades, ge, color=colors, edgecolor='white')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Energy")
        title_parts = ["Grade Energy"]
        if "grade_entropy" in gs:
            title_parts.append(f"H={gs['grade_entropy']:.3f}")
        if "closure_error" in gs:
            title_parts.append(f"closure={gs['closure_error']:.3f}")
        ax.set_title(" | ".join(title_parts))
    else:
        ax.text(0.5, 0.5, "No grade energy\ndata", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "geometric_controller.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_topology_map(
    model: nn.Module,
    landscape: LandscapeMap,
    trajectory: List[Tuple[float, float]],
    modes: List[str],
    output_dir: str = "gdo_plots",
):
    """Contour plot of 2D loss surface + critical points + trajectory."""
    _ensure_output_dir(output_dir)
    if not hasattr(model, 'a'):
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    xr = np.linspace(-2.5, 2.5, 100)
    yr = np.linspace(-1.5, 3.5, 100)
    X, Y = np.meshgrid(xr, yr)
    Z = (model.a - X) ** 2 + model.b * (Y - X ** 2) ** 2
    ax.contourf(X, Y, np.log10(Z + 1e-10), levels=30, cmap='terrain', alpha=0.7)
    ax.contour(X, Y, np.log10(Z + 1e-10), levels=15, colors='gray',
               linewidths=0.3, alpha=0.5)

    cp_markers = {
        CriticalPointType.MINIMUM: ('o', 'blue', 'Minimum'),
        CriticalPointType.SADDLE: ('^', 'red', 'Saddle'),
        CriticalPointType.MAXIMUM: ('s', 'gray', 'Maximum'),
    }
    for cp in landscape.critical_points:
        if cp.params.shape[0] >= 2:
            marker, color, cp_label = cp_markers.get(
                cp.point_type, ('x', 'black', 'Unknown'))
            ax.scatter(cp.params[0].item(), cp.params[1].item(),
                       marker=marker, c=color, s=80, zorder=5,
                       edgecolors='white', linewidths=1, label=cp_label)

    if trajectory:
        mode_colors_traj = {"explore": "blue", "navigate": "green", "sprint": "orange"}
        for i in range(1, len(trajectory)):
            m = modes[i] if i < len(modes) else "explore"
            ax.plot([trajectory[i - 1][0], trajectory[i][0]],
                    [trajectory[i - 1][1], trajectory[i][1]],
                    color=mode_colors_traj.get(m, "blue"), linewidth=0.5, alpha=0.7)
        ax.scatter(*trajectory[0], marker='*', c='lime', s=150, zorder=6,
                   edgecolors='black', label='Start')
        ax.scatter(*trajectory[-1], marker='*', c='red', s=150, zorder=6,
                   edgecolors='black', label='End')
        ax.scatter(1.0, 1.0, marker='D', c='gold', s=100, zorder=6,
                   edgecolors='black', label='Optimum')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Loss Landscape & Optimization Trajectory")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "topology_map.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_three_way_comparison(
    results: Dict[str, ExperimentResult],
    title: str = "Optimizer Comparison",
    output_dir: str = "gdo_plots",
):
    """4-panel: loss curves, final loss bars, wall time bars, convergence rate."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {
        "GDO": ("b-", 2.0), "RiemannianAdam": ("r--", 1.5),
        "Adam": ("g:", 1.2), "Adam (no algebra)": ("g:", 1.2),
        "ExponentialSGD": ("m-.", 1.2), "SGD": ("m-.", 1.2),
    }
    color_map = {
        "GDO": "steelblue", "RiemannianAdam": "salmon",
        "Adam": "seagreen", "Adam (no algebra)": "seagreen",
        "ExponentialSGD": "plum", "SGD": "plum",
    }

    ax = axes[0, 0]
    for name, res in results.items():
        style, lw = styles.get(name, ("k-", 1.0))
        ax.semilogy(res.losses, style, label=name, linewidth=lw)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    names = list(results.keys())
    finals = [results[n].final_loss for n in names]
    bar_colors = [color_map.get(n, 'gray') for n in names]
    ax.bar(names, finals, color=bar_colors, edgecolor='white')
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(finals):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1, 0]
    wall_times = [results[n].total_wall_time for n in names]
    ax.bar(names, wall_times, color=bar_colors, edgecolor='white')
    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title("Wall Time")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(wall_times):
        ax.text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontsize=8)

    ax = axes[1, 1]
    for name, res in results.items():
        cum_time = np.cumsum(res.wall_times)
        style, lw = styles.get(name, ("k-", 1.0))
        ax.semilogy(cum_time, res.losses, style, label=name, linewidth=lw)
    ax.set_xlabel("Cumulative Wall Time (s)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Wall Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace('/', '_').lower()
    path = os.path.join(output_dir, f"comparison_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_convergence_rate(
    results: Dict[str, ExperimentResult],
    title: str = "Convergence Rate",
    output_dir: str = "gdo_plots",
):
    """3-panel: loss vs step, loss vs wall-time, convergence rate."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {"GDO": "b-", "RiemannianAdam": "r--", "Adam": "g:", "Adam (no algebra)": "g:"}

    ax = axes[0]
    for name, res in results.items():
        ax.semilogy(res.losses, styles.get(name, "k-"), label=name, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, res in results.items():
        cum_time = np.cumsum(res.wall_times)
        ax.semilogy(cum_time, res.losses, styles.get(name, "k-"), label=name, linewidth=1.2)
    ax.set_xlabel("Wall Time (s)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Wall Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    window = 50
    for name, res in results.items():
        if len(res.losses) > window:
            losses_arr = np.array(res.losses)
            rate = -(losses_arr[window:] - losses_arr[:-window]) / window
            smoothed = np.convolve(rate, np.ones(20)/20, mode='valid')
            ax.plot(smoothed, styles.get(name, "k-"), label=name, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Convergence Rate (loss drop/step)")
    ax.set_title("Smoothed Convergence Rate")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"convergence_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_timing_breakdown(
    results: Dict[str, ExperimentResult],
    title: str = "Timing Breakdown",
    output_dir: str = "gdo_plots",
):
    """2-panel: per-step wall time, cumulative time vs loss."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {"GDO": "b", "RiemannianAdam": "r", "Adam": "g", "Adam (no algebra)": "g"}

    ax = axes[0]
    for name, res in results.items():
        wt = np.array(res.wall_times) * 1000
        window = max(1, len(wt) // 100)
        if window > 1:
            wt_smooth = np.convolve(wt, np.ones(window)/window, mode='valid')
        else:
            wt_smooth = wt
        ax.plot(wt_smooth, color=styles.get(name, 'k'), label=name, linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Wall Time (ms)")
    ax.set_title("Per-Step Wall Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, res in results.items():
        cum_time = np.cumsum(res.wall_times)
        ax.plot(cum_time, res.losses, color=styles.get(name, 'k'), label=name, linewidth=1.2)
    ax.set_xlabel("Cumulative Time (s)")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.set_title("Cumulative Time vs Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"timing_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_bivector_trajectory(
    results: Dict[str, ExperimentResult],
    title: str = "Bivector Trajectory",
    output_dir: str = "gdo_plots",
):
    """Bivector param norm evolution across optimizers."""
    _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {"GDO": "b-", "RiemannianAdam": "r--", "Adam": "g:", "Adam (no algebra)": "g:"}
    for name, res in results.items():
        if res.bivector_norms:
            ax.plot(res.bivector_norms, styles.get(name, "k-"), label=name, linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Bivector Param Norm")
    ax.set_title("Bivector Parameter Evolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"bivector_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_optimizer_state_dashboard(
    gdo_result: ExperimentResult,
    title: str = "GDO State Dashboard",
    output_dir: str = "gdo_plots",
):
    """4-panel: mode timeline, topology summary, warp beta/gamma, lift events."""
    _ensure_output_dir(output_dir)
    diag = gdo_result.gdo_diagnostics
    if diag is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    mode_hist = gdo_result.mode_history or diag.get("mode_history", [])
    if mode_hist:
        mode_to_int = {"explore": 0, "navigate": 1, "sprint": 2}
        mode_ints = [mode_to_int.get(m, 0) for m in mode_hist]
        mode_colors = {0: '#4a90d9', 1: '#66bb6a', 2: '#ffa726'}
        for i in range(len(mode_ints)):
            ax.bar(i, 1, color=mode_colors.get(mode_ints[i], 'gray'), width=1.0)
        ax.set_xlabel("Step")
        ax.set_yticks([])
        ax.set_title("Mode Timeline (blue=explore, green=navigate, orange=sprint)")
    else:
        ax.text(0.5, 0.5, "No mode data", ha='center', va='center', transform=ax.transAxes)

    ax = axes[0, 1]
    topo = diag.get("topology_map", {})
    ax.axis('off')
    lines = [
        f"Critical points detected: {topo.get('critical_points', 0)}",
        f"Plateau episodes: {len(topo.get('plateau_episodes', []))}",
        f"Curvature samples: {len(topo.get('curvature_history', []))}",
    ]
    warp = diag.get("warp", {})
    lines.append(f"\nWarp beta: {warp.get('beta', 0):.4f}")
    lines.append(f"Warp gamma: {warp.get('gamma', 1):.4f}")
    lines.append(f"On plateau: {warp.get('on_plateau', False)}")

    lift = diag.get("lift_oracle", {})
    lines.append(f"\nLift count: {lift.get('lift_count', 0)}")
    lines.append(f"Consecutive fails: {lift.get('consecutive_fails', 0)}")
    lines.append(f"Current sigma: {lift.get('current_sigma', 0):.4f}")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    ax.set_title("GDO State Summary")

    ax = axes[1, 0]
    curv_hist = topo.get("curvature_history", [])
    if curv_hist:
        ax.plot(curv_hist, 'b.-', linewidth=0.8)
        ax.set_xlabel("Probe Index")
        ax.set_ylabel("Mean Curvature")
        ax.set_title("Curvature History")
    else:
        ax.text(0.5, 0.5, "No curvature data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    grad_hist = topo.get("gradient_norm_history", [])
    if grad_hist:
        ax.semilogy(grad_hist, 'r.-', linewidth=0.8)
        ax.set_xlabel("Probe Index")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm History")
    else:
        ax.text(0.5, 0.5, "No gradient data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"gdo_state_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_loss_landscape_2d_slice(
    model: nn.Module,
    loss_fn: Callable,
    n_grid: int = 40,
    radius: float = 1.0,
    title: str = "Loss Landscape Slice",
    output_dir: str = "gdo_plots",
):
    """Contour plot along 2 random orthogonal directions in param space."""
    _ensure_output_dir(output_dir)
    params = list(model.parameters())
    flat_center = torch.cat([p.detach().reshape(-1) for p in params])
    n = flat_center.shape[0]
    device = flat_center.device

    d1 = F.normalize(torch.randn(n, device=device), dim=0)
    d2 = torch.randn(n, device=device)
    d2 = d2 - (d2 @ d1) * d1
    d2 = F.normalize(d2, dim=0)

    alphas = np.linspace(-radius, radius, n_grid)
    betas = np.linspace(-radius, radius, n_grid)
    Z = np.zeros((n_grid, n_grid))

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                flat_p = flat_center + a * d1 + b * d2
                idx = 0
                for p in params:
                    sz = p.numel()
                    p.data.copy_(flat_p[idx:idx+sz].reshape(p.shape))
                    idx += sz
                Z[j, i] = loss_fn().item()

    idx = 0
    for p in params:
        sz = p.numel()
        p.data.copy_(flat_center[idx:idx+sz].reshape(p.shape))
        idx += sz

    fig, ax = plt.subplots(figsize=(8, 7))
    A, B = np.meshgrid(alphas, betas)
    cs = ax.contourf(A, B, np.log10(Z + 1e-10), levels=30, cmap='viridis')
    fig.colorbar(cs, ax=ax, label='log10(loss)')
    ax.scatter([0], [0], c='red', s=100, marker='*', zorder=5, label='Current')
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"landscape_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig
