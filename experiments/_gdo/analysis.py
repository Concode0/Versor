"""Cross-experiment analysis: convergence metrics, overhead ratios, report generation."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from experiments._lib import ensure_output_dir

from .config import ExperimentResult


def compute_convergence_metrics(result: ExperimentResult) -> Dict:
    """Steps to 50%/90%/99% of total improvement, AUC."""
    losses = np.array(result.losses)
    if len(losses) < 2:
        return {}
    initial = losses[0]
    final = losses[-1]
    total_improvement = initial - final
    if total_improvement <= 0:
        return {"auc": float(losses.sum()), "improvement": 0.0}

    metrics = {"auc": float(losses.sum()), "improvement": float(total_improvement)}
    for pct, label in [(0.5, "steps_to_50pct"), (0.9, "steps_to_90pct"), (0.99, "steps_to_99pct")]:
        threshold = initial - pct * total_improvement
        reached = np.where(losses <= threshold)[0]
        metrics[label] = int(reached[0]) if len(reached) > 0 else len(losses)

    return metrics


def compute_overhead_ratio(
    gdo_result: ExperimentResult,
    baseline_result: ExperimentResult,
) -> float:
    """Wall-time ratio GDO / baseline."""
    if baseline_result.total_wall_time < 1e-6:
        return float('inf')
    return gdo_result.total_wall_time / baseline_result.total_wall_time


def analyze_experiment_results(
    all_results: Dict[str, Dict[str, ExperimentResult]],
    output_dir: str = "gdo_plots",
) -> str:
    """Cross-experiment analysis. Returns formatted report text."""
    lines = []
    lines.append("=" * 70)
    lines.append("GDO ANALYSIS REPORT")
    lines.append("=" * 70)

    lines.append("\n1. WIN/LOSS MATRIX (lowest final loss)")
    lines.append("-" * 50)
    wins = {}
    for task_name, task_results in all_results.items():
        if not task_results:
            continue
        best_name = min(task_results.keys(), key=lambda k: task_results[k].final_loss)
        wins.setdefault(best_name, []).append(task_name)
        final_strs = [f"  {k}: {v.final_loss:.6f}" for k, v in task_results.items()]
        lines.append(f"\n  {task_name}:")
        lines.extend(final_strs)
        lines.append(f"  Winner: {best_name}")

    lines.append("\n  Summary:")
    for opt_name, tasks in sorted(wins.items()):
        lines.append(f"    {opt_name}: {len(tasks)} wins ({', '.join(tasks)})")

    lines.append("\n2. CONVERGENCE SPEED")
    lines.append("-" * 50)
    for task_name, task_results in all_results.items():
        lines.append(f"\n  {task_name}:")
        for opt_name, res in task_results.items():
            cm = compute_convergence_metrics(res)
            if cm:
                lines.append(
                    f"    {opt_name}: 50%@{cm.get('steps_to_50pct', '?')}, "
                    f"90%@{cm.get('steps_to_90pct', '?')}, "
                    f"99%@{cm.get('steps_to_99pct', '?')}"
                )

    lines.append("\n3. WALL-TIME EFFICIENCY")
    lines.append("-" * 50)
    for task_name, task_results in all_results.items():
        lines.append(f"\n  {task_name}:")
        for opt_name, res in task_results.items():
            ms_per_step = (res.total_wall_time / max(len(res.losses), 1)) * 1000
            lines.append(f"    {opt_name}: {res.total_wall_time:.1f}s total, {ms_per_step:.1f}ms/step")

    lines.append("\n4. GDO OVERHEAD RATIO (vs RiemannianAdam)")
    lines.append("-" * 50)
    for task_name, task_results in all_results.items():
        gdo_res = task_results.get("GDO")
        riem_res = task_results.get("RiemannianAdam")
        if gdo_res and riem_res:
            ratio = compute_overhead_ratio(gdo_res, riem_res)
            lines.append(f"  {task_name}: {ratio:.2f}x")

    lines.append("\n5. GDO ADVANTAGE ANALYSIS")
    lines.append("-" * 50)
    gdo_better = []
    gdo_worse = []
    for task_name, task_results in all_results.items():
        gdo_res = task_results.get("GDO")
        riem_res = task_results.get("RiemannianAdam")
        if gdo_res and riem_res:
            if gdo_res.final_loss < riem_res.final_loss * 0.95:
                gdo_better.append(task_name)
            elif gdo_res.final_loss > riem_res.final_loss * 1.05:
                gdo_worse.append(task_name)
    lines.append(f"  GDO better (>5% lower loss): {gdo_better or 'none'}")
    lines.append(f"  GDO worse (>5% higher loss): {gdo_worse or 'none'}")

    report = "\n".join(lines)

    ensure_output_dir(output_dir)
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    return report
