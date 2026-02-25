# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Feynman Sweep Task - runs all (or filtered) equations and produces a benchmark report."""

import os
import time
import csv
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf

from datasets.feynman import FEYNMAN_EQUATIONS, get_equations_by_tier
from tasks.feynman import FeynmanTask
from log import get_logger

logger = get_logger(__name__)


class FeynmanSweepTask:
    """Orchestrates multiple FeynmanTask runs across equations.

    NOT a BaseTask subclass - it creates and runs individual FeynmanTask instances.

    Config keys:
        dataset.equations: "all" | "tier1"-"tier4" | list of equation IDs
        sweep.results_dir: directory for output files
        sweep.save_checkpoints: keep per-equation .pt files
        sweep.visualize: generate per-equation plots
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.equations = self._resolve_equations()
        self.results_dir = cfg.get("sweep", {}).get("results_dir", "./results")

    def _resolve_equations(self):
        """Parse dataset.equations into a list of equation ID strings."""
        sel = self.cfg.dataset.get("equations", "all")

        if isinstance(sel, str):
            if sel == "all":
                return list(FEYNMAN_EQUATIONS.keys())
            elif sel.startswith("tier"):
                tier_num = int(sel.replace("tier", ""))
                return get_equations_by_tier(tier_num)
            else:
                return [sel]
        else:
            return list(sel)

    def run(self):
        """Run each equation sequentially and collect results."""
        n_total = len(self.equations)
        logger.info(f"Feynman Sweep: {n_total} equations")
        logger.info(f"Epochs per equation: {self.cfg.training.epochs}")
        logger.info(f"Results dir: {self.results_dir}")

        results = []
        save_ckpt = self.cfg.get("sweep", {}).get("save_checkpoints", False)
        do_viz = self.cfg.get("sweep", {}).get("visualize", False)

        for i, eq_id in enumerate(self.equations):
            spec = FEYNMAN_EQUATIONS.get(eq_id)
            if spec is None:
                logger.warning(f"[{i+1}/{n_total}] SKIP unknown equation: {eq_id}")
                continue

            logger.info(f"[{i+1}/{n_total}] {eq_id} - {spec['desc']} "
                        f"(tier {spec.get('tier', '?')}, {spec['n_vars']} vars)")

            eq_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
            eq_cfg.dataset.equation = eq_id
            eq_cfg.name = "feynman"

            t0 = time.time()
            try:
                task = FeynmanTask(eq_cfg)
                train_loader, val_loader, test_loader = task.get_data()

                # Minimal training loop (reuse task.run() internals)
                best_val_mae = float("inf")
                for epoch in range(task.epochs):
                    task._epoch = epoch + 1
                    task.model.train()
                    for batch in train_loader:
                        task.train_step(batch)

                    task.model.eval()
                    val_mae, val_r2 = task.evaluate(val_loader)
                    task.scheduler.step(val_mae)

                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        if save_ckpt:
                            os.makedirs(self.results_dir, exist_ok=True)
                            ckpt = os.path.join(
                                self.results_dir,
                                f"feynman_{eq_id.replace('.', '_')}_best.pt",
                            )
                            task.save_checkpoint(ckpt)

                # Final test evaluation
                task.model.eval()
                test_mae, test_r2 = task.evaluate(test_loader)

                if do_viz:
                    try:
                        task.visualize(test_loader)
                    except Exception as viz_err:
                        logger.warning(f"Visualization failed: {viz_err}")

                wall_time = time.time() - t0

                results.append({
                    "equation": eq_id,
                    "desc": spec["desc"],
                    "n_vars": spec["n_vars"],
                    "tier": spec.get("tier", 0),
                    "test_mae": test_mae,
                    "test_r2": test_r2,
                    "best_val_mae": best_val_mae,
                    "wall_time": wall_time,
                    "status": "ok",
                })
                logger.info(f"MAE={test_mae:.6f}  R**2={test_r2:.4f}  "
                            f"({wall_time:.1f}s)")

            except Exception as e:
                wall_time = time.time() - t0
                results.append({
                    "equation": eq_id,
                    "desc": spec["desc"],
                    "n_vars": spec["n_vars"],
                    "tier": spec.get("tier", 0),
                    "test_mae": float("nan"),
                    "test_r2": float("nan"),
                    "best_val_mae": float("nan"),
                    "wall_time": wall_time,
                    "status": f"error: {e}",
                })
                logger.warning(f"FAILED: {e} ({wall_time:.1f}s)")

        self._save_results(results)
        self._print_summary(results)

    def _save_results(self, results):
        """Save CSV and markdown report."""
        os.makedirs(self.results_dir, exist_ok=True)

        # CSV
        csv_path = os.path.join(self.results_dir, "feynman_benchmark.csv")
        fieldnames = [
            "equation", "desc", "n_vars", "tier",
            "test_mae", "test_r2", "best_val_mae", "wall_time", "status",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(results, key=lambda x: (x["tier"], -x["test_r2"])):
                writer.writerow(r)
        logger.info(f"Saved CSV: {csv_path}")

        # Markdown
        md_path = os.path.join(self.results_dir, "feynman_benchmark.md")
        with open(md_path, "w") as f:
            f.write("# Feynman Symbolic Regression Benchmark\n\n")
            f.write(f"Epochs: {self.cfg.training.epochs} | "
                    f"LR: {self.cfg.training.lr} | "
                    f"Algebra: Cl({self.cfg.algebra.p},{self.cfg.algebra.get('q', 0)})\n\n")

            f.write("| Equation | Description | Vars | Tier | Test MAE | Test R**2 | Time (s) | Status |\n")
            f.write("|----------|-------------|------|------|----------|---------|----------|--------|\n")
            for r in sorted(results, key=lambda x: (x["tier"], x["equation"])):
                mae_s = f"{r['test_mae']:.6f}" if r["status"] == "ok" else "-"
                r2_s = f"{r['test_r2']:.4f}" if r["status"] == "ok" else "-"
                t_s = f"{r['wall_time']:.1f}"
                status = "ok" if r["status"] == "ok" else "FAIL"
                f.write(f"| {r['equation']} | {r['desc']} | {r['n_vars']} | "
                        f"{r['tier']} | {mae_s} | {r2_s} | {t_s} | {status} |\n")

            # Aggregate stats
            ok = [r for r in results if r["status"] == "ok"]
            if ok:
                import statistics
                maes = [r["test_mae"] for r in ok]
                r2s = [r["test_r2"] for r in ok]
                total_time = sum(r["wall_time"] for r in results)

                f.write(f"\n## Summary\n\n")
                f.write(f"- **Equations tested**: {len(ok)}/{len(results)}\n")
                f.write(f"- **Mean MAE**: {statistics.mean(maes):.6f}\n")
                f.write(f"- **Median MAE**: {statistics.median(maes):.6f}\n")
                f.write(f"- **Mean R**2**: {statistics.mean(r2s):.4f}\n")
                f.write(f"- **Median R**2**: {statistics.median(r2s):.4f}\n")
                f.write(f"- **R**2 >= 0.99**: {sum(1 for r in r2s if r >= 0.99)}/{len(ok)}\n")
                f.write(f"- **R**2 >= 0.95**: {sum(1 for r in r2s if r >= 0.95)}/{len(ok)}\n")
                f.write(f"- **R**2 >= 0.90**: {sum(1 for r in r2s if r >= 0.90)}/{len(ok)}\n")
                f.write(f"- **Total wall time**: {total_time:.1f}s\n")

        logger.info(f"Saved Markdown: {md_path}")

    def _print_summary(self, results):
        """Print aggregate statistics to stdout."""
        ok = [r for r in results if r["status"] == "ok"]
        failed = len(results) - len(ok)

        logger.info("\n" + "=" * 64)
        logger.info("  FEYNMAN SWEEP SUMMARY")
        logger.info("=" * 64)
        logger.info(f"  Equations: {len(ok)} succeeded, {failed} failed, {len(results)} total")

        if not ok:
            logger.info("  No successful runs.")
            return

        import statistics
        maes = [r["test_mae"] for r in ok]
        r2s = [r["test_r2"] for r in ok]
        total_time = sum(r["wall_time"] for r in results)

        logger.info(f"  Mean MAE:   {statistics.mean(maes):.6f}")
        logger.info(f"  Median MAE: {statistics.median(maes):.6f}")
        logger.info(f"  Mean R**2:    {statistics.mean(r2s):.4f}")
        logger.info(f"  Median R**2:  {statistics.median(r2s):.4f}")
        logger.info(f"  R**2 >= 0.99: {sum(1 for r in r2s if r >= 0.99)}/{len(ok)}")
        logger.info(f"  R**2 >= 0.95: {sum(1 for r in r2s if r >= 0.95)}/{len(ok)}")
        logger.info(f"  R**2 >= 0.90: {sum(1 for r in r2s if r >= 0.90)}/{len(ok)}")
        logger.info(f"  Total time: {total_time:.1f}s")

        # Top 5 and Bottom 5
        by_r2 = sorted(ok, key=lambda x: x["test_r2"], reverse=True)
        logger.info("\n  Top 5 (by R**2):")
        for r in by_r2[:5]:
            logger.info(f"    {r['equation']:12s} R**2={r['test_r2']:.4f}  MAE={r['test_mae']:.6f}  [{r['desc']}]")
        logger.info("\n  Bottom 5 (by R**2):")
        for r in by_r2[-5:]:
            logger.info(f"    {r['equation']:12s} R**2={r['test_r2']:.4f}  MAE={r['test_mae']:.6f}  [{r['desc']}]")

        logger.info("=" * 64)
