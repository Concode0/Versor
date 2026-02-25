# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Feynman Symbolic Regression Task.

Learns physics equations from the Feynman Lectures via Multi-Rotor GBN.
Sparsity loss on rotor weights discovers the symbolic structure.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig

from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from datasets.feynman import FEYNMAN_EQUATIONS, get_feynman_loaders
from models.feynman_net import FeynmanGBN
from layers.multi_rotor import MultiRotorLayer
from experiments.orthogonality import StrictOrthogonality, OrthogonalitySettings
from log import get_logger

logger = get_logger(__name__)


class FeynmanTask(BaseTask):
    """Feynman Physics Equation Regression.

    Learns one equation at a time using geometric algebra.
    The Multi-Rotor superposition is encouraged to be sparse, mirroring
    the parsimonious nature of symbolic expressions.

    Config keys (under dataset / model / training):
        dataset.equation        : one of FEYNMAN_EQUATIONS keys (e.g. "I.6.20")
        dataset.n_samples       : number of generated / loaded samples
        dataset.noise           : Gaussian noise std fraction
        dataset.cache_dir       : where to cache .pt files
        model.hidden_channels   : channel count C
        model.num_layers        : residual block count
        model.num_rotors        : K rotors per MultiRotorLayer
        model.embed_grade2      : enable grade-2 pairwise embedding
        model.use_decomposition : bivector decomposition in rotors
        training.sparsity_weight: weight for rotor sparsity loss
        orthogonality.*         : optional grade-orthogonality enforcement
    """

    def __init__(self, cfg: DictConfig):
        self.equation        = cfg.dataset.get("equation", "I.6.20")
        self.sparsity_weight = cfg.training.get("sparsity_weight", 0.01)

        # n_vars must be known before setup_model() is called inside super().__init__
        if self.equation not in FEYNMAN_EQUATIONS:
            raise ValueError(
                f"Unknown equation '{self.equation}'. "
                f"Available: {list(FEYNMAN_EQUATIONS)}"
            )
        self.n_vars = FEYNMAN_EQUATIONS[self.equation]["n_vars"]

        # Placeholders - filled in get_data()
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

        # Epoch counter (updated in run(), read by train_step/evaluate)
        self._epoch = 0

        super().__init__(cfg)

        # ------------------------------------------------------------------
        # Optional orthogonality enforcement (set up after super().__init__
        # so self.algebra and self.device are available)
        # ------------------------------------------------------------------
        ortho_cfg = cfg.get("orthogonality")
        if ortho_cfg is not None and ortho_cfg.get("enabled", False):
            tg = (list(ortho_cfg.get("target_grades"))
                  if ortho_cfg.get("target_grades") is not None else None)
            settings = OrthogonalitySettings(
                enabled=True,
                mode=str(ortho_cfg.get("mode", "loss")),
                weight=float(ortho_cfg.get("weight", 0.05)),
                target_grades=tg,
                monitor_interval=int(ortho_cfg.get("monitor_interval", 10)),
                coupling_warn_threshold=float(
                    ortho_cfg.get("coupling_warn_threshold", 0.3)
                ),
            )
            self.ortho = StrictOrthogonality(self.algebra, settings).to(self.device)
            self.ortho_warmup = int(ortho_cfg.get("warmup_epochs", 10))
        else:
            self.ortho = None
            self.ortho_warmup = 0

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def setup_algebra(self) -> CliffordAlgebra:
        """Use Cl(p, 0) as configured (default Cl(4,0))."""
        return CliffordAlgebra(
            p=self.cfg.algebra.p,
            q=self.cfg.algebra.get("q", 0),
            device=self.device,
        )

    def setup_model(self) -> FeynmanGBN:
        """Build FeynmanGBN with config parameters."""
        return FeynmanGBN(
            algebra=self.algebra,
            in_features=self.n_vars,
            channels=self.cfg.model.get("hidden_channels", 16),
            num_layers=self.cfg.model.get("num_layers", 3),
            num_rotors=self.cfg.model.get("num_rotors", 8),
            embed_grade2=self.cfg.model.get("embed_grade2", False),
            use_decomposition=self.cfg.model.get("use_decomposition", False),
        )

    def setup_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def get_data(self):
        """Load Feynman equation data and store normalisation stats."""
        train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std = \
            get_feynman_loaders(
                equation=self.equation,
                n_samples=self.cfg.dataset.get("n_samples", 10000),
                batch_size=self.cfg.training.batch_size,
                noise=self.cfg.dataset.get("noise", 0.0),
                cache_dir=self.cfg.dataset.get("cache_dir", "./data/feynman"),
                seed=self.cfg.training.get("seed", 42),
            )

        self.x_mean = x_mean.to(self.device)
        self.x_std  = x_std.to(self.device)
        self.y_mean = y_mean.to(self.device)
        self.y_std  = y_std.to(self.device)

        return train_loader, val_loader, test_loader

    def train_step(self, batch) -> tuple:
        """One optimisation step.

        Data in loaders is already normalised.  Computes MSE + sparsity + ortho.

        Args:
            batch: (x_norm [B,k], y_norm [B,1]) from DataLoader.

        Returns:
            loss (float), logs (dict with MSE, Sparsity, MAE_orig, Ortho)
        """
        x_norm, y_norm = batch
        x_norm = x_norm.to(self.device)
        y_norm = y_norm.to(self.device)

        self.optimizer.zero_grad()

        pred_norm = self.model(x_norm)   # [B, 1] - also sets model._last_hidden

        mse_loss = self.criterion(pred_norm, y_norm)
        sparsity = self.sparsity_weight * self.model.total_sparsity_loss()

        # Orthogonality penalty (uses detached _last_hidden - monitoring metric)
        ortho_loss = torch.tensor(0.0, device=self.device)
        if self.ortho is not None and hasattr(self.model, "_last_hidden"):
            eff_w = self.ortho.anneal_weight(
                self._epoch, self.ortho_warmup, self.epochs
            )
            ortho_loss = eff_w * self.ortho.parasitic_energy(self.model._last_hidden)

        loss = mse_loss + sparsity + ortho_loss
        loss.backward()
        self.optimizer.step()

        # MAE in original (un-normalised) units
        with torch.no_grad():
            pred_orig = pred_norm.detach() * self.y_std + self.y_mean
            y_orig    = y_norm.detach()    * self.y_std + self.y_mean
            mae_orig  = torch.abs(pred_orig - y_orig).mean().item()

        return loss.item(), {
            "MSE":      mse_loss.item(),
            "Sparsity": sparsity.item(),
            "MAE":      mae_orig,
            "Ortho":    ortho_loss.item(),
        }

    def evaluate(self, loader) -> tuple:
        """Compute MAE (original units) and R**2 on a loader.

        Also prints ASCII orthogonality diagnostics every
        ``ortho.settings.monitor_interval`` epochs (when ortho is enabled).

        Returns:
            (mean_mae, r2)
        """
        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for x_norm, y_norm in loader:
                x_norm = x_norm.to(self.device)
                y_norm = y_norm.to(self.device)
                pred_norm = self.model(x_norm)

                # Denormalise
                pred_orig   = pred_norm * self.y_std + self.y_mean
                target_orig = y_norm    * self.y_std + self.y_mean
                preds.append(pred_orig)
                targets.append(target_orig)

        preds   = torch.cat(preds,   dim=0)   # [N, 1]
        targets = torch.cat(targets, dim=0)

        mae = torch.abs(preds - targets).mean().item()

        ss_res = ((preds - targets) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        logger.info(f"MAE (orig units): {mae:.6f}  |  R**2: {r2:.4f}")

        # Periodic ortho diagnostics
        if (self.ortho is not None
                and self._epoch % self.ortho.settings.monitor_interval == 0):
            x_b, _ = next(iter(loader))
            with torch.no_grad():
                self.model(x_b.to(self.device))   # populates _last_hidden
            logger.debug(self.ortho.format_diagnostics(self.model._last_hidden))

        return mae, r2

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def variable_importance(self, x_sample: torch.Tensor) -> torch.Tensor:
        """Gradient-based variable importance: mean |dy_hat / dx_i| across a batch.

        Args:
            x_sample: [B, k] normalised scalar inputs (CPU or device).

        Returns:
            Tensor [k] - mean absolute gradient per input variable.
        """
        self.model.eval()
        x = x_sample.to(self.device).detach().requires_grad_(True)
        self.model(x).sum().backward()
        return x.grad.abs().mean(0).detach().cpu()   # [k]

    def symbolic_summary(self, loader):
        """Print an ASCII interpretability report to stdout.

        Covers:
          - Variable importance bar chart
          - Active rotation planes per layer
          - Top-5 output blade decomposition
          - Ortho diagnostics (if enabled)
        """
        self.model.eval()

        x_b, _ = next(iter(loader))
        x_b = x_b.to(self.device)

        eq_desc = FEYNMAN_EQUATIONS[self.equation]["desc"]
        bar_width = 20

        logger.info("\n" + "=" * 64)
        logger.info("=== SYMBOLIC HINTS ===")
        logger.info(f"Equation : {self.equation}  [{eq_desc}]")
        logger.info("=" * 64)

        # -- 1. Variable importance --
        imp    = self.variable_importance(x_b)           # [k]
        total  = imp.sum().item() + 1e-12
        imp_pct = (imp / total * 100).numpy()
        logger.info("\nVariable Importance  (|dy_hat / dx_i|):")
        for i, pct in enumerate(imp_pct):
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            logger.info(f"  x{i + 1}: [{bar}] {pct:.1f}%")

        # -- 2. Active rotation planes --
        analysis = self.model.get_rotor_analysis()
        logger.info("\nActive Rotation Planes  (threshold > 0.05):")
        for info in analysis:
            pairs   = sorted(zip(info["rotor_activity"], info["dominant_planes"]),
                             reverse=True)
            active  = [(a, p) for a, p in pairs if a > 0.05]
            if active:
                desc = ", ".join(f"{p}({a:.3f})" for a, p in active[:5])
            else:
                desc = "(all below threshold)"
            logger.info(f"  Layer {info['layer']}: {desc}")

        # -- 3. Top-5 output blade decomposition --
        blade_weights = self.model.get_output_blade_weights(self.algebra)
        top5 = sorted(blade_weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        max_bw = max(abs(w) for _, w in top5) + 1e-12
        logger.info("\nOutput Blade Decomposition  (top-5):")
        for name, w in top5:
            sign   = "+" if w >= 0 else "-"
            filled = int(bar_width * abs(w) / max_bw)
            bar    = "█" * filled + "░" * (bar_width - filled)
            logger.info(f"  {name:6s}: {sign}[{bar}] {w:+.4f}")

        # -- 4. Ortho diagnostics --
        if self.ortho is not None:
            logger.info("\nOrthogonality Report:")
            with torch.no_grad():
                self.model(x_b)   # refresh _last_hidden
            logger.debug(self.ortho.format_diagnostics(self.model._last_hidden))

        logger.info("=" * 64 + "\n")

    # ------------------------------------------------------------------
    # Visualization - 6-panel figure
    # ------------------------------------------------------------------

    def visualize(self, loader):
        """Save a 6-panel analysis figure.

        Panels:
          [0,0] Pred vs Actual scatter
          [0,1] Rotor activity heatmap  (rotors x layers)
          [0,2] Grade energy spectrum
          [1,0] Variable importance horizontal bars
          [1,1] Cross-grade coupling heatmap
          [1,2] Output blade decomposition

        Saves to ``feynman_{eq}_analysis.png``.
        """
        self.model.eval()

        # ---- collect predictions + last_hidden ----
        preds, targets = [], []
        first_x = None

        with torch.no_grad():
            for idx, (x_norm, y_norm) in enumerate(loader):
                x_norm = x_norm.to(self.device)
                pred   = self.model(x_norm) * self.y_std + self.y_mean
                truth  = y_norm.to(self.device) * self.y_std + self.y_mean
                preds.append(pred.cpu())
                targets.append(truth.cpu())
                if idx == 0:
                    first_x = x_norm.cpu()

        preds   = torch.cat(preds).squeeze(-1).numpy()
        targets = torch.cat(targets).squeeze(-1).numpy()
        last_hidden = self.model._last_hidden   # [B, C, dim] from last batch

        # ---- grade energy spectrum ----
        n_grades = self.algebra.n + 1
        dim      = self.algebra.dim
        grade_energies = []
        for g in range(n_grades):
            g_idx = [i for i in range(dim) if bin(i).count("1") == g]
            e = last_hidden[..., g_idx].pow(2).mean().item() if g_idx else 0.0
            grade_energies.append(e)
        total_e    = sum(grade_energies) + 1e-12
        grade_fracs = [e / total_e for e in grade_energies]

        # ---- rotor analysis ----
        rotor_analysis = self.model.get_rotor_analysis()

        # ---- variable importance ----
        var_imp = self.variable_importance(first_x) if first_x is not None else None

        # ---- coupling matrix ----
        ortho_viz = self.ortho
        if ortho_viz is None:
            ortho_viz = StrictOrthogonality(
                self.algebra, OrthogonalitySettings()
            ).to(self.device)
        coupling = (
            ortho_viz.cross_grade_coupling(last_hidden).detach().cpu().numpy()
        )

        # ---- output blade weights ----
        blade_weights = self.model.get_output_blade_weights(self.algebra)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            eq_desc = FEYNMAN_EQUATIONS[self.equation]["desc"]
            fig.suptitle(
                f"Feynman {self.equation}: {eq_desc}",
                fontsize=14, fontweight="bold"
            )

            # -- Panel 0,0: Pred vs Actual --
            ax = axes[0, 0]
            ax.scatter(targets, preds, alpha=0.4, s=8, label="samples")
            lo = min(targets.min(), preds.min())
            hi = max(targets.max(), preds.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect")
            ax.set_xlabel("Actual value")
            ax.set_ylabel("Predicted value")
            ax.set_title("Pred vs Actual")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # -- Panel 0,1: Rotor Activity heatmap --
            ax = axes[0, 1]
            n_layers = len(rotor_analysis)
            n_rotors = (
                len(rotor_analysis[0]["rotor_activity"]) if rotor_analysis else 0
            )
            if n_layers > 0 and n_rotors > 0:
                heatmap  = np.zeros((n_rotors, n_layers))
                dominant = [[""] * n_layers for _ in range(n_rotors)]
                for info in rotor_analysis:
                    l = info["layer"]
                    for k, act in enumerate(info["rotor_activity"]):
                        heatmap[k, l]  = act
                        dominant[k][l] = info["dominant_planes"][k]
                im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
                plt.colorbar(im, ax=ax, label="Mean |weight|")
                for k in range(n_rotors):
                    for l in range(n_layers):
                        if heatmap[k, l] > 0.05:
                            ax.text(l, k, dominant[k][l],
                                    ha="center", va="center",
                                    fontsize=7, color="white")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Rotor")
                ax.set_xticks(range(n_layers))
                ax.set_yticks(range(n_rotors))
            ax.set_title("Rotor Activity")

            # -- Panel 0,2: Grade Energy Spectrum --
            ax = axes[0, 2]
            grades   = list(range(n_grades))
            colors3  = plt.cm.tab10(np.linspace(0, 1, max(n_grades, 2)))
            ax.bar(grades, grade_fracs, color=colors3[:n_grades])
            ax.set_xlabel("Grade")
            ax.set_ylabel("Energy fraction")
            ax.set_title("Grade Energy Spectrum")
            ax.set_xticks(grades)
            ax.grid(True, alpha=0.3, axis="y")

            # -- Panel 1,0: Variable Importance --
            ax = axes[1, 0]
            if var_imp is not None:
                k_vars   = var_imp.shape[0]
                imp_np   = var_imp.numpy()
                ax.barh(range(k_vars), imp_np, color="steelblue")
                ax.set_yticks(range(k_vars))
                ax.set_yticklabels([f"x{i + 1}" for i in range(k_vars)])
                ax.set_xlabel("|dy_hat / dx_i|")
                ax.set_title("Variable Importance")
                ax.grid(True, alpha=0.3, axis="x")
            else:
                ax.set_visible(False)

            # -- Panel 1,1: Cross-Grade Coupling --
            ax = axes[1, 1]
            n_g = coupling.shape[0]
            im5 = ax.imshow(coupling, cmap="RdBu_r", vmin=-1, vmax=1,
                            aspect="auto")
            plt.colorbar(im5, ax=ax, label="Correlation")
            for i in range(n_g):
                for j in range(n_g):
                    ax.text(j, i, f"{coupling[i, j]:.2f}",
                            ha="center", va="center", fontsize=8)
            ax.set_xticks(range(n_g))
            ax.set_yticks(range(n_g))
            ax.set_xticklabels([f"G{g}" for g in range(n_g)])
            ax.set_yticklabels([f"G{g}" for g in range(n_g)])
            target_gs = (
                getattr(ortho_viz.settings, "target_grades", None)
                or list(range(n_g))
            )
            for g in (target_gs or []):
                if 0 <= g < n_g:
                    rect = plt.Rectangle(
                        (g - 0.5, g - 0.5), 1, 1,
                        linewidth=2, edgecolor="lime", facecolor="none"
                    )
                    ax.add_patch(rect)
            ax.set_title("Cross-Grade Coupling")

            # -- Panel 1,2: Output Blade Decomposition --
            ax = axes[1, 2]
            top10 = sorted(
                blade_weights.items(), key=lambda kv: abs(kv[1]), reverse=True
            )[:10]
            if top10:
                grade_color_map = {
                    0: "blue", 1: "orange", 2: "green", 3: "red", 4: "purple"
                }

                def _grade_from_name(name):
                    return 0 if name == "1" else len(name) - 1

                names6   = [b[0] for b in top10]
                weights6 = [b[1] for b in top10]
                colors6  = [
                    grade_color_map.get(_grade_from_name(n), "gray")
                    for n in names6
                ]
                ax.bar(range(len(names6)), weights6, color=colors6)
                ax.set_xticks(range(len(names6)))
                ax.set_xticklabels(names6, rotation=45, ha="right", fontsize=8)
                ax.axhline(0, color="black", lw=0.5)
                ax.set_ylabel("Weight")
                ax.set_title("Output Blade Decomposition")
                ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            fname = f"feynman_{self.equation.replace('.', '_')}_analysis.png"
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            logger.info(f"Saved analysis to {fname}")
            plt.close()

        except ImportError:
            logger.warning("Matplotlib not found. Skipping visualization.")

    # ------------------------------------------------------------------
    # Custom run loop
    # ------------------------------------------------------------------

    def run(self):
        """Full training / validation / test loop with checkpoint saving."""
        eq_desc = FEYNMAN_EQUATIONS[self.equation]["desc"]
        logger.info(f"Starting FeynmanTask: {self.equation}  [{eq_desc}]")

        train_loader, val_loader, test_loader = self.get_data()

        best_val_mae = float("inf")
        ckpt_path = f"{self.cfg.name}_best.pt"

        pbar = tqdm(range(self.epochs))

        for epoch in pbar:
            self._epoch = epoch + 1   # 1-indexed for anneal_weight / diagnostics

            # ---- Train ----
            self.model.train()
            total_mse = total_sparsity = total_mae = total_ortho = 0.0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_mse      += logs["MSE"]
                total_sparsity += logs["Sparsity"]
                total_mae      += logs["MAE"]
                total_ortho    += logs.get("Ortho", 0.0)

            n = len(train_loader)
            avg_mse      = total_mse      / n
            avg_sparsity = total_sparsity / n
            avg_mae      = total_mae      / n
            avg_ortho    = total_ortho    / n

            # ---- Validate ----
            self.model.eval()
            val_mae, val_r2 = self.evaluate(val_loader)

            self.scheduler.step(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                self.save_checkpoint(ckpt_path)

            desc = (
                f"MSE: {avg_mse:.4f} | Spar: {avg_sparsity:.4f} | "
                f"TrainMAE: {avg_mae:.4f} | ValMAE: {val_mae:.4f} | "
                f"ValR2: {val_r2:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            if self.ortho is not None:
                desc += f" | Ortho: {avg_ortho:.5f}"
            pbar.set_description(desc)

        logger.info(f"Training complete. Best val MAE: {best_val_mae:.6f}")

        # ---- Final test ----
        logger.info("Loading best checkpoint for test evaluation...")
        self.load_checkpoint(ckpt_path)
        self.model.eval()
        test_mae, test_r2 = self.evaluate(test_loader)
        logger.info(f"TEST  MAE: {test_mae:.6f}  |  R**2: {test_r2:.4f}")

        self.symbolic_summary(test_loader)
        self.visualize(test_loader)
