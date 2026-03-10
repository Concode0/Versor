# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm, hermitian_grade_spectrum
from tasks.base import BaseTask
from datalib.md17 import get_md17_loaders
from models.md17 import MD17ForceNet
from functional.loss import ConservativeLoss, HermitianGradeRegularization
from log import get_logger

logger = get_logger(__name__)


class MD17Task(BaseTask):
    """rMD17 / MD17 Molecular Dynamics Task.

    Predicts both energy and forces for molecular configurations.
    Multi-task learning with weighted loss combination including
    sparsity regularization and conservative force constraint.

    Uses revised MD17 (rMD17) by default with standard 1000/1000/Rest
    train/val/test split. Split sizes are configurable via dataset config.

    rMD17 molecules: aspirin, azobenzene, benzene, ethanol,
    malonaldehyde, naphthalene, paracetamol, salicylic_acid, toluene, uracil.
    Targets: Energy (kcal/mol), Forces (kcal/mol/A)
    """

    def __init__(self, cfg):
        self.molecule = cfg.dataset.get('molecule', 'aspirin')
        self.revised = cfg.dataset.get('revised', True)
        self.n_train = cfg.dataset.get('n_train', 1000)
        self.n_val = cfg.dataset.get('n_val', 1000)
        self.data_root = "./data/rMD17" if self.revised else "./data/MD17"
        self.loss_weights = cfg.training.get('loss_weights', {'energy': 1.0, 'force': 10.0})
        super().__init__(cfg)
        self.conservative_loss = ConservativeLoss()
        # Hermitian grade regularization for Cl(3,0,1): 5 grades
        target_spectrum = cfg.training.get('target_spectrum', [0.35, 0.30, 0.20, 0.10, 0.05])
        self.grade_reg = HermitianGradeRegularization(self.algebra, target_spectrum=target_spectrum)

    def setup_algebra(self):
        """Use Cl(3,0,1) PGA for SE(3) rigid-body motions."""
        return CliffordAlgebra(p=3, q=0, r=self.cfg.algebra.get("r", 1), device=self.device)

    def setup_model(self):
        """Build MD17ForceNet model with PGA motors, dynamic rotors, and RBF."""
        return MD17ForceNet(
            self.algebra,
            hidden_dim=self.cfg.model.hidden_dim,
            num_layers=self.cfg.model.layers,
            num_static_rotors=self.cfg.model.get('num_static_rotors', 8),
            num_dynamic_rotors=self.cfg.model.get('num_dynamic_rotors', 4),
            max_z=self.cfg.model.get('max_z', 100),
            num_rbf=self.cfg.model.get('num_rbf', 20),
            rbf_cutoff=self.cfg.model.get('rbf_cutoff', 5.0),
            use_decomposition=self.cfg.model.get('use_decomposition', False),
            decomp_k=self.cfg.model.get('decomp_k', 10),
            use_rotor_backend=self.cfg.model.get('use_rotor_backend', False),
            use_geo_square=self.cfg.model.get('use_geo_square', True),
            use_checkpoint=self.cfg.model.get('use_checkpoint', False),
        )

    def setup_criterion(self):
        """Multi-task loss: energy MSE + force MSE."""
        return nn.MSELoss()

    def get_data(self):
        """Load rMD17/MD17 dataset with normalization stats."""
        train_loader, val_loader, test_loader, e_mean, e_std, f_mean, f_std = get_md17_loaders(
            root=self.data_root,
            molecule=self.molecule,
            batch_size=self.cfg.training.batch_size,
            max_samples=self.cfg.dataset.get('samples', None),
            revised=self.revised,
            n_train=self.n_train,
            n_val=self.n_val,
        )

        self.energy_mean = torch.tensor(e_mean, device=self.device)
        self.energy_std = torch.tensor(e_std, device=self.device)
        self.force_mean = torch.tensor(f_mean, device=self.device)
        self.force_std = torch.tensor(f_std, device=self.device)

        return train_loader, val_loader, test_loader

    def train_step(self, batch):
        batch = batch.to(self.device)

        energy_target = batch.energy  # [B]
        force_target = batch.force    # [N, 3]

        energy_norm = (energy_target - self.energy_mean) / (self.energy_std + 1e-6)
        force_norm = (force_target - self.force_mean) / (self.force_std + 1e-6)

        pos = batch.pos.clone().requires_grad_(True)

        self.optimizer.zero_grad()
        # Use forward_energy to avoid retain_graph=True inside forward().
        # Forces are computed here via a single autograd.grad call; the graph
        # is freed normally when loss.backward() runs.
        energy_pred = self.model.forward_energy(batch.z, pos, batch.batch, batch.edge_index)
        force_pred = -torch.autograd.grad(
            outputs=energy_pred,
            inputs=pos,
            grad_outputs=torch.ones_like(energy_pred),
            create_graph=True,
        )[0]

        energy_loss = self.criterion(energy_pred, energy_norm)
        force_loss = self.criterion(force_pred, force_norm)
        sparsity_loss = self.model.total_sparsity_loss()

        w_conservative = self.loss_weights.get('conservative', 0.0)
        if w_conservative > 0:
            conservative_loss = self.conservative_loss(energy_pred, force_pred, pos)
        else:
            conservative_loss = torch.tensor(0.0, device=self.device)

        w_grade_reg = self.loss_weights.get('grade_reg', 0.0)
        if w_grade_reg > 0:
            latent = self.model.get_latent_features()
            if latent is not None:
                grade_reg_loss = self.grade_reg(latent)
            else:
                grade_reg_loss = torch.tensor(0.0, device=self.device)
        else:
            grade_reg_loss = torch.tensor(0.0, device=self.device)

        w_sparsity = self.loss_weights.get('sparsity', 0.0)
        loss = (self.loss_weights['energy'] * energy_loss +
                self.loss_weights['force'] * force_loss +
                w_sparsity * sparsity_loss +
                w_conservative * conservative_loss +
                w_grade_reg * grade_reg_loss)

        loss.backward()
        self.optimizer.step()

        energy_pred_denorm = energy_pred.detach() * self.energy_std + self.energy_mean
        force_pred_denorm = force_pred.detach() * self.force_std + self.force_mean
        energy_mae = torch.abs(energy_pred_denorm - energy_target).mean()
        force_mae = torch.abs(force_pred_denorm - force_target).mean()

        latent = self.model.get_latent_features()
        h_norm = hermitian_norm(self.algebra, latent).mean().item() if latent is not None else 0.0

        return loss.item(), {
            "Loss": loss.item(),
            "E_Loss": energy_loss.item(),
            "F_Loss": force_loss.item(),
            "Sparsity": sparsity_loss.item() if torch.is_tensor(sparsity_loss) else sparsity_loss,
            "E_MAE": energy_mae.item(),
            "F_MAE": force_mae.item(),
            "H_Norm": h_norm,
        }

    def evaluate(self, val_loader):
        self.model.eval()
        total_energy_mae = 0
        total_force_mae = 0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                energy_target = batch.energy
                force_target = batch.force

                energy_pred_norm, force_pred_norm = self.model(
                    batch.z, batch.pos, batch.batch, batch.edge_index
                )

                energy_pred = energy_pred_norm * self.energy_std + self.energy_mean
                force_pred = force_pred_norm * self.force_std + self.force_mean

                total_energy_mae += torch.abs(energy_pred - energy_target).sum().item()
                total_force_mae += torch.abs(force_pred - force_target).sum().item()
                count += energy_target.size(0)

        avg_energy_mae = total_energy_mae / count
        avg_force_mae = total_force_mae / (count * force_target.size(-2))  # Normalize by num_atoms

        return {
            'Energy_MAE': avg_energy_mae,
            'Force_MAE': avg_force_mae
        }

    def visualize(self, val_loader):
        self.model.eval()
        batch = next(iter(val_loader))
        batch = batch.to(self.device)

        energy_target = batch.energy
        force_target = batch.force

        with torch.no_grad():
            energy_pred_norm, force_pred_norm = self.model(
                batch.z, batch.pos, batch.batch, batch.edge_index
            )
            energy_pred = energy_pred_norm * self.energy_std + self.energy_mean
            force_pred = force_pred_norm * self.force_std + self.force_mean

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            e_true = energy_target.cpu().numpy()
            e_pred = energy_pred.cpu().numpy()
            axes[0].scatter(e_true, e_pred, alpha=0.5, label='Predictions')
            axes[0].set_xlabel("Actual Energy (kcal/mol)")
            axes[0].set_ylabel("Predicted Energy (kcal/mol)")
            min_val = min(e_true.min(), e_pred.min())
            max_val = max(e_true.max(), e_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
            axes[0].set_title(f"MD17 Energy Prediction ({self.molecule})")
            axes[0].grid(True)
            axes[0].legend()

            f_true = force_target.cpu().numpy().flatten()
            f_pred = force_pred.cpu().numpy().flatten()
            axes[1].hist(f_pred - f_true, bins=50, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel("Force Error (kcal/mol/A)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title(f"MD17 Force Error Distribution ({self.molecule})")
            axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("md17_prediction.png")
            logger.info("Saved visualization to md17_prediction.png")
            plt.close()
        except ImportError:
            logger.warning("Matplotlib not found. Skipping visualization.")

    def run(self):
        """Execute the main training loop."""
        variant = "rMD17" if self.revised else "MD17"
        logger.info(f"Starting Task: {variant} ({self.molecule})")
        train_loader, val_loader, test_loader = self.get_data()

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))

        best_val_metric = float('inf')

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            total_e_mae = 0
            total_f_mae = 0

            inner_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in inner_pbar:
                loss, logs = self.train_step(batch)
                total_loss += loss
                total_e_mae += logs['E_MAE']
                total_f_mae += logs['F_MAE']
                inner_pbar.set_postfix(E_MAE=f"{logs['E_MAE']:.4f}")

            avg_loss = total_loss / len(train_loader)
            avg_e_mae = total_e_mae / len(train_loader)
            avg_f_mae = total_f_mae / len(train_loader)

            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['Energy_MAE'] + val_metrics['Force_MAE']

            self.scheduler.step(val_loss)

            if val_loss < best_val_metric:
                best_val_metric = val_loss
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            logs = {
                'Loss': avg_loss,
                'E_MAE': avg_e_mae,
                'F_MAE': avg_f_mae,
                'Val_E_MAE': val_metrics['Energy_MAE'],
                'Val_F_MAE': val_metrics['Force_MAE'],
                'LR': self.optimizer.param_groups[0]['lr']
            }
            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

        logger.info(f"Training Complete. Best Val Metric: {best_val_metric:.4f}")

        logger.info("Loading best model for Test Set evaluation...")
        self.load_checkpoint(f"{self.cfg.name}_best.pt")

        test_metrics = self.evaluate(test_loader)
        logger.info(f"FINAL TEST Energy MAE: {test_metrics['Energy_MAE']:.4f}")
        logger.info(f"FINAL TEST Force MAE: {test_metrics['Force_MAE']:.4f}")

        self.visualize(test_loader)
