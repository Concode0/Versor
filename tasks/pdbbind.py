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
from datasets.pdbbind import get_pdbbind_loaders
from models.pdbbind_net import PDBBindNet
from functional.loss import HermitianGradeRegularization


class PDBBindTask(BaseTask):
    """PDBbind Protein-Ligand Binding Affinity Task.

    Predicts binding affinity (pKd/pKi) from 3D protein-ligand complex
    structures using dual-graph encoding with geometric cross-attention.

    Uses Cl(3,0) for 3D Euclidean molecular geometry.
    """

    def __init__(self, cfg):
        self.data_root = "./data/PDBbind"
        self.loss_weights = cfg.training.get('loss_weights', {'affinity': 1.0, 'sparsity': 0.01})
        super().__init__(cfg)
        # Hermitian grade regularization: balanced for Cl(3,0)
        target_spectrum = cfg.training.get('target_spectrum', [0.25, 0.25, 0.25, 0.25])
        self.grade_reg = HermitianGradeRegularization(self.algebra, target_spectrum=target_spectrum)

    def setup_algebra(self):
        return CliffordAlgebra(p=3, q=0, device=self.device)

    def setup_model(self):
        return PDBBindNet(
            self.algebra,
            protein_hidden_dim=self.cfg.model.get('protein_hidden_dim', 64),
            ligand_hidden_dim=self.cfg.model.get('ligand_hidden_dim', 32),
            interaction_dim=self.cfg.model.get('interaction_dim', 64),
            num_protein_layers=self.cfg.model.get('layers', 4),
            num_ligand_layers=self.cfg.model.get('ligand_layers', 3),
            num_rotors=self.cfg.model.get('num_rotors', 8),
            use_decomposition=self.cfg.model.get('use_decomposition', True),
            decomp_k=self.cfg.model.get('decomp_k', 10),
            use_rotor_backend=self.cfg.model.get('use_rotor_backend', True)
        )

    def setup_criterion(self):
        return nn.MSELoss()

    def get_data(self):
        train_loader, val_loader, test_loader, aff_mean, aff_std = get_pdbbind_loaders(
            root=self.data_root,
            version=self.cfg.dataset.get('version', 'refined'),
            batch_size=self.cfg.training.batch_size,
            max_samples=self.cfg.dataset.get('samples', None),
            pocket_cutoff=self.cfg.dataset.get('cutoff', 10.0),
            max_protein_atoms=self.cfg.dataset.get('max_protein_atoms', 1000),
            max_ligand_atoms=self.cfg.dataset.get('max_ligand_atoms', 100)
        )

        self.aff_mean = torch.tensor(aff_mean, device=self.device)
        self.aff_std = torch.tensor(aff_std, device=self.device)

        return train_loader, val_loader, test_loader

    def _to_device(self, batch):
        """Move batch dict to device."""
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def train_step(self, batch):
        batch = self._to_device(batch)

        affinity_target = batch['affinity']
        affinity_norm = (affinity_target - self.aff_mean) / (self.aff_std + 1e-6)

        batch_size = affinity_target.size(0)

        self.optimizer.zero_grad()
        affinity_pred = self.model(
            batch['protein_pos'], batch['protein_z'], batch['protein_aa'],
            batch['protein_edge_index'], batch['protein_batch'],
            batch['ligand_pos'], batch['ligand_z'],
            batch['ligand_edge_index'], batch['ligand_batch'],
            batch_size
        )

        loss_affinity = self.criterion(affinity_pred, affinity_norm)
        loss_sparsity = self.model.total_sparsity_loss()

        # Grade regularization on latent features
        w_grade_reg = self.loss_weights.get('grade_reg', 0.0)
        if w_grade_reg > 0:
            prot_feat, lig_feat = self.model.get_latent_features()
            if prot_feat is not None:
                grade_reg_loss = (self.grade_reg(prot_feat) + self.grade_reg(lig_feat)) / 2
            else:
                grade_reg_loss = torch.tensor(0.0, device=self.device)
        else:
            grade_reg_loss = torch.tensor(0.0, device=self.device)

        w = self.loss_weights
        loss = (w.get('affinity', 1.0) * loss_affinity +
                w.get('sparsity', 0.01) * loss_sparsity +
                w_grade_reg * grade_reg_loss)

        loss.backward()
        self.optimizer.step()

        # Denormalize for metrics
        aff_pred_denorm = affinity_pred.detach() * self.aff_std + self.aff_mean
        mae = torch.abs(aff_pred_denorm - affinity_target).mean()

        # Pearson correlation
        if affinity_target.size(0) > 2:
            stacked = torch.stack([aff_pred_denorm.flatten(), affinity_target.flatten()])
            pearson = torch.corrcoef(stacked)[0, 1].item()
        else:
            pearson = 0.0

        # Compute Hermitian norm of latent features
        prot_feat, lig_feat = self.model.get_latent_features()
        h_norm = 0.0
        if prot_feat is not None:
            h_norm = hermitian_norm(self.algebra, prot_feat).mean().item()

        return loss.item(), {
            "Loss": loss.item(),
            "MAE": mae.item(),
            "Pearson": pearson if not (pearson != pearson) else 0.0,  # NaN check
            "H_Norm": h_norm,
        }

    def evaluate(self, val_loader):
        self.model.eval()
        total_mae = 0.0
        all_pred = []
        all_target = []
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                affinity_target = batch['affinity']
                batch_size = affinity_target.size(0)

                aff_pred_norm = self.model(
                    batch['protein_pos'], batch['protein_z'], batch['protein_aa'],
                    batch['protein_edge_index'], batch['protein_batch'],
                    batch['ligand_pos'], batch['ligand_z'],
                    batch['ligand_edge_index'], batch['ligand_batch'],
                    batch_size
                )

                aff_pred = aff_pred_norm * self.aff_std + self.aff_mean
                total_mae += torch.abs(aff_pred - affinity_target).sum().item()
                all_pred.append(aff_pred)
                all_target.append(affinity_target)
                count += batch_size

        avg_mae = total_mae / max(count, 1)

        all_pred = torch.cat(all_pred)
        all_target = torch.cat(all_target)
        if all_pred.size(0) > 2:
            pearson = torch.corrcoef(torch.stack([all_pred, all_target]))[0, 1].item()
        else:
            pearson = 0.0

        return {
            'MAE': avg_mae,
            'Pearson': pearson if not (pearson != pearson) else 0.0,
        }

    def visualize(self, val_loader):
        self.model.eval()
        batch = next(iter(val_loader))
        batch = self._to_device(batch)
        batch_size = batch['affinity'].size(0)

        with torch.no_grad():
            aff_pred_norm = self.model(
                batch['protein_pos'], batch['protein_z'], batch['protein_aa'],
                batch['protein_edge_index'], batch['protein_batch'],
                batch['ligand_pos'], batch['ligand_z'],
                batch['ligand_edge_index'], batch['ligand_batch'],
                batch_size
            )
            aff_pred = aff_pred_norm * self.aff_std + self.aff_mean

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            true = batch['affinity'].cpu().numpy()
            pred = aff_pred.cpu().numpy()

            ax.scatter(true, pred, alpha=0.6)
            mn, mx = min(true.min(), pred.min()), max(true.max(), pred.max())
            ax.plot([mn, mx], [mn, mx], 'r--', label='Perfect')
            ax.set_xlabel("Actual pKd")
            ax.set_ylabel("Predicted pKd")
            ax.set_title("PDBbind Binding Affinity Prediction")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("pdbbind_prediction.png")
            print(">>> Saved visualization to pdbbind_prediction.png")
            plt.close()
        except ImportError:
            print("Matplotlib not found. Skipping visualization.")

    def run(self):
        print(f">>> Starting Task: PDBbind")
        train_loader, val_loader, test_loader = self.get_data()

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))
        best_val_metric = float('inf')

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            total_mae = 0
            n_batches = 0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                total_mae += logs['MAE']
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_mae = total_mae / max(n_batches, 1)

            val_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_metrics['MAE'])

            if val_metrics['MAE'] < best_val_metric:
                best_val_metric = val_metrics['MAE']
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            desc = f"Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f} | Val_MAE: {val_metrics['MAE']:.4f} | Val_R: {val_metrics['Pearson']:.4f}"
            pbar.set_description(desc)

        print(f">>> Training Complete. Best Val MAE: {best_val_metric:.4f}")
        self.load_checkpoint(f"{self.cfg.name}_best.pt")

        test_metrics = self.evaluate(test_loader)
        print(f">>> FINAL TEST MAE: {test_metrics['MAE']:.4f}")
        print(f">>> FINAL TEST Pearson: {test_metrics['Pearson']:.4f}")

        self.visualize(test_loader)
