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
from core.cga import ConformalAlgebra
from core.metric import hermitian_norm, hermitian_grade_spectrum
from tasks.base import BaseTask
from datasets.abc import get_abc_loaders
from models.cad_net import CADAutoEncoder
from functional.loss import ChamferDistance, HermitianGradeRegularization


class ABCTask(BaseTask):
    """ABC Dataset CAD Parametric Model Task.

    Autoencoder for CAD model reconstruction using conformal geometric algebra.
    Supports point cloud reconstruction and primitive parameter regression.

    Uses Cl(4,1) conformal space for unified primitive representation.
    """

    def __init__(self, cfg):
        self.data_root = "./data/ABC"
        self.task_type = cfg.dataset.get('task', 'reconstruction')
        self.num_points = cfg.dataset.get('num_points', 2048)
        self.loss_weights = cfg.training.get('loss_weights', {
            'reconstruction': 1.0, 'sparsity': 0.01
        })
        super().__init__(cfg)
        # Hermitian grade regularization: vector+bivector for conformal Cl(4,1)
        target_spectrum = cfg.training.get('target_spectrum', [0.1, 0.25, 0.3, 0.2, 0.1, 0.05])
        self.grade_reg = HermitianGradeRegularization(self.algebra, target_spectrum=target_spectrum)

    def setup_algebra(self):
        """Cl(4,1) conformal signature for 3D CGA."""
        return CliffordAlgebra(p=4, q=1, device=self.device)

    def setup_model(self):
        cga = ConformalAlgebra(euclidean_dim=3, device=self.device)

        return CADAutoEncoder(
            self.algebra,
            cga,
            latent_dim=self.cfg.model.get('latent_dim', 128),
            num_rotors=self.cfg.model.get('num_rotors', 16),
            output_points=self.num_points,
            use_decomposition=self.cfg.model.get('use_decomposition', True),
            decomp_k=self.cfg.model.get('decomp_k', 10),
            use_rotor_backend=self.cfg.model.get('use_rotor_backend', True),
            decoder_type=self.task_type
        )

    def setup_criterion(self):
        if self.task_type == 'reconstruction':
            return ChamferDistance()
        else:
            return nn.MSELoss()

    def get_data(self):
        train_loader, val_loader, test_loader = get_abc_loaders(
            root=self.data_root,
            task=self.task_type,
            num_points=self.num_points,
            augment=self.cfg.dataset.get('augment', True),
            batch_size=self.cfg.training.batch_size,
            max_samples=self.cfg.dataset.get('samples', None)
        )
        return train_loader, val_loader, test_loader

    def train_step(self, batch):
        points = batch['points'].to(self.device)

        self.optimizer.zero_grad()

        if self.task_type == 'reconstruction':
            recon_points = self.model(points)
            loss_recon = self.criterion(recon_points, points)
            loss_sparsity = self.model.total_sparsity_loss()

            # Grade regularization on latent features
            w_grade_reg = self.loss_weights.get('grade_reg', 0.0)
            if w_grade_reg > 0:
                latent = self.model.get_latent_features()
                if latent is not None:
                    grade_reg_loss = self.grade_reg(latent)
                else:
                    grade_reg_loss = torch.tensor(0.0, device=self.device)
            else:
                grade_reg_loss = torch.tensor(0.0, device=self.device)

            w = self.loss_weights
            loss = (w.get('reconstruction', 1.0) * loss_recon +
                    w.get('sparsity', 0.01) * loss_sparsity +
                    w_grade_reg * grade_reg_loss)

            loss.backward()
            self.optimizer.step()

            # Compute Hermitian norm of latent features
            latent = self.model.get_latent_features()
            h_norm = hermitian_norm(self.algebra, latent).mean().item() if latent is not None else 0.0

            return loss.item(), {
                "Loss": loss.item(),
                "Chamfer": loss_recon.item(),
                "H_Norm": h_norm,
            }
        else:
            target_params = batch['primitive_params'].to(self.device)
            types, params = self.model(points)
            loss = self.criterion(params.view(params.size(0), -1),
                                  target_params.view(target_params.size(0), -1).expand_as(params.view(params.size(0), -1)))

            loss.backward()
            self.optimizer.step()

            return loss.item(), {"Loss": loss.item()}

    def evaluate(self, val_loader):
        self.model.eval()
        total_metric = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(self.device)
                batch_size = points.size(0)

                if self.task_type == 'reconstruction':
                    recon = self.model(points)
                    chamfer = self.criterion(recon, points)
                    total_metric += chamfer.item() * batch_size
                else:
                    target_params = batch['primitive_params'].to(self.device)
                    types, params = self.model(points)
                    loss = self.criterion(params.view(batch_size, -1),
                                          target_params.view(batch_size, -1))
                    total_metric += loss.item() * batch_size

                count += batch_size

        metric_name = 'Chamfer' if self.task_type == 'reconstruction' else 'MSE'
        return {metric_name: total_metric / max(count, 1)}

    def visualize(self, val_loader):
        if self.task_type != 'reconstruction':
            print(">>> Visualization only supported for reconstruction task")
            return

        self.model.eval()
        batch = next(iter(val_loader))
        points = batch['points'].to(self.device)

        with torch.no_grad():
            recon = self.model(points)

        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(14, 6))

            # Input point cloud
            ax1 = fig.add_subplot(121, projection='3d')
            pts = points[0].cpu().numpy()
            ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.5)
            ax1.set_title("Input Point Cloud")

            # Reconstructed point cloud
            ax2 = fig.add_subplot(122, projection='3d')
            rpts = recon[0].cpu().numpy()
            ax2.scatter(rpts[:, 0], rpts[:, 1], rpts[:, 2], s=1, alpha=0.5, c='orange')
            ax2.set_title("Reconstructed Point Cloud")

            plt.suptitle("ABC CAD Reconstruction")
            plt.tight_layout()
            plt.savefig("abc_reconstruction.png")
            print(">>> Saved visualization to abc_reconstruction.png")
            plt.close()
        except ImportError:
            print("Matplotlib not found. Skipping visualization.")

    def run(self):
        print(f">>> Starting Task: ABC Dataset ({self.task_type}, {self.num_points} points)")
        train_loader, val_loader, test_loader = self.get_data()

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))
        best_val = float('inf')
        metric_name = 'Chamfer' if self.task_type == 'reconstruction' else 'MSE'

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            total_metric = 0
            n_batches = 0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                total_metric += logs.get('Chamfer', logs.get('Loss', 0))
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_metric = total_metric / max(n_batches, 1)

            val_metrics = self.evaluate(val_loader)
            val_score = val_metrics[metric_name]
            self.scheduler.step(val_score)

            if val_score < best_val:
                best_val = val_score
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            desc = f"Loss: {avg_loss:.6f} | {metric_name}: {avg_metric:.6f} | Val_{metric_name}: {val_score:.6f}"
            pbar.set_description(desc)

        print(f">>> Training Complete. Best Val {metric_name}: {best_val:.6f}")
        self.load_checkpoint(f"{self.cfg.name}_best.pt")

        test_metrics = self.evaluate(test_loader)
        print(f">>> FINAL TEST {metric_name}: {test_metrics[metric_name]:.6f}")

        self.visualize(test_loader)
