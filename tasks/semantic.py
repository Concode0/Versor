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
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.projection import BladeSelector
from functional.activation import GeometricGELU
from functional.loss import GeometricMSELoss
from datasets.newsgroups import NewsgroupsClassificationDataset
from tasks.base import BaseTask
from torch.utils.data import DataLoader


class SemanticAutoEncoder(nn.Module):
    """Semantic Unbender. Autoencoder with geometric disentanglement.

    Encoder: CliffordLinear → GeometricGELU → CliffordLinear → RotorLayer
    Selector: BladeSelector (soft per-blade gate)
    Decoder: CliffordLinear → GeometricGELU → CliffordLinear → RotorLayer

    The rotor learns to rotate semantic content into grade-1 (vector)
    subspace. BladeSelector suppresses higher-grade noise.
    """

    def __init__(self, algebra, in_channels=8, hidden_channels=16):
        super().__init__()
        self.algebra = algebra

        self.encoder = nn.Sequential(
            CliffordLinear(algebra, in_channels, hidden_channels),
            GeometricGELU(algebra, channels=hidden_channels),
            CliffordLinear(algebra, hidden_channels, in_channels),
            RotorLayer(algebra, channels=in_channels)
        )

        self.selector = BladeSelector(algebra, channels=in_channels)

        self.decoder = nn.Sequential(
            CliffordLinear(algebra, in_channels, hidden_channels),
            GeometricGELU(algebra, channels=hidden_channels),
            CliffordLinear(algebra, hidden_channels, in_channels),
            RotorLayer(algebra, channels=in_channels)
        )

    def forward(self, x):
        latent_full = self.encoder(x)
        latent_proj = self.selector(latent_full)
        recon = self.decoder(latent_proj)
        return recon, latent_full, latent_proj


class SemanticTask(BaseTask):
    """Semantic Disentanglement on 20 Newsgroups.

    Tests whether a rotor can geometrically unbend the semantic manifold:
    push meaning into grade-1 (vector) subspace while reconstructing faithfully.

    Metrics: grade purity, reconstruction loss, noise robustness.
    """

    def __init__(self, cfg):
        self.embedding_dim = cfg.dataset.embedding_dim
        self.in_channels = cfg.model.get('in_channels', 8)
        super().__init__(cfg)

    def setup_algebra(self):
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        return SemanticAutoEncoder(
            self.algebra,
            in_channels=self.in_channels,
            hidden_channels=self.cfg.model.hidden_dim,
        )

    def setup_criterion(self):
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        cache_dir = self.cfg.dataset.get('cache_dir', './data/newsgroups')
        max_samples = self.cfg.dataset.get('max_samples', 2000) or 2000

        train_dataset = NewsgroupsClassificationDataset(
            self.algebra, embedding_dim=self.embedding_dim,
            in_channels=self.in_channels,
            split='train', max_samples=max_samples, cache_dir=cache_dir
        )
        val_dataset = NewsgroupsClassificationDataset(
            self.algebra, embedding_dim=self.embedding_dim,
            in_channels=self.in_channels,
            split='test', max_samples=max_samples, cache_dir=cache_dir
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.training.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.training.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def _grade_purity(self, latent):
        """Fraction of energy in grade-1 (vector) subspace."""
        grade1 = self.algebra.grade_projection(latent, 1)
        g1_energy = (grade1 ** 2).sum(dim=-1)
        total_energy = (latent ** 2).sum(dim=-1) + 1e-9
        return (g1_energy / total_energy).mean().item()

    def train_step(self, batch):
        data, _ = batch  # labels unused for autoencoder
        data = data.to(self.device)

        self.optimizer.zero_grad()
        recon, latent_full, latent_proj = self.model(data)

        # Reconstruction loss
        loss_recon = self.criterion(recon, data)

        # Grade purity regularization: penalize energy outside grade-1
        total_energy = (latent_full ** 2).sum(dim=-1).mean()
        grade1 = self.algebra.grade_projection(latent_full, 1)
        grade1_energy = (grade1 ** 2).sum(dim=-1).mean()
        loss_impurity = total_energy - grade1_energy

        # Blade selector sparsity
        loss_sparsity = self.model.selector.weights.abs().mean()

        loss = loss_recon + 0.1 * loss_impurity + 0.01 * loss_sparsity

        loss.backward()
        self.optimizer.step()

        purity = self._grade_purity(latent_full)

        return loss.item(), {
            "Recon": loss_recon.item(),
            "Purity": purity,
        }

    def evaluate(self, val_loader):
        self.model.eval()
        total_recon = 0
        total_purity = 0
        n_batches = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                recon, latent_full, _ = self.model(data)

                total_recon += self.criterion(recon, data).item()
                total_purity += self._grade_purity(latent_full)
                n_batches += 1

        avg_recon = total_recon / n_batches
        avg_purity = total_purity / n_batches

        print(f">>> Val Recon Loss: {avg_recon:.6f} | Grade Purity: {avg_purity:.4f}")

        # Noise robustness
        sample_data = next(iter(val_loader))[0].to(self.device)
        baseline_loss = self.criterion(self.model(sample_data)[0], sample_data).item()
        print(f"    Noise Robustness (baseline): {baseline_loss:.6f}")

        for noise_level in [0.05, 0.1, 0.2]:
            noisy = sample_data + torch.randn_like(sample_data) * noise_level
            denoised = self.model(noisy)[0]
            denoise_loss = self.criterion(denoised, sample_data).item()
            print(f"    Noise {noise_level:.2f}: Denoising Loss = {denoise_loss:.6f}")

        return avg_purity

    def visualize(self, val_loader):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(">>> matplotlib not found, skipping visualization.")
            return

        from core.visualizer import GeneralVisualizer
        viz = GeneralVisualizer(self.algebra)

        self.model.eval()
        sample_data = next(iter(val_loader))[0].to(self.device)

        with torch.no_grad():
            _, latent_full, _ = self.model(sample_data)

        # Grade energy heatmap
        viz.plot_grade_heatmap(latent_full, title="Semantic Latent Grade Energy")
        viz.save("semantic_grade_energy.png")

        # Component activation heatmap
        viz.plot_components_heatmap(latent_full, title="Semantic Components (After Unbending)")
        viz.save("semantic_components.png")

        # Bivector meaning map
        rotor_layer = self.model.encoder[-1]  # Last layer is RotorLayer
        self._plot_bivector_map(rotor_layer)

    def _plot_bivector_map(self, rotor_layer):
        """Visualize which rotation planes the rotor activates."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return

        # Average across channels
        w = rotor_layer.bivector_weights.detach().cpu().abs().mean(dim=0).numpy()
        indices = rotor_layer.bivector_indices.cpu().numpy()

        names = []
        for idx in indices:
            name = "e"
            temp = int(idx)
            i = 1
            while temp > 0:
                if temp & 1:
                    name += str(i)
                temp >>= 1
                i += 1
            names.append(name)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=names, y=w, hue=names, palette="magma", legend=False)
        plt.title("Bivector Meaning Map (Rotational Activity)")
        plt.xticks(rotation=45)
        plt.ylabel("Mean Rotor Coefficient Magnitude")
        plt.tight_layout()
        plt.savefig("semantic_bivector_map.png")
        plt.close()
        print(">>> Saved semantic_bivector_map.png")

    def run(self):
        """Train/val loop."""
        print(f">>> Starting Task: {self.cfg.name}")
        train_loader, val_loader = self.get_data()

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))

        best_purity = 0.0

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            total_purity = 0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                total_purity += logs['Purity']

            n_batches = len(train_loader)
            avg_loss = total_loss / n_batches
            avg_purity = total_purity / n_batches

            val_purity = self.evaluate(val_loader)
            self.scheduler.step(1.0 - val_purity)

            if val_purity > best_purity:
                best_purity = val_purity
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            logs = {
                'Loss': avg_loss,
                'Purity': avg_purity,
                'ValPur': val_purity,
                'LR': self.optimizer.param_groups[0]['lr']
            }
            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

        print(f">>> Training Complete. Best Val Grade Purity: {best_purity:.4f}")
        self.save_checkpoint(f"{self.cfg.name}_final.pt")

        self.model.eval()
        with torch.no_grad():
            self.visualize(val_loader)
