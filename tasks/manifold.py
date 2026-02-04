import torch
import torch.nn as nn
import math
import numpy as np
from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.projection import BladeSelector
from functional.activation import GeometricGELU
from core.visualizer import GeneralVisualizer
from functional.loss import GeometricMSELoss, SubspaceLoss
from tasks.base import BaseTask

class ManifoldAutoEncoder(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        
        # Deep Encoder: 3D -> Hidden -> Latent
        self.encoder = nn.Sequential(
            CliffordLinear(algebra, 1, 4), 
            GeometricGELU(algebra, channels=4),
            CliffordLinear(algebra, 4, 1),
            RotorLayer(algebra, channels=1)
        )
        
        # Learnable Blade Selection (Dimensionality Reduction)
        self.selector = BladeSelector(algebra, channels=1)
        
        # Decoder: Latent -> Hidden -> 3D
        self.decoder = nn.Sequential(
            CliffordLinear(algebra, 1, 4),
            GeometricGELU(algebra, channels=4),
            CliffordLinear(algebra, 4, 1),
            RotorLayer(algebra, channels=1)
        )

    def forward(self, x):
        latent_full = self.encoder(x)
        latent_proj = self.selector(latent_full)
        recon = self.decoder(latent_proj)
        return recon, latent_full, latent_proj

class ManifoldTask(BaseTask):
    def setup_algebra(self):
        return CliffordAlgebra(3, 0, device=self.device)

    def setup_model(self):
        return ManifoldAutoEncoder(self.algebra)

    def setup_criterion(self):
        # We handle multiple losses in train_step, but return primary here
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        num_samples = 1000
        noise = 0.05
        t = torch.linspace(0, 2*math.pi, num_samples)
        x = torch.sin(t)
        y = torch.sin(t) * torch.cos(t)
        z = 0.5 * x * y
        points = torch.stack([x, y, z], dim=1)
        points += torch.randn_like(points) * noise
        
        # Convert to Multivector
        mv_data = torch.zeros(num_samples, 1, self.algebra.dim)
        mv_data[:, 0, 1] = points[:, 0]
        mv_data[:, 0, 2] = points[:, 1]
        mv_data[:, 0, 4] = points[:, 2]
        return mv_data

    def train_step(self, data):
        self.optimizer.zero_grad()
        recon, latent, latent_proj = self.model(data)
        
        # Losses
        loss_recon = self.criterion(recon, data)
        
        # Manifold Regularization: Minimize e3 (index 4) energy in latent
        manifold_reg = SubspaceLoss(self.algebra, exclude_indices=[4]) 
        z_energy = manifold_reg(latent)
        
        # Sparsity
        sparsity = self.model.selector.weights.abs().mean()
        
        total_loss = loss_recon + 1.0 * z_energy + 0.01 * sparsity
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), {"Recon": loss_recon.item(), "Z": z_energy.item()}

    def evaluate(self, data):
        recon, latent, latent_proj = self.model(data)
        loss = self.criterion(recon, data)
        print(f"Final Reconstruction Loss: {loss.item():.6f}")

    def visualize(self, data):
        recon, latent, latent_proj = self.model(data)
        viz = GeneralVisualizer(self.algebra)
        
        # 1. 3D Manifold
        viz.plot_3d(data, title="Original (Bended)")
        viz.save("manifold_original.png")
        
        viz.plot_3d(latent, title="Latent Space (Unbent)")
        viz.save("manifold_latent.png")
        
        # 2. Latent Analysis
        viz.plot_latent_projection(latent, method='pca', title="Latent PCA")
        viz.save("manifold_pca.png")
        
        viz.plot_grade_heatmap(latent, title="Latent Grade Energy")
        viz.save("manifold_grades.png")
        
        viz.plot_components_heatmap(latent, title="Latent Components")
        viz.save("manifold_components.png")
        print("Saved visualizations: manifold_*.png")

def run_manifold_task(epochs=800, lr=0.02):
    task = ManifoldTask(epochs=epochs, lr=lr, device='cpu')
    task.run()
