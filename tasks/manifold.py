import torch
import torch.nn as nn
import torch.optim as optim
import math
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.projection import BladeSelector
from functional.activation import GeometricGELU
from core.visualizer import GeneralVisualizer

class ManifoldAutoEncoder(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        
        # Deep Encoder: 3D -> Hidden -> Latent
        self.encoder = nn.Sequential(
            CliffordLinear(algebra, 1, 4), 
            GeometricGELU(algebra, channels=4), # Geometric Activation
            CliffordLinear(algebra, 4, 1),
            RotorLayer(algebra, channels=1)
        )
        
        # Learnable Blade Selection (Dimensionality Reduction)
        self.selector = BladeSelector(algebra, channels=1)
        
        # Decoder: Latent -> Hidden -> 3D
        self.decoder = nn.Sequential(
            CliffordLinear(algebra, 1, 4),
            GeometricGELU(algebra, channels=4), # Geometric Activation
            CliffordLinear(algebra, 4, 1),
            RotorLayer(algebra, channels=1)
        )

    def forward(self, x):
        # 1. Encode
        latent_full = self.encoder(x)
        
        # 2. Bottleneck: Soft Projection
        latent_proj = self.selector(latent_full)
        
        # 3. Decode
        recon = self.decoder(latent_proj)
        
        return recon, latent_full, latent_proj

def generate_figure8_data(num_samples=1000, noise=0.05):
    t = torch.linspace(0, 2*math.pi, num_samples)
    x = torch.sin(t)
    y = torch.sin(t) * torch.cos(t)
    z = 0.5 * x * y
    points = torch.stack([x, y, z], dim=1)
    points += torch.randn_like(points) * noise
    return points

def tensor_to_multivector(points, algebra):
    batch_size = points.shape[0]
    mv_data = torch.zeros(batch_size, 1, algebra.dim)
    mv_data[:, 0, 1] = points[:, 0]
    mv_data[:, 0, 2] = points[:, 1]
    mv_data[:, 0, 4] = points[:, 2]
    return mv_data

from functional.loss import GeometricMSELoss, SubspaceLoss

def run_manifold_task(epochs=800, lr=0.02):
    device = 'cpu'
    algebra = CliffordAlgebra(3, 0, device=device)
    
    print(">>> [Task] Manifold Restoration (Unbending Figure-8)...")
    points_3d = generate_figure8_data(1000)
    data = tensor_to_multivector(points_3d, algebra).to(device)
    
    model = ManifoldAutoEncoder(algebra).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Losses
    recon_criterion = GeometricMSELoss(algebra)
    # Penalize e3 (index 4) to force 2D alignment
    manifold_reg = SubspaceLoss(algebra, exclude_indices=[4]) 
    
    pbar = tqdm.tqdm(range(epochs))
    
    for epoch in pbar:
        optimizer.zero_grad()
        recon, latent, latent_proj = model(data)
        
        loss = recon_criterion(recon, data)
        z_energy = manifold_reg(latent)
        sparsity = model.selector.weights.abs().mean()
        
        total_loss = loss + 1.0 * z_energy + 0.01 * sparsity
        
        total_loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f} | Z: {z_energy.item():.4f}")
        
    print("\n>>> Evaluation & Visualization...")
    model.eval()
    
    viz = GeneralVisualizer(algebra)
    with torch.no_grad():
        recon, latent, latent_proj = model(data)
        
        # 1. 3D Manifold (Original vs Latent vs Recon)
        # We can reuse plot_3d but maybe we want subplots.
        # GeneralVisualizer.plot_3d returns a figure. 
        # Let's save them individually for clarity.
        
        viz.plot_3d(data, title="Original (Bended)")
        viz.save("manifold_original.png")
        
        viz.plot_3d(latent, title="Latent Space (Unbent)")
        viz.save("manifold_latent.png")
        
        # 2. PCA of Latent Space
        viz.plot_latent_projection(latent, method='pca', title="Latent PCA")
        viz.save("manifold_pca.png")
        
        # 3. Grade Energy Heatmap
        viz.plot_grade_heatmap(latent, title="Latent Grade Energy")
        viz.save("manifold_grades.png")
        
        # 4. Component Heatmap (Top 100 samples)
        viz.plot_components_heatmap(latent, title="Latent Components")
        viz.save("manifold_components.png")

        print("Saved visualizations: manifold_*.png")
