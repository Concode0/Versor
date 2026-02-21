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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from core.algebra import CliffordAlgebra

class GeneralVisualizer:
    """Visualization toolkit. Visualization tools for geometric algebra.

    Plots 3D projections, latent spaces, and energy heatmaps.
    """

    def __init__(self, algebra: CliffordAlgebra):
        """Sets up the visualizer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
        """
        self.algebra = algebra
        self.basis_names = self._generate_basis_names()
        
        # Set Seaborn theme
        sns.set_theme(style="whitegrid")
        self.img_counter = 1

    def _generate_basis_names(self):
        """Generates names for basis blades. e1, e2, e12..."""
        names = []
        for i in range(self.algebra.dim):
            if i == 0:
                names.append("1")
                continue
            
            name = "e"
            temp = i
            idx = 1
            while temp > 0:
                if temp & 1:
                    name += str(idx)
                temp >>= 1
                idx += 1
            names.append(name)
        return names

    def save(self, filename=None):
        """Save the current figure to disk.

        Args:
            filename (str, optional): Output path. Defaults to auto-incrementing.
        """
        if filename is None:
            filename = f"viz_{self.img_counter}.png"
            self.img_counter += 1
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()

    def plot_3d(self, data: torch.Tensor, dims=(1, 2, 4), title="3D Projection"):
        """Plot a 3D scatter projection of multivector components.

        Args:
            data (torch.Tensor): Multivectors.
            dims (tuple, optional): Components to map to X, Y, Z.
            title (str, optional): Title.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim)
            
        x = data[:, dims[0]].cpu().numpy()
        y = data[:, dims[1]].cpu().numpy()
        z = data[:, dims[2]].cpu().numpy()
        
        fig = plt.figure(figsize=(12, 10), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by phase angle in XY plane
        c = np.arctan2(y, x)
        
        ax.scatter(x, y, z, c=c, cmap='twilight', s=50, alpha=0.6, edgecolors='w', linewidth=0.3)
        
        ax.set_xlabel(self.basis_names[dims[0]], fontsize=12)
        ax.set_ylabel(self.basis_names[dims[1]], fontsize=12)
        ax.set_zlabel(self.basis_names[dims[2]], fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Minimalist style
        ax.grid(False) 
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=30, azim=45)
        
        return fig

    def plot_latent_projection(self, data: torch.Tensor, method='pca', title=None):
        """Dimensionality reduction for visualization.

        Args:
            data (torch.Tensor): Input data.
            method (str): 'pca' or 'tsne'. Defaults to 'pca'.
            title (str, optional): Title.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim) # Flatten batch
            
        X = data.detach().cpu().numpy()
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            title = title or "Latent Space (PCA)"
            xlabel, ylabel = "PC 1", "PC 2"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
            title = title or "Latent Space (t-SNE)"
            xlabel, ylabel = "t-SNE Dim 1", "t-SNE Dim 2"
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        X_embedded = reducer.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], alpha=0.6, edgecolor=None)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return plt.gcf()

    def plot_grade_heatmap(self, data: torch.Tensor, title="Grade Energy Distribution"):
        """Plot per-grade energy distribution.

        Args:
            data (torch.Tensor): Input multivectors.
            title (str): Title.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim)
            
        energy_per_grade = []
        grade_labels = []
        
        for k in range(self.algebra.n + 1):
            mask = self.algebra.grade_projection(torch.ones(1, self.algebra.dim, device=self.algebra.device), k).bool()
            mask = mask.view(-1)
            
            if not mask.any():
                continue
                
            comps = data[:, mask]
            # Mean energy of this grade across the batch
            energy = (comps ** 2).sum(dim=1).mean().item()
            energy_per_grade.append(energy)
            grade_labels.append(f"Grade {k}")
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x=grade_labels, y=energy_per_grade, hue=grade_labels, palette="viridis", legend=False)
        plt.title(title)
        plt.ylabel("Average Energy")
        plt.yscale('log') # Use log scale for dynamic range
        return plt.gcf()

    def plot_components_heatmap(self, data: torch.Tensor, title="Component Activation Heatmap"):
        """Plot basis blade activation magnitudes.

        Args:
            data (torch.Tensor): Input data.
            title (str): Title.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim)
            
        # Subset for visibility
        if data.shape[0] > 100:
            data = data[:100]
            
        X = data.abs().detach().cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(X.T, yticklabels=self.basis_names, cmap="magma", cbar_kws={'label': 'Magnitude'})
        plt.title(title)
        plt.xlabel("Sample Index")
        return plt.gcf()