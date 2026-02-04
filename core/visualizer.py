import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from core.algebra import CliffordAlgebra

class GeneralVisualizer:
    def __init__(self, algebra: CliffordAlgebra):
        self.algebra = algebra
        self.basis_names = self._generate_basis_names()
        
        # Configure plotting style
        sns.set_theme(style="whitegrid")
        self.img_counter = 1

    def _generate_basis_names(self):
        # Generate names based on bits
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
        if filename is None:
            filename = f"viz_{self.img_counter}.png"
            self.img_counter += 1
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()

    def plot_3d(self, data: torch.Tensor, dims=(1, 2, 4), title="3D Projection"):
        """
        Scatter plot of 3 specific components of the multivector.
        Default dims=(1, 2, 4) corresponds to e1, e2, e3.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim)
            
        x = data[:, dims[0]].cpu().numpy()
        y = data[:, dims[1]].cpu().numpy()
        z = data[:, dims[2]].cpu().numpy()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by angle in xy plane
        c = np.arctan2(y, x)
        
        ax.scatter(x, y, z, c=c, cmap='hsv', s=5, alpha=0.7)
        ax.set_xlabel(self.basis_names[dims[0]])
        ax.set_ylabel(self.basis_names[dims[1]])
        ax.set_zlabel(self.basis_names[dims[2]])
        ax.set_title(title)
        
        # Equal aspect ratio hack
        # max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        # mid_x = (x.max()+x.min()) * 0.5
        # mid_y = (y.max()+y.min()) * 0.5
        # mid_z = (z.max()+z.min()) * 0.5
        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return fig

    def plot_latent_projection(self, data: torch.Tensor, method='pca', title=None):
        """
        Project high-dimensional multivector coefficients to 2D using PCA or t-SNE.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim) # Flatten batch
            
        X = data.detach().cpu().numpy()
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            title = title or "Latent Space (PCA)"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
            title = title or "Latent Space (t-SNE)"
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        X_embedded = reducer.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], alpha=0.6, edgecolor=None)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        return plt.gcf()

    def plot_grade_heatmap(self, data: torch.Tensor, title="Grade Energy Distribution"):
        """
        Visualizes the average energy (magnitude squared) of each grade.
        Rows: Samples (or time), Cols: Grades (0 to n).
        If data is just a batch, we show a bar chart of average energy per grade.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim)
            
        # Compute energy per grade
        # We need a mask for each grade
        energy_per_grade = []
        grade_labels = []
        
        for k in range(self.algebra.n + 1):
            mask = self.algebra.grade_projection(torch.ones(1, self.algebra.dim, device=self.algebra.device), k).bool()
            mask = mask.view(-1)
            
            if not mask.any():
                continue
                
            # Extract components for this grade
            # data: [Batch, Dim]
            comps = data[:, mask]
            
            # Energy: mean of squares sum
            # Average over batch
            energy = (comps ** 2).sum(dim=1).mean().item()
            energy_per_grade.append(energy)
            grade_labels.append(f"Grade {k}")
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x=grade_labels, y=energy_per_grade, palette="viridis")
        plt.title(title)
        plt.ylabel("Average Energy")
        plt.yscale('log') # Log scale often helps if Scalar/Vector dominate
        return plt.gcf()

    def plot_components_heatmap(self, data: torch.Tensor, title="Component Activation Heatmap"):
        """
        Heatmap of all component magnitudes.
        """
        if data.ndim > 2:
            data = data.reshape(-1, self.algebra.dim)
            
        # Take a subset of samples if too large
        if data.shape[0] > 100:
            data = data[:100]
            
        X = data.abs().detach().cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(X.T, yticklabels=self.basis_names, cmap="magma", cbar_kws={'label': 'Magnitude'})
        plt.title(title)
        plt.xlabel("Sample Index")
        return plt.gcf()
