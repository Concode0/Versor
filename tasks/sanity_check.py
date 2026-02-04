import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import tqdm

from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.projection import BladeSelector
from functional.activation import GeometricGELU
from functional.loss import GeometricMSELoss, SubspaceLoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer

class SemanticAutoEncoder(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        self.encoder = nn.Sequential(
            CliffordLinear(algebra, 1, 4),
            GeometricGELU(algebra, channels=4),
            CliffordLinear(algebra, 4, 1),
            RotorLayer(algebra, channels=1)
        )
        self.selector = BladeSelector(algebra, channels=1)
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

class SanityCheckTask(BaseTask):
    def __init__(self, epochs=100, lr=0.01, device='cpu', embedding_dim=6):
        self.embedding_dim = embedding_dim
        super().__init__(epochs=epochs, lr=lr, device=device)

    def setup_algebra(self):
        return CliffordAlgebra(p=self.embedding_dim, q=0, device=self.device)

    def setup_model(self):
        return SemanticAutoEncoder(self.algebra)

    def setup_criterion(self):
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        print("Generating Random Gaussian Noise (No Semantics)...")
        
        # 3 "Topics" worth of random noise
        n_per_topic = 10
        n_topics = 3
        total = n_per_topic * n_topics
        
        # Random vectors in 768 dim (simulating BERT space)
        noise_embeddings = np.random.randn(total, 768)
        
        # PCA Projection to GA dims (to match pipeline process)
        # Even noise has principal components (random directions)
        print(f"Projecting Noise to {self.embedding_dim} dims...")
        pca = PCA(n_components=self.embedding_dim)
        reduced_embeddings = pca.fit_transform(noise_embeddings)
        reduced_embeddings = reduced_embeddings / np.abs(reduced_embeddings).max()
        
        vector_indices = [1 << i for i in range(self.embedding_dim)]
        
        def to_mv(data):
            mv = torch.zeros(len(data), 1, self.algebra.dim)
            for i in range(self.embedding_dim):
                mv[:, 0, vector_indices[i]] = torch.tensor(data[:, i])
            return mv.to(self.device)

        # Split
        train_end = 2 * n_per_topic
        train_data = to_mv(reduced_embeddings[:train_end])
        val_data = to_mv(reduced_embeddings[train_end:])
        
        return train_data, val_data

    def train_step(self, data):
        self.optimizer.zero_grad()
        recon, latent, latent_proj = self.model(data)
        
        loss_recon = self.criterion(recon, data)
        
        # Regularization (Same as Semantic Task)
        total_energy = (latent ** 2).sum(dim=-1).mean()
        grade1 = self.algebra.grade_projection(latent, 1)
        grade1_energy = (grade1 ** 2).sum(dim=-1).mean()
        impurity_loss = total_energy - grade1_energy
        
        sparsity = self.model.selector.weights.abs().mean()
        
        loss = loss_recon + 0.1 * impurity_loss + 0.01 * sparsity
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Recon": loss_recon.item(), "Purity": impurity_loss.item()}

    def run(self):
        print(f">>> Starting Task: {self.__class__.__name__}")
        train_data, val_data = self.get_data()

        self.model.train()
        pbar = tqdm.tqdm(range(self.epochs))
        
        for epoch in pbar:
            loss, metrics = self.train_step(train_data)
            pbar.set_description(f"Loss: {loss:.4f} | Recon: {metrics['Recon']:.4f}")
            
        print("\n>>> Training Complete.")
        
        self.model.eval()
        with torch.no_grad():
            self.evaluate(train_data, "Train (Noise)")
            self.evaluate(val_data, "Val (Noise)")
            
            viz = GeneralVisualizer(self.algebra)
            _, latent, _ = self.model(train_data)
            viz.plot_latent_projection(latent, title="Latent Space (Random Noise)")
            viz.save("sanity_noise_pca.png")
            print("Saved sanity check visualization.")

    def evaluate(self, data, name):
        recon, latent, latent_proj = self.model(data)
        loss = self.criterion(recon, data)
        
        grade1 = self.algebra.grade_projection(latent, 1)
        g1_energy = (grade1 ** 2).sum(dim=-1).mean()
        total_energy = (latent ** 2).sum(dim=-1).mean() + 1e-9
        purity = (g1_energy / total_energy).item()
        
        print(f"[{name}] Reconstruction Loss: {loss.item():.6f}")
        print(f"[{name}] Grade Purity: {purity:.4f}")

    def visualize(self, data):
        # We handle visualization in run() already for simplicity, but BaseTask requires this.
        # Let's move the PCA plotting here.
        # data is just passed as argument but we might not need it if we visualize training data.
        # But BaseTask calls visualize(data) where data is from get_data().
        # get_data returns tuple (train, val).
        
        train_data, val_data = data
        viz = GeneralVisualizer(self.algebra)
        _, latent, _ = self.model(train_data)
        viz.plot_latent_projection(latent, title="Latent Space (Random Noise)")
        viz.save("sanity_noise_pca.png")
        print("Saved sanity check visualization.")

def run_sanity_check(epochs=100, lr=0.01):
    task = SanityCheckTask(epochs=epochs, lr=lr, device='cpu')
    task.run()
