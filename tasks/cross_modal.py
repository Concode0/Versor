import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from functional.activation import GeometricGELU
from functional.loss import GeometricMSELoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer

class ModalityEncoder(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.net = nn.Sequential(
            CliffordLinear(algebra, 1, 4),
            GeometricGELU(algebra, channels=4),
            CliffordLinear(algebra, 4, 1),
            RotorLayer(algebra, channels=1) 
        )
    def forward(self, x):
        return self.net(x)

class CrossModalBinder(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        
        # Two independent encoders trying to map to a shared space
        self.encoder_A = ModalityEncoder(algebra) # e.g. Text
        self.encoder_B = ModalityEncoder(algebra) # e.g. Image
        
    def forward(self, x_a, x_b):
        z_a = self.encoder_A(x_a)
        z_b = self.encoder_B(x_b)
        return z_a, z_b

class CrossModalTask(BaseTask):
    def __init__(self, epochs=100, lr=0.01, device='cpu', embedding_dim=6):
        self.embedding_dim = embedding_dim
        super().__init__(epochs=epochs, lr=lr, device=device)

    def setup_algebra(self):
        return CliffordAlgebra(p=self.embedding_dim, q=0, device=self.device)

    def setup_model(self):
        return CrossModalBinder(self.algebra)

    def setup_criterion(self):
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        print("Loading BERT for Source A (Text)...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
        
        sentences = [
            "A photograph of a dog", "A cat sitting on a mat",
            "A red sports car", "A blue ocean wave",
            "A delicious pizza", "A mountain landscape",
            "A computer screen", "A running horse"
        ] # 8 unique samples
        
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert(**inputs)
        embeddings_A = outputs.last_hidden_state[:, 0, :].numpy()
        
        # PCA to GA dims
        print(f"Projecting to {self.embedding_dim} dims...")
        pca = PCA(n_components=self.embedding_dim)
        reduced_A = pca.fit_transform(embeddings_A)
        reduced_A = reduced_A / np.abs(reduced_A).max()
        
        # Convert to MV A
        vector_indices = [1 << i for i in range(self.embedding_dim)]
        data_A = torch.zeros(len(sentences), 1, self.algebra.dim)
        for i in range(self.embedding_dim):
            data_A[:, 0, vector_indices[i]] = torch.tensor(reduced_A[:, i])
            
        # Generate Source B (Synthetic Image Modality)
        # Apply a fixed, random rotation (Misalignment) + Noise to A
        print("Generating Source B (Synthetic Misaligned View)...")
        
        # Fixed Rotor for distortion
        # Rotate e1-e2 plane by 45 degrees
        phi = torch.tensor(0.78) # 45 deg
        B_dist = torch.zeros(1, self.algebra.dim)
        B_dist[0, 3] = 1.0 # e1e2
        
        R_dist = self.algebra.exp(-0.5 * phi * B_dist)
        R_dist_rev = self.algebra.reverse(R_dist)
        
        # Apply R A R~
        data_B = self.algebra.geometric_product(R_dist.expand_as(data_A), data_A)
        data_B = self.algebra.geometric_product(data_B, R_dist_rev.expand_as(data_A))
        
        # Add Noise (Modality gap)
        data_B = data_B + 0.1 * torch.randn_like(data_B)
        
        return data_A.to(self.device), data_B.to(self.device)

    def train_step(self, data):
        data_A, data_B = data
        self.optimizer.zero_grad()
        
        z_a, z_b = self.model(data_A, data_B)
        
        # Contrastive/Alignment Loss
        # We want z_a and z_b to be close in the shared geometric space
        loss_align = self.criterion(z_a, z_b)
        
        # Regularization: Prevent collapse to zero
        # Hinge loss or simply ensure norm is close to 1?
        # Let's try Isometry regularization: keep norm of z_a close to norm of x_a
        norm_a = (z_a**2).sum(dim=-1).mean()
        norm_input = (data_A**2).sum(dim=-1).mean()
        loss_norm = (norm_a - norm_input).pow(2)
        
        loss = loss_align + 0.1 * loss_norm
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Align": loss_align.item(), "Norm": loss_norm.item()}

    def evaluate(self, data):
        data_A, data_B = data
        z_a, z_b = self.model(data_A, data_B)
        
        # 1. Alignment Distance
        final_dist = torch.norm(z_a - z_b, dim=-1).mean().item()
        print(f"Final Alignment Distance: {final_dist:.6f}")
        
        # 2. Alignment Ratio (Improvement)
        initial_dist = torch.norm(data_A - data_B, dim=-1).mean().item()
        ratio = final_dist / (initial_dist + 1e-9)
        print(f"Alignment Ratio (Final/Initial): {ratio:.4f} (Lower is better)")
        
        # 3. Semantic Consistency (Top-1 Retrieval Accuracy)
        # For each z_a[i], is z_b[i] the closest among all z_b?
        # Normalize for cosine similarity
        z_a_norm = z_a / (z_a.norm(dim=-1, keepdim=True) + 1e-9)
        z_b_norm = z_b / (z_b.norm(dim=-1, keepdim=True) + 1e-9)
        
        # Similarity matrix [N, N]
        sim_matrix = torch.mm(z_a_norm.squeeze(1), z_b_norm.squeeze(1).t())
        
        # Top-1 Accuracy
        # Indices of max similarity
        preds = sim_matrix.argmax(dim=1)
        targets = torch.arange(len(preds), device=self.device)
        accuracy = (preds == targets).float().mean().item()
        
        print(f"Semantic Consistency (Top-1 Retrieval): {accuracy*100:.2f}%")
        
    def visualize(self, data):
        data_A, data_B = data
        z_a, z_b = self.model(data_A, data_B)
        
        viz = GeneralVisualizer(self.algebra)
        
        # 1. Before Alignment (Overlay A and B)
        # Use PCA
        # We need to concatenate and color code
        A_flat = data_A.reshape(-1, self.algebra.dim)
        B_flat = data_B.reshape(-1, self.algebra.dim)
        
        combined = torch.cat([A_flat, B_flat], dim=0)
        
        # Manually plot PCA using sklearn to handle colors
        pca = PCA(n_components=2)
        emb = pca.fit_transform(combined.detach().cpu().numpy())
        n = A_flat.shape[0]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(emb[:n, 0], emb[:n, 1], c='blue', label='Source A (Text)', alpha=0.6)
        plt.scatter(emb[n:, 0], emb[n:, 1], c='red', label='Source B (Img/Distorted)', alpha=0.6)
        plt.legend()
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Before Unification (Misaligned)")
        plt.savefig("crossmodal_before.png")
        plt.close()
        
        # 2. After Alignment
        zA_flat = z_a.reshape(-1, self.algebra.dim)
        zB_flat = z_b.reshape(-1, self.algebra.dim)
        combined_z = torch.cat([zA_flat, zB_flat], dim=0)
        
        emb_z = pca.fit_transform(combined_z.detach().cpu().numpy())
        
        plt.figure(figsize=(10, 6))
        plt.scatter(emb_z[:n, 0], emb_z[:n, 1], c='blue', label='Latent A', alpha=0.6)
        plt.scatter(emb_z[n:, 0], emb_z[n:, 1], c='red', label='Latent B', alpha=0.6)
        plt.legend()
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("After Unification (Aligned)")
        plt.savefig("crossmodal_after.png")
        plt.close()
        
        print("Saved cross-modal visualizations.")

def run_crossmodal_task(epochs=100, lr=0.01):
    task = CrossModalTask(epochs=epochs, lr=lr, device='cpu')
    task.run()
