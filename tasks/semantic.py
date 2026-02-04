import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # Encoder: Align semantic manifold
        self.encoder = nn.Sequential(
            CliffordLinear(algebra, 1, 4),
            GeometricGELU(algebra, channels=4),
            CliffordLinear(algebra, 4, 1),
            RotorLayer(algebra, channels=1) # The "Unbender"
        )
        
        # Selector: Filter noise/complex grades
        self.selector = BladeSelector(algebra, channels=1)
        
        # Decoder: Reconstruct
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

class SemanticTask(BaseTask):
    def __init__(self, epochs=100, lr=0.01, device='cpu', embedding_dim=6):
        # We limit embedding dim to keep GA tractable (2^6 = 64 dims)
        self.embedding_dim = embedding_dim 
        super().__init__(epochs=epochs, lr=lr, device=device)

    def setup_algebra(self):
        # Euclidean metric for semantic space
        return CliffordAlgebra(p=self.embedding_dim, q=0, device=self.device)

    def setup_model(self):
        return SemanticAutoEncoder(self.algebra)

    def setup_criterion(self):
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        # 1. Load Pre-trained BERT
        print("Loading BERT for semantic projection...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 2. Datasets
        # Train: Tech (0) and Nature (1)
        train_sentences = [
            "Computer algorithm processing data", "Artificial intelligence neural network",
            "Server cloud computing storage", "Python programming language code",
            "Digital circuit transistor logic", "Operating system linux kernel",
            "Database sql query optimization", "Cybersecurity firewall encryption",
            
            "Forest tree green leaf", "River flowing water stream",
            "Mountain peak snow sky", "Flower garden bloom spring",
            "Ocean blue wave sand", "Rainforest amazon jungle bio",
            "Desert cactus dry sand", "Volcano lava eruption magma"
        ]
        
        # Validation: Same topics, new sentences
        val_sentences = [
            "Laptop keyboard mouse screen", "Machine learning deep learning",
            "Sunset horizon red sky", "Waterfall cascade rock stone"
        ]
        
        # Unseen: Space (2)
        unseen_sentences = [
            "Galaxy star planet orbit", "Universe black hole gravity",
            "Astronaut rocket moon landing", "Solar system sun earth mars"
        ]
        
        all_sentences = train_sentences + val_sentences + unseen_sentences
        
        # 3. Get Embeddings
        inputs = tokenizer(all_sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 4. PCA Projection
        print(f"Projecting 768 dims to {self.embedding_dim} dims...")
        pca = PCA(n_components=self.embedding_dim)
        reduced_embeddings = pca.fit_transform(embeddings.numpy())
        
        # Normalize
        reduced_embeddings = reduced_embeddings / np.abs(reduced_embeddings).max()
        
        # 5. Embed into Algebra
        vector_indices = [1 << i for i in range(self.embedding_dim)]
        
        def to_mv(indices):
            subset = reduced_embeddings[indices]
            mv = torch.zeros(len(indices), 1, self.algebra.dim)
            for i in range(self.embedding_dim):
                mv[:, 0, vector_indices[i]] = torch.tensor(subset[:, i])
            return mv.to(self.device)

        n_train = len(train_sentences)
        n_val = len(val_sentences)
        
        train_data = to_mv(range(0, n_train))
        val_data = to_mv(range(n_train, n_train + n_val))
        unseen_data = to_mv(range(n_train + n_val, len(all_sentences)))
        
        return train_data, val_data, unseen_data

    def run(self):
        """Override run to handle split data."""
        print(f">>> Starting Task: {self.__class__.__name__}")
        
        train_data, val_data, unseen_data = self.get_data()

        # Training Loop
        self.model.train()
        pbar = tqdm.tqdm(range(self.epochs))
        
        for epoch in pbar:
            metrics = self.train_step(train_data)
            loss, metric_dict = metrics
            
            # Periodic Validation
            if epoch % 100 == 0:
                with torch.no_grad():
                    val_recon, _, _ = self.model(val_data)
                    val_loss = self.criterion(val_recon, val_data)
                    metric_dict["ValLoss"] = val_loss.item()
            
            desc = f"Loss: {loss:.4f}"
            for k, v in metric_dict.items():
                desc += f" | {k}: {v:.4f}"
            pbar.set_description(desc)
            
        print("\n>>> Training Complete.")
        
        self.model.eval()
        with torch.no_grad():
            print("\n--- Evaluation on Train Data ---")
            self.evaluate(train_data, name="Train")
            
            print("\n--- Evaluation on Unseen Topic (Space) ---")
            self.evaluate(unseen_data, name="Unseen")
            
            print("\n--- Noise Robustness Test ---")
            self.evaluate_robustness(val_data)
            
            self.visualize(train_data, unseen_data)

    def train_step(self, data):
        self.optimizer.zero_grad()
        recon, latent, latent_proj = self.model(data)
        
        loss_recon = self.criterion(recon, data)
        
        # Regularization
        total_energy = (latent ** 2).sum(dim=-1).mean()
        grade1 = self.algebra.grade_projection(latent, 1)
        grade1_energy = (grade1 ** 2).sum(dim=-1).mean()
        impurity_loss = total_energy - grade1_energy
        
        sparsity = self.model.selector.weights.abs().mean()
        
        loss = loss_recon + 0.1 * impurity_loss + 0.01 * sparsity
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Recon": loss_recon.item(), "Purity": impurity_loss.item()}

    def evaluate(self, data, name="Data"):
        recon, latent, latent_proj = self.model(data)
        
        # 1. Reconstruction
        loss = self.criterion(recon, data)
        print(f"[{name}] Reconstruction Loss: {loss.item():.6f}")
        
        # 2. Grade Purity
        grade1 = self.algebra.grade_projection(latent, 1)
        g1_energy = (grade1 ** 2).sum(dim=-1)
        total_energy = (latent ** 2).sum(dim=-1) + 1e-9
        purity = (g1_energy / total_energy).mean().item()
        print(f"[{name}] Semantic Grade Purity: {purity:.4f}")

    def evaluate_robustness(self, data, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        print(f"Baseline Loss (Noise=0.0): {self.criterion(self.model(data)[0], data).item():.6f}")
        
        for nl in noise_levels:
            # Add Gaussian noise to input multivector coefficients
            noisy_data = data + torch.randn_like(data) * nl
            recon, _, _ = self.model(noisy_data)
            
            # Loss against ORIGINAL (Denoising capability)
            loss = self.criterion(recon, data)
            print(f"Noise {nl:.2f}: Denoising Loss = {loss.item():.6f}")

    def visualize(self, train_data, unseen_data):
        viz = GeneralVisualizer(self.algebra)
        
        # 1. Latent Space (Train vs Unseen)
        _, train_latent, _ = self.model(train_data)
        _, unseen_latent, _ = self.model(unseen_data)
        
        # Combine for PCA to see relative position
        combined_latent = torch.cat([train_latent, unseen_latent], dim=0)
        
        viz.plot_latent_projection(combined_latent, title="Latent Space (Train + Unseen Space)")
        viz.save("semantic_robustness_pca.png")
        
        # 2. Unseen Grade Energy
        viz.plot_grade_heatmap(unseen_latent, title="Unseen Topic Grade Energy")
        viz.save("semantic_unseen_grades.png")
        
        # 3. Bivector Meaning Map
        self._plot_bivector_map(self.model.encoder[-1].bivector_weights)
        
        print("Saved robustness visualizations.")

    def _plot_bivector_map(self, weights):
        w = weights.detach().cpu().numpy().flatten()
        indices = self.model.encoder[-1].bivector_indices.cpu().numpy()
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
        sns.barplot(x=names, y=w, palette="magma")
        plt.title("Bivector Meaning Map (Rotational Activity)")
        plt.xticks(rotation=45)
        plt.ylabel("Rotor Coefficient Magnitude")
        plt.tight_layout()
        plt.savefig("semantic_bivector_map.png")
        plt.close()

def run_semantic_task(epochs=100, lr=0.01):
    task = SemanticTask(epochs=epochs, lr=lr, device='cpu')
    task.run()
