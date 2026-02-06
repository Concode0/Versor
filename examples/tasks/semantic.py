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
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.decomposition import PCA

from core.algebra import CliffordAlgebra
from layers.rotor import RotorLayer
from layers.projection import BladeSelector
from functional.loss import SubspaceLoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer

class SemanticNetwork(nn.Module):
    """Meaning Unbender. Words are vectors.

    Aligns semantic concepts with geometric grades.
    """

    def __init__(self, algebra):
        """Sets up the network."""
        super().__init__()
        self.rotor = RotorLayer(algebra, channels=1)
        self.selector = BladeSelector(algebra, channels=1)

    def forward(self, x):
        """Forward pass."""
        x_rot = self.rotor(x)
        return self.selector(x_rot)

class SemanticTask(BaseTask):
    """Semantic Disentanglement. Tech is vector, Nature is bivector.

    Forces meanings into orthogonal subspaces.
    """

    def __init__(self, cfg):
        self.embedding_dim = cfg.dataset.embedding_dim
        super().__init__(cfg)

    def setup_algebra(self):
        """High-dim Euclidean."""
        return CliffordAlgebra(p=self.embedding_dim, q=0, device=self.device)

    def setup_model(self):
        """The Unbender."""
        return SemanticNetwork(self.algebra)

    def setup_criterion(self):
        """Subspace Loss. Flatten to vectors."""
        target = []
        for i in range(self.algebra.dim):
            if bin(i).count('1') == 1:
                target.append(i)
        return SubspaceLoss(self.algebra, target_indices=target)

    def get_data(self):
        """Fetches BERT embeddings. Adds noise."""
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
        
        topics = {
            "tech": ["computer", "software", "algorithm", "network", "database"],
            "nature": ["tree", "flower", "river", "mountain", "forest"]
        }
        
        all_embeddings = []
        labels = []
        
        for topic, words in topics.items():
            inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = bert(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(emb)
            labels.extend([topic] * len(words))
            
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # PCA to 6D
        pca = PCA(n_components=self.embedding_dim)
        reduced = pca.fit_transform(all_embeddings)
        reduced = reduced / np.abs(reduced).max()
        
        # Embed into Multivectors
        data = torch.zeros(len(labels), 1, self.algebra.dim)
        vector_indices = [1 << i for i in range(self.embedding_dim)]
        
        for i in range(self.embedding_dim):
            data[:, 0, vector_indices[i]] = torch.tensor(reduced[:, i])
            
        # Add noise to bivectors (simulate entanglement)
        noise = torch.randn_like(data) * 0.2
        mask = torch.zeros(self.algebra.dim, dtype=torch.bool)
        for i in range(self.algebra.dim):
            if bin(i).count('1') == 2:
                mask[i] = True
        
        data[..., mask] += noise[..., mask]
        
        return data.to(self.device)

    def train_step(self, data):
        """Clean up the meaning."""
        self.optimizer.zero_grad()
        output = self.model(data)
        
        loss = self.criterion(output)
        
        # Calculate Grade Purity
        total_energy = (output**2).sum()
        
        vec_indices = [1 << i for i in range(self.embedding_dim)]
        vec_energy = (output[..., vec_indices]**2).sum()
        
        purity = vec_energy / (total_energy + 1e-9)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Recon": loss.item(), "Purity": purity.item()}

    def evaluate(self, data):
        """How pure is it?"""
        output = self.model(data)
        
        vec_indices = [1 << i for i in range(self.embedding_dim)]
        vec_energy = (output[..., vec_indices]**2).sum()
        total_energy = (output**2).sum()
        purity = vec_energy / (total_energy + 1e-9)
        
        print(f"Final Semantic Grade Purity: {purity.item():.4f}")

    def visualize(self, data):
        """Heatmap of meaning."""
        output = self.model(data)
        viz = GeneralVisualizer(self.algebra)
        
        viz.plot_components_heatmap(output, title="Semantic Components (After Unbending)")
        viz.save("semantic_bivector_map.png")