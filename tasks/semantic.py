# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

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
    """Network for semantic manifold alignment (unbending).

    Learns to rotate high-dimensional semantic embeddings (e.g., from BERT)
    such that distinct topics align with orthogonal geometric basis blades (grades).
    """

    def __init__(self, algebra):
        """Initializes the network.

        Args:
            algebra: The algebra instance.
        """
        super().__init__()
        self.rotor = RotorLayer(algebra, channels=1)
        self.selector = BladeSelector(algebra, channels=1)

    def forward(self, x):
        """Forward pass."""
        x_rot = self.rotor(x)
        return self.selector(x_rot)

class SemanticTask(BaseTask):
    """Task for disentangling semantic topics into geometric grades.

    For example, aligning "Technology" words with vectors (Grade 1) and
    "Nature" words with bivectors (Grade 2).
    """

    def __init__(self, cfg):
        self.embedding_dim = cfg.dataset.embedding_dim
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up Cl(6, 0)."""
        return CliffordAlgebra(p=self.embedding_dim, q=0, device=self.device)

    def setup_model(self):
        """Sets up the SemanticNetwork."""
        return SemanticNetwork(self.algebra)

    def setup_criterion(self):
        """Sets up SubspaceLoss.

        We want the output to be pure Grade 1 (Vector) for Topic A,
        and pure Grade 2 (Bivector) for Topic B.
        However, SubspaceLoss takes a static target.
        For this demo, we'll assume we want to align ALL topics to Grade 1 (Vectors),
        effectively 'flattening' the semantic space to a vector space.
        """
        # Target: Vectors
        target = []
        for i in range(self.algebra.dim):
            if bin(i).count('1') == 1:
                target.append(i)
        return SubspaceLoss(self.algebra, target_indices=target)

    def get_data(self):
        """Loads BERT embeddings for two distinct topics."""
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
            # [Batch, Hidden]
            emb = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(emb)
            labels.extend([topic] * len(words))
            
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # PCA to 6D
        pca = PCA(n_components=self.embedding_dim)
        reduced = pca.fit_transform(all_embeddings)
        # Normalize
        reduced = reduced / np.abs(reduced).max()
        
        # Embed into Multivectors
        # We put them initially into mixed grades or just vectors?
        # Let's put them into vectors but add noise to other grades to simulate "entanglement"
        
        data = torch.zeros(len(labels), 1, self.algebra.dim)
        vector_indices = [1 << i for i in range(self.embedding_dim)]
        
        for i in range(self.embedding_dim):
            data[:, 0, vector_indices[i]] = torch.tensor(reduced[:, i])
            
        # Add noise to bivectors (entanglement)
        noise = torch.randn_like(data) * 0.2
        # Mask only bivectors
        mask = torch.zeros(self.algebra.dim, dtype=torch.bool)
        for i in range(self.algebra.dim):
            if bin(i).count('1') == 2:
                mask[i] = True
        
        data[..., mask] += noise[..., mask]
        
        return data.to(self.device)

    def train_step(self, data):
        """Executes one training step."""
        self.optimizer.zero_grad()
        output = self.model(data)
        
        # Minimize non-vector components
        loss = self.criterion(output)
        
        # Calculate Grade Purity (Energy in Grade 1 / Total Energy)
        total_energy = (output**2).sum()
        
        vec_indices = [1 << i for i in range(self.embedding_dim)]
        vec_energy = (output[..., vec_indices]**2).sum()
        
        purity = vec_energy / (total_energy + 1e-9)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Recon": loss.item(), "Purity": purity.item()}

    def evaluate(self, data):
        """Evaluates final grade purity."""
        output = self.model(data)
        
        vec_indices = [1 << i for i in range(self.embedding_dim)]
        vec_energy = (output[..., vec_indices]**2).sum()
        total_energy = (output**2).sum()
        purity = vec_energy / (total_energy + 1e-9)
        
        print(f"Final Semantic Grade Purity: {purity.item():.4f}")

    def visualize(self, data):
        """Visualizes the bivector map."""
        output = self.model(data)
        viz = GeneralVisualizer(self.algebra)
        
        viz.plot_components_heatmap(output, title="Semantic Components (After Unbending)")
        viz.save("semantic_bivector_map.png")