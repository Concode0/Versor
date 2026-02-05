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
from torch.utils.data import Dataset
import numpy as np
from core.algebra import CliffordAlgebra
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

class Figure8Dataset(Dataset):
    """Synthetic dataset generating a distorted Figure-8 manifold in 3D.

    Generates points on a 2D curve embedded in 3D, distorted by a non-linear Z term.
    """

    def __init__(self, algebra: CliffordAlgebra, num_samples=1000):
        """Initializes the dataset.

        Args:
            algebra (CliffordAlgebra): Algebra instance for multivector construction.
            num_samples (int, optional): Number of points. Defaults to 1000.
        """
        self.algebra = algebra
        self.data = self._generate(num_samples)

    def _generate(self, n):
        """Generates the data points."""
        t = torch.linspace(0, 2*np.pi, n)
        x = torch.sin(t)
        y = torch.sin(t) * torch.cos(t)
        z = 0.5 * x * y  # Distorted manifold (z = 0.5xy)
        
        # Convert to Multivector (vectors at indices 1, 2, 4 for 3D)
        # e1=1, e2=2, e3=4
        data = torch.zeros(n, 1, self.algebra.dim)
        data[..., 1] = x.unsqueeze(-1)
        data[..., 2] = y.unsqueeze(-1)
        data[..., 4] = z.unsqueeze(-1)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CrossModalDataset(Dataset):
    """Synthetic Multi-Modal Dataset.

    Generates pairs of (Text Embedding, Distorted Image Embedding).
    Source A: Real BERT embeddings of sentences.
    Source B: Source A rotated + noisy to simulate a different modality view.
    """

    def __init__(self, algebra: CliffordAlgebra, embedding_dim=6):
        """Initializes the dataset.

        Args:
            algebra (CliffordAlgebra): Algebra instance.
            embedding_dim (int, optional): Dimension of the embedding space. Defaults to 6.
        """
        self.algebra = algebra
        self.embedding_dim = embedding_dim
        self.data_A, self.data_B = self._generate()

    def _generate(self):
        """Generates synthetic multi-modal data."""
        # Load BERT (using small model for demo speed)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
        
        sentences = [
            "A photograph of a dog", "A cat sitting on a mat",
            "A red sports car", "A blue ocean wave",
            "A delicious pizza", "A mountain landscape",
            "A computer screen", "A running horse"
        ]
        
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert(**inputs)
        embeddings_A = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Project to lower dimension using PCA
        pca = PCA(n_components=self.embedding_dim)
        reduced_A = pca.fit_transform(embeddings_A)
        reduced_A = reduced_A / (np.abs(reduced_A).max() + 1e-6)
        
        # Convert to Multivector A (Basis Vectors)
        vector_indices = [1 << i for i in range(self.embedding_dim)]
        data_A = torch.zeros(len(sentences), 1, self.algebra.dim)
        for i in range(self.embedding_dim):
            data_A[:, 0, vector_indices[i]] = torch.tensor(reduced_A[:, i])
            
        # Generate Source B (Synthetic Image Modality)
        # Apply a fixed rotation (Misalignment) + Noise
        phi = torch.tensor(0.78) # ~45 degrees
        B_dist = torch.zeros(1, self.algebra.dim)
        B_dist[0, 3] = 1.0 # e1^e2 plane rotation
        
        R_dist = self.algebra.exp(-0.5 * phi * B_dist)
        R_dist_rev = self.algebra.reverse(R_dist)
        
        # Apply transformation R A R~
        data_B = self.algebra.geometric_product(R_dist.expand_as(data_A), data_A)
        data_B = self.algebra.geometric_product(data_B, R_dist_rev.expand_as(data_A))
        
        # Add modality gap noise
        data_B = data_B + 0.1 * torch.randn_like(data_B)
        
        return data_A, data_B

    def __len__(self):
        return len(self.data_A)

    def __getitem__(self, idx):
        return self.data_A[idx], self.data_B[idx]
