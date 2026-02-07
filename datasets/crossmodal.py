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

class CrossModalDataset(Dataset):
    """Synthetic cross-modal dataset.

    A: Real BERT embeddings.
    B: Rotated A + Noise.
    """

    def __init__(self, algebra: CliffordAlgebra, embedding_dim=6):
        """Sets up the dataset."""
        self.algebra = algebra
        self.embedding_dim = embedding_dim
        self.data_A, self.data_B = self._generate()

    def _generate(self):
        """Downloads BERT, applies geometric distortions to the embeddings."""
        from transformers import BertTokenizer, BertModel
        from sklearn.decomposition import PCA

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

        pca = PCA(n_components=self.embedding_dim)
        reduced_A = pca.fit_transform(embeddings_A)
        reduced_A = reduced_A / (np.abs(reduced_A).max() + 1e-6)

        vector_indices = [1 << i for i in range(self.embedding_dim)]
        data_A = torch.zeros(len(sentences), 1, self.algebra.dim)
        for i in range(self.embedding_dim):
            data_A[:, 0, vector_indices[i]] = torch.tensor(reduced_A[:, i])

        phi = torch.tensor(0.78) # ~45 degrees
        B_dist = torch.zeros(1, self.algebra.dim)
        B_dist[0, 3] = 1.0

        R_dist = self.algebra.exp(-0.5 * phi * B_dist)
        R_dist_rev = self.algebra.reverse(R_dist)

        data_B = self.algebra.geometric_product(R_dist.expand_as(data_A), data_A)
        data_B = self.algebra.geometric_product(data_B, R_dist_rev.expand_as(data_A))

        data_B = data_B + 0.1 * torch.randn_like(data_B)

        return data_A, data_B

    def __len__(self):
        return len(self.data_A)

    def __getitem__(self, idx):
        return self.data_A[idx], self.data_B[idx]
