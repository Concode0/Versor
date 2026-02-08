# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class NewsgroupsClassificationDataset(Dataset):
    """20 Newsgroups for semantic classification.

    BERT [CLS] embeddings → PCA → multivectors.
    Uses multiple channels to preserve discriminative information:
    PCA to (in_channels * embedding_dim) dims, split across channels.
    Disk-cached for speed after first run.
    """

    def __init__(self, algebra, embedding_dim=6, in_channels=8, split='train',
                 max_samples=2000, cache_dir='./data/newsgroups'):
        self.algebra = algebra
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.n_components = in_channels * embedding_dim
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.data, self.labels = self._load_or_compute()

    def _cache_path(self, name):
        return os.path.join(
            self.cache_dir,
            f'{name}_{self.split}_{self.embedding_dim}_ch{self.in_channels}.pt'
        )

    def _load_or_compute(self):
        bert_cache = self._cache_path('bert_cls')
        labels_cache = self._cache_path('labels')

        if os.path.exists(bert_cache) and os.path.exists(labels_cache):
            print(f">>> Loading cached BERT embeddings from {bert_cache}")
            data = torch.load(bert_cache, weights_only=True)
            labels = torch.load(labels_cache, weights_only=True)
            return data, labels

        print(f">>> Computing BERT embeddings for 20 Newsgroups ({self.split})...")
        print(f"    PCA: 768 → {self.n_components} ({self.in_channels} channels × {self.embedding_dim} dims)")
        from sklearn.datasets import fetch_20newsgroups
        from transformers import BertTokenizer, BertModel
        from sklearn.decomposition import PCA

        subset = 'train' if self.split == 'train' else 'test'
        newsgroups = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))

        texts = newsgroups.data[:self.max_samples]
        targets = newsgroups.target[:self.max_samples]

        # BERT encoding
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
        bert.eval()

        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors='pt', padding=True,
                truncation=True, max_length=128
            )
            with torch.no_grad():
                outputs = bert(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)

        embeddings = np.concatenate(embeddings, axis=0)

        # PCA reduction to in_channels * embedding_dim components
        import pickle
        pca_cache_path = os.path.join(self.cache_dir, f'pca_{self.n_components}_ch{self.in_channels}.pkl')

        if self.split == 'train':
            pca = PCA(n_components=self.n_components)
            reduced = pca.fit_transform(embeddings)
            print(f"    PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            with open(pca_cache_path, 'wb') as f:
                pickle.dump(pca, f)
            print(f"    Saved PCA model to {pca_cache_path}")
        else:
            if not os.path.exists(pca_cache_path):
                raise FileNotFoundError(f"PCA model not found at {pca_cache_path}. Run with split='train' first to fit PCA.")
            print(f"    Loading PCA model from {pca_cache_path}...")
            with open(pca_cache_path, 'rb') as f:
                pca = pickle.load(f)
            reduced = pca.transform(embeddings)

        reduced = reduced / (np.abs(reduced).max() + 1e-6)

        # Embed into multivectors: each channel gets embedding_dim components
        vector_indices = [1 << i for i in range(self.embedding_dim)]
        data = torch.zeros(len(targets), self.in_channels, self.algebra.dim)

        for ch in range(self.in_channels):
            offset = ch * self.embedding_dim
            for i in range(self.embedding_dim):
                data[:, ch, vector_indices[i]] = torch.tensor(
                    reduced[:, offset + i], dtype=torch.float32
                )

        labels = torch.tensor(targets, dtype=torch.long)

        # Cache to disk
        torch.save(data, bert_cache)
        torch.save(labels, labels_cache)
        print(f">>> Cached embeddings to {self.cache_dir}")

        return data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


