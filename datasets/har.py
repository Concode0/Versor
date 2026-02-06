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
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from core.algebra import CliffordAlgebra

class HARDataset(Dataset):
    """HAR Dataset. Phones sensing walking.
    
    561-dim feature vectors.
    """

    def __init__(self, algebra: CliffordAlgebra, root='./data/HAR', split='train'):
        """Loads the CSV."""
        self.algebra = algebra
        self.root = root
        self.split = split
        
        file_path = f"{root}/{split}.csv"
        try:
            self.df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"HAR dataset not found at {file_path}.")
            
        feature_cols = [c for c in self.df.columns if c not in ['subject', 'Activity']]
        
        self.data = torch.tensor(self.df[feature_cols].values, dtype=torch.float32)
        
        self.labels = self._encode_labels(self.df['Activity'])
        self.classes = sorted(self.df['Activity'].unique())
        
    def _encode_labels(self, series):
        """String to Int."""
        unique = sorted(series.unique())
        mapping = {label: i for i, label in enumerate(unique)}
        return torch.tensor([mapping[l] for l in series], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]