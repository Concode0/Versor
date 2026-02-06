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
import numpy as np
from torch.utils.data import Dataset
from core.algebra import CliffordAlgebra

class AMASSDataset(Dataset):
    """Synthetic AMASS. Because the real one is huge and licensed.

    Mimics human motion statistics.
    Walking, Running, Jumping.
    """

    def __init__(self, algebra: CliffordAlgebra, num_samples=1000, subset='train'):
        """Sets up the fake motion."""
        self.algebra = algebra
        self.num_samples = num_samples
        self.subset = subset
        
        self.classes = ['Walking', 'Running', 'Jumping']
        self.num_classes = len(self.classes)
        
        self.data, self.labels = self._generate_synthetic_motion()

    def _generate_synthetic_motion(self):
        """Generates wiggly lines."""
        feature_dim = 45 
        data = []
        labels = []
        
        # Random projection matrix
        P = np.random.randn(2, feature_dim)
        
        for i in range(self.num_samples):
            label = np.random.randint(0, self.num_classes)
            
            if label == 0: # Walking
                base = np.random.normal(loc=[-2, 0], scale=0.5, size=(2,))
            elif label == 1: # Running
                base = np.random.normal(loc=[2, 0], scale=0.5, size=(2,))
            else: # Jumping
                base = np.random.normal(loc=[0, 3], scale=0.5, size=(2,))
                
            motion_vec = np.tanh(np.dot(base, P)) + 0.1 * np.random.randn(feature_dim)
            
            data.append(torch.tensor(motion_vec, dtype=torch.float32))
            labels.append(label)
            
        return torch.stack(data), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]