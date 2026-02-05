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
import numpy as np
from torch.utils.data import Dataset
from core.algebra import CliffordAlgebra

class AMASSDataset(Dataset):
    """Dataset handler for AMASS (Archive of Motion Capture as Surface Shapes).
    
    Since the actual dataset is large and requires a license/download, this class
    implements a synthetic generator that mimics the statistical properties of 
    human motion data for prototyping.

    Structure:
    - Input: Skeleton joint rotations/positions [Frames, Joints, 3].
    - Labels: Action classes (0: Walking, 1: Running, 2: Jumping, etc.).
    """

    def __init__(self, algebra: CliffordAlgebra, num_samples=1000, subset='train'):
        """Initializes the dataset.

        Args:
            algebra (CliffordAlgebra): Algebra instance.
            num_samples (int): Number of motion sequences to generate.
            subset (str): 'train' or 'test'.
        """
        self.algebra = algebra
        self.num_samples = num_samples
        self.subset = subset
        
        # Action classes
        self.classes = ['Walking', 'Running', 'Jumping']
        self.num_classes = len(self.classes)
        
        # Generate data
        self.data, self.labels = self._generate_synthetic_motion()

    def _generate_synthetic_motion(self):
        """Generates synthetic high-dimensional motion signatures."""
        # Feature dim: 15 joints * 3 coordinates = 45 dims
        feature_dim = 45 
        data = []
        labels = []
        
        # Non-linear projection to "Joint Space"
        # Random projection matrix
        P = np.random.randn(2, feature_dim)
        
        for i in range(self.num_samples):
            # Randomly assign a class
            label = np.random.randint(0, self.num_classes)
            
            # Base motion signature (latent vector)
            # Walking: Periodic, low frequency
            # Running: Periodic, high frequency
            # Jumping: Impulsive
            
            # We simulate this by creating a latent vector 'z' that is perfectly separable
            # then projecting it to high-dim space non-linearly.
            
            if label == 0: # Walking
                base = np.random.normal(loc=[-2, 0], scale=0.5, size=(2,))
            elif label == 1: # Running
                base = np.random.normal(loc=[2, 0], scale=0.5, size=(2,))
            else: # Jumping
                base = np.random.normal(loc=[0, 3], scale=0.5, size=(2,))
                
            # x = tanh(z * P) + noise
            motion_vec = np.tanh(np.dot(base, P)) + 0.1 * np.random.randn(feature_dim)
            
            data.append(torch.tensor(motion_vec, dtype=torch.float32))
            labels.append(label)
            
        return torch.stack(data), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
