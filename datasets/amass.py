# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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
            
            # Convert to Multivector form
            # Map features to vector basis e_1 ... e_45
            # Requires high-dim algebra, but usually we project to lower dim first.
            # For this task, we treat input as raw features to be embedded.
            
            # We'll store it as a tensor [1, Dim] where Dim matches Algebra
            # Note: Algebra dim grows as 2^n. For n=45, it's impossible.
            # We must use a lower-dimensional embedding algebra (e.g. p=4, q=0 -> dim=16)
            # and project input features into it, OR use the input as raw features 
            # for the first layer (Linear).
            
            data.append(torch.tensor(motion_vec, dtype=torch.float32))
            labels.append(label)
            
        return torch.stack(data), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
