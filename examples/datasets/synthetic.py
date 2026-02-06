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

class Figure8Dataset(Dataset):
    """Fake data. Figure-8 distorted by z=0.5xy.

    Tests if the model can flatten a manifold.
    """

    def __init__(self, algebra: CliffordAlgebra, num_samples=1000):
        """Sets up the dataset."""
        self.algebra = algebra
        self.data = self._generate(num_samples)

    def _generate(self, n):
        """Math generator."""
        t = torch.linspace(0, 2*np.pi, n)
        x = torch.sin(t)
        y = torch.sin(t) * torch.cos(t)
        z = 0.5 * x * y

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