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
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

class VersorModelNet(ModelNet):
    """Wrapper for ModelNet10."""
    pass

def get_modelnet_loader(root, batch_size=32, subset='train', rotated=False):
    """
    Args:
        subset: 'train' or 'test'.
        rotated: If True, apply random SO(3) rotations.
    """
    pre_transform = T.NormalizeScale()
    transform_list = [T.SamplePoints(1024)]
    
    if rotated:
        transform_list.append(T.RandomRotate(degrees=180, axis=0))
        transform_list.append(T.RandomRotate(degrees=180, axis=1))
        transform_list.append(T.RandomRotate(degrees=180, axis=2))
        
    transform = T.Compose(transform_list)
    
    dataset = ModelNet(root=root, name='10', train=(subset=='train'), 
                       transform=transform, pre_transform=pre_transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=(subset=='train'))