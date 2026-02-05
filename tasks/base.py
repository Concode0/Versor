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
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
from omegaconf import DictConfig

class BaseTask(ABC):
    """Abstract base class for all geometric learning tasks.

    Standardizes the lifecycle: setup -> data loading -> training loop -> evaluation -> visualization.

    Attributes:
        cfg (DictConfig): Hydra configuration.
        device (str): Computation device.
        algebra (CliffordAlgebra): The algebra instance.
        model (nn.Module): The neural network.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        epochs (int): Number of training epochs.
    """

    def __init__(self, cfg: DictConfig):
        """Initializes the task.

        Args:
            cfg (DictConfig): Configuration object containing task and training parameters.
        """
        self.cfg = cfg
        self.device = cfg.algebra.device
        self.algebra = self.setup_algebra()
        self.model = self.setup_model().to(self.device)
        self.criterion = self.setup_criterion()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.training.lr)
        self.epochs = cfg.training.epochs

    @abstractmethod
    def setup_algebra(self):
        """Initializes and returns the Clifford Algebra instance."""
        pass

    @abstractmethod
    def setup_model(self):
        """Initializes and returns the Neural Network model."""
        pass

    @abstractmethod
    def setup_criterion(self):
        """Initializes and returns the loss function."""
        pass

    @abstractmethod
    def get_data(self):
        """Loads or generates data. Must return a DataLoader or Tensor."""
        pass

    @abstractmethod
    def train_step(self, data):
        """Performs a single training step (forward + backward).

        Args:
            data: Input batch.

        Returns:
            tuple: (scalar loss, dict of log metrics).
        """
        pass

    @abstractmethod
    def evaluate(self, data):
        """Runs evaluation metrics after training."""
        pass

    @abstractmethod
    def visualize(self, data):
        """Generates and saves visualizations of the results."""
        pass

    def run(self):
        """Executes the full task lifecycle."""
        print(f">>> Starting Task: {self.cfg.name}")
        dataloader = self.get_data()
        
        # Training Loop
        self.model.train()
        pbar = tqdm(range(self.epochs))
        
        # Determine if we are using a DataLoader or a single Tensor batch
        is_loader = not isinstance(dataloader, (torch.Tensor, tuple, list))
        
        for epoch in pbar:
            if is_loader:
                total_loss = 0
                for batch in dataloader:
                    loss, logs = self.train_step(batch)
                    total_loss += loss
                avg_loss = total_loss / len(dataloader)
                logs['Loss'] = avg_loss
            else:
                loss, logs = self.train_step(dataloader)
                
            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

        print(">>> Training Complete.")
        
        self.model.eval()
        with torch.no_grad():
            # Evaluate on a sample/batch for simplicity
            sample_data = next(iter(dataloader)) if is_loader else dataloader
            self.evaluate(sample_data)
            self.visualize(sample_data)
