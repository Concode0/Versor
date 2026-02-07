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
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
from omegaconf import DictConfig

class BaseTask(ABC):
    """The Template. Every task follows this ritual.

    Setup -> Load -> Train -> Eval -> Visualize.
    Follows the standard lifecycle.

    Attributes:
        cfg (DictConfig): Config.
        device (str): Device.
        algebra (CliffordAlgebra): Math kernel.
        model (nn.Module): The brain.
        criterion (nn.Module): The judge.
        optimizer (optim.Optimizer): The teacher.
    """

    def __init__(self, cfg: DictConfig):
        """Sets up the task.

        Args:
            cfg (DictConfig): Hydra config.
        """
        self.cfg = cfg
        self.device = cfg.algebra.device
        self.algebra = self.setup_algebra()
        self.model = self.setup_model().to(self.device)
        self.criterion = self.setup_criterion()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.training.lr)
        self.epochs = cfg.training.epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        if cfg.get('checkpoint'):
            self.load_checkpoint(cfg.checkpoint)

    @abstractmethod
    def setup_algebra(self):
        """Spawns the algebra."""
        pass

    @abstractmethod
    def setup_model(self):
        """Builds the net."""
        pass

    @abstractmethod
    def setup_criterion(self):
        """Defines the loss."""
        pass

    @abstractmethod
    def get_data(self):
        """Fetches data."""
        pass

    @abstractmethod
    def train_step(self, data):
        """One step of optimization."""
        pass

    @abstractmethod
    def evaluate(self, data):
        """How bad is it?"""
        pass

    @abstractmethod
    def visualize(self, data):
        """Draws pretty pictures."""
        pass

    def save_checkpoint(self, path: str):
        """Dumps state to disk."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.cfg
        }
        torch.save(checkpoint, path)
        print(f">>> Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Resurrects state from disk."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f">>> Checkpoint loaded from {path}")

    def run(self):
        """Runs the show."""
        print(f">>> Starting Task: {self.cfg.name}")
        dataloader = self.get_data()
        
        # Training Loop
        self.model.train()
        pbar = tqdm(range(self.epochs))
        
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
                avg_loss, logs = self.train_step(dataloader)
            
            self.scheduler.step(avg_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logs['LR'] = current_lr
                
            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

        print(">>> Training Complete.")
        
        self.model.eval()
        with torch.no_grad():
            sample_data = next(iter(dataloader)) if is_loader else dataloader
            self.evaluate(sample_data)
            self.visualize(sample_data)