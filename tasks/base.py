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
from log import get_logger
from core.device import DeviceConfig, resolve_device

logger = get_logger(__name__)

class BaseTask(ABC):
    """Abstract base class for all training tasks.

    Lifecycle: setup_algebra → setup_model → setup_criterion → get_data → train → evaluate → visualize.

    Attributes:
        cfg (DictConfig): Hydra configuration.
        device (str): Computation device.
        device_config (DeviceConfig): Full device/backend configuration.
        algebra (CliffordAlgebra): Clifford algebra kernel.
        model (nn.Module): Neural network model.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Parameter optimizer.
    """

    def __init__(self, cfg: DictConfig):
        """Sets up the task.

        Args:
            cfg (DictConfig): Hydra config.
        """
        self.cfg = cfg

        # Build DeviceConfig from Hydra config
        self.device_config = DeviceConfig(
            device=cfg.algebra.get('device', 'auto'),
            pin_memory=cfg.training.get('pin_memory', None),
            num_workers=cfg.training.get('num_workers', None),
            compile_model=cfg.training.get('compile', False),
            amp=cfg.training.get('amp', False),
            cudnn_benchmark=cfg.training.get('cudnn_benchmark', None),
        )
        self.device = self.device_config.device
        self.device_config.apply_backend_settings()

        self.algebra = self.setup_algebra()
        self.model = self.setup_model().to(self.device)
        self.model = self.device_config.maybe_compile(self.model)
        self.criterion = self.setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.epochs = cfg.training.epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        # AMP scaler (None when AMP is disabled)
        self._scaler = self.device_config.get_scaler()

        if cfg.get('checkpoint'):
            self.load_checkpoint(cfg.checkpoint)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to best available accelerator.

        Priority: cuda > mps > cpu.

        .. deprecated::
            Use :func:`core.device.resolve_device` instead.
        """
        return resolve_device(device)

    def _setup_optimizer(self):
        """Sets up optimizer based on config.

        Default: RiemannianAdam (true manifold optimization on Spin(n))

        Supports:
        - 'riemannian_adam': Adam with exponential retraction (Riemannian, DEFAULT)
        - 'exponential_sgd': SGD with exponential retraction (Riemannian)
        - 'adamw': Standard AdamW (Euclidean, for ablation experiments only)

        Returns:
            Configured optimizer instance.
        """
        opt_type = self.cfg.training.get('optimizer_type', 'riemannian_adam')
        lr = self.cfg.training.lr

        if opt_type == 'exponential_sgd':
            from optimizers.riemannian import ExponentialSGD
            return ExponentialSGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.cfg.training.get('momentum', 0.9),
                algebra=self.algebra,
                max_bivector_norm=self.cfg.training.get('max_bivector_norm', 10.0)
            )
        elif opt_type == 'riemannian_adam':
            from optimizers.riemannian import RiemannianAdam
            return RiemannianAdam(
                self.model.parameters(),
                lr=lr,
                betas=self.cfg.training.get('betas', (0.9, 0.999)),
                algebra=self.algebra,
                max_bivector_norm=self.cfg.training.get('max_bivector_norm', 10.0)
            )
        else:
            # Euclidean AdamW (for ablation experiments only)
            # Note: Treats Spin(n) as flat space, theoretically incorrect for rotor parameters
            return optim.AdamW(self.model.parameters(), lr=lr)

    @abstractmethod
    def setup_algebra(self):
        """Initialize the Clifford algebra."""
        pass

    @abstractmethod
    def setup_model(self):
        """Construct the neural network model."""
        pass

    @abstractmethod
    def setup_criterion(self):
        """Define the loss function."""
        pass

    @abstractmethod
    def get_data(self):
        """Load and return the dataset."""
        pass

    @abstractmethod
    def train_step(self, data):
        """One step of optimization."""
        pass

    @abstractmethod
    def evaluate(self, data):
        """Evaluate the model and return metrics."""
        pass

    @abstractmethod
    def visualize(self, data):
        """Generate visualizations of model outputs."""
        pass

    def _backward(self, loss: torch.Tensor) -> None:
        """Backward pass with optional AMP grad scaling."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        """Optimizer step with optional AMP scaler unscaling."""
        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

    def save_checkpoint(self, path: str):
        """Save model, optimizer, and scheduler state to disk."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.cfg
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str):
        """Restore model, optimizer, and scheduler state from disk."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Checkpoint loaded from %s", path)

    def run(self):
        """Execute the full training loop."""
        logger.info("Starting Task: %s", self.cfg.name)
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

        logger.info("Training Complete.")
        
        self.model.eval()
        with torch.no_grad():
            sample_data = next(iter(dataloader)) if is_loader else dataloader
            self.evaluate(sample_data)
            self.visualize(sample_data)