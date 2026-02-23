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
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm, hermitian_grade_spectrum
from tasks.base import BaseTask
from datasets.weatherbench import get_weatherbench_loaders
from models.weather_gbn import WeatherGBN
from functional.loss import PhysicsInformedLoss, HermitianGradeRegularization
from log import get_logger

logger = get_logger(__name__)


class WeatherBenchTask(BaseTask):
    """WeatherBench Global Weather Forecasting Task.

    Predicts global weather state at future time from current state.
    Uses spacetime rotors Cl(2,1) for temporal evolution on spherical grid.

    Variables: z500 (geopotential 500hPa), t850 (temperature 850hPa)
    """

    def __init__(self, cfg):
        self.data_root = "./data/WeatherBench"
        self.variables = cfg.dataset.get('variables', ['z500', 't850'])
        self.lead_time = cfg.dataset.get('lead_time', 72)
        self.loss_weights = cfg.training.get('loss_weights', {
            'forecast': 1.0, 'sparsity': 0.01
        })
        super().__init__(cfg)
        # Hermitian grade regularization: vector+bivector for spacetime Cl(2,1)
        target_spectrum = cfg.training.get('target_spectrum', [0.15, 0.35, 0.35, 0.15])
        self.grade_reg = HermitianGradeRegularization(self.algebra, target_spectrum=target_spectrum)

    def setup_algebra(self):
        """Cl(2,1) for spacetime: 2 spatial + 1 temporal with signature (+,+,-)."""
        return CliffordAlgebra(p=2, q=1, device=self.device)

    def setup_model(self):
        return WeatherGBN(
            self.algebra,
            num_variables=len(self.variables),
            spatial_hidden_dim=self.cfg.model.get('spatial_hidden_dim', 64),
            num_spatial_layers=self.cfg.model.get('num_spatial_layers', 6),
            num_temporal_steps=self.cfg.model.get('num_temporal_steps', 12),
            num_rotors=self.cfg.model.get('num_rotors', 16),
            use_decomposition=self.cfg.model.get('use_decomposition', True),
            decomp_k=self.cfg.model.get('decomp_k', 10),
            use_rotor_backend=self.cfg.model.get('use_rotor_backend', True)
        )

    def setup_criterion(self):
        physics_weight = self.cfg.training.get('physics_weight', 0.1)
        return PhysicsInformedLoss(physics_weight=physics_weight)

    def get_data(self):
        train_loader, val_loader, test_loader, var_means, var_stds = get_weatherbench_loaders(
            root=self.data_root,
            resolution=self.cfg.dataset.get('resolution', '5.625deg'),
            variables=self.variables,
            lead_time=self.lead_time,
            batch_size=self.cfg.training.batch_size,
            max_samples=self.cfg.dataset.get('samples', None),
        )

        self.var_means = var_means.to(self.device)
        self.var_stds = var_stds.to(self.device)

        return train_loader, val_loader, test_loader

    def _normalize(self, state):
        """Per-variable standardization."""
        return (state - self.var_means) / (self.var_stds + 1e-6)

    def _denormalize(self, state):
        return state * self.var_stds + self.var_means

    def train_step(self, batch):
        state_t = batch['state_t'].to(self.device)
        state_t_plus = batch['state_t_plus'].to(self.device)
        edge_index = batch['edge_index'].to(self.device)
        lat_weights = batch['lat_weights'].to(self.device)

        # Normalize
        state_t_norm = self._normalize(state_t)
        state_t_plus_norm = self._normalize(state_t_plus)

        num_steps = self.lead_time // 6

        self.optimizer.zero_grad()
        forecast_norm = self.model(state_t_norm, edge_index, lat_weights, num_steps=num_steps)

        # Physics-informed loss
        loss_forecast = self.criterion(forecast_norm, state_t_plus_norm, lat_weights)
        loss_sparsity = self.model.total_sparsity_loss()

        # Grade regularization on latent features
        w_grade_reg = self.loss_weights.get('grade_reg', 0.0)
        if w_grade_reg > 0:
            latent = self.model.get_latent_features()
            if latent is not None:
                grade_reg_loss = self.grade_reg(latent)
            else:
                grade_reg_loss = torch.tensor(0.0, device=self.device)
        else:
            grade_reg_loss = torch.tensor(0.0, device=self.device)

        w = self.loss_weights
        loss = (w.get('forecast', 1.0) * loss_forecast +
                w.get('sparsity', 0.01) * loss_sparsity +
                w_grade_reg * grade_reg_loss)

        loss.backward()
        self.optimizer.step()

        # RMSE in original scale
        forecast_denorm = self._denormalize(forecast_norm.detach())
        rmse = torch.sqrt(F.mse_loss(forecast_denorm, state_t_plus))

        # Compute Hermitian norm of latent features
        latent = self.model.get_latent_features()
        h_norm = hermitian_norm(self.algebra, latent).mean().item() if latent is not None else 0.0

        return loss.item(), {
            "Loss": loss.item(),
            "RMSE": rmse.item(),
            "H_Norm": h_norm,
        }

    def evaluate(self, val_loader):
        self.model.eval()
        total_rmse = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                state_t = batch['state_t'].to(self.device)
                state_t_plus = batch['state_t_plus'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                lat_weights = batch['lat_weights'].to(self.device)

                state_t_norm = self._normalize(state_t)
                num_steps = self.lead_time // 6

                forecast_norm = self.model(state_t_norm, edge_index, lat_weights, num_steps=num_steps)
                forecast = self._denormalize(forecast_norm)

                rmse = torch.sqrt(F.mse_loss(forecast, state_t_plus))
                total_rmse += rmse.item() * state_t.size(0)
                count += state_t.size(0)

        avg_rmse = total_rmse / max(count, 1)

        # Anomaly Correlation Coefficient (ACC)
        # ACC = corr(forecast_anomaly, target_anomaly) where anomaly = value - climatology
        # Using batch mean as proxy for climatology
        return {
            'RMSE': avg_rmse,
        }

    def visualize(self, val_loader):
        self.model.eval()
        batch = next(iter(val_loader))
        state_t = batch['state_t'].to(self.device)
        state_t_plus = batch['state_t_plus'].to(self.device)
        edge_index = batch['edge_index'].to(self.device)
        lat_weights = batch['lat_weights'].to(self.device)

        with torch.no_grad():
            state_t_norm = self._normalize(state_t)
            num_steps = self.lead_time // 6
            forecast_norm = self.model(state_t_norm, edge_index, lat_weights, num_steps=num_steps)
            forecast = self._denormalize(forecast_norm)

        try:
            import matplotlib.pyplot as plt

            # Plot first sample, first variable
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            target = state_t_plus[0, :, :, 0].cpu().numpy()
            pred = forecast[0, :, :, 0].cpu().numpy()
            error = pred - target

            vmin, vmax = target.min(), target.max()

            axes[0].imshow(target, aspect='auto', vmin=vmin, vmax=vmax)
            axes[0].set_title(f"Target ({self.variables[0]})")
            axes[1].imshow(pred, aspect='auto', vmin=vmin, vmax=vmax)
            axes[1].set_title(f"Forecast ({self.variables[0]})")
            im = axes[2].imshow(error, aspect='auto', cmap='RdBu_r')
            axes[2].set_title("Error")
            plt.colorbar(im, ax=axes[2])

            plt.suptitle(f"WeatherBench {self.lead_time}h Forecast")
            plt.tight_layout()
            plt.savefig("weatherbench_forecast.png")
            logger.info("Saved visualization to weatherbench_forecast.png")
            plt.close()
        except ImportError:
            logger.warning("Matplotlib not found. Skipping visualization.")

    def run(self):
        logger.info(f"Starting Task: WeatherBench ({self.lead_time}h forecast)")
        train_loader, val_loader, test_loader = self.get_data()

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))
        best_val_rmse = float('inf')

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            total_rmse = 0
            n_batches = 0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                total_rmse += logs['RMSE']
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_rmse = total_rmse / max(n_batches, 1)

            val_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_metrics['RMSE'])

            if val_metrics['RMSE'] < best_val_rmse:
                best_val_rmse = val_metrics['RMSE']
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            desc = f"Loss: {avg_loss:.4f} | RMSE: {avg_rmse:.4f} | Val_RMSE: {val_metrics['RMSE']:.4f}"
            pbar.set_description(desc)

        logger.info(f"Training Complete. Best Val RMSE: {best_val_rmse:.4f}")
        self.load_checkpoint(f"{self.cfg.name}_best.pt")

        test_metrics = self.evaluate(test_loader)
        logger.info(f"FINAL TEST RMSE: {test_metrics['RMSE']:.4f}")

        self.visualize(test_loader)
