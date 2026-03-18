# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""PonderNet-style adaptive computation controller.

Takes per-step halting probabilities and produces:
- Mixing weights for per-step outputs (geometric distribution)
- KL divergence against a geometric prior for regularization
- Expected number of computation steps per example
"""

import torch
import torch.nn as nn


class AdaptiveHalt(nn.Module):
    """PonderNet adaptive computation time controller.

    Computes mixing weights from per-step halt probabilities using a
    geometric distribution: p(halt at t) = lambda_t * prod_{s<t}(1 - lambda_s).

    KL regularization pushes the learned distribution toward a geometric prior
    p_prior(t) = lambda_p * (1 - lambda_p)^(t-1).
    """

    def __init__(self, lambda_p: float = 0.5, max_steps: int = 20, eps: float = 1e-7):
        super().__init__()
        self.lambda_p = lambda_p
        self.max_steps = max_steps
        self.eps = eps

        # Precompute geometric prior
        prior = torch.zeros(max_steps)
        for t in range(max_steps):
            prior[t] = lambda_p * ((1.0 - lambda_p) ** t)
        prior = prior / prior.sum()  # normalize
        self.register_buffer('prior', prior)

    def forward(self, halt_probs: list) -> dict:
        """Compute mixing weights and KL loss from per-step halt probabilities.

        Args:
            halt_probs: List of T tensors, each [B] (mean halt prob per example).

        Returns:
            dict with:
                'weights': [B, T] mixing weights for per-step outputs
                'expected_steps': [B] expected computation depth
                'kl_loss': scalar KL divergence against geometric prior
        """
        T = len(halt_probs)
        B = halt_probs[0].shape[0]
        device = halt_probs[0].device
        eps = self.eps

        # Stack halt probs: [T, B]
        lambdas = torch.stack(halt_probs, dim=0)  # [T, B]
        lambdas = lambdas.clamp(eps, 1.0 - eps)

        # Compute geometric distribution weights
        # p(halt at t) = lambda_t * prod_{s<t}(1 - lambda_s)
        log_survive = torch.log(1.0 - lambdas)            # [T, B]
        cumulative_log_survive = torch.cumsum(log_survive, dim=0)  # [T, B]
        # Shift: at t=0, no prior survival needed
        shifted = torch.zeros_like(cumulative_log_survive)
        shifted[1:] = cumulative_log_survive[:-1]

        log_weights = torch.log(lambdas) + shifted  # [T, B]
        weights = torch.exp(log_weights)             # [T, B]

        # Normalize (handles numerical issues at boundaries)
        weights = weights / (weights.sum(dim=0, keepdim=True) + eps)
        weights = weights.permute(1, 0)  # [B, T]

        # Expected steps: E[t] = sum_t t * p(halt=t), 1-indexed
        step_indices = torch.arange(1, T + 1, device=device, dtype=weights.dtype)
        expected_steps = (weights * step_indices.unsqueeze(0)).sum(dim=1)  # [B]

        # KL divergence: KL(q || p_prior) where q = learned halt distribution
        prior = self.prior[:T]  # truncate to actual steps used
        prior = prior / (prior.sum() + eps)  # renormalize
        prior = prior.unsqueeze(0).expand(B, -1)  # [B, T]

        kl_loss = (weights * (torch.log(weights + eps) - torch.log(prior + eps))).sum(dim=1)
        kl_loss = kl_loss.mean()  # average over batch

        return {
            'weights': weights,
            'expected_steps': expected_steps,
            'kl_loss': kl_loss,
        }
