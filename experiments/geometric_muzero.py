# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

r"""Geometric MuZero: Replacing MCTS with Geometric Superposition Search.

HYPOTHESIS

Rotor-based dynamics + multi-rotor superposition search can match or exceed
Monte Carlo Tree Search (MCTS) sample efficiency while providing geometric
guarantees that MCTS cannot:

  1. **Equivariant policy**: Rotors form a group, so rotating the input state
     by any rotor produces a correspondingly rotated policy — structural, not
     learned.
  2. **Invariant value**: The value function extracts grade-0 (scalar)
     components, which are invariant under sandwich products by construction.
  3. **Uncertainty from bivector energy**: High grade-2 norm in a state signals
     rich unexplored relational structure — a geometric exploration signal
     that replaces UCB heuristics.
  4. **Superposition search**: Instead of sequential tree expansion, K
     hypothetical futures are evaluated simultaneously via independent
     rotor transformations — Geometric Superposition Search (GSS).

Core idea — Cl(1,1) as search algebra:
  The bivector e₊e₋ in Cl(1,1) squares to +1, giving a *hyperbolic* rotor:
      R_search = exp(θ · e₊e₋) = cosh(θ) + sinh(θ) · e₊e₋
  tanh(θ) smoothly interpolates [0, 1) as an exploration rate, replacing
  UCB's c·√(ln N / n) with a geometrically principled alternative.

State algebra — Cl(3,0):
  Grade 0 (idx 0):     scalar invariants  → value
  Grade 1 (idx 1,2,4): directional        → policy / actions
  Grade 2 (idx 3,5,6): relational         → dynamics / uncertainty
  Grade 3 (idx 7):     pseudoscalar        → topological info

Current State:

Experimental prototype. Validates on a synthetic grid planning environment
(no external dependencies beyond PyTorch). The architecture implements all
three MuZero functions (representation, dynamics, prediction) in Clifford
algebra, with GSS replacing MCTS.

==============================================================================
CALL FOR PARTICIPANTS
==============================================================================
This is an open experiment. We welcome contributions extending GSS to:
  - Continuous control (MuJoCo, DMControl) via higher-dimensional algebras
  - Board games (Go, Chess) where tree search is dominant
  - Atari / visual domains with convolutional representation networks
  - Theoretical analysis of GSS convergence properties

If you use ideas from this experiment, please cite the Versor framework:

@software{Kim_Versor_Universal_Geometric_2026,
author = {Kim, Eunkyum},
doi = {10.5281/zenodo.18939519},
month = mar,
title = {{Versor: Universal Geometric Algebra Neural Network}},
url = {https://github.com/Concode0/versor},
version = {1.0.0},
year = {2026}
}
"""

from __future__ import annotations

import sys
import os
import argparse
import collections
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from core.metric import hermitian_distance, hermitian_grade_spectrum
from layers import CliffordLinear, RotorLayer, CliffordLayerNorm
from layers.primitives.base import CliffordModule
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam


class GridPlanningEnv:
    """Deterministic grid world with obstacles and a goal.

    Observations are 3-channel grids (agent / goal / obstacles) flattened
    to a 1-D vector.  No external dependencies.

    Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
    Rewards: +1.0 goal, -0.1 per step, -0.5 wall/obstacle collision
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

    def __init__(self, grid_size: int = 5, num_obstacles: int = 3,
                 seed: int = 42, max_steps: int = 50):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.num_actions = 5
        self.obs_dim = 3 * grid_size * grid_size

        rng = random.Random(seed)
        cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        rng.shuffle(cells)
        self.goal = cells[0]
        self.obstacles = set(cells[1:1 + num_obstacles])
        self._start_candidates = [
            c for c in cells[1 + num_obstacles:]
            if c != self.goal and c not in self.obstacles
        ]
        self._rng = rng
        self.agent = None
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.agent = self._rng.choice(self._start_candidates)
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        dr, dc = self.ACTIONS[action]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        self.steps += 1

        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            reward = -0.5
        elif (nr, nc) in self.obstacles:
            reward = -0.5
        else:
            self.agent = (nr, nc)
            reward = -0.1

        done = False
        if self.agent == self.goal:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._obs(), reward, done

    def _obs(self) -> np.ndarray:
        g = self.grid_size
        obs = np.zeros((3, g, g), dtype=np.float32)
        obs[0, self.agent[0], self.agent[1]] = 1.0
        obs[1, self.goal[0], self.goal[1]] = 1.0
        for (r, c) in self.obstacles:
            obs[2, r, c] = 1.0
        return obs.reshape(-1)

    def render(self) -> str:
        lines = []
        g = self.grid_size
        for r in range(g):
            row = []
            for c in range(g):
                if (r, c) == self.agent:
                    row.append('A')
                elif (r, c) == self.goal:
                    row.append('G')
                elif (r, c) in self.obstacles:
                    row.append('#')
                else:
                    row.append('.')
            lines.append(' '.join(row))
        return '\n'.join(lines)


class ReplayBuffer:
    """Simple deque-based replay buffer with n-step return computation."""

    def __init__(self, capacity: int = 10000):
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool):
        self.buffer.append((obs, action, reward, next_obs, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, device: str = 'cpu'
               ) -> Tuple[torch.Tensor, ...]:
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        obs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        next_obs = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        return obs, actions, rewards, next_obs, dones

    def compute_n_step_returns(self, gamma: float, n: int, device: str = 'cpu'
                               ) -> Tuple[torch.Tensor, ...]:
        """Sample transitions with n-step bootstrapped returns."""
        if len(self.buffer) < n + 1:
            return self.sample(len(self.buffer), device)

        valid = len(self.buffer) - n
        indices = random.sample(range(valid), min(64, valid))

        obs_list, action_list, return_list, next_obs_list, done_list = [], [], [], [], []
        for i in indices:
            G = 0.0
            final_done = False
            for k in range(n):
                _, _, r, _, d = self.buffer[i + k]
                G += (gamma ** k) * r
                if d:
                    final_done = True
                    break
            obs_list.append(self.buffer[i][0])
            action_list.append(self.buffer[i][1])
            return_list.append(G)
            last = min(i + n, len(self.buffer) - 1)
            next_obs_list.append(self.buffer[last][3])
            done_list.append(float(final_done))

        obs = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        actions = torch.tensor(action_list, dtype=torch.long, device=device)
        returns = torch.tensor(return_list, dtype=torch.float32, device=device).unsqueeze(-1)
        next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=device)
        dones = torch.tensor(done_list, dtype=torch.float32, device=device).unsqueeze(-1)
        return obs, actions, returns, next_obs, dones


class _GABlock(nn.Module):
    """CliffordLayerNorm -> GeometricGELU -> RotorLayer -> CliffordLinear."""

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        super().__init__()
        self.norm = CliffordLayerNorm(algebra, channels)
        self.act = GeometricGELU(algebra, channels)
        self.rotor = RotorLayer(algebra, channels)
        self.linear = CliffordLinear(algebra, channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(h)
        h = self.rotor(h)
        h = self.linear(h)
        return x + h


class RepresentationNet(nn.Module):
    """Lifts flat observations into multivector states in Cl(p,q)."""

    def __init__(self, algebra: CliffordAlgebra, obs_dim: int,
                 hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.algebra = algebra
        self.hidden_channels = hidden_channels
        self.lift = nn.Linear(obs_dim, hidden_channels * algebra.dim)
        self.blocks = nn.ModuleList([
            _GABlock(algebra, hidden_channels) for _ in range(num_layers)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [B, obs_dim] -> state: [B, C, D]"""
        B = obs.shape[0]
        x = self.lift(obs)                                       # [B, C*D]
        x = x.reshape(B, self.hidden_channels, self.algebra.dim) # [B, C, D]
        for block in self.blocks:
            x = block(x)
        return x


class DynamicsNet(nn.Module):
    """Applies action as bivector-rotor to state, then refines.

    Actions are encoded as learnable bivectors.  The dynamics step is:
      1. Look up action bivector B_a
      2. Compute rotor R_a = exp(-B_a / 2)
      3. Sandwich: s' = R_a s R~_a   (geometric state transition)
      4. Refine: 1x GA block for non-isometric corrections
      5. Predict reward from grade-0 of refined state
    """

    def __init__(self, algebra: CliffordAlgebra, num_actions: int = 5,
                 hidden_channels: int = 32):
        super().__init__()
        self.algebra = algebra
        self.hidden_channels = hidden_channels

        # Grade-2 bivector indices
        bv_mask = algebra.grade_masks[2]
        self.register_buffer('bv_indices',
                             bv_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_bv = len(self.bv_indices)

        # Learnable action bivectors [num_actions, num_bv]
        self.action_bivectors = nn.Parameter(torch.randn(num_actions, self.num_bv) * 0.1)

        # Post-rotor refinement
        self.refine = _GABlock(algebra, hidden_channels)

        # Reward head: grade-0 -> scalar
        self.reward_head = nn.Linear(hidden_channels, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state: [B, C, D], action: [B] (long)
        returns: (reward [B, 1], next_state [B, C, D])
        """
        B = state.shape[0]
        D = self.algebra.dim

        # Build full bivector multivector
        bv_coeffs = self.action_bivectors[action]        # [B, num_bv]
        bv_full = torch.zeros(B, D, device=state.device, dtype=state.dtype)
        idx = self.bv_indices.unsqueeze(0).expand(B, -1)
        bv_full.scatter_(1, idx, bv_coeffs)

        # Rotor: R = exp(-B/2)
        R = self.algebra.exp(-0.5 * bv_full)             # [B, D]
        R_rev = self.algebra.reverse(R)                   # [B, D]

        # Sandwich product: R state R~
        next_state = self.algebra.sandwich_product(R, state, R_rev)  # [B, C, D]

        # Refine
        next_state = self.refine(next_state)

        # Reward from grade-0
        grade0 = next_state[:, :, 0]                     # [B, C]
        reward = self.reward_head(grade0)                 # [B, 1]

        return reward, next_state


class PredictionNet(nn.Module):
    """Extracts policy and value from a multivector state.

    Value head uses grade-0 (scalar) — invariant under rotor transforms.
    Policy head uses grade-1 (vector) — directional action features.
    """

    def __init__(self, algebra: CliffordAlgebra, hidden_channels: int = 32,
                 num_actions: int = 5):
        super().__init__()
        self.algebra = algebra
        mid = hidden_channels // 2

        self.norm = CliffordLayerNorm(algebra, hidden_channels)
        self.linear = CliffordLinear(algebra, hidden_channels, mid)
        self.act = GeometricGELU(algebra, mid)

        # Grade-1 indices for policy
        g1_mask = algebra.grade_masks[1]
        self.register_buffer('grade1_indices',
                             g1_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_g1 = len(self.grade1_indices)

        # Value: grade-0 -> scalar
        self.value_head = nn.Linear(mid, 1)

        # Policy: grade-1 components -> action logits
        self.policy_head = nn.Linear(mid * self.num_g1, num_actions)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state: [B, C, D]
        returns: (policy_logits [B, A], value [B, 1])
        """
        x = self.norm(state)
        x = self.linear(x)
        x = self.act(x)                                  # [B, mid, D]

        # Value from grade-0
        grade0 = x[:, :, 0]                              # [B, mid]
        value = self.value_head(grade0)                   # [B, 1]

        # Policy from grade-1
        g1 = x[:, :, self.grade1_indices]                 # [B, mid, num_g1]
        g1_flat = g1.reshape(x.shape[0], -1)              # [B, mid * num_g1]
        policy = self.policy_head(g1_flat)                # [B, A]

        return policy, value


class HypothesisGenerator(CliffordModule):
    """Generates H hypothesis states via independent rotors.

    Like MultiRotorLayer but returns individual R_h x R~_h outputs
    *before* the weighted einsum reduction.  Each rotor produces an
    independent view of the state — the "superposition" of futures.
    """

    def __init__(self, algebra: CliffordAlgebra, num_hypotheses: int = 8):
        super().__init__(algebra)
        self.num_hypotheses = num_hypotheses

        bv_mask = algebra.grade_masks[2]
        self.register_buffer('bv_indices',
                             bv_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_bv = len(self.bv_indices)

        self.rotor_bivectors = nn.Parameter(
            torch.randn(num_hypotheses, self.num_bv) * 0.05
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, C, D] -> hypotheses: [B, H, C, D]
        """
        H = self.num_hypotheses
        D = self.algebra.dim
        device, dtype = state.device, state.dtype

        # Build bivector multivectors [H, D]
        bv_full = torch.zeros(H, D, device=device, dtype=dtype)
        idx = self.bv_indices.unsqueeze(0).expand(H, -1)
        bv_full.scatter_(1, idx, self.rotor_bivectors)

        # Rotors
        R = self.algebra.exp(-0.5 * bv_full)             # [H, D]
        R_rev = self.algebra.reverse(R)                   # [H, D]

        # Sandwich each hypothesis: R_h state R~_h
        # state: [B, C, D] -> [B, 1, C, D]
        # R:     [H, D]    -> [1, H, 1, D]
        x = state.unsqueeze(1)                            # [B, 1, C, D]
        R_exp = R.view(1, H, 1, D)
        R_rev_exp = R_rev.view(1, H, 1, D)

        Rx = self.algebra.geometric_product(R_exp, x)
        hypotheses = self.algebra.geometric_product(Rx, R_rev_exp)  # [B, H, C, D]

        return hypotheses


class SearchParameterNet(nn.Module):
    """Maps state to exploration parameter theta.

    Uses grade norms as features, outputs theta such that tanh(theta)
    gives the exploration rate.  Conceptually this lives in Cl(1,1):
      R_search = cosh(theta) + sinh(theta) * e_{+-}
    where tanh(theta) is the explore/exploit ratio.
    """

    def __init__(self, algebra: CliffordAlgebra, hidden_channels: int = 32):  # noqa: ARG002
        super().__init__()
        num_grades = algebra.n + 1
        self.mlp = nn.Sequential(
            nn.Linear(num_grades, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state: torch.Tensor, algebra: CliffordAlgebra
                ) -> torch.Tensor:
        """
        state: [B, C, D] -> theta: [B, 1]
        """
        # Grade norms: [B, C, num_grades]
        flat = state.reshape(-1, state.shape[-1])
        norms = algebra.get_grade_norms(flat)             # [B*C, G]
        norms = norms.reshape(state.shape[0], state.shape[1], -1)
        pooled = norms.mean(dim=1)                        # [B, G]
        theta = self.mlp(pooled)                          # [B, 1]
        return theta


class GeometricSuperpositionSearch(nn.Module):
    """Multi-hypothesis search using rotor superposition.

    Instead of sequential MCTS tree expansion:
      1. Generate H hypothesis states via independent rotors
      2. For each depth step, predict policy/value, select actions,
         advance through dynamics
      3. Use bivector energy as uncertainty signal for exploration
      4. Coherence check: if hypotheses collapse, perturb with random rotor
      5. Collapse: weight hypotheses by accumulated value, aggregate actions
    """

    def __init__(self, state_algebra: CliffordAlgebra,
                 search_algebra: CliffordAlgebra,
                 dynamics: DynamicsNet, prediction: PredictionNet,
                 num_hypotheses: int = 8, search_depth: int = 3,
                 num_actions: int = 5, discount: float = 0.99,
                 temperature: float = 1.0):
        super().__init__()
        self.state_algebra = state_algebra
        self.search_algebra = search_algebra
        self.dynamics = dynamics
        self.prediction = prediction
        self.num_hypotheses = num_hypotheses
        self.search_depth = search_depth
        self.num_actions = num_actions
        self.discount = discount
        self.temperature = temperature

        self.hypothesis_gen = HypothesisGenerator(state_algebra, num_hypotheses)
        self.search_param = SearchParameterNet(state_algebra)

        # Min Hermitian distance before coherence perturbation
        self.coherence_threshold = 0.1

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Run GSS and return improved action probabilities.

        state: [B, C, D] -> action_probs: [B, num_actions]
        """
        B = state.shape[0]
        H = self.num_hypotheses
        D = self.search_depth
        A = self.num_actions
        device = state.device

        # 1. Generate hypotheses
        hypotheses = self.hypothesis_gen(state)           # [B, H, C, D_alg]

        # 2. Search parameter (explore/exploit balance)
        theta = self.search_param(state, self.state_algebra)  # [B, 1]
        explore_weight = torch.abs(torch.tanh(theta))                     # [B, 1]

        # Track values and first-step actions
        hyp_values = torch.zeros(B, H, device=device)
        first_actions = torch.zeros(B, H, dtype=torch.long, device=device)

        # 3. Search loop
        for d in range(D):
            for h in range(H):
                h_state = hypotheses[:, h]                # [B, C, D_alg]

                # Predict
                policy_logits, value = self.prediction(h_state)  # [B, A], [B, 1]

                # Bivector energy as uncertainty
                spectrum = hermitian_grade_spectrum(
                    self.state_algebra,
                    h_state.reshape(-1, h_state.shape[-1])
                )  # [B*C, num_grades]
                spectrum = spectrum.reshape(B, h_state.shape[1], -1)
                bv_energy = spectrum[:, :, 2].mean(dim=1, keepdim=True)  # [B, 1]

                # Exploration-modulated action selection
                noise = torch.randn(B, A, device=device) * 0.3
                scores = policy_logits + explore_weight * bv_energy * noise
                action = scores.argmax(dim=-1)            # [B]

                if d == 0:
                    first_actions[:, h] = action

                # Advance through dynamics
                reward, next_state = self.dynamics(h_state, action)
                hyp_values[:, h] += (self.discount ** d) * reward.squeeze(-1)
                hypotheses[:, h] = next_state

            # Coherence check: perturb collapsed hypotheses
            self._coherence_check(hypotheses)

        # 4. Terminal value bootstrap
        for h in range(H):
            _, terminal_value = self.prediction(hypotheses[:, h])
            hyp_values[:, h] += (self.discount ** D) * terminal_value.squeeze(-1)

        # 5. Collapse: aggregate first-step actions weighted by value
        weights = F.softmax(hyp_values / self.temperature, dim=-1)  # [B, H]
        action_probs = torch.zeros(B, A, device=device)
        for h in range(H):
            one_hot = F.one_hot(first_actions[:, h], A).float()     # [B, A]
            action_probs += weights[:, h:h+1] * one_hot

        # Normalize (in case multiple hypotheses chose same action)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return action_probs

    def _coherence_check(self, hypotheses: torch.Tensor):
        """Perturb hypotheses that are too similar (in-place).

        hypotheses: [B, H, C, D]
        """
        B, H, C, D = hypotheses.shape
        for i in range(H):
            for j in range(i + 1, H):
                # Mean Hermitian distance across channels
                diff = hypotheses[:, i] - hypotheses[:, j]   # [B, C, D]
                flat = diff.reshape(-1, D)
                dist = hermitian_distance(self.state_algebra, flat,
                                          torch.zeros_like(flat))  # [B*C, 1]
                mean_dist = dist.reshape(B, C).mean()

                if mean_dist < self.coherence_threshold:
                    # Perturb the worse hypothesis with a random rotor
                    bv = torch.randn(1, D, device=hypotheses.device) * 0.1
                    bv = self.state_algebra.grade_projection(bv, 2)
                    R = self.state_algebra.exp(-0.5 * bv)          # [1, D]
                    R_rev = self.state_algebra.reverse(R)           # [1, D]
                    # Use sandwich_product: R [N, D], x [N, C, D]
                    # Expand R to [B, D]
                    R_b = R.expand(B, -1)
                    R_rev_b = R_rev.expand(B, -1)
                    perturbed = self.state_algebra.sandwich_product(
                        R_b, hypotheses[:, j], R_rev_b)            # [B, C, D]
                    hypotheses[:, j] = perturbed


class GeometricMuZero(nn.Module):
    """Full Geometric MuZero agent.

    Composes:
      - RepresentationNet: observations -> multivector state
      - DynamicsNet: state + action -> (reward, next_state) via action-rotors
      - PredictionNet: state -> (policy, value) via grade extraction
      - GSS: multi-hypothesis search replacing MCTS
    """

    def __init__(self, state_algebra: CliffordAlgebra,
                 search_algebra: CliffordAlgebra,
                 obs_dim: int, num_actions: int = 5,
                 hidden_channels: int = 32, num_rep_layers: int = 2,
                 num_hypotheses: int = 8, search_depth: int = 3,
                 discount: float = 0.99, temperature: float = 1.0):
        super().__init__()
        self.state_algebra = state_algebra
        self.search_algebra = search_algebra

        self.representation = RepresentationNet(
            state_algebra, obs_dim, hidden_channels, num_rep_layers)
        self.dynamics = DynamicsNet(
            state_algebra, num_actions, hidden_channels)
        self.prediction = PredictionNet(
            state_algebra, hidden_channels, num_actions)
        self.gss = GeometricSuperpositionSearch(
            state_algebra, search_algebra,
            self.dynamics, self.prediction,
            num_hypotheses, search_depth, num_actions,
            discount, temperature)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: obs -> (policy, value, gss_policy).

        Returns both the raw network predictions and the GSS-improved policy.
        """
        state = self.representation(obs)
        policy, value = self.prediction(state)
        gss_policy = self.gss(state)
        return policy, value, gss_policy

    def predict(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Network-only prediction (no search)."""
        state = self.representation(obs)
        return self.prediction(state)

    def predict_dynamics(self, obs: torch.Tensor, action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state and reward from observation + action."""
        state = self.representation(obs)
        reward, next_state = self.dynamics(state, action)
        return state, reward, next_state


def _self_play(model: GeometricMuZero, env: GridPlanningEnv,
               replay: ReplayBuffer, num_episodes: int, device: str,
               epsilon: float = 0.3):
    """Collect episodes using GSS-guided policy with epsilon-greedy."""
    model.eval()
    total_returns = []

    for _ in range(num_episodes):
        obs = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, env.num_actions - 1)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    state = model.representation(obs_t)
                    gss_policy = model.gss(state)             # [1, A]
                action = gss_policy.squeeze(0).argmax().item()

            next_obs, reward, done = env.step(action)
            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward

        total_returns.append(ep_return)

    model.train()
    return total_returns


def _train_step(model: GeometricMuZero, replay: ReplayBuffer,
                optimizer: torch.optim.Optimizer,
                gamma: float, n_step: int, device: str) -> Dict[str, float]:
    """Single gradient update on a batch from the replay buffer.

    Uses actor-critic policy gradient (not GSS targets) so that the policy
    learns from actual environment experience rather than its own
    (initially uninformative) search output.
    """
    if len(replay) < 64:
        return {}

    obs, actions, returns, next_obs, dones = replay.compute_n_step_returns(
        gamma, n_step, device)

    # Network predictions
    state = model.representation(obs)
    policy_logits, value = model.prediction(state)

    # Dynamics prediction
    pred_reward, pred_next_state = model.dynamics(state, actions)
    with torch.no_grad():
        target_next_state = model.representation(next_obs)

    # Value target: n-step return + bootstrap
    with torch.no_grad():
        _, next_value = model.prediction(target_next_state)
        value_target = returns + (1.0 - dones) * (gamma ** n_step) * next_value

    # Policy gradient with advantage
    log_probs = F.log_softmax(policy_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1))  # [B, 1]
    with torch.no_grad():
        advantage = value_target - value
    loss_policy = -(action_log_probs * advantage).mean()

    # Entropy bonus for exploration
    probs = F.softmax(policy_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    loss_entropy = -0.01 * entropy

    loss_value = F.mse_loss(value, value_target)
    loss_reward = F.mse_loss(pred_reward, returns[:, :1])
    loss_dynamics = F.mse_loss(pred_next_state, target_next_state.detach())

    # Bivector regularization
    loss_bv_reg = model.dynamics.action_bivectors.abs().mean()

    loss = (loss_policy + loss_entropy + loss_value
            + loss_reward + 0.5 * loss_dynamics + 0.01 * loss_bv_reg)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'total': loss.item(),
        'policy': loss_policy.item(),
        'value': loss_value.item(),
        'reward': loss_reward.item(),
        'dynamics': loss_dynamics.item(),
        'bv_reg': loss_bv_reg.item(),
    }


@torch.no_grad()
def _evaluate(model: GeometricMuZero, env: GridPlanningEnv,
              num_episodes: int, device: str, use_gss: bool = True
              ) -> Dict[str, float]:
    """Evaluate agent with greedy policy (optionally with GSS)."""
    model.eval()
    returns = []
    steps_list = []
    goals_reached = 0

    for _ in range(num_episodes):
        obs = env.reset()
        ep_return = 0.0
        done = False
        ep_steps = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if use_gss:
                state = model.representation(obs_t)
                policy = model.gss(state)
            else:
                policy, _ = model.predict(obs_t)
                policy = F.softmax(policy, dim=-1)
            action = policy.squeeze(0).argmax().item()
            obs, reward, done = env.step(action)
            ep_return += reward
            ep_steps += 1
        returns.append(ep_return)
        steps_list.append(ep_steps)
        if reward == 1.0:
            goals_reached += 1

    model.train()
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_steps': np.mean(steps_list),
        'goal_rate': goals_reached / num_episodes,
    }


def _save_plots(history: Dict[str, List], output_dir: str):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Training losses
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    loss_keys = ['total', 'policy', 'value', 'reward', 'dynamics', 'bv_reg']
    for ax, key in zip(axes.flat, loss_keys):
        if key in history and history[key]:
            ax.plot(history[key], linewidth=0.8)
            ax.set_title(f'Loss: {key}')
            ax.set_xlabel('Update')
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'losses.png'), dpi=150)
    plt.close()

    # 2. Episode returns
    if 'returns' in history and history['returns']:
        fig, ax = plt.subplots(figsize=(10, 5))
        rets = history['returns']
        ax.plot(rets, alpha=0.3, color='blue', linewidth=0.5)
        # Running mean
        window = min(20, len(rets))
        if len(rets) >= window:
            running = np.convolve(rets, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rets)), running, color='blue', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Episode Returns')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'returns.png'), dpi=150)
        plt.close()

    # 3. Evaluation metrics
    if 'eval_return' in history and history['eval_return']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs = history.get('eval_epochs', list(range(len(history['eval_return']))))
        ax1.plot(epochs, history['eval_return'], 'o-', color='green')
        ax1.fill_between(epochs,
                         np.array(history['eval_return']) - np.array(history.get('eval_std', [0]*len(epochs))),
                         np.array(history['eval_return']) + np.array(history.get('eval_std', [0]*len(epochs))),
                         alpha=0.2, color='green')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Return')
        ax1.set_title('Evaluation Returns')
        ax1.grid(True, alpha=0.3)

        if 'eval_goal_rate' in history:
            ax2.plot(epochs, history['eval_goal_rate'], 'o-', color='orange')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Goal Rate')
            ax2.set_title('Goal Reach Rate')
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation.png'), dpi=150)
        plt.close()

    # 4. Grade spectrum evolution
    if 'grade_spectrum' in history and history['grade_spectrum']:
        spectra = np.array(history['grade_spectrum'])     # [T, num_grades]
        fig, ax = plt.subplots(figsize=(10, 5))
        for g in range(spectra.shape[1]):
            ax.plot(spectra[:, g], label=f'Grade {g}', linewidth=1.5)
        ax.set_xlabel('Evaluation step')
        ax.set_ylabel('Mean Hermitian Energy')
        ax.set_title('Grade Spectrum Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grade_spectrum.png'), dpi=150)
        plt.close()

    print(f"[INFO] Plots saved to {output_dir}/")


def train(args):
    """Orchestrate self-play, training, and evaluation."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] State algebra: Cl({args.state_p},{args.state_q})")
    print(f"[CONFIG] Search algebra: Cl(1,1)")
    print(f"[CONFIG] Grid: {args.grid_size}x{args.grid_size}, "
          f"{args.num_obstacles} obstacles")
    print(f"[CONFIG] Hypotheses: {args.num_hypotheses}, "
          f"Search depth: {args.search_depth}")

    # Algebras
    state_algebra = CliffordAlgebra(args.state_p, args.state_q, device=device)
    search_algebra = CliffordAlgebra(1, 1, device=device)
    print(f"[INFO] State dim: {state_algebra.dim}, "
          f"Search dim: {search_algebra.dim}")

    # Environment
    env = GridPlanningEnv(
        grid_size=args.grid_size,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        max_steps=args.max_episode_steps,
    )
    print(f"[INFO] Obs dim: {env.obs_dim}, Actions: {env.num_actions}")
    env.reset()
    print(f"[ENV]\n{env.render()}\n")

    # Model
    model = GeometricMuZero(
        state_algebra=state_algebra,
        search_algebra=search_algebra,
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        hidden_channels=args.hidden_channels,
        num_rep_layers=args.num_rep_layers,
        num_hypotheses=args.num_hypotheses,
        search_depth=args.search_depth,
        discount=args.discount,
        temperature=args.temperature,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {num_params:,}")

    # Optimizer
    optimizer = RiemannianAdam(
        model.parameters(), lr=args.lr, algebra=state_algebra)

    # Replay buffer
    replay = ReplayBuffer(capacity=args.replay_buffer_size)

    # History tracking
    history: Dict[str, List] = {
        'total': [], 'policy': [], 'value': [], 'reward': [],
        'dynamics': [], 'bv_reg': [],
        'returns': [],
        'eval_epochs': [], 'eval_return': [], 'eval_std': [],
        'eval_goal_rate': [], 'grade_spectrum': [],
    }

    # Random baseline
    print("[EVAL] Random baseline...")
    random_returns = []
    for _ in range(args.eval_episodes):
        obs = env.reset()
        ret = 0.0
        done = False
        while not done:
            obs, r, done = env.step(random.randint(0, env.num_actions - 1))
            ret += r
        random_returns.append(ret)
    print(f"  Random: return={np.mean(random_returns):.3f} +/- "
          f"{np.std(random_returns):.3f}")

    # Training
    for epoch in range(1, args.epochs + 1):
        # Epsilon decays from 1.0 to 0.05 over training
        epsilon = max(0.05, 1.0 - (epoch - 1) / (args.epochs * 0.7))

        # Self-play
        ep_returns = _self_play(
            model, env, replay, args.episodes_per_epoch, device, epsilon)
        history['returns'].extend(ep_returns)

        # Train
        model.train()
        for _ in range(args.updates_per_epoch):
            losses = _train_step(
                model, replay, optimizer,
                args.discount, args.n_step_return, device)
            for k, v in losses.items():
                if k in history:
                    history[k].append(v)

        # Diagnostics
        if epoch % args.diag_interval == 0 or epoch == 1:
            metrics = _evaluate(
                model, env, args.eval_episodes, device, use_gss=True)

            history['eval_epochs'].append(epoch)
            history['eval_return'].append(metrics['mean_return'])
            history['eval_std'].append(metrics['std_return'])
            history['eval_goal_rate'].append(metrics['goal_rate'])

            # Grade spectrum snapshot
            with torch.no_grad():
                sample_obs = torch.tensor(
                    env.reset(), dtype=torch.float32, device=device).unsqueeze(0)
                sample_state = model.representation(sample_obs)
                spectrum = hermitian_grade_spectrum(
                    state_algebra, sample_state.reshape(-1, state_algebra.dim))
                mean_spectrum = spectrum.mean(dim=0).cpu().numpy()
                history['grade_spectrum'].append(mean_spectrum)

            print(f"[Epoch {epoch:4d}] "
                  f"return={metrics['mean_return']:+.3f} "
                  f"goal={metrics['goal_rate']:.0%} "
                  f"steps={metrics['mean_steps']:.1f} "
                  f"buf={len(replay)}")

    # Final evaluation
    print("\n[FINAL] GSS policy:")
    final_gss = _evaluate(model, env, args.eval_episodes * 2, device, use_gss=True)
    print(f"  Return: {final_gss['mean_return']:+.3f} +/- {final_gss['std_return']:.3f}")
    print(f"  Goal rate: {final_gss['goal_rate']:.0%}")
    print(f"  Mean steps: {final_gss['mean_steps']:.1f}")

    print("\n[FINAL] Network-only (no search):")
    final_net = _evaluate(model, env, args.eval_episodes * 2, device, use_gss=False)
    print(f"  Return: {final_net['mean_return']:+.3f} +/- {final_net['std_return']:.3f}")
    print(f"  Goal rate: {final_net['goal_rate']:.0%}")

    print(f"\n[FINAL] Random baseline: {np.mean(random_returns):+.3f}")

    # Plots
    if args.save_plots:
        _save_plots(history, args.output_dir)

    # Show a sample trajectory
    print("\n[TRAJECTORY] Sample episode with GSS:")
    model.eval()
    obs = env.reset()
    print(env.render())
    done = False
    step = 0
    while not done and step < 20:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        state = model.representation(obs_t)
        policy = model.gss(state)
        action = policy.squeeze(0).argmax().item()
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        obs, reward, done = env.step(action)
        step += 1
        print(f"  Step {step}: {action_names[action]} -> reward={reward:.1f}")
    print(env.render())
    if done and reward == 1.0:
        print("  -> GOAL REACHED!")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Geometric MuZero — Geometric Superposition Search replaces MCTS')

    # Environment
    p.add_argument('--grid-size', type=int, default=5)
    p.add_argument('--num-obstacles', type=int, default=3)
    p.add_argument('--max-episode-steps', type=int, default=50)

    # Algebra
    p.add_argument('--state-p', type=int, default=3,
                   help='Positive dimensions for state algebra Cl(p,q)')
    p.add_argument('--state-q', type=int, default=0,
                   help='Negative dimensions for state algebra Cl(p,q)')

    # Model
    p.add_argument('--hidden-channels', type=int, default=32)
    p.add_argument('--num-rep-layers', type=int, default=2)
    p.add_argument('--num-hypotheses', type=int, default=8)
    p.add_argument('--search-depth', type=int, default=3)
    p.add_argument('--discount', type=float, default=0.99)
    p.add_argument('--temperature', type=float, default=1.0)

    # Training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--replay-buffer-size', type=int, default=10000)
    p.add_argument('--episodes-per-epoch', type=int, default=10)
    p.add_argument('--updates-per-epoch', type=int, default=20)
    p.add_argument('--n-step-return', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')

    # Evaluation
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--diag-interval', type=int, default=20)

    # Output
    p.add_argument('--save-plots', action='store_true')
    p.add_argument('--output-dir', type=str, default='muzero_plots')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
