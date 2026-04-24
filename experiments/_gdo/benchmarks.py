"""Benchmark models for GDO experiments: analytic, geometric, GA-neural, manifold."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from functional.activation import GeometricGELU
from layers import (
    CliffordLayerNorm,
    CliffordLinear,
    CliffordModule,
    MultiRotorLayer,
    RotorLayer,
)
from models.blocks.gbn import GeometricBladeNetwork


# --- Category: Analytic Functions ---

class RosenbrockModel(nn.Module):
    """2D Rosenbrock function. Famous narrow curved valley."""
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b
        self.x = nn.Parameter(torch.tensor([-1.5]))
        self.y = nn.Parameter(torch.tensor([1.5]))

    def forward(self) -> torch.Tensor:
        return (self.a - self.x) ** 2 + self.b * (self.y - self.x ** 2) ** 2


class RastriginModel(nn.Module):
    """N-dimensional Rastrigin function. Many local minima; global at x=0."""
    def __init__(self, n_dims: int = 4, A: float = 10.0):
        super().__init__()
        self.A = A
        self.n = n_dims
        self.x = nn.Parameter(torch.randn(n_dims) * 3.0)

    def forward(self) -> torch.Tensor:
        x = self.x
        return self.A * self.n + (x ** 2 - self.A * torch.cos(2 * math.pi * x)).sum()


class AckleyModel(nn.Module):
    """N-dimensional Ackley function. Nearly flat plateau with narrow central well.
    Tests Lorentz warp effectiveness."""
    def __init__(self, n_dims: int = 10, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi):
        super().__init__()
        self._a = a
        self._b = b
        self._c = c
        self.x = nn.Parameter(torch.randn(n_dims) * 2.0)

    def forward(self) -> torch.Tensor:
        x = self.x
        sum_sq = (x ** 2).mean()
        sum_cos = (torch.cos(self._c * x)).mean()
        return -self._a * torch.exp(-self._b * sum_sq.sqrt()) - torch.exp(sum_cos) + self._a + math.e


class StyblinskiTangModel(nn.Module):
    """N-dimensional Styblinski-Tang. Multiple asymmetric wells.
    Tests topology search for finding global basin."""
    def __init__(self, n_dims: int = 6):
        super().__init__()
        self.x = nn.Parameter(torch.randn(n_dims) * 3.0)

    def forward(self) -> torch.Tensor:
        x = self.x
        return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x).sum()


# --- Category: Geometric Primitives ---

class SmallGBNModel(CliffordModule):
    """Small Geometric Blade Network for testing optimizer on actual GA model."""

    def __init__(self, p: int = 3, q: int = 0, channels: int = 4, device: str = 'cpu'):
        algebra = CliffordAlgebra(p, q, device=device)
        super().__init__(algebra)
        dim = 2 ** (p + q)
        self.norm = CliffordLayerNorm(self.algebra, channels)
        self.rotor = RotorLayer(self.algebra, channels)
        self.linear = CliffordLinear(self.algebra, channels, channels)
        self.act = GeometricGELU(self.algebra)
        self._channels = channels
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.rotor(x)
        x = self.act(x)
        x = self.linear(x)
        return x


class RotorRegistrationModel(CliffordModule):
    """Fit a rotor in Cl(3,0) to align a source point cloud to a rotated+noised target."""

    def __init__(
        self,
        n_points: int = 50,
        noise_std: float = 0.05,
        rotation_angle: float = 2.5,
        device: str = 'cpu',
    ):
        algebra = CliffordAlgebra(3, 0, device=device)
        super().__init__(algebra)

        torch.manual_seed(42)
        raw = torch.randn(n_points, 3, device=device)
        raw = F.normalize(raw, dim=-1)
        self.register_buffer('source', raw)

        axis = torch.tensor([1.0, 1.0, 1.0], device=device)
        axis = axis / axis.norm()
        gt_bv = self._axis_angle_to_bivector(axis, rotation_angle)
        self.register_buffer('gt_bivector', gt_bv)

        gt_rotor = self.algebra.exp(-0.5 * gt_bv.unsqueeze(0))
        source_mv = self.algebra.embed_vector(raw)
        rotated = self.algebra.sandwich_product(
            gt_rotor.expand(n_points, -1),
            source_mv.unsqueeze(1),
        ).squeeze(1)
        target_pts = self._extract_vector(rotated)
        target_pts = target_pts + noise_std * torch.randn_like(target_pts)
        self.register_buffer('target', target_pts)

        self.rotor = RotorLayer(self.algebra, channels=1)

    def _axis_angle_to_bivector(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        bv = torch.zeros(self.algebra.dim, device=axis.device)
        bv[3] = angle * axis[2]
        bv[5] = -angle * axis[1]
        bv[6] = angle * axis[0]
        return bv

    def _extract_vector(self, mv: torch.Tensor) -> torch.Tensor:
        return torch.stack([mv[..., 1], mv[..., 2], mv[..., 4]], dim=-1)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source)
        source_mv = source_mv.unsqueeze(1)
        rotated_mv = self.rotor(source_mv)
        pred_pts = self._extract_vector(rotated_mv.squeeze(1))
        return F.mse_loss(pred_pts, self.target)

    def angular_error(self) -> float:
        with torch.no_grad():
            learned_bv = torch.zeros(
                self.algebra.dim, device=self.gt_bivector.device
            )
            learned_bv[self.rotor.grade_indices] = self.rotor.grade_weights[0]
            r_learned = self.algebra.exp(-0.5 * learned_bv.unsqueeze(0))
            r_gt = self.algebra.exp(-0.5 * self.gt_bivector.unsqueeze(0))
            r_gt_rev = self.algebra.reverse(r_gt)
            product = self.algebra.geometric_product(r_learned, r_gt_rev)
            cos_half = product[0, 0].abs().clamp(max=1.0).item()
            return 2.0 * math.acos(cos_half)


class MinkowskiRotorModel(CliffordModule):
    """Fit a Lorentz boost in Cl(2,1) to align spacetime events.
    Tests optimizer on indefinite signature (mixed exp map regime)."""
    def __init__(self, n_events: int = 30, boost_rapidity: float = 0.8, device: str = 'cpu'):
        algebra = CliffordAlgebra(2, 1, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim  # 8

        torch.manual_seed(42)
        raw = torch.randn(n_events, 2, device=device)
        spatial = F.normalize(raw, dim=-1) * 0.5
        events_3d = torch.cat([spatial, torch.ones(n_events, 1, device=device)], dim=-1)
        self.register_buffer('source', events_3d)

        gt_bv = torch.zeros(dim, device=device)
        gt_bv[5] = boost_rapidity
        self.register_buffer('gt_bivector', gt_bv)

        gt_rotor = self.algebra.exp(-0.5 * gt_bv.unsqueeze(0))
        source_mv = self.algebra.embed_vector(events_3d)
        boosted = self.algebra.sandwich_product(
            gt_rotor.expand(n_events, -1),
            source_mv.unsqueeze(1),
        ).squeeze(1)
        target_3d = torch.stack([boosted[..., 1], boosted[..., 2], boosted[..., 4]], dim=-1)
        target_3d = target_3d + 0.02 * torch.randn_like(target_3d)
        self.register_buffer('target', target_3d)

        self.rotor = RotorLayer(self.algebra, channels=1)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source).unsqueeze(1)
        boosted_mv = self.rotor(source_mv).squeeze(1)
        pred = torch.stack([boosted_mv[..., 1], boosted_mv[..., 2], boosted_mv[..., 4]], dim=-1)
        return F.mse_loss(pred, self.target)

    def rapidity_error(self) -> float:
        with torch.no_grad():
            learned_bv = torch.zeros(self.algebra.dim, device=self.gt_bivector.device)
            learned_bv[self.rotor.grade_indices] = self.rotor.grade_weights[0]
            return (learned_bv - self.gt_bivector).norm().item()


class ConformalRegistrationModel(CliffordModule):
    """Fit a conformal rotor in Cl(4,1) for rotation+translation.
    Tests optimizer on 32-dimensional multivectors."""
    def __init__(self, n_points: int = 40, device: str = 'cpu'):
        algebra = CliffordAlgebra(4, 1, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim  # 32

        torch.manual_seed(42)
        raw = torch.randn(n_points, 3, device=device) * 0.5
        self.register_buffer('source_pts', raw)

        gt_bv = torch.zeros(dim, device=device)
        bv_indices = [i for i in range(dim) if bin(i).count('1') == 2]
        if len(bv_indices) > 0:
            gt_bv[bv_indices[0]] = 0.4

        self.register_buffer('gt_bivector', gt_bv)

        gt_rotor = self.algebra.exp(-0.5 * gt_bv.unsqueeze(0))
        src_5d = torch.zeros(n_points, 5, device=device)
        src_5d[:, :3] = raw
        source_mv = self.algebra.embed_vector(src_5d)
        rotated = self.algebra.sandwich_product(
            gt_rotor.expand(n_points, -1),
            source_mv.unsqueeze(1),
        ).squeeze(1)
        target_mv = rotated.clone()
        target_mv[:, 1] += 0.3
        target_mv += 0.01 * torch.randn_like(target_mv)
        self.register_buffer('target_mv', target_mv)
        self.register_buffer('source_mv', source_mv)

        self.rotor = RotorLayer(self.algebra, channels=1)

    def forward(self) -> torch.Tensor:
        src = self.source_mv.unsqueeze(1)
        pred = self.rotor(src).squeeze(1)
        return F.mse_loss(pred, self.target_mv)


class MultiRotorRegistrationModel(CliffordModule):
    """Fit a MultiRotorLayer to align multi-cluster point clouds.
    Tests commutator scheduling and multi-modal optimization."""
    def __init__(self, n_clusters: int = 3, points_per_cluster: int = 20, device: str = 'cpu'):
        algebra = CliffordAlgebra(3, 0, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim

        torch.manual_seed(42)
        sources = []
        targets = []

        for c in range(n_clusters):
            center = torch.randn(3, device=device)
            pts = center + 0.2 * torch.randn(points_per_cluster, 3, device=device)
            sources.append(pts)

            angle = 0.5 + c * 1.0
            axis = F.normalize(torch.randn(3, device=device), dim=0)
            bv = torch.zeros(dim, device=device)
            bv[3] = angle * axis[2]
            bv[5] = -angle * axis[1]
            bv[6] = angle * axis[0]
            rotor = self.algebra.exp(-0.5 * bv.unsqueeze(0))
            pts_mv = self.algebra.embed_vector(pts)
            rotated = self.algebra.sandwich_product(
                rotor.expand(points_per_cluster, -1),
                pts_mv.unsqueeze(1),
            ).squeeze(1)
            tgt_pts = torch.stack([rotated[..., 1], rotated[..., 2], rotated[..., 4]], dim=-1)
            targets.append(tgt_pts + 0.03 * torch.randn_like(tgt_pts))

        self.register_buffer('source', torch.cat(sources))
        self.register_buffer('target', torch.cat(targets))

        self.multi_rotor = MultiRotorLayer(self.algebra, channels=1, num_rotors=n_clusters)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source).unsqueeze(1)
        rotated_mv = self.multi_rotor(source_mv).squeeze(1)
        pred = torch.stack([rotated_mv[..., 1], rotated_mv[..., 2], rotated_mv[..., 4]], dim=-1)
        return F.mse_loss(pred, self.target)


# --- Category: GA Neural Networks ---

class MediumGBNModel(CliffordModule):
    """Medium GBN using GeometricBladeNetwork. 3 layers, 16ch.
    Task: learn regression on multivector inputs."""
    def __init__(self, p=3, q=0, channels=16, layers=3, n_samples=64, device='cpu'):
        algebra = CliffordAlgebra(p, q, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim
        self.gbn = GeometricBladeNetwork(
            self.algebra, in_channels=channels,
            hidden_channels=channels, out_channels=channels,
            layers=layers,
        )
        torch.manual_seed(42)
        self.register_buffer('X', torch.randn(n_samples, channels, dim, device=device) * 0.3)
        self.register_buffer('y', self.X[:, :, 0].mean(dim=1, keepdim=True))

    def forward(self) -> torch.Tensor:
        out = self.gbn(self.X)
        pred = out[:, :, 0].mean(dim=1, keepdim=True)
        return F.mse_loss(pred, self.y)


class MultiSigGBNModel(CliffordModule):
    """GBN in Minkowski signature Cl(2,1). 2 layers, 8ch.
    Tests optimizer with mixed exp map regime."""
    def __init__(self, channels=8, layers=2, n_samples=48, device='cpu'):
        algebra = CliffordAlgebra(2, 1, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim
        self.gbn = GeometricBladeNetwork(
            self.algebra, in_channels=channels,
            hidden_channels=channels, out_channels=channels,
            layers=layers,
        )
        torch.manual_seed(42)
        self.register_buffer('X', torch.randn(n_samples, channels, dim, device=device) * 0.3)
        self.register_buffer('y', self.X[:, :, 0].mean(dim=1, keepdim=True))

    def forward(self) -> torch.Tensor:
        out = self.gbn(self.X)
        pred = out[:, :, 0].mean(dim=1, keepdim=True)
        return F.mse_loss(pred, self.y)


# --- Category: Manifold Tasks ---

class SO3InterpolationModel(CliffordModule):
    """Learn a smooth rotor trajectory through waypoints on SO(3).
    Tests geodesic integrator on curved manifold."""
    def __init__(self, n_waypoints: int = 8, device: str = 'cpu'):
        algebra = CliffordAlgebra(3, 0, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim
        self.n_waypoints = n_waypoints

        torch.manual_seed(42)
        waypoint_bivectors = []
        for i in range(n_waypoints):
            bv = torch.zeros(dim, device=device)
            angle = 0.3 + i * 0.5
            axis = F.normalize(torch.randn(3, device=device), dim=0)
            bv[3] = angle * axis[2]
            bv[5] = -angle * axis[1]
            bv[6] = angle * axis[0]
            waypoint_bivectors.append(bv)
        waypoint_bvs = torch.stack(waypoint_bivectors)
        waypoint_rotors = self.algebra.exp(-0.5 * waypoint_bvs)
        self.register_buffer('target_rotors', waypoint_rotors)

        test_vec = torch.zeros(dim, device=device)
        test_vec[1] = 1.0
        self.register_buffer('test_vec', test_vec)

        targets = []
        for i in range(n_waypoints):
            R = waypoint_rotors[i:i+1]
            v = test_vec.unsqueeze(0)
            rotated = self.algebra.sandwich_product(R, v.unsqueeze(1)).squeeze(1)
            targets.append(rotated)
        self.register_buffer('target_points', torch.cat(targets))

        self.rotor_bank = RotorLayer(self.algebra, channels=n_waypoints)

    def forward(self) -> torch.Tensor:
        test_expanded = self.test_vec.unsqueeze(0).unsqueeze(0).expand(1, self.n_waypoints, -1)
        rotated = self.rotor_bank(test_expanded).squeeze(0)
        return F.mse_loss(rotated, self.target_points)

    def geodesic_deviation(self) -> float:
        """Measure how far learned rotors are from ground-truth on SO(3)."""
        with torch.no_grad():
            learned_bv = torch.zeros(self.n_waypoints, self.algebra.dim,
                                     device=self.target_rotors.device)
            learned_bv[:, self.rotor_bank.grade_indices] = self.rotor_bank.grade_weights
            r_learned = self.algebra.exp(-0.5 * learned_bv)
            r_gt_rev = self.algebra.reverse(self.target_rotors)
            product = self.algebra.geometric_product(r_learned, r_gt_rev)
            cos_half = product[:, 0].abs().clamp(max=1.0)
            angles = 2.0 * torch.acos(cos_half)
            return angles.mean().item()
