# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Minkowski's Theorem and Lattice Geometry via Geometric Algebra.

This experiment demonstrates Minkowski's First and Second Fundamental Theorems
using the Clifford Algebra Cl(n,0). 

Innovations in this script:
1. GA-Sphere Decoding: Uses Geometric Algebra blade rejection to prune 
   the search tree for lattice points (Divide & Conquer). Optimized on CPU.
2. Rotor-Guided Continuous Integer Relaxation: A custom PyTorch layer
   that uses GD and rotor alignment to discover short lattice vectors. 
   Accelerated on MPS/CUDA.
3. Minkowski Theorem 2 Verification: Finds n successive minima and validates
   their linear independence dynamically using the wedge product.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from core.metric import induced_norm
from core.visualizer import GeneralVisualizer


class LatticeSumLayer(nn.Module):
    """Rotor-Guided Continuous Integer Relaxation for Lattice Search.
    
    Relaxes integer coefficients to continuous values, optimizes them via GD,
    and uses a Rotor to guide the continuous vector towards favorable descent spaces.
    """
    def __init__(self, n: int, algebra: CliffordAlgebra, mv_basis: list):
        super().__init__()
        self.n = n
        self.algebra = algebra
        self.mv_basis = mv_basis
        self.device = mv_basis[0].device
        
        # Continuous coefficients (initialized near zero but non-zero)
        self.c = nn.Parameter(torch.randn(n, device=self.device) * 0.5 + 0.5)
        
        # Learnable Bivector for Rotor Guidance
        bv_mask = self.algebra.grade_masks[2]
        self.num_bivectors = bv_mask.sum().item()
        if self.num_bivectors > 0:
            self.bivector_coeffs = nn.Parameter(torch.randn(self.num_bivectors, device=self.device) * 0.1)
        else:
            self.bivector_coeffs = None
            
    def forward(self):
        # 1. Build continuous point
        p = torch.zeros(self.algebra.dim, device=self.device)
        for i in range(self.n):
            p = p + self.c[i] * self.mv_basis[i]
            
        # 2. Apply Rotor Guidance
        if self.bivector_coeffs is not None:
            B = torch.zeros(self.algebra.dim, device=self.device)
            bv_mask = self.algebra.grade_masks[2]
            B[bv_mask] = self.bivector_coeffs
            R = self.algebra.exp(B)
            R_rev = self.algebra.reverse(R)
            
            # Rotate point to align with favorable descent directions
            p_guided = self.algebra.geometric_product(
                self.algebra.geometric_product(R, p), R_rev
            )
        else:
            p_guided = p
            
        return self.c, p, p_guided


class MinkowskiLattice:
    """Handles GA-based lattice operations and Minkowski theorem verification."""

    def __init__(self, n: int = 3, seed: int = 42, device: str = 'cpu', skew_factor: float = 0.0):
        self.n = n
        self.device = device
        # We maintain two algebras: one for GPU/MPS acceleration, one for CPU recursion
        self.algebra = CliffordAlgebra(p=n, q=0, device=device)
        self.cpu_algebra = CliffordAlgebra(p=n, q=0, device='cpu')
        
        torch.manual_seed(seed)
        # Create a random basis. We start with identity and add noise.
        base = torch.eye(n, device='cpu') # Always generate basis on CPU first
        noise = torch.randn(n, n, device='cpu') * 0.1
        basis = base + noise
        
        # Apply skew factor: B = B * (I + skew * UpperTriangle)
        if skew_factor != 0:
            skew_mat = torch.eye(n, device='cpu')
            for i in range(n):
                for j in range(i + 1, n):
                    skew_mat[i, j] = skew_factor * torch.randn(1).item()
            basis = basis @ skew_mat
            
        self.basis_vectors_cpu = basis
        self.basis_vectors = basis.to(device)
        
        # Embed basis vectors into the Grade-1 subspace
        self.mv_basis_cpu = self.cpu_algebra.embed_vector(self.basis_vectors_cpu)
        self.mv_basis = self.algebra.embed_vector(self.basis_vectors)
        
        print(f"Initialized L in Cl({n},0) with skew_factor={skew_factor}")
        print(f"  Training Device: {device}")
        for i in range(n):
            print(f"  b_{i+1}: {self.basis_vectors_cpu[i].numpy()}")

    def get_determinant(self, basis_mvs=None) -> float:
        """Compute det(L) = ||b_1 ^ b_2 ^ ... ^ b_n||."""
        if basis_mvs is None:
            basis_mvs = self.mv_basis
            alg = self.algebra
        else:
            # Detect which algebra to use
            alg = self.cpu_algebra if basis_mvs[0].device == torch.device('cpu') else self.algebra
            
        vol_blade = basis_mvs[0]
        for i in range(1, self.n):
            vol_blade = alg.wedge(vol_blade, basis_mvs[i])
        
        det = induced_norm(alg, vol_blade).item()
        return det

    def verify_volume_invariance(self):
        """Proof of geometric invariance of volume under shear transformations."""
        print("\nVerifying Geometric Invariance of Volume (on CPU):")
        det_orig = self.get_determinant(self.mv_basis_cpu)
        print(f"  Original det(L): {det_orig:.6f}")
        
        i, j = 0, 1 if self.n > 1 else (0, 0)
        if i == j: return
        
        alpha = 1.5
        print(f"  Applying shear: b_{i+1} -> b_{i+1} + {alpha}*b_{j+1}")
        
        new_mv_basis = [mv.clone() for mv in self.mv_basis_cpu]
        new_mv_basis[i] = self.mv_basis_cpu[i] + alpha * self.mv_basis_cpu[j]
        
        det_new = self.get_determinant(new_mv_basis)
        print(f"  New det(L'):      {det_new:.6f}")
        print(f"  Invariance Proof: |det(L) - det(L')| = {abs(det_orig - det_new):.2e}")
        assert math.isclose(det_orig, det_new, rel_tol=1e-5)

    def unit_lp_ball_volume(self, p: float = 2.0) -> float:
        """Volume of a unit n-ball in L_p norm."""
        return (2 * math.gamma(1/p + 1))**self.n / math.gamma(self.n/p + 1)

    def minkowski_bound_radius(self, p: float = 2.0) -> float:
        """Radius R such that Vol(Lp_Ball_R) = 2^n * det(L)."""
        det = self.get_determinant(self.mv_basis_cpu)
        v_unit = self.unit_lp_ball_volume(p)
        return (2**self.n * det / v_unit) ** (1 / self.n)

    def get_lattice_point(self, coeffs: tuple, device='cpu') -> torch.Tensor:
        """Generate a lattice point multivector from integer coefficients."""
        alg = self.cpu_algebra if device == 'cpu' else self.algebra
        basis = self.mv_basis_cpu if device == 'cpu' else self.mv_basis
        point = torch.zeros(alg.dim, device=device)
        for i, c in enumerate(coeffs):
            point += c * basis[i]
        return point

    def ga_sphere_decode(self, search_range: int = 3, p: float = 2.0):
        """Search for non-zero lattice points using GA-Sphere Decoding on CPU.
        
        Uses blade rejection (projection) to drastically prune the search tree.
        """
        det = self.get_determinant(self.mv_basis_cpu)
        bound_r = self.minkowski_bound_radius(p)
        
        body_name = f"L_{p}" if p < float('inf') else "L_inf"
        print(f"\nGA-Sphere Decoding (Convex Body: {body_name}) on CPU:")
        print(f"  Determinant det(L):    {det:.4f}")
        print(f"  Critical Radius R:     {bound_r:.4f}")
        
        # Precompute blades B_k on CPU
        blades = {0: None}
        current_blade = self.mv_basis_cpu[0]
        blades[1] = current_blade
        for k in range(2, self.n + 1):
            current_blade = self.cpu_algebra.wedge(current_blade, self.mv_basis_cpu[k-1])
            blades[k] = current_blade

        found_points = []
        
        def search(k, current_q, current_coeffs):
            # Pruning via GA Projection:
            if k > 0 and k < self.n:
                B_k = blades[k]
                bk_norm = induced_norm(self.cpu_algebra, B_k).item()
                if bk_norm > 1e-6:
                    q_reject = self.cpu_algebra.blade_reject(current_q, B_k)
                    dist = induced_norm(self.cpu_algebra, q_reject).item()
                    if dist > bound_r + 1e-4:  # Prune search tree
                        return

            if k == 0:
                # Leaf node
                vec_coords = torch.stack([current_q[1 << i] for i in range(self.n)])
                if p == float('inf'):
                    norm_val = torch.max(torch.abs(vec_coords)).item()
                else:
                    norm_val = torch.pow(torch.sum(torch.pow(torch.abs(vec_coords), p)), 1/p).item()

                if 0 < norm_val <= bound_r:
                    found_points.append({
                        'coeffs': tuple(current_coeffs),
                        'norm': norm_val,
                        'mv': current_q
                    })
                return

            for z in range(-search_range, search_range + 1):
                next_q = current_q + z * self.mv_basis_cpu[k-1]
                search(k - 1, next_q, [z] + current_coeffs)

        initial_q = torch.zeros(self.cpu_algebra.dim, device='cpu')
        search(self.n, initial_q, [])
        
        found_points.sort(key=lambda x: x['norm'])
        print(f"  Found {len(found_points)} non-zero points within radius R.")
        return found_points

    def optimize_continuous_relaxation(self):
        """Finds short lattice vectors using Rotor-Guided GD on the accelerated device."""
        print(f"\nRunning Rotor-Guided Continuous Integer Relaxation (LatticeSumLayer) on {self.device}...")
        layer = LatticeSumLayer(self.n, self.algebra, self.mv_basis).to(self.device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
        
        best_norm = float('inf')
        best_coeffs = None
        best_mv = None
        
        for step in range(250):
            optimizer.zero_grad()
            c, p_raw, p_guided = layer()
            
            norm_sq = induced_norm(self.algebra, p_guided).pow(2)
            int_penalty = torch.sum(torch.sin(math.pi * c)**2)
            origin_repel = torch.exp(-5.0 * norm_sq)
            
            loss = norm_sq + 8.0 * int_penalty + 15.0 * origin_repel
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                c_round = c.round().int().cpu().numpy()
                if not np.all(c_round == 0):
                    p_round = self.get_lattice_point(c_round, device='cpu')
                    n_round = induced_norm(self.cpu_algebra, p_round).item()
                    if n_round < best_norm:
                        best_norm = n_round
                        best_coeffs = tuple(c_round)
                        best_mv = p_round

        print(f"  GD Discovered Coefficients: {best_coeffs}")
        print(f"  Resulting Norm:             {best_norm:.4f}")
        return {'coeffs': best_coeffs, 'norm': best_norm, 'mv': best_mv}

    def verify_minkowski_second_theorem(self, points):
        """Finds n successive minima and verifies Minkowski's Second Theorem on CPU."""
        print("\nVerifying Minkowski's Second Theorem (Successive Minima) on CPU:")
        minima = []
        current_blade = None
        
        for p in points:
            v = p['mv'].to('cpu')
            if current_blade is None:
                minima.append(p)
                current_blade = v
            else:
                new_blade = self.cpu_algebra.wedge(current_blade, v)
                indep_measure = induced_norm(self.cpu_algebra, new_blade).item()
                if indep_measure > 1e-4:
                    minima.append(p)
                    current_blade = new_blade
                    
            if len(minima) == self.n:
                break
                
        if len(minima) < self.n:
            print(f"  Warning: Only found {len(minima)} linearly independent points.")
            return minima
            
        print(f"  Found {self.n} linearly independent successive minima:")
        lambdas = []
        for i, m in enumerate(minima):
            lambdas.append(m['norm'])
            print(f"    lambda_{i+1}: norm={m['norm']:.4f}, coeffs={m['coeffs']}")
            
        lambda_prod = np.prod(lambdas)
        det = self.get_determinant(self.mv_basis_cpu)
        v_unit = self.unit_lp_ball_volume(2.0)
        bound = (2**self.n) * det / v_unit
        
        print(f"  Product of lambdas: {lambda_prod:.4f}")
        print(f"  Minkowski Bound (2^n det(L) / V_n): {bound:.4f}")
        
        if lambda_prod <= bound + 1e-4:
            print("  Theorem 2 Holds: lambda_1 * ... * lambda_n <= Bound")
        else:
            print("  Theorem 2 Verification Failed.")
            
        return minima


def run_experiment(args):
    print(f"\n{'='*60}")
    print(f" Minkowski's Theorem Experiment (Dimension n={args.dim})")
    print(f"{'='*60}")
    
    # Auto-detect best device for GD training
    training_device = args.device
    if training_device == 'cpu':
        if torch.cuda.is_available(): training_device = 'cuda'
        elif torch.backends.mps.is_available(): training_device = 'mps'

    lat = MinkowskiLattice(n=args.dim, seed=args.seed, device=training_device, skew_factor=args.skew_factor)
    lat.verify_volume_invariance()
    
    # 1. GD Search (Accelerated)
    gd_point = lat.optimize_continuous_relaxation()
    
    # 2. Tree Search (CPU optimized)
    p_norms = [1.0, 2.0, float('inf')]
    results = {}
    for p in p_norms:
        points = lat.ga_sphere_decode(search_range=args.search_range, p=p)
        results[p] = points
            
    # 3. Theorem 2 (CPU)
    points_l2 = results[2.0]
    if points_l2:
        successive_minima = lat.verify_minkowski_second_theorem(points_l2)
    else:
        successive_minima = []

    # 4. Visualization
    if args.save_plots and args.dim in [2, 3]:
        print("\nGenerating visualizations...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        all_mvs = []
        labels = []
        grid_range = 3
        
        import itertools
        ranges = [range(-grid_range, grid_range + 1)] * args.dim
        for coeffs in itertools.product(*ranges):
            p_mv = lat.get_lattice_point(coeffs, device='cpu')
            all_mvs.append(p_mv)
            
            is_special = any(p['coeffs'] == coeffs for p in points_l2)
            is_gd = (coeffs == gd_point['coeffs']) if gd_point['coeffs'] else False
            is_minima = any(m['coeffs'] == coeffs for m in successive_minima)
            
            if all(c == 0 for c in coeffs):
                labels.append("Origin")
            elif is_minima:
                labels.append("Successive Minima")
            elif is_gd:
                labels.append("GD Discovered")
            elif is_special:
                labels.append("Inside L2 Bound")
            else:
                labels.append("Lattice Point")
        
        all_mvs_tensor = torch.stack(all_mvs)
        
        if args.dim == 2:
            plt.figure(figsize=(10, 10))
            pts_np = all_mvs_tensor.cpu().numpy()
            x, y = pts_np[:, 1], pts_np[:, 2]
            
            colors = []
            for l in labels:
                if l == "Origin": colors.append('black')
                elif l == "Successive Minima": colors.append('gold')
                elif l == "GD Discovered": colors.append('magenta')
                elif l == "Inside L2 Bound": colors.append('red')
                else: colors.append('blue')
            
            plt.scatter(x, y, c=colors, alpha=0.6, s=50)
            
            styles = {1.0: ('green', '--', 'L1'), 2.0: ('red', '-', 'L2'), float('inf'): ('blue', ':', 'Linf')}
            for p in p_norms:
                r = lat.minkowski_bound_radius(p)
                color, ls, name = styles[p]
                if p == 1.0:
                    pts = np.array([[r, 0], [0, r], [-r, 0], [0, -r], [r, 0]])
                    plt.plot(pts[:, 0], pts[:, 1], color=color, linestyle=ls, label=name)
                elif p == 2.0:
                    circle = plt.Circle((0, 0), r, color=color, fill=False, linestyle=ls, label=name)
                    plt.gca().add_patch(circle)
                elif p == float('inf'):
                    square = plt.Rectangle((-r, -r), 2*r, 2*r, color=color, fill=False, linestyle=ls, label=name)
                    plt.gca().add_patch(square)
            
            plt.axhline(0, color='grey', lw=1); plt.axvline(0, color='grey', lw=1)
            plt.title(f"Minkowski (n=2, MPS={training_device!='cpu'})")
            plt.legend(); plt.axis('equal')
            plt.savefig(os.path.join(args.output_dir, "minkowski_2d_diversified.png"), dpi=150)
            
        elif args.dim == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            pts_np = all_mvs_tensor.cpu().numpy()
            x, y, z = pts_np[:, 1], pts_np[:, 2], pts_np[:, 4]
            
            b = lat.basis_vectors_cpu.numpy()
            for i, j, k in itertools.product(range(-1, 1), repeat=3):
                for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                    p1 = i*b[0] + j*b[1] + k*b[2]
                    p2 = (i+di)*b[0] + (j+dj)*b[1] + (k+dk)*b[2]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.3, lw=0.5)

            r = lat.minkowski_bound_radius(p=2.0)
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
            ax.plot_surface(r*np.cos(u)*np.sin(v), r*np.sin(u)*np.sin(v), r*np.cos(v), color='cyan', alpha=0.1)
            
            for p in points_l2:
                coords = [p['mv'][1].item(), p['mv'][2].item(), p['mv'][4].item()]
                is_min = any(m['coeffs'] == p['coeffs'] for m in successive_minima)
                color = 'gold' if is_min else 'red'
                ax.plot([0, coords[0]], [0, coords[1]], [0, coords[2]], color=color, alpha=0.6, lw=2)
                ax.scatter([coords[0]], [coords[1]], [coords[2]], color=color, s=80)
            
            ax.scatter(x, y, z, color='blue', alpha=0.1, s=15)
            ax.scatter([0], [0], [0], color='black', s=150, marker='*')
            ax.set_title(f"3D Minkowski (MPS={training_device!='cpu'})")
            plt.savefig(os.path.join(args.output_dir, "minkowski_3d_penetration.png"), dpi=150)

    print(f"\nExperiment complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minkowski's Theorem via Geometric Algebra")
    parser.add_argument('--dim', type=int, default=3, help='Dimension n')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--search-range', type=int, default=5, help='Search range')
    parser.add_argument('--skew-factor', type=float, default=0.5, help='Skew factor')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    parser.add_argument('--output-dir', type=str, default='minkowski_plots', help='Output dir')
    
    args = parser.parse_args()
    run_experiment(args)
