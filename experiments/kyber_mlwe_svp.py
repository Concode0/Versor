# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

# ===========================================================================
# [ACADEMIC EXPERIMENT & PERSONAL ARCHIVE]
#
# This file was part of a private experimental branch exploring Geometric 
# Algebra's potential in high-dimensional optimization (SVP/CVP). 
#
# NOTICE:
# 1. This is NOT a production-grade solver or an attack tool. It was 
#    uploaded accidentally during a mass-push of the Versor engine.
# 2. Any aggressive terminology or tone in comments/logs reflects a 
#    highly informal, private development process and "passionate 
#    prototyping" stage. Please disregard any non-technical rhetoric.
# 3. This script is provided "as-is" for theoretical curiosity only. 
#    It fails to scale to real-world LWE parameters and is no longer 
#    actively maintained.
# ===========================================================================

# Just for fun

"""Geometric Algebra Approach to Kyber Module-LWE Lattice Problems.

Investigates the application of Geometric Algebra networks to lattice problems
arising in the Kyber key-encapsulation mechanism (NIST FIPS 203).  The solver
pipeline consists of five components:

  1. KyberInstance    — Module-LWE instance generation and Kannan embedding (CVP→SVP)
  2. GALatticeReducer — LLL/BKZ basis reduction with GSO Lovász condition
  3. BladeEnumerator  — Branch-and-bound enumeration with blade-rejection pruning
  4. RotorSearchLayer — Neural SVP candidate generation via rotor-based feature extraction
  5. KyberSolver      — End-to-end orchestration (reduction → search → extraction)

Technical highlights:
  - Kyber-512 parameters: eta1=3 (secret distribution), eta2=2 (error distribution)
    per NIST FIPS 203
  - Full Kannan embedding of dimension (2m+1); omitting truncation preserves the
    target vector (e, -s, M) in the embedded lattice
  - Negacyclic convolution matrix: M[i,j] = p[i-j] if i>=j, else -p[n+i-j],
    consistent with the quotient ring Z_q[X]/(X^n+1)
  - LLL reduction with GSO-based Lovász condition following Cohen (Algorithm 2.6.3)
  - Three-term training objective with two hard constraints
    (unit Kannan coefficient; periodic integrality projection via rounding)
  - Norm-projection layer to suppress large-magnitude false-positive candidates
  - Numerically stable LAPACK operations via pre-scaled QR decomposition,
    mitigating DLASCL overflow at high lattice dimensions
  - Phase 4 verifies shortest basis vectors and pairwise linear combinations
    before evaluating neural search candidates

Computational notes:
  - LLL uses incremental GSO updates (Cohen 2.6.3): O(n) per step vs O(n²d)
  - Vectorized Gram-Schmidt and negacyclic matrix construction
  - CUDA-compatible throughout; all arithmetic in torch float64
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from core.metric import induced_norm
from layers.primitives.multi_rotor import MultiRotorLayer
from layers.primitives.normalization import CliffordLayerNorm
from layers.primitives.projection import GeometricNeutralizer

try:
    from fpylll import IntegerMatrix, LLL, BKZ, GSO, FPLLL
    HAS_FPYLLL = True
except ImportError:
    HAS_FPYLLL = False


def resolve_device(requested: str) -> str:
    """Resolve device string, with auto-detection for 'auto'."""
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    return requested


# ---------------------------------------------------------------------------
# 1. KyberInstance — Module-LWE Instance Generation and Kannan Embedding
# ---------------------------------------------------------------------------

class KyberInstance:
    """Generate a Kyber-style Module-LWE instance and construct the Kannan embedding.

    The polynomial ring R_q = Z_q[X]/(X^n + 1) is represented via anti-circulant
    (negacyclic convolution) matrices.  The secret s is sampled from CBD(eta1)
    and the error e from CBD(eta2).

    Standard Kyber-512 parameters (NIST FIPS 203): n=256, k=2, q=3329, eta1=3, eta2=2.

    The Kannan embedding has dimension 2m+1 where m = k*n.  The embedding is
    constructed without truncation, as truncation would displace the target
    vector (e, -s, M) outside the embedded lattice.

    Note: Kyber applies compress/decompress to ciphertext components (du, dv),
    which affects ciphertext size but not the underlying lattice structure.
    The Kannan embedding operates on the algebraic relation b = As + e (mod q)
    prior to compression.
    """

    def __init__(self, n: int = 256, k: int = 2, q: int = 3329,
                 eta1: int = 3, eta2: int = 2, eta: int = None,
                 seed: int = 42, device: str = 'cpu'):
        if eta is not None:  # backward compat: single eta overrides both
            eta1, eta2 = eta, eta
        self.n, self.k, self.q, self.eta1, self.eta2 = n, k, q, eta1, eta2
        self.device = device
        self.m = k * n
        self.full_dim = 2 * self.m + 1

        torch.manual_seed(seed)

        # Secret via CBD(eta1), error via CBD(eta2) — per NIST FIPS 203
        self.s = self._cbd(self.m, self.eta1)
        self.e = self._cbd(self.m, self.eta2)

        # Public matrix A (block-negacyclic)
        self.A = self._random_module_matrix()

        # b = A s + e  (mod q)
        self.b = (self.A @ self.s + self.e) % self.q

        # Adaptive M: expected error norm from CBD(eta2) over m dimensions
        # Var(CBD(eta)) = eta/2, so E[||e||] = sqrt(m * eta2 / 2)
        self.M_embed = max(math.sqrt(self.m * self.eta2 / 2.0), 1.0)

        # Full Kannan embedding basis
        self.basis = self._kannan_embedding()

        # Precompute the correction vector k = floor((As + e) / q) for verification
        As_plus_e = self.A.long() @ self.s.long() + self.e.long()
        self.k_correction = torch.div(As_plus_e, self.q, rounding_mode='floor')

        # Target vector norm: ||(e, -s, M)||
        self.target_norm = math.sqrt(
            torch.norm(self.e.double()).item()**2 +
            torch.norm(self.s.double()).item()**2 +
            self.M_embed**2
        )

        print(f"  KyberInstance: n={n}, k={k}, q={q}, eta1={eta1}, eta2={eta2}")
        print(f"  Ring dim m={self.m}, full lattice dim={self.full_dim}")
        print(f"  ||s||={torch.norm(self.s.double()).item():.2f}, "
              f"||e||={torch.norm(self.e.double()).item():.2f}, "
              f"M={self.M_embed:.2f}")
        print(f"  Expected target vector norm: {self.target_norm:.2f}")

    @staticmethod
    def _cbd(length: int, eta: int) -> torch.Tensor:
        """Centered binomial distribution CBD(eta)."""
        a = torch.randint(0, 2, (length, eta), dtype=torch.long)
        b = torch.randint(0, 2, (length, eta), dtype=torch.long)
        return a.sum(dim=1) - b.sum(dim=1)

    def _negacyclic_matrix(self, poly: torch.Tensor) -> torch.Tensor:
        """Anti-circulant matrix for R_q = Z_q[X]/(X^n + 1).

        M[i,j] = poly[i - j]       if i >= j   (no wraparound)
        M[i,j] = -poly[n + i - j]  if i < j    (X^n = -1 wraparound)

        Verified against NIST FIPS 203 (Kyber spec): negacyclic NTT matrix
        M[i,j] = f[(i-j) mod n] * (-1)^{floor((i-j)/n)} matches X^n + 1 quotient.

        Vectorized: (row - col) % n gives the correct index for both cases.
        """
        n = len(poly)
        rows = torch.arange(n, dtype=torch.long).unsqueeze(1)
        cols = torch.arange(n, dtype=torch.long).unsqueeze(0)
        idx = (rows - cols) % n
        lower = rows >= cols
        vals = poly[idx]
        M = torch.where(lower, vals, -vals) % self.q
        return M

    def _random_module_matrix(self) -> torch.Tensor:
        """Build k×k block matrix of negacyclic polynomials → m×m matrix."""
        A = torch.zeros(self.m, self.m, dtype=torch.long)
        for bi in range(self.k):
            for bj in range(self.k):
                poly = torch.randint(0, self.q, (self.n,), dtype=torch.long)
                block = self._negacyclic_matrix(poly)
                A[bi*self.n:(bi+1)*self.n, bj*self.n:(bj+1)*self.n] = block
        return A

    def _kannan_embedding(self) -> torch.Tensor:
        """Construct full Kannan embedding lattice basis.

        Layout (2m+1) × (2m+1):
            [[q*I_m,  0,    0 ],    rows 0..m-1
             [ A^T,  I_m,   0 ],    rows m..2m-1
             [ b^T,   0,    M ]]    row 2m

        The short vector in this lattice is (e, -s, M) with coefficients:
            c[0:m]   = -k  (mod-q correction)
            c[m:2m]  = -s  (secret)
            c[2m]    =  1  (Kannan embedding coefficient)

        Lattice vector: c @ B gives:
            cols 0..m-1:  q*(-k) + A^T*(-s) + b = e  (since b = As + e + qk)
            cols m..2m-1: -s
            col 2m:       M
        """
        m, dim = self.m, self.full_dim
        dev = self.device

        B = torch.zeros(dim, dim, dtype=torch.float64, device=dev)

        # Top-left: q * I_m
        B[:m, :m] = self.q * torch.eye(m, dtype=torch.float64, device=dev)

        # Middle block: A^T (transpose gives correct product A @ c_secret)
        B[m:2*m, :m] = self.A.to(dtype=torch.float64, device=dev).T

        # Middle diagonal: I_m
        B[m:2*m, m:2*m] = torch.eye(m, dtype=torch.float64, device=dev)

        # Bottom row: (b, 0, M)
        B[2*m, :m] = self.b.to(dtype=torch.float64, device=dev)
        B[2*m, 2*m] = self.M_embed

        return B

    def verify_solution(self, s_recovered: torch.Tensor,
                        e_recovered: torch.Tensor) -> bool:
        """Check As + e ≡ b (mod q)."""
        b_check = (self.A.long() @ s_recovered.long() +
                   e_recovered.long()) % self.q
        b_ref = self.b.long() % self.q
        return torch.all(b_check == b_ref).item()


# ---------------------------------------------------------------------------
# 2. GALatticeReducer — LLL/BKZ Basis Reduction with GSO Lovász Condition
# ---------------------------------------------------------------------------

class GALatticeReducer:
    """Lattice basis reduction via LLL with Gram-Schmidt orthogonalization and BKZ enumeration.

    Size reduction is performed using GSO mu coefficients.  Basis swaps are
    governed by the Lovász condition on projected norms.  BKZ enumeration
    blocks delegate to BladeEnumerator for Geometric Algebra-guided pruning.
    """

    def __init__(self, block_dim: int = 8, device: str = 'cpu', use_fpylll: bool = True):
        self.block_dim = block_dim
        self.device = device
        self.use_fpylll = use_fpylll and HAS_FPYLLL

    def _torch_to_imatrix(self, basis: torch.Tensor):
        """Convert float64 torch basis to fpylll IntegerMatrix (rounds to nearest int)."""
        return IntegerMatrix.from_matrix(basis.round().long().tolist())

    def _imatrix_to_torch(self, A, device: str) -> torch.Tensor:
        """Convert fpylll IntegerMatrix back to float64 torch tensor."""
        n, d = A.nrows, A.ncols
        rows = [[A[i, j] for j in range(d)] for i in range(n)]
        return torch.tensor(rows, dtype=torch.float64, device=device)

    def _fpylll_reduce(self, basis: torch.Tensor, rounds: int) -> torch.Tensor:
        """LLL + progressive BKZ reduction via fpylll backend.

        Uses MPFR extended precision for large lattices (dim > 160) to
        maintain GSO numerical stability.  Progressive BKZ warms up from
        block_size 4 to the target, giving each stage a better starting
        point.
        """
        A = self._torch_to_imatrix(basis)
        n = A.nrows

        # Select float type: MPFR for large lattices
        ft = "double"
        if n > 160:
            try:
                FPLLL.set_precision(max(150, n))
                ft = "mpfr"
            except Exception:
                ft = "double"

        M = GSO.Mat(A, float_type=ft)
        M.update_gso()

        # LLL reduction
        lll_obj = LLL.Reduction(M)
        lll_obj()

        # Progressive BKZ: warm up from small block sizes
        if self.block_dim >= 4:
            for bs in range(4, self.block_dim, 2):
                params = BKZ.Param(block_size=bs, max_loops=max(1, rounds // 2))
                BKZ.Reduction(M, lll_obj, params)()

        # Final pass at target block size
        if self.block_dim >= 2:
            params = BKZ.Param(block_size=self.block_dim, max_loops=rounds)
            BKZ.Reduction(M, lll_obj, params)()

        return self._imatrix_to_torch(A, self.device)

    def _compute_gso(self, basis: torch.Tensor):
        """Gram-Schmidt orthogonalization with mu coefficients.

        Vectorized: inner loop replaced with batched dot products and
        a single matrix-vector multiply per row.

        Returns:
            gso: Orthogonal vectors [n, d].
            mu: Projection coefficients [n, n] (lower triangular).
            B_sq: Squared norms of GSO vectors [n].
        """
        n = basis.shape[0]
        gso = basis.clone()
        mu = torch.zeros(n, n, dtype=basis.dtype, device=basis.device)
        B_sq = torch.zeros(n, dtype=basis.dtype, device=basis.device)

        B_sq[0] = (gso[0] ** 2).sum()
        for i in range(1, n):
            # Batched dot products: basis[i] against all previous GSO vectors
            dots = torch.mv(gso[:i], basis[i])  # [i]
            valid = B_sq[:i] > 1e-30
            mu[i, :i] = torch.where(valid, dots / B_sq[:i].clamp(min=1e-30),
                                    torch.zeros_like(dots))
            # Subtract all projections at once
            gso[i] = basis[i] - torch.mv(gso[:i].T, mu[i, :i])
            B_sq[i] = (gso[i] ** 2).sum()

        return gso, mu, B_sq

    def _lll_reduce(self, basis: torch.Tensor, delta: float = 0.99) -> torch.Tensor:
        """LLL reduction with incremental GSO updates (Cohen Algorithm 2.6.3).

        Key optimization: GSO is computed once upfront. Size reduction updates
        only mu coefficients (O(k) per step instead of O(n²d) full recompute).
        Swaps use the standard incremental GSO update formulas.
        """
        n = basis.shape[0]
        basis = basis.clone()
        _, mu, B_sq = self._compute_gso(basis)  # Single full GSO computation
        k = 1
        iterations = 0
        max_iter = n * n * 10  # Termination bound (O(n²) worst-case swaps)

        while k < n and iterations < max_iter:
            iterations += 1

            # Size reduce b_k against b_{k-1}, ..., b_0
            for j in range(k - 1, -1, -1):
                if abs(mu[k, j].item()) > 0.5:
                    r = torch.round(mu[k, j])
                    basis[k] = basis[k] - r * basis[j]
                    # Incremental mu update: O(j) instead of O(n²d) GSO recompute
                    # basis[j] = gso[j] + sum_{i<j} mu[j,i]*gso[i], so
                    # <basis_new[k], gso[m]>/B_sq[m] decreases by r*mu[j,m] for m<j
                    # and by r for m=j (since <basis[j], gso[j]>/B_sq[j] = 1).
                    if j > 0:
                        mu[k, :j] -= r * mu[j, :j]
                    mu[k, j] -= r

            # Lovász condition: ||b*_k||^2 >= (delta - mu[k,k-1]^2) * ||b*_{k-1}||^2
            if B_sq[k-1] > 1e-30:
                lovasz_ok = B_sq[k] >= (delta - mu[k, k-1]**2) * B_sq[k-1]
            else:
                lovasz_ok = True

            if lovasz_ok:
                k += 1
            else:
                # Swap b_k and b_{k-1}
                basis[[k, k-1]] = basis[[k-1, k]]

                # Incremental GSO update after swap (standard LLL formulas)
                mu_bar = mu[k, k-1].clone()
                B = B_sq[k] + mu_bar ** 2 * B_sq[k-1]

                if B > 1e-30:
                    old_Bk = B_sq[k].clone()
                    mu[k, k-1] = mu_bar * B_sq[k-1] / B
                    B_sq[k] = B_sq[k-1] * old_Bk / B
                    B_sq[k-1] = B

                    # Swap mu rows for j < k-1
                    if k >= 2:
                        temp = mu[k-1, :k-1].clone()
                        mu[k-1, :k-1] = mu[k, :k-1]
                        mu[k, :k-1] = temp

                    # Update mu for all rows i > k (vectorized)
                    if k + 1 < n:
                        t = mu[k+1:, k].clone()
                        mu[k+1:, k] = mu[k+1:, k-1] - mu_bar * t
                        mu[k+1:, k-1] = t + mu[k, k-1] * mu[k+1:, k]

                k = max(k - 1, 1)

        return basis

    def reduce(self, basis: torch.Tensor, rounds: int = 5) -> torch.Tensor:
        """LLL + BKZ reduction.

        Args:
            basis: Lattice basis [dim, dim], float64.
            rounds: Number of BKZ tours after initial LLL.

        Returns:
            Reduced basis [dim, dim].
        """
        n = basis.shape[0]
        basis = basis.clone().to(dtype=torch.float64, device=self.device)

        if self.use_fpylll:
            print(f"  Using fpylll backend (LLL + BKZ-{self.block_dim})")
            basis = self._fpylll_reduce(basis, rounds)
            metrics = self._compute_metrics(basis)
            print(f"  fpylll done: shortest={metrics['shortest']:.4f}, "
                  f"log_defect={metrics['log_defect']:.2f}, "
                  f"rhf={metrics['rhf']:.6f}")
            return basis

        # Phase 1: Full LLL reduction
        basis = self._lll_reduce(basis)
        metrics = self._compute_metrics(basis)
        print(f"  LLL done: shortest={metrics['shortest']:.4f}, "
              f"log_defect={metrics['log_defect']:.2f}, "
              f"rhf={metrics['rhf']:.6f}")

        # Phase 2: BKZ tours with block enumeration
        for rnd in range(rounds):
            improved = False

            for start in range(0, n - 1, max(1, self.block_dim // 2)):
                end = min(start + self.block_dim, n)
                block_size = end - start
                if block_size < 2:
                    continue

                # Extract block and enumerate
                block = basis[start:end, :].clone()
                enumerator = BladeEnumerator(block, device=self.device)
                short_vec, short_norm = enumerator.enumerate()

                if short_vec is not None:
                    current_norm = torch.norm(basis[start]).item()
                    if short_norm < current_norm * 0.999:
                        basis[start] = short_vec
                        # Local LLL re-reduction around insertion point
                        lo = max(0, start - 2)
                        hi = min(n, end + 2)
                        local = basis[lo:hi].clone()
                        local = self._lll_reduce(local)
                        basis[lo:hi] = local
                        improved = True

            metrics = self._compute_metrics(basis)
            print(f"  BKZ round {rnd+1}/{rounds}: "
                  f"shortest={metrics['shortest']:.4f}, "
                  f"log_defect={metrics['log_defect']:.2f}, "
                  f"rhf={metrics['rhf']:.6f}"
                  f"{' (improved)' if improved else ''}")

            if not improved:
                print(f"  BKZ converged at round {rnd+1}.")
                break

        return basis

    def _compute_metrics(self, basis: torch.Tensor) -> dict:
        """Compute reduction quality metrics with numerically safe LAPACK calls.

        Uses QR decomposition (Householder reflections) instead of slogdet
        (LU) to avoid DLASCL errors on ill-conditioned lattices at high
        dimensions. log|det(B)| = sum(log|R_ii|) + sum(log(row_norms)).
        """
        norms = torch.norm(basis, dim=1)
        shortest = norms.min().item()
        log_norms_sum = torch.log(norms.clamp(min=1e-100)).sum().item()

        # QR-based log|det|: det(B) = det(B/norms) * prod(norms)
        # QR of B_scaled gives |det(B_scaled)| = prod|R_ii|
        try:
            norms_safe = norms.clamp(min=1e-100)
            basis_scaled = basis / norms_safe.unsqueeze(1)
            _Q, R = torch.linalg.qr(basis_scaled)
            diag_abs = torch.abs(torch.diag(R)).clamp(min=1e-100)
            log_det_scaled = torch.log(diag_abs).sum().item()
            log_det = log_det_scaled + torch.log(norms_safe).sum().item()
        except Exception:
            log_det = log_norms_sum  # Fallback: assume orthogonal (defect ≈ 0)

        log_defect = log_norms_sum - log_det
        n = basis.shape[0]
        det_root = math.exp(log_det / n) if log_det > -1e30 else 1e-30
        rhf = (shortest / max(det_root, 1e-100)) ** (1.0 / n)
        return {'log_defect': log_defect, 'rhf': rhf, 'shortest': shortest,
                'log_det': log_det}


# ---------------------------------------------------------------------------
# 3. BladeEnumerator — Geometric Algebra-Augmented Shortest Vector Enumeration
# ---------------------------------------------------------------------------

class BladeEnumerator:
    """Branch-and-bound enumeration with Geometric Algebra blade-rejection pruning.

    Precomputes the blade hierarchy B_k = b_1 ∧ ... ∧ b_k and uses the
    blade-rejection norm as an analogue of the Schnorr-Euchner pruning criterion.
    """

    def __init__(self, block_basis: torch.Tensor, search_range: int = 3,
                 device: str = 'cpu'):
        self.device = device
        self.search_range = search_range
        self.block_dim = block_basis.shape[0]
        self.vec_dim = block_basis.shape[1]
        self.block_basis = block_basis.to(dtype=torch.float64, device=device)

        # Cap algebra dimension for tractability (Cl(p,0) with p <= 10)
        self.alg_dim = min(self.vec_dim, 10)
        self.algebra = CliffordAlgebra(p=self.alg_dim, q=0, device=device)

        # Embed basis vectors as multivectors (using first alg_dim components)
        self.mv_basis = []
        for i in range(self.block_dim):
            v = self.block_basis[i, :self.alg_dim]
            self.mv_basis.append(
                self.algebra.embed_vector(v.unsqueeze(0)).squeeze(0)
            )

        # Blade hierarchy
        self.blades = self._precompute_blades()

        # Gaussian heuristic bound (QR-based log|det| to avoid DLASCL errors)
        sq = min(self.block_dim, self.vec_dim)
        sub = self.block_basis[:, :sq]
        try:
            row_norms = torch.norm(sub, dim=1).clamp(min=1e-100)
            sub_scaled = sub / row_norms.unsqueeze(1)
            _Q, R = torch.linalg.qr(sub_scaled)
            diag_abs = torch.abs(torch.diag(R)).clamp(min=1e-100)
            log_det_scaled = torch.log(diag_abs).sum().item()
            log_det = log_det_scaled + torch.log(row_norms).sum().item()
            det_root = math.exp(log_det / sq)
            self.bound = math.sqrt(sq / (2 * math.pi * math.e)) * det_root * 1.05
        except Exception:
            self.bound = float('inf')

    def _precompute_blades(self) -> dict:
        blades = {}
        if self.block_dim == 0:
            return blades
        current = self.mv_basis[0]
        blades[1] = current
        for k in range(2, min(self.alg_dim, self.block_dim) + 1):
            current = self.algebra.wedge(
                current.unsqueeze(0), self.mv_basis[k-1].unsqueeze(0)
            ).squeeze(0)
            if induced_norm(self.algebra, current.unsqueeze(0)).item() < 1e-12:
                break
            blades[k] = current
        return blades

    def enumerate(self) -> tuple:
        """Enumerate with blade rejection pruning.

        Returns:
            (best_vector, best_norm) or (None, inf).
        """
        best_vec = None
        best_norm = float('inf')
        sr = self.search_range

        def search(depth, current_vec, current_mv):
            nonlocal best_vec, best_norm

            # Blade rejection pruning
            k = self.block_dim - depth
            if 0 < k <= len(self.blades) and k in self.blades:
                B_k = self.blades[k]
                q_reject = self.algebra.blade_reject(
                    current_mv.unsqueeze(0), B_k.unsqueeze(0)
                )
                if induced_norm(self.algebra, q_reject).item() > min(self.bound, best_norm):
                    return

            if depth == 0:
                norm = torch.norm(current_vec).item()
                if 0 < norm < best_norm:
                    best_norm = norm
                    best_vec = current_vec.clone()
                return

            idx = depth - 1
            for z in range(-sr, sr + 1):
                next_vec = current_vec + z * self.block_basis[idx]
                next_mv = (current_mv + z * self.mv_basis[idx]
                           if idx < len(self.mv_basis) else current_mv)
                search(depth - 1, next_vec, next_mv)

        zero_vec = torch.zeros(self.vec_dim, dtype=torch.float64, device=self.device)
        zero_mv = torch.zeros(self.algebra.dim, dtype=torch.float64, device=self.device)
        search(self.block_dim, zero_vec, zero_mv)
        return best_vec, best_norm


# ---------------------------------------------------------------------------
# 4. RotorSearchLayer — Neural SVP Candidate Generation via Rotor Features
# ---------------------------------------------------------------------------

class RotorSearchLayer(nn.Module):
    """Neural short-vector candidate generator via rotor-based feature extraction.

    Processing pipeline:
        embed_vector → channel expansion → CliffordLayerNorm →
        MultiRotorLayer(K=4) → GeometricNeutralizer → grade-1 extraction
        → norm-projection constraint

    Multivectors are pre-normalized to unit magnitude before the neutralizer
    to avoid ill-conditioned covariance estimates in the EMA statistics.
    A norm-projection step suppresses outputs whose magnitude exceeds a
    multiple of the target norm, preventing large-vector false positives.
    """

    def __init__(self, block_dim: int = 8, channels: int = 2,
                 num_rotors: int = 4, target_norm: float = None,
                 norm_multiplier: float = 2.0, device: str = 'cpu'):
        super().__init__()
        self.block_dim = block_dim
        self.channels = channels
        self.device = device
        self.target_norm = target_norm
        self.norm_multiplier = norm_multiplier

        alg_dim = min(block_dim, 10)
        self.algebra = CliffordAlgebra(p=alg_dim, q=0, device=device)

        self.norm_layer = CliffordLayerNorm(
            self.algebra, channels=channels, recover=False)
        self.multi_rotor = MultiRotorLayer(
            self.algebra, channels=channels, num_rotors=num_rotors)
        self.neutralizer = GeometricNeutralizer(
            self.algebra, channels=channels, momentum=0.3)

    def forward(self, blocks: torch.Tensor):
        """Process blocks through GA pipeline.

        Args:
            blocks: [B, L, block_dim].

        Returns:
            guided_blocks: [B, L, block_dim].
        """
        B, L, D = blocks.shape
        self.algebra.ensure_device(blocks.device)

        d = min(D, self.algebra.n)
        x_mv = self.algebra.embed_vector(
            blocks[..., :d].reshape(-1, d)
        )  # [B*L, 2^n]

        # Expand to channels
        x_mv = x_mv.unsqueeze(1).expand(-1, self.channels, -1).contiguous()

        # CliffordLayerNorm → MultiRotor
        x_mv = self.norm_layer(x_mv)
        x_mv = self.multi_rotor(x_mv)

        # Pre-normalize to unit multivectors before the neutralizer,
        # then restore the original scale afterward.  This stabilizes
        # the EMA covariance estimate when multivector magnitudes vary widely.
        mv_norms = x_mv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_normed = x_mv / mv_norms
        try:
            x_normed = self.neutralizer(x_normed)
        except Exception:
            pass  # Covariance is degenerate; neutralization is skipped
        x_mv = x_normed * mv_norms

        # Extract grade-1
        g1_idx = [1 << i for i in range(self.algebra.n)]
        guided = x_mv[..., g1_idx].mean(dim=1)  # [B*L, n]

        if d < D:
            pad = torch.zeros(guided.shape[0], D - d,
                              dtype=guided.dtype, device=guided.device)
            guided = torch.cat([guided, pad], dim=-1)

        # Norm-projection constraint: scale down outputs whose magnitude exceeds
        # norm_multiplier * target_norm, analogous to the bivector norm clipping
        # in optimizers/riemannian.py.
        if self.target_norm is not None:
            max_allowed = self.target_norm * self.norm_multiplier
            out_norms = torch.norm(guided, dim=-1, keepdim=True).clamp(min=1e-12)
            scale = torch.clamp(out_norms / max_allowed, min=1.0)
            guided = guided / scale

        return guided.view(B, L, D)


# ---------------------------------------------------------------------------
# 5. KyberSolver — End-to-End Orchestration with Hard Constraints
# ---------------------------------------------------------------------------

class KyberSolver:
    """End-to-end solver for Kyber Module-LWE instances.

    Phase 1: Module-LWE instance generation and full Kannan embedding
    Phase 2: LLL + BKZ basis reduction
    Phase 3: Neural short-vector search with hard constraints
             (unit Kannan coefficient; periodic integrality projection;
              three-term training objective)
    Phase 4: Secret extraction from candidate short vectors and algebraic verification

    Compatible with CPU and CUDA backends.  All arithmetic in torch float64.
    """

    def __init__(self, n: int = 256, k: int = 2, q: int = 3329,
                 eta1: int = 3, eta2: int = 2, eta: int = None,
                 block_dim: int = 8, bkz_rounds: int = 5,
                 search_steps: int = 300, num_trials: int = 10,
                 seed: int = 42, device: str = 'cpu',
                 use_fpylll: bool = True):
        if eta is not None:
            eta1, eta2 = eta, eta
        self.n, self.k, self.q, self.eta1, self.eta2 = n, k, q, eta1, eta2
        self.block_dim = block_dim
        self.bkz_rounds = bkz_rounds
        self.search_steps = search_steps
        self.num_trials = num_trials
        self.seed = seed
        self.device = device
        self.use_fpylll = use_fpylll
        self.batch_size = 2
        self.stride = block_dim

    def solve(self):
        print(f"\n{'='*60}")
        print(f" Kyber MLWE Solver via Kannan Embedding + GA")
        print(f" n={self.n}, k={self.k}, q={self.q}, eta1={self.eta1}, eta2={self.eta2}")
        print(f" device={self.device}")
        print(f"{'='*60}")
        start_total = time.time()

        # Phase 1: Generate instance (full embedding, no truncation)
        print(f"\n--- Phase 1: MLWE Instance Generation ---")
        instance = KyberInstance(
            n=self.n, k=self.k, q=self.q, eta1=self.eta1, eta2=self.eta2,
            seed=self.seed, device=self.device
        )
        basis = instance.basis.clone()
        wd = basis.shape[0]  # = 2*m + 1
        m = instance.m

        # Phase 2: LLL + BKZ reduction
        print(f"\n--- Phase 2: Lattice Reduction ---")
        reducer = GALatticeReducer(block_dim=self.block_dim, device=self.device,
                                   use_fpylll=self.use_fpylll)
        basis = reducer.reduce(basis, rounds=self.bkz_rounds)
        metrics = reducer._compute_metrics(basis)
        print(f"  Post-reduction: shortest={metrics['shortest']:.4f}, "
              f"rhf={metrics['rhf']:.6f}")

        # Phase 3: Neural search
        print(f"\n--- Phase 3: Neural SVP Search ---")

        # Norm bound: minimum of the instance's target vector norm and the Gaussian heuristic
        gh_norm = (math.sqrt(wd / (2 * math.pi * math.e)) *
                   math.exp(metrics['log_det'] / wd)) if metrics['log_det'] > -1e30 else float('inf')
        target_norm = min(instance.target_norm, gh_norm) if gh_norm > 0 else instance.target_norm
        print(f"  Norm bound: target={target_norm:.4f} "
              f"(instance={instance.target_norm:.4f}, GH={gh_norm:.4f})")

        model = RotorSearchLayer(
            block_dim=self.block_dim, channels=2, num_rotors=4,
            target_norm=target_norm, norm_multiplier=2.0,
            device=self.device
        ).to(device=self.device, dtype=torch.float64)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  RotorSearchLayer: {num_params} params")

        num_blocks = max(1, (wd - self.block_dim) // self.stride + 1)
        found_vectors = []
        existing_blades = [None] * num_blocks

        def _make_model():
            return RotorSearchLayer(
                block_dim=self.block_dim, channels=2, num_rotors=4,
                target_norm=target_norm, norm_multiplier=2.0,
                device=self.device
            ).to(device=self.device, dtype=torch.float64)

        for trial in range(self.num_trials):
            print(f"\n  Trial {trial+1}/{self.num_trials}")
            t0 = time.time()

            # Reinitialize model if any parameters contain NaN values from the previous trial
            if any(torch.isnan(p).any() for p in model.parameters()):
                model = _make_model()

            # Try both signs for the Kannan embedding coefficient
            best_result = None
            for sign in [1.0, -1.0]:
                result = self._neural_search(
                    basis, model, num_blocks, existing_blades,
                    attempt=trial, embed_sign=sign
                )
                if result is not None:
                    _, _, norm = result
                    if best_result is None or norm < best_result[2]:
                        best_result = result

            if best_result is not None:
                coeffs, vec, norm = best_result

                # Unimodular basis update
                new_basis = self._unimodular_update(basis, coeffs, vec)
                if new_basis is not None:
                    old_m = reducer._compute_metrics(basis)
                    new_m = reducer._compute_metrics(new_basis)
                    if new_m['log_defect'] < old_m['log_defect'] - 1e-4:
                        basis = new_basis
                        basis = reducer.reduce(basis, rounds=1)
                        print(f"    Basis updated: log_defect "
                              f"{old_m['log_defect']:.2f} → {new_m['log_defect']:.2f}")

                found_vectors.append({'norm': norm, 'vector': vec, 'coeffs': coeffs})
                print(f"    Found: norm={norm:.4f} (time={time.time()-t0:.1f}s)")
            else:
                print(f"    No improvement.")

        # Phase 4: Solution Extraction
        print(f"\n--- Phase 4: Solution Extraction ---")
        solution_found = False

        # Sub-phase 4a: Check reduced basis rows directly (cheapest check first)
        basis_norms = torch.norm(basis, dim=1)
        sorted_basis_idx = torch.argsort(basis_norms)
        n_check = min(20, wd)
        print(f"  Scanning {n_check} shortest basis vectors...")

        for rank, idx in enumerate(sorted_basis_idx[:n_check]):
            bvec = basis[idx.item()]
            bnorm = basis_norms[idx.item()].item()

            e_cand = bvec[:m].round().long()
            s_cand = -bvec[m:2*m].round().long()

            if instance.verify_solution(s_cand, e_cand):
                print(f"  *** SOLUTION FROM BASIS ROW {idx.item()} "
                      f"(norm={bnorm:.4f}) ***")
                print(f"  ||e||={torch.norm(e_cand.double()).item():.2f}, "
                      f"||s||={torch.norm(s_cand.double()).item():.2f}")
                solution_found = True
                break

            s_cand2 = bvec[m:2*m].round().long()
            if instance.verify_solution(s_cand2, e_cand):
                print(f"  *** SOLUTION FROM BASIS ROW {idx.item()} "
                      f"(alt sign, norm={bnorm:.4f}) ***")
                solution_found = True
                break

        # Sub-phase 4b: Check pairwise combinations of shortest basis vectors
        if not solution_found:
            top_k = min(10, wd)
            top_idx = sorted_basis_idx[:top_k]
            print(f"  Scanning pairwise combinations of shortest {top_k} vectors...")

            for i in range(top_k):
                if solution_found:
                    break
                for j in range(i + 1, top_k):
                    if solution_found:
                        break
                    for ci in [-1, 1]:
                        if solution_found:
                            break
                        for cj in [-1, 1]:
                            combo = (ci * basis[top_idx[i].item()] +
                                     cj * basis[top_idx[j].item()])

                            e_cand = combo[:m].round().long()
                            s_cand = -combo[m:2*m].round().long()

                            if instance.verify_solution(s_cand, e_cand):
                                print(f"  *** SOLUTION FROM COMBINATION "
                                      f"({ci}*row{top_idx[i].item()}"
                                      f"+{cj}*row{top_idx[j].item()}) "
                                      f"norm={torch.norm(combo).item():.4f} ***")
                                solution_found = True
                                break

                            s_cand2 = combo[m:2*m].round().long()
                            if instance.verify_solution(s_cand2, e_cand):
                                print(f"  *** SOLUTION FROM COMBINATION "
                                      f"(alt sign) ***")
                                solution_found = True
                                break

        # Sub-phase 4c: Check neural search found_vectors
        if not solution_found:
            found_vectors.sort(key=lambda x: x['norm'])

            for i, fv in enumerate(found_vectors[:10]):
                vec = fv['vector']
                print(f"  Candidate {i+1}: norm={fv['norm']:.4f}")

                e_cand = vec[:m].round().long()
                s_cand = -vec[m:2*m].round().long()

                if instance.verify_solution(s_cand, e_cand):
                    print(f"  *** SOLUTION VERIFIED: As + e ≡ b (mod q) ***")
                    print(f"  ||e||={torch.norm(e_cand.double()).item():.2f}, "
                          f"||s||={torch.norm(s_cand.double()).item():.2f}")
                    solution_found = True
                    break

                s_cand2 = vec[m:2*m].round().long()
                if instance.verify_solution(s_cand2, e_cand):
                    print(f"  *** SOLUTION VERIFIED (alt sign): "
                          f"As + e ≡ b (mod q) ***")
                    solution_found = True
                    break

        if not solution_found:
            print(f"  Solution not extracted from found vectors.")
            if found_vectors:
                sv = found_vectors[0]['vector']
                res = self._lattice_membership_check(instance.basis, sv)
                print(f"  Lattice membership residual: {res:.2e}")

        total_time = time.time() - start_total
        print(f"\nTotal time: {total_time:.1f}s")
        final = reducer._compute_metrics(basis)
        print(f"Final: shortest={final['shortest']:.4f}, "
              f"log_defect={final['log_defect']:.2f}, rhf={final['rhf']:.6f}")

    def _slice_blocks(self, v: torch.Tensor, num_blocks: int) -> torch.Tensor:
        """Slice vector into non-overlapping blocks."""
        blocks = []
        for i in range(num_blocks):
            start = i * self.stride
            end = start + self.block_dim
            if end <= v.shape[1]:
                blocks.append(v[:, start:end])
        if not blocks:
            return v[:, :self.block_dim].unsqueeze(1)
        return torch.stack(blocks, dim=1)

    def _neural_search(self, basis: torch.Tensor, model: nn.Module,
                       num_blocks: int, existing_blades: list,
                       attempt: int = 0, embed_sign: float = 1.0) -> tuple:
        """Gradient-based short-vector search with hard constraints and a three-term objective.

        Hard constraints enforced at each step:
          1. Kannan embedding coefficient fixed to ±1
          2. Periodic integrality projection: near-integer coefficients are
             rounded in-place every 50 steps
          3. Element-wise coefficient clamp to [-100, 100] for numerical stability

        Training objective:
          L = norm_sq + 20·guide_loss + 10·ortho_penalty

        The guide loss is the cosine dissimilarity between the current candidate
        and the model's suggested direction (scale-invariant).  The orthogonality
        penalty promotes linear independence across search attempts.
        """
        wd = basis.shape[0]
        dev = self.device

        # Stabilize basis
        norms = torch.norm(basis, dim=1, keepdim=True).clamp(min=1e-12)
        basis_stab = basis / norms
        norms_flat = norms.squeeze(1)

        # Learnable free coefficients (all except last)
        c_free = nn.Parameter(torch.zeros(
            self.batch_size, wd - 1, dtype=torch.float64, device=dev
        ))
        # Last coefficient: hard constraint ±1
        c_embed = embed_sign * torch.ones(
            self.batch_size, 1, dtype=torch.float64, device=dev
        )

        # Seed initialization
        with torch.no_grad():
            sorted_idx = torch.argsort(norms_flat[:-1])
            for b in range(self.batch_size):
                seed_idx = sorted_idx[(attempt * self.batch_size + b) % (wd - 1)].item()
                c_free[b, seed_idx] = 1.0
            c_free.add_(torch.randn_like(c_free) * 0.001)

        optimizer = torch.optim.Adam(list(model.parameters()) + [c_free], lr=0.002)
        alg = model.algebra

        best_norm = float('inf')
        best_coeffs = None
        best_vec = None
        patience_count = 0
        patience = max(self.search_steps // 2, 100)

        for step in range(self.search_steps):
            optimizer.zero_grad()

            # Assemble full coefficient vector with hard constraint
            c = torch.cat([c_free, c_embed], dim=1)  # [B, wd]

            # Reconstruct lattice vector
            v = (c * norms_flat.unsqueeze(0)) @ basis_stab  # [B, wd]

            # Block processing through GA model
            blocks = self._slice_blocks(v, num_blocks)
            guided = model(blocks)

            # Reconstruct guided vector
            guided_v = torch.zeros_like(v)
            counts = torch.zeros_like(v)
            for i in range(guided.shape[1]):
                start = i * self.stride
                end = start + self.block_dim
                if end <= wd:
                    guided_v[:, start:end] += guided[:, i]
                    counts[:, start:end] += 1
            guided_v = guided_v / counts.clamp(min=1)

            # === Training objective: three-term loss ===
            norm_sq = (v ** 2).sum(dim=1, keepdim=True)

            # Cosine dissimilarity (scale-invariant): measures directional
            # agreement between the candidate v and the model's guidance signal.
            v_n = torch.norm(v, dim=1, keepdim=True).clamp(min=1e-8)
            gv_n = torch.norm(guided_v, dim=1, keepdim=True).clamp(min=1e-8)
            guide_loss = 1.0 - (v * guided_v).sum(dim=1, keepdim=True) / (v_n * gv_n)

            # Orthogonality penalty via blade rejection
            ortho_penalty = torch.zeros(
                self.batch_size, 1, dtype=torch.float64, device=dev)
            for i in range(min(guided.shape[1], len(existing_blades))):
                if existing_blades[i] is not None:
                    start = i * self.stride
                    end = start + self.block_dim
                    if end <= wd:
                        d = min(self.block_dim, alg.n)
                        mv_v = alg.embed_vector(v[:, start:start+d])
                        blade = existing_blades[i].to(
                            dtype=torch.float64, device=dev
                        ).expand(self.batch_size, -1)
                        w = alg.wedge(blade, mv_v)
                        ortho_penalty += 1.0 / (induced_norm(alg, w) + 1e-4)

            loss = (norm_sq + 20.0 * guide_loss + 10.0 * ortho_penalty).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [c_free], max_norm=1.0)
            optimizer.step()

            # Constrain bivector parameter norms to avoid overflow in exp(),
            # consistent with RiemannianAdam's max_bivector_norm=10.0.
            with torch.no_grad():
                for p in model.parameters():
                    p_norm = torch.norm(p)
                    if p_norm > 10.0:
                        p.mul_(10.0 / p_norm)

            # === Hard constraint 3: element-wise coefficient clamp ===
            # Bounds free coefficients to [-100, 100] to prevent numerical
            # overflow while preserving per-element direction (unlike uniform
            # norm scaling, this leaves the fixed c_embed unaffected).
            with torch.no_grad():
                c_free.data.clamp_(-100, 100)

            # === Hard constraint 2: periodic integrality projection ===
            if step > 0 and step % 50 == 0:
                with torch.no_grad():
                    residual = (c_free - c_free.round()).abs()
                    snap_mask = residual < 0.15
                    c_free.data[snap_mask] = c_free.data[snap_mask].round()

            # Track best rounded vector
            improved = False
            with torch.no_grad():
                c_full = torch.cat([c_free.round(), c_embed], dim=1)
                for b in range(self.batch_size):
                    if torch.all(c_full[b] == 0):
                        continue
                    v_round = (c_full[b] * norms_flat) @ basis_stab
                    n_round = torch.norm(v_round).item()
                    if n_round < best_norm * 0.99999 and n_round > 0:
                        best_norm = n_round
                        best_coeffs = c_full[b].clone()
                        best_vec = v_round.clone()
                        improved = True

            if improved:
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= patience:
                break

            if step % 100 == 0:
                print(f"      Step {step:3d} | Loss: {loss.item():.2e} "
                      f"| Best: {best_norm:.4f}")

        # Update existing blades for independence tracking
        if best_vec is not None:
            blks = self._slice_blocks(best_vec.unsqueeze(0), num_blocks).squeeze(0)
            for i in range(min(blks.shape[0], len(existing_blades))):
                d = min(blks.shape[1], alg.n)
                mv_v = alg.embed_vector(blks[i, :d].unsqueeze(0))
                if existing_blades[i] is None:
                    existing_blades[i] = mv_v.squeeze(0)
                else:
                    nb = alg.wedge(existing_blades[i].unsqueeze(0), mv_v).squeeze(0)
                    if induced_norm(alg, nb.unsqueeze(0)).item() > 1e-4:
                        existing_blades[i] = nb

        if best_coeffs is not None:
            return best_coeffs, best_vec, best_norm
        return None

    def _unimodular_update(self, basis: torch.Tensor, coeffs: torch.Tensor,
                           new_vec: torch.Tensor) -> torch.Tensor:
        """Replace pivot basis row with new short vector + size reduce."""
        c = coeffs.round().long()
        if torch.all(c == 0):
            return None

        pivot_idx = torch.argmax(torch.abs(c)).item()
        if c[pivot_idx].abs() == 0:
            return None

        new_basis = basis.clone()
        new_basis[pivot_idx] = new_vec

        # Size-reduce other rows against the new vector
        n = new_basis.shape[0]
        bi_sq = (new_basis[pivot_idx] ** 2).sum()
        if bi_sq > 1e-30:
            for j in range(n):
                if j == pivot_idx:
                    continue
                mu = (new_basis[j] * new_basis[pivot_idx]).sum() / bi_sq
                r = torch.round(mu)
                if r.abs() > 0:
                    new_basis[j] = new_basis[j] - r * new_basis[pivot_idx]

        return new_basis

    def _lattice_membership_check(self, original_basis: torch.Tensor,
                                  vec: torch.Tensor) -> float:
        """Check approximate lattice membership using QR decomposition.

        Uses pre-conditioned QR solve instead of lstsq to avoid DLASCL
        errors on ill-conditioned lattice bases.
        """
        try:
            B_T = original_basis.T
            col_norms = torch.norm(B_T, dim=0).clamp(min=1e-100)
            B_scaled = B_T / col_norms.unsqueeze(0)
            Q, R = torch.linalg.qr(B_scaled)
            rhs = Q.T @ vec
            coeffs_scaled = torch.linalg.solve_triangular(
                R, rhs.unsqueeze(1), upper=True).squeeze(1)
            coeffs = coeffs_scaled / col_norms
            residual = torch.norm(B_T @ coeffs - vec).item()
            int_residual = torch.norm(coeffs - coeffs.round()).item()
            return residual + int_residual
        except Exception:
            return float('inf')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kyber MLWE Lattice Solver via Kannan Embedding + GA"
    )
    parser.add_argument('--n', type=int, default=256,
                        help='Polynomial degree (default: 256)')
    parser.add_argument('--k', type=int, default=2,
                        help='Module rank (default: 2)')
    parser.add_argument('--q', type=int, default=3329,
                        help='Modulus (default: 3329)')
    parser.add_argument('--eta1', type=int, default=3,
                        help='CBD parameter for secret (Kyber-512: 3)')
    parser.add_argument('--eta2', type=int, default=2,
                        help='CBD parameter for error (Kyber-512: 2)')
    parser.add_argument('--eta', type=int, default=None,
                        help='Set both eta1 and eta2 (overrides individual values)')
    parser.add_argument('--block-dim', type=int, default=8,
                        help='BKZ block dimension (default: 8)')
    parser.add_argument('--bkz-rounds', type=int, default=5,
                        help='BKZ reduction rounds (default: 5)')
    parser.add_argument('--search-steps', type=int, default=300,
                        help='Neural search GD steps (default: 300)')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of neural search restarts (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, cuda, or auto (default: cpu)')
    parser.add_argument('--no-fpylll', action='store_true',
                        help='Disable fpylll backend, use PyTorch LLL/BKZ')
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    eta1 = args.eta if args.eta is not None else args.eta1
    eta2 = args.eta if args.eta is not None else args.eta2

    solver = KyberSolver(
        n=args.n, k=args.k, q=args.q, eta1=eta1, eta2=eta2,
        block_dim=args.block_dim, bkz_rounds=args.bkz_rounds,
        search_steps=args.search_steps, num_trials=args.num_trials,
        seed=args.seed, device=args.device,
        use_fpylll=not args.no_fpylll
    )
    solver.solve()


if __name__ == '__main__':
    main()