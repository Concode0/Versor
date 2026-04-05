# What Makes Versor Unique: Code Examples

Ten innovations that distinguish Versor from standard deep learning frameworks, each illustrated with actual source code.

---

## 1. Signature-Aware Exponential Map

**What it does**: A single vectorized formula exponentiates bivectors across all three metric regimes — elliptic (cos/sin for Euclidean rotations), hyperbolic (cosh/sinh for Lorentz boosts), and parabolic (1+B for null dimensions) — with no branching per element.

**How standard DL differs**: Separate rotation matrix construction, no mixed-signature support.

From `core/algebra.py` — `CliffordAlgebra._exp_bivector_closed`:

```python
# Signed squared norm: alpha = Sum_k b_k^2 . (e_k)^2
# alpha < 0 -> elliptic (Euclidean-like), alpha > 0 -> hyperbolic
alpha = (bv_coeffs * bv_coeffs * bv_sq).sum(dim=-1, keepdim=True)

abs_alpha = alpha.abs().clamp(min=1e-12)
theta = torch.sqrt(abs_alpha)

# Elliptic branch: cos(theta) and sin(theta)/theta
cos_theta = torch.cos(theta)
sinc_theta = torch.where(
    theta > 1e-7,
    torch.sin(theta) / theta,
    1.0 - abs_alpha / 6.0,
)

# Hyperbolic branch: cosh(theta) and sinh(theta)/theta
cosh_theta = torch.cosh(theta)
sinhc_theta = torch.where(
    theta > 1e-7,
    torch.sinh(theta) / theta,
    1.0 + abs_alpha / 6.0,
)

# Select branch based on sign of alpha
is_elliptic = alpha < -1e-12
is_hyperbolic = alpha > 1e-12

scalar_part = torch.where(
    is_elliptic, cos_theta,
    torch.where(is_hyperbolic, cosh_theta, torch.ones_like(theta))
)
coeff_part = torch.where(
    is_elliptic, sinc_theta,
    torch.where(is_hyperbolic, sinhc_theta, torch.ones_like(theta))
)

result = coeff_part * B
result[..., 0] = scalar_part.squeeze(-1)
```

Zero geometric products. Exact for simple bivectors in any `Cl(p,q,r)`.

---

## 2. Rotor Sandwich Product

**What it does**: Learns bivector parameters B, computes R = exp(-B/2), and applies the isometry x' = RxR~ — a pure geometric rotation that preserves lengths, angles, and the origin.

**How standard DL differs**: `nn.Linear` applies an unconstrained weight matrix W that can stretch, shear, and deform.

From `layers/primitives/rotor.py` — `RotorLayer`:

```python
def _compute_rotors(self, device, dtype):
    """Compute R and R~ from bivector weights."""
    B = torch.zeros(self.channels, self.algebra.dim, device=device, dtype=dtype)
    indices = self.bivector_indices.unsqueeze(0).expand(self.channels, -1)
    B.scatter_(1, indices, self.bivector_weights)

    R = self.algebra.exp(-0.5 * B)
    R_rev = self.algebra.reverse(R)
    return R, R_rev

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the sandwich product x' = RxR~."""
    R, R_rev = self._compute_rotors(x.device, x.dtype)

    R_expanded = R.unsqueeze(0)
    R_rev_expanded = R_rev.unsqueeze(0)

    Rx = self.algebra.geometric_product(R_expanded, x)
    res = self.algebra.geometric_product(Rx, R_rev_expanded)
    return res
```

Every parameter is a bivector component — a specific plane of rotation with direct geometric meaning.

---

## 3. Direction-Preserving Activation (GeometricGELU)

**What it does**: Scales the magnitude of a multivector via GELU while preserving its direction (the unit multivector). The activation cannot rotate the data — only the RotorLayer does that.

**How standard DL differs**: ReLU/GELU applied coefficient-wise destroys geometric direction information.

From `functional/activation.py` — `GeometricGELU.forward`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True)
    eps = 1e-6
    scale = F.gelu(norm + self.bias.view(1, -1, 1)) / (norm + eps)
    return x * scale
```

Five lines. The direction `x / ||x||` is untouched; only the scalar magnitude changes.

---

## 4. Riemannian Adam

**What it does**: Runs Adam momentum in the Lie algebra (bivector space) with bivector norm clipping. Combined with `exp(-B/2)` in the forward pass, this gives a Riemannian update on the Spin(n) manifold.

**How standard DL differs**: Standard Adam updates unconstrained Euclidean parameters with no manifold awareness.

From `optimizers/riemannian.py` — `RiemannianAdam.step`:

```python
# Adam update in Lie algebra (bivector space)
# Combined with exp(-B/2) in forward pass, this gives Riemannian update
denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
p.addcdiv_(exp_avg, denom, value=-step_size)

# Clip bivector norm for numerical stability in exp()
if self.max_bivector_norm is not None:
    p_norm = p.norm(dim=-1, keepdim=True)
    scale = torch.clamp(p_norm / self.max_bivector_norm, min=1.0)
    p.div_(scale)
```

The key insight: because Versor parameterizes rotors via bivectors (the Lie algebra), Euclidean gradient updates in bivector space ARE geometrically meaningful. The `exp(-B/2)` in the forward pass completes the manifold retraction.

---

## 5. Automatic Metric Search

**What it does**: Lifts data into a conformal algebra `Cl(X+1, 1)`, trains multiple GBN probes with biased initialization, then reads the learned bivector energy distribution to infer the optimal `(p, q, r)` signature.

**How standard DL differs**: Architecture hyperparameters (hidden sizes, attention heads) are manually chosen or grid-searched with no geometric interpretation.

From `core/analysis/signature.py` — `MetricSearch._analyze_bivector_energy`:

```python
# For each basis bivector e_ab, look up bv_sq_scalar:
#   -1 -> elliptic, +1 -> hyperbolic, 0 -> null
sq_val = bv_sq[bv_idx_pos].item()
if sq_val < -0.5:
    sig_type = 'elliptic'
elif sq_val > 0.5:
    sig_type = 'hyperbolic'
else:
    sig_type = 'null'
```

The probe's learned rotation planes directly reveal which metric regime the data lives in.

---

## 6. Hermitian Metrics for Mixed Signatures

**What it does**: Constructs a positive-definite inner product for any `Cl(p,q,r)` via Clifford conjugation + metric signs. This ensures gradient-based optimization works even in Minkowski or degenerate algebras where the standard norm can be negative.

**How standard DL differs**: Euclidean L2 norm only — breaks when applied to Minkowski-signature data.

From `core/metric.py` — `_hermitian_signs` and `hermitian_inner_product`:

```python
def _hermitian_signs(algebra: CliffordAlgebra) -> torch.Tensor:
    signs = torch.ones(algebra.dim, device=algebra.device)
    pq = algebra.p + algebra.q
    for i in range(algebra.dim):
        k = bin(i).count('1')  # grade
        conj_sign = ((-1) ** k) * ((-1) ** (k * (k - 1) // 2))
        metric_product = 1
        has_null = False
        for bit in range(algebra.n):
            if i & (1 << bit):
                if bit >= pq:
                    has_null = True
                    break
                metric_product *= (1 if bit < algebra.p else -1)
        if has_null:
            signs[i] = 0
        else:
            metric_sign = ((-1) ** (k * (k - 1) // 2)) * metric_product
            signs[i] = conj_sign * metric_sign
    return signs

def hermitian_inner_product(algebra, A, B):
    signs = _hermitian_signs(algebra).to(device=A.device, dtype=A.dtype)
    return (signs * A * B).sum(dim=-1, keepdim=True)
```

Precomputed once per algebra. For Euclidean `Cl(p,0)`, all signs are +1 and this reduces to the standard dot product.

---

## 7. Bivector Decomposition via Power Iteration

**What it does**: Decomposes a non-simple bivector (one that cannot be written as a single wedge product) into simple components via GA power iteration, then exponentiates each in closed form and composes via geometric product.

**How standard DL differs**: Matrix exponentials typically use Taylor series or Pade approximation — no geometric decomposition.

From `core/decomposition.py` — `ga_power_iteration`:

```python
def ga_power_iteration(algebra, b, v_init=None, threshold=1e-6, max_iterations=100):
    """Find the dominant simple bivector component."""
    if v_init is None:
        v_raw = torch.randn(*batch_shape, algebra.n, device=device, dtype=dtype)
        v = algebra.embed_vector(v_raw)

    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    for _ in range(max_iterations):
        v_prev = v
        v = algebra.right_contraction(b, v)          # Key: GA right contraction
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        if (v - v_prev).norm(dim=-1).max() < threshold:
            break

    u = algebra.right_contraction(b, v)
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    sigma = b.norm(dim=-1, keepdim=True)
    b_s = sigma * algebra.wedge(u, v)                 # Simple projection
    return b_s, v
```

Reference: Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers from Irreducibles." arXiv:2507.11688v1

---

## 8. Rotor Gadget (Parameter-Efficient Linear)

**What it does**: Replaces `nn.Linear` with left/right rotor pairs and block-diagonal channel routing. Uses `O(K * n(n-1)/2)` parameters instead of `O(in_channels * out_channels)`, achieving ~63% parameter reduction.

**How standard DL differs**: `nn.Linear` uses a dense weight matrix with no geometric structure.

From `layers/primitives/rotor_gadget.py` — `RotorGadget.forward`:

```python
def _compute_rotors(self):
    B_left = self._bivector_to_multivector(self.bivector_left)    # [pairs, dim]
    B_right = self._bivector_to_multivector(self.bivector_right)  # [pairs, dim]

    R_left = self.algebra.exp(-0.5 * B_left)        # Left rotor
    R_right = self.algebra.exp(-0.5 * B_right)      # Right rotor
    R_right_rev = self.algebra.reverse(R_right)      # Reverse for sandwich

    return R_left, R_right_rev

def forward(self, x):
    R_left, R_right_rev = self._compute_rotors()

    # Two batched GPs instead of 2*K sequential GPs
    temp = self.algebra.geometric_product(R_left_expanded, x)
    out = self.algebra.geometric_product(temp, R_right_expanded)
    return self._aggregate_to_output_channels(out)
```

The transformation `psi(x) = r . x . s~` where r, s are independent rotors — more expressive than a single sandwich product.

---

## 9. Rotor-to-Formula Translation

**What it does**: Reads trained bivector weights as symbolic rotation angles and planes, then maps each to its closed-form trigonometric/hyperbolic action — producing a human-readable formula from a trained neural network.

**How standard DL differs**: Black-box neural networks require post-hoc interpretation tools (SHAP, LIME). Here the formula is a direct readout.

From `models/sr/translator.py` — `RotorTranslator._plane_to_action`:

```python
def _plane_to_action(self, plane: SimplePlane) -> sympy.Expr:
    """Closed-form sandwich product action for a single plane."""
    xi = self.symbols[plane.var_i]
    xj = self.symbols[plane.var_j]
    theta = plane.angle

    if plane.sig_type == "elliptic":
        return xi * sympy.cos(2 * theta) - xj * sympy.sin(2 * theta)
    elif plane.sig_type == "hyperbolic":
        return xi * sympy.cosh(2 * theta) + xj * sympy.sinh(2 * theta)
    else:  # parabolic
        return xi + 2 * theta * xj
```

A trained rotation angle of 0.785 rad in the e12 plane becomes `cos(1.57)*x1 - sin(1.57)*x2`. The algebra guarantees this mapping is exact.

---

## 10. Iterative Geometric Unbending

**What it does**: A 4-phase pipeline for symbolic regression that uses GA blade rejection instead of numerical subtraction to iteratively extract formula terms.

**How standard DL differs**: Genetic programming (PySR) or transformer-based equation search. No geometric structure in the search.

From `models/sr/unbender.py` — pipeline summary:

```
Phase 0: Data Preparation
  - SVD alignment, variable grouping, implicit probe

Phase 1: Per-Group Iterative Extraction
  - Single-rotor-per-stage with GA orthogonal elimination
  - blade rejection (NOT numerical subtraction)

Phase 2: Mother Algebra Cross-Term Discovery
  - GPCA in Cl(P,Q,R) for cross-group interactions

Phase 3: SymPy Refinement
  - lstsq reweight, implicit solve, simplify
```

Key advantage: GA blade rejection (`algebra.blade_reject`) orthogonally removes discovered components from the residual, preserving geometric structure that numerical subtraction would corrupt. Each stage's rotor directly encodes a formula term via the translator (Innovation #9).
