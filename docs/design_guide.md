# Geometric Algebra Model Design Guide

A practitioner's guide to building models with Versor. Assumes familiarity with PyTorch; no prior Clifford algebra knowledge required.

---

## 1. Does Your Task Have Geometric Structure?

Versor adds inductive bias: it constrains certain layers to isometries (length- and angle-preserving maps). This is useful when you know the symmetry group of your data; it is overhead when you don't.

**Use Versor when your task has one or more of these properties:**

- **Spatial coordinates as inputs** — positions, orientations, distances, angles, velocities (molecules, point clouds, robot kinematics, weather fields).
- **Equivariance requirement** — a rotation of the input should produce a predictable, structured transformation of the output, not just a scalar similarity score.
- **Known manifold structure** — EEG phase-amplitude coupling, Lorentz-invariant physics, hyperbolic embeddings, projective geometry.

**Standard PyTorch is fine when:**

- Inputs are tokens, pixels, or tabular features with no geometric interpretation.
- Channels are arbitrary feature slots with no spatial relationship.
- You are prototyping and want to minimize debugging surface area.

---

## 2. Choosing a Metric Signature

The signature $Cl(p, q, r)$ determines what the algebra "knows" about your geometry. Use the table below as a starting point.

| Signature | Geometry | Typical tasks |
|---|---|---|
| $Cl(3, 0)$ | Euclidean 3D | Molecules (QM9, MD17), point clouds |
| $Cl(3, 0, 1)$ | Projective GA / SE(3) | Molecular dynamics with translations, robotics |
| $Cl(3, 1)$ or $Cl(1, 3)$ | Minkowski / spacetime | EEG phase-amplitude, relativistic physics |
| $Cl(4, 1)$ | Conformal GA | Logic, CAD, translations-as-rotations |
| $Cl(n, 0)$ | High-dimensional Euclidean | Semantic embeddings, symbolic regression |

**When none of the above fits**, let the data decide:

```python
from core.search import MetricSearch

best_p, best_q, best_r = MetricSearch(device='cpu').search(your_data_tensor)
algebra = CliffordAlgebra(best_p, best_q, best_r, device='cpu')
```

**Apple Silicon / CPU note:** always pass `device='cpu'` explicitly to `CliffordAlgebra`. It does not default to MPS.

---

## 3. Layer Decision Map

Every GBN model uses a mix of geometric and non-geometric layers. This table shows which layer to reach for and when to skip it.

| Purpose | Layer | Key property | When to skip |
|---|---|---|---|
| Geometric rotation (even versor) | `RotorLayer` | Isometry — preserves lengths, angles; Spin group | No manifold structure in the input |
| Reflection (odd versor) | `ReflectionLayer` | Learns unit vectors; applies `x' = -nxn⁻¹`; Pin group | Task has no reflection symmetry |
| Multi-scale rotation | `MultiRotorLayer` | K-rotor superposition | Simple tasks; use `RotorLayer` first |
| Channel mixing | `CliffordLinear` (traditional backend) | Standard scalar weight matrix | Never — always needed alongside rotors |
| Constrained channel mixing | `CliffordLinear(backend='rotor')` | ~63% fewer params, bivector-constrained | Need full cross-channel expressivity |
| Normalization | `CliffordLayerNorm` | Preserves direction, normalizes magnitude | Very shallow models (1–2 layers) |
| Non-linearity | `GeometricGELU` | Magnitude gating, preserves direction | When coefficient-wise activation is intentional |
| Grade filtering | `BladeSelector` | Soft attention over basis blades | No a priori grade structure in the task |
| Task readout | `nn.Linear` on flattened multivector | Unconstrained projection to output space | Never — always use standard linear for readout |

The key principle: **`RotorLayer` rotates; `CliffordLinear` mixes channels; `nn.Linear` projects to outputs.** These are three different jobs. Do not conflate them.

---

## 4. The Standard GBN Stack

The canonical Geometric Blade Network block, annotated:

```python
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.primitives.linear import CliffordLinear
from layers.primitives.rotor import RotorLayer
from layers.primitives.normalization import CliffordLayerNorm
from functional.activation import GeometricGELU

algebra = CliffordAlgebra(p=3, q=0, device='cpu')  # Euclidean 3D

class GBNBlock(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        # Channel mixing: scalar weight matrix, O(channels^2) params
        # This is NOT a geometric operation — channels are feature slots
        self.linear_in = CliffordLinear(algebra, channels, channels)

        # Normalization: preserves the direction of each multivector
        self.norm = CliffordLayerNorm(algebra, channels)

        # Non-linearity: gates the magnitude via GELU, direction unchanged
        self.act = GeometricGELU(algebra, channels)

        # Geometric rotation: isometry, O(n^2/2) bivector params per channel
        # This IS the geometric operation — constrains to the rotation group
        self.rotor = RotorLayer(algebra, channels)

    def forward(self, x):
        # x: [Batch, channels, 2^n]
        x = self.linear_in(x)   # channel mixing (standard linear algebra)
        x = self.norm(x)         # normalize magnitudes
        x = self.act(x)          # non-linearity
        x = self.rotor(x)        # geometric rotation
        return x
```

Stack multiple `GBNBlock` instances for depth. The final readout is always a standard `nn.Linear`:

```python
# Readout: flatten multivector dimension, project to task output
# [Batch, channels, 2^n] → [Batch, channels * 2^n] → [Batch, out_dim]
self.head = nn.Linear(channels * algebra.dim, out_dim)

# In forward:
out = x.flatten(1)   # flatten channels + blade dimensions
out = self.head(out)
```

---

## 5. Hybrid Design: Versor + Standard PyTorch

Versor models are intentionally hybrid. Standard `nn.Linear` and `CliffordLinear` are expected — not a compromise.

**Use `nn.Linear` for:**
- Embedding raw scalar features into the multivector channel space
- Attention weight computation (no geometric meaning)
- Final task-specific readout heads
- Any projection between non-geometric feature spaces

**Use `CliffordLinear` (traditional backend) for:**
- Mixing channels within the multivector representation
- Situations where you want the weight matrix to see all blade components together

**Use `RotorLayer` / `MultiRotorLayer` for:**
- Any step where the input carries spatial/geometric meaning that must be preserved
- Message passing over graphs when edge features are spatial
- Equivariant transformations in the network backbone

**Minimal hybrid model:**

```python
import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.primitives.linear import CliffordLinear
from layers.primitives.rotor import RotorLayer
from layers.primitives.normalization import CliffordLayerNorm
from functional.activation import GeometricGELU

class HybridModel(nn.Module):
    """
    Hybrid model: standard nn.Linear for embedding and readout,
    Versor geometric layers for the transformation backbone.
    """
    def __init__(self, in_dim, hidden_channels, out_dim, algebra):
        super().__init__()
        self.algebra = algebra
        dim = algebra.dim  # 2^n blade dimensions

        # Standard: project raw scalar features into channel space
        # Output is treated as channel 0 of the multivector (grade 0)
        self.embed = nn.Linear(in_dim, hidden_channels)

        # Geometric backbone: channel mix → normalize → activate → rotate
        self.linear = CliffordLinear(algebra, hidden_channels, hidden_channels)
        self.norm = CliffordLayerNorm(algebra, hidden_channels)
        self.act = GeometricGELU(algebra, hidden_channels)
        self.rotor = RotorLayer(algebra, hidden_channels)

        # Standard: flatten and project to task output
        self.head = nn.Linear(hidden_channels * dim, out_dim)

    def forward(self, x_scalar, x_mv):
        # x_scalar: [B, in_dim] — raw scalar features (e.g., atom types)
        # x_mv: [B, hidden_channels, dim] — multivector geometric features

        # Embed scalars to channel dimension; add as grade-0 component
        scalar_feat = self.embed(x_scalar)  # [B, hidden_channels]
        x_mv = x_mv + scalar_feat.unsqueeze(-1) * torch.zeros_like(x_mv)

        # Geometric transformation
        x_mv = self.linear(x_mv)
        x_mv = self.norm(x_mv)
        x_mv = self.act(x_mv)
        x_mv = self.rotor(x_mv)

        # Readout
        return self.head(x_mv.flatten(1))
```

For a production example of this hybrid pattern, see `models/md17.py` (`MD17InteractionBlock`), which uses `nn.Linear` for edge projections alongside `RotorLayer` for spatial message passing.

---

## 6. When NOT to Use Rotors

Rotors are an *inductive bias*, not a universal improvement. There are clear cases where they hurt:

**Channels with no geometric relationship.** If your 128 channels are learned feature slots with no spatial interpretation, a rotor over them does not correspond to any meaningful rotation. Use `CliffordLinear(backend='traditional')` instead.

**Tasks that need arbitrary cross-channel amplification.** A rotor is an isometry — it cannot learn to scale one channel relative to another. If your task requires the network to suppress or amplify specific feature dimensions, use an unconstrained linear layer.

**Unknown or mismatched metric signature.** A rotor in $Cl(3, 0)$ on data that lives in $Cl(3, 1)$ will produce geometrically incorrect transformations. If you are unsure of the signature and `MetricSearch` is too expensive, default to standard layers until you have a hypothesis.

**Very shallow networks.** A 1–2 layer model may not benefit from the Cayley table overhead. The geometric inductive bias pays off over depth; for shallow models, a standard `nn.Linear` is usually faster and simpler.

**Rule of thumb:** start with a standard PyTorch baseline. Add `RotorLayer` where you can articulate *why* a rotation group is the right constraint for that step.

---

## 7. Setting Up Training

For models where all backbone weights are bivectors (i.e., `backend='rotor'` throughout), use `RiemannianAdam`:

```python
from optimizers.riemannian import RiemannianAdam

optimizer = RiemannianAdam(model.parameters(), lr=1e-3, algebra=algebra)
```

`RiemannianAdam` runs Adam updates in bivector space (the Lie algebra of the rotation group). The `exp(-B/2)` map provides the manifold retraction automatically — the update stays on the Spin manifold without needing a projection step. Bivector norm clipping (default `max_norm=10.0`) prevents instability in deep networks.

**Standard `torch.optim.Adam` also works** and is the right choice when your model mixes bivector and non-bivector parameters (e.g., `nn.Linear` readout heads). `RiemannianAdam` is most beneficial when the *entire* parameter space lives on the Bivector Manifold.

---

## 8. Complete Minimal Example

End-to-end: choose algebra, build a 3-layer GBN, train, evaluate. Runs in under 60 seconds on CPU with synthetic data.

```python
import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.primitives.linear import CliffordLinear
from layers.primitives.rotor import RotorLayer
from layers.primitives.normalization import CliffordLayerNorm
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam

# --- 1. Algebra ---
algebra = CliffordAlgebra(p=3, q=0, device='cpu')  # Cl(3,0): Euclidean 3D
dim = algebra.dim  # 8 = 2^3 blade components

# --- 2. Model ---
hidden = 16
out_dim = 1

class SimpleGBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = CliffordLinear(algebra, hidden, hidden)
        self.n1 = CliffordLayerNorm(algebra, hidden)
        self.a1 = GeometricGELU(algebra, hidden)
        self.r1 = RotorLayer(algebra, hidden)

        self.l2 = CliffordLinear(algebra, hidden, hidden)
        self.n2 = CliffordLayerNorm(algebra, hidden)
        self.a2 = GeometricGELU(algebra, hidden)
        self.r2 = RotorLayer(algebra, hidden)

        self.head = nn.Linear(hidden * dim, out_dim)

    def forward(self, x):
        x = self.r1(self.a1(self.n1(self.l1(x))))
        x = self.r2(self.a2(self.n2(self.l2(x))))
        return self.head(x.flatten(1))

model = SimpleGBN()

# --- 3. Synthetic data ---
B = 32  # batch size
x = torch.randn(B, hidden, dim)   # [Batch, Channels, 2^n]
y = torch.randn(B, out_dim)

# --- 4. Training loop ---
optimizer = RiemannianAdam(model.parameters(), lr=1e-3, algebra=algebra)
loss_fn = nn.MSELoss()

for step in range(200):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

print("Done.")
```

**Expected output:** loss decreasing from ~1.0 toward ~0.0 over 200 steps on synthetic random targets.

---

## 9. Where to Go Next

| If you want... | Read... |
|---|---|
| All layers with annotated code examples | `docs/innovations.md` |
| Step-by-step tutorial with each layer | `docs/tutorial.md` |
| Formal mathematical definitions | `docs/mathematical.md` |
| Task-specific configurations (MD17, SR, LQA, EEG) | `docs/tutorial.md` §6 |
| Design philosophy and motivation | `docs/philosophy.md` |
| Common errors and troubleshooting | `docs/faq.md` |
| CLAUDE.md developer reference | `CLAUDE.md` |
