# Tutorial: Building with Versor

A step-by-step guide to using Versor's geometric layers in your own models.

## 1. Create a Clifford Algebra

Everything starts with a `CliffordAlgebra` instance. The signature $(p, q)$ determines the geometry:

```python
from core.algebra import CliffordAlgebra

# 3D Euclidean (rotations in 3-space)
algebra = CliffordAlgebra(p=3, q=0, device='cpu')
print(algebra.dim)  # 8 = 2^3 basis blades

# 2D Minkowski (Lorentz boosts)
algebra_mink = CliffordAlgebra(p=1, q=1, device='cpu')
print(algebra_mink.dim)  # 4 = 2^2 basis blades
```

Key properties:
- `algebra.n` — total dimensions ($p + q$)
- `algebra.dim` — total basis blades ($2^n$)
- `algebra.num_grades` — number of grades ($n + 1$)

## 2. Understand Multivector Tensors

All data in Versor is represented as multivectors with shape `[Batch, Channels, 2^n]`.

```python
import torch

# Embed raw 3D vectors into multivectors
vectors = torch.randn(32, 3)  # [Batch, 3]
mv = algebra.embed_vector(vectors)  # [Batch, 8]
# Components: [scalar, e1, e2, e12, e3, e13, e23, e123]
# Indices:      0      1    2   3    4    5     6     7

# Add a channel dimension for neural layers
mv = mv.unsqueeze(1)  # [32, 1, 8] — 1 channel
```

Basis blade indexing uses binary representation:
- Index 0 (`000`) = scalar (grade 0)
- Index 1 (`001`) = $e_1$, Index 2 (`010`) = $e_2$, Index 4 (`100`) = $e_3$ (grade 1)
- Index 3 (`011`) = $e_{12}$, Index 5 (`101`) = $e_{13}$, Index 6 (`110`) = $e_{23}$ (grade 2)
- Index 7 (`111`) = $e_{123}$ (grade 3)

## 3. Core Algebra Operations

```python
A = torch.randn(4, 8)  # 4 multivectors
B = torch.randn(4, 8)

# Geometric product
AB = algebra.geometric_product(A, B)

# Grade projection (extract vectors only)
vectors_only = algebra.grade_projection(A, grade=1)

# Reverse (Clifford conjugate): flips sign based on grade
A_rev = algebra.reverse(A)

# Exponentiate a bivector to get a rotor
bivector = torch.zeros(1, 8)
bivector[0, 3] = 0.5  # rotation in e12 plane
R = algebra.exp(bivector)
```

### Hermitian Metrics

For measurement in mixed-signature algebras, use the Hermitian inner product:

```python
from core.metric import hermitian_inner_product, hermitian_norm, hermitian_grade_spectrum

# Works correctly in any signature — Cl(3,0), Cl(2,1), Cl(4,1), etc.
ip = hermitian_inner_product(algebra, A, B)   # Signed: <bar{A}B>_0
norm = hermitian_norm(algebra, A)             # sqrt(|<A,A>_H|)
spectrum = hermitian_grade_spectrum(algebra, A)  # Per-grade energy
```

## 4. Using Layers

### RotorLayer — Learned Geometric Rotation

```python
from layers.rotor import RotorLayer

rotor = RotorLayer(algebra, channels=4)

x = torch.randn(32, 4, 8)  # [Batch, Channels, Dim]
y = rotor(x)  # Same shape, rotated

# After training, inspect learned bivectors:
print(rotor.bivector_weights)  # [4, 3] — 4 channels, 3 bivector planes

# Prune small bivectors for sparsity
n_pruned = rotor.prune_bivectors(threshold=1e-4)
```

### MultiRotorLayer — Spectral Decomposition

```python
from layers.multi_rotor import MultiRotorLayer

multi = MultiRotorLayer(algebra, channels=4, num_rotors=8)

x = torch.randn(32, 4, 8)
y = multi(x)  # Superposition of 8 sandwich products

# Get invariant features (grade norms)
invariants = multi(x, return_invariants=True)  # [32, 4, n+1]
```

### CliffordLinear — Channel Mixing

```python
from layers.linear import CliffordLinear

# Traditional backend (dense matrix)
linear = CliffordLinear(algebra, in_channels=4, out_channels=8)

x = torch.randn(32, 4, 8)
y = linear(x)  # [32, 8, 8] — channels mixed, blades preserved

# Rotor backend (parameter-efficient)
linear_rotor = CliffordLinear(
    algebra, in_channels=16, out_channels=32,
    backend='rotor',       # Use rotor compositions
    num_rotor_pairs=4,     # Number of rotor pairs
    shuffle='fixed'        # Optional: channel shuffle
)

x = torch.randn(32, 16, 8)
y = linear_rotor(x)  # [32, 32, 8] — 60-88% fewer parameters
```

### RotorGadget — Parameter-Efficient Rotor Compositions

The `RotorGadget` layer implements the "Generalized Rotor Gadget" from [Pence et al., 2025](https://arxiv.org/abs/2507.11688). It replaces dense weight matrices with rotor sandwich products for dramatic parameter reduction.

```python
from layers.rotor_gadget import RotorGadget

# Basic usage
gadget = RotorGadget(
    algebra,
    in_channels=16,
    out_channels=32,
    num_rotor_pairs=4,      # More pairs = more expressive
    aggregation='mean'      # or 'sum', 'learned'
)

x = torch.randn(32, 16, 8)
y = gadget(x)  # [32, 32, 8]

# With bivector decomposition (more efficient)
gadget_decomp = RotorGadget(
    algebra,
    in_channels=16,
    out_channels=32,
    num_rotor_pairs=4,
    use_decomposition=True,  # Use power iteration method
    decomp_k=10              # Decomposition iterations
)

# With input shuffle (regularization)
gadget_shuffle = RotorGadget(
    algebra,
    in_channels=16,
    out_channels=32,
    num_rotor_pairs=4,
    shuffle='random'         # 'none', 'fixed', or 'random'
)
```

**Shuffle modes for regularization**:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `'none'` (default) | Sequential block assignment | Channels have meaningful order |
| `'fixed'` | Random permutation at init (consistent) | Unordered features, want deterministic routing |
| `'random'` | Different permutation each forward | Small datasets, need regularization |

All shuffle modes have **zero parameter overhead** and are fully differentiable.

**Example usage**:
```python
# No shuffle (default)
gadget_none = RotorGadget(algebra, 16, 32, num_rotor_pairs=4, shuffle='none')

# Fixed shuffle - consistent but non-sequential routing
gadget_fixed = RotorGadget(algebra, 16, 32, num_rotor_pairs=4, shuffle='fixed')
print(gadget_fixed.channel_permutation)  # e.g., tensor([3, 7, 1, 15, ...])

# Random shuffle - built-in regularization
gadget_random = RotorGadget(algebra, 16, 32, num_rotor_pairs=4, shuffle='random')
# Different permutation each forward pass
```

**Parameter comparison**:
```python
# Traditional: 16 × 32 = 512 parameters
linear_trad = CliffordLinear(algebra, 16, 32)

# Rotor: 4 pairs × 3 bivectors × 2 (left+right) = 24 parameters (95% reduction)
linear_rotor = CliffordLinear(algebra, 16, 32, backend='rotor', num_rotor_pairs=4)
```

**Performance characteristics**:
- **Memory**: Fixed shuffle adds 4 × in_channels bytes (permutation indices)
- **Compute**: Minimal overhead (one tensor indexing operation)
- **Gradient flow**: All modes preserve gradients correctly

### CliffordLayerNorm — Direction-Preserving Normalization

```python
from layers.normalization import CliffordLayerNorm

norm = CliffordLayerNorm(algebra, channels=4)

x = torch.randn(32, 4, 8)
y = norm(x)  # Normalized to unit magnitude, direction preserved
```

### GeometricGELU — Magnitude-Based Activation

```python
from functional.activation import GeometricGELU

act = GeometricGELU(algebra, channels=4)

x = torch.randn(32, 4, 8)
y = act(x)  # Magnitude scaled by GELU, direction preserved
```

### BladeSelector — Grade Attention

```python
from layers.projection import BladeSelector

selector = BladeSelector(algebra, channels=1)

x = torch.randn(32, 1, 8)
y = selector(x)  # Soft per-blade gate (learned)
```

## 5. Composing a Model

```python
import torch.nn as nn
from layers.rotor import RotorLayer
from layers.linear import CliffordLinear
from layers.normalization import CliffordLayerNorm
from functional.activation import GeometricGELU

class MyGBN(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.net = nn.Sequential(
            CliffordLinear(algebra, 1, 4),
            CliffordLayerNorm(algebra, channels=4),
            GeometricGELU(algebra, channels=4),
            RotorLayer(algebra, channels=4),
            CliffordLinear(algebra, 4, 1),
        )

    def forward(self, x):
        return self.net(x)

algebra = CliffordAlgebra(p=3, q=0, device='cpu')
model = MyGBN(algebra)
x = torch.randn(32, 1, 8)
y = model(x)  # [32, 1, 8]
```

## 6. Riemannian Optimization (Default)

Versor uses **true manifold optimization** on the Spin group Spin(n) **by default**. Since bivector parameters live in the Lie algebra so(n), updates respect the curved geometry of rotations.

### Why Riemannian Optimization?

**Problem with Euclidean optimization**: Standard gradient descent treats parameter space as flat Euclidean space:
```
θ_new = θ_old - η∇θ  (ignores manifold curvature)
```

**Solution (Versor's default)**: Riemannian optimizers perform geodesic updates on the Spin(n) manifold:
- Bivector parameters B live in the Lie algebra so(n) (tangent space at identity)
- Forward pass applies exponential map: R = exp(-B/2)
- This completes the Riemannian update: updating B in tangent space + exp in forward pass = geodesic motion

### Available Optimizers

#### ExponentialSGD
SGD with momentum in the Lie algebra:

```python
from optimizers.riemannian import ExponentialSGD

optimizer = ExponentialSGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    algebra=algebra,
    max_bivector_norm=10.0  # Numerical stability
)
```

#### RiemannianAdam
Adam with momentum accumulation in the Lie algebra:

```python
from optimizers.riemannian import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    algebra=algebra,
    max_bivector_norm=10.0
)
```

### Configuration

Riemannian optimization is enabled by **default** for all tasks. The config is:

```yaml
training:
  optimizer_type: "riemannian_adam"  # DEFAULT: Riemannian optimization
                                     # Options: "riemannian_adam", "exponential_sgd", "adamw" (ablation only)
  lr: 0.001
  max_bivector_norm: 10.0  # Prevents exp() overflow
  momentum: 0.9  # For SGD variants
  betas: [0.9, 0.999]  # For Adam variants
```

Or use a preset:

```bash
# Default: RiemannianAdam
uv run main.py task=qm9

# Use Exponential SGD
uv run main.py task=qm9 optimizer=exponential_sgd

# For ablation experiments with Euclidean optimization:
uv run main.py task=qm9 training.optimizer_type=adamw training.lr=0.01
```

### When to Use Riemannian vs Euclidean

**Riemannian Optimizers (Default, Recommended)**:
- ✅ Training rotor-heavy architectures (RotorLayer, MultiRotorLayer, RotorGadget)
- ✅ Large rotations expected (bivector norms >> 0.1)
- ✅ Theoretical correctness (true geometric optimization on Spin(n))
- ✅ Numerical stability with bivector norm clipping
- ✅ Pure end-to-end geometric operations

**Euclidean Optimizers (AdamW) - For Ablation Experiments Only**:
- ⚠️ Baseline comparison / ablation studies
- ⚠️ Reproducing legacy results
- ⚠️ Model has predominantly scalar parameters (non-rotor layers)
- ⚠️ Note: Treats Spin(n) as flat space, theoretically incorrect for rotor parameters

### Hyperparameter Tuning

Riemannian optimizers often need **lower learning rates** than Euclidean:
- Euclidean AdamW: `lr = 0.001` → Riemannian Adam: `lr = 0.0005`
- Euclidean SGD: `lr = 0.01` → Exponential SGD: `lr = 0.005`

The `max_bivector_norm` parameter (default: 10.0) clips bivector norms after each update to prevent numerical overflow in exp(). Adjust if needed:
- Too low (< 5.0): May limit expressiveness, slow convergence
- Too high (> 20.0): Risk of NaN/Inf in exp() computation
- Sweet spot: 5.0 - 15.0

### Mathematical Background

The Spin group Spin(n) is a curved manifold (not flat ℝⁿ):
- **Lie algebra so(n)**: Tangent space at identity, isomorphic to bivector space
- **Exponential map**: exp: so(n) → Spin(n) takes bivectors to rotors
- **Riemannian gradient**: Already in tangent space since parameters are bivectors

Our layers parameterize: B ∈ so(n) → R = exp(-B/2) ∈ Spin(n)

So updating B via gradient descent, then computing R, is equivalent to a first-order Riemannian update on the manifold.

## 7. Creating a Task

All tasks inherit from `BaseTask` and implement 7 methods:

```python
from tasks.base import BaseTask
from core.algebra import CliffordAlgebra
from functional.loss import GeometricMSELoss

class MyTask(BaseTask):
    def setup_algebra(self):
        return CliffordAlgebra(p=3, q=0, device=self.device)

    def setup_model(self):
        return MyGBN(self.algebra)

    def setup_criterion(self):
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        # Return a DataLoader or raw tensor
        return torch.randn(100, 1, self.algebra.dim).to(self.device)

    def train_step(self, data):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, data)  # autoencoder
        loss.backward()
        self.optimizer.step()
        return loss.item(), {"Loss": loss.item()}

    def evaluate(self, data):
        output = self.model(data)
        loss = self.criterion(output, data)
        print(f"Eval loss: {loss.item():.4f}")

    def visualize(self, data):
        pass  # Optional
```

Then register it in `main.py`:

```python
# In main.py task_map:
'mytask': MyTask,
```

Create a config `conf/task/mytask.yaml`:

```yaml
# @package _global_
name: "mytask"
algebra:
  p: 3
  q: 0
  device: "cpu"
training:
  epochs: 100
  lr: 0.001
  batch_size: 32
  seed: 42
```

Run it:

```bash
uv run main.py task=mytask training.epochs=200
```

## 8. Losses

```python
from functional.loss import (
    GeometricMSELoss,    # Standard MSE on multivector coefficients
    SubspaceLoss,        # Penalizes energy outside target blades
    IsometryLoss,        # Enforces norm preservation
    BivectorRegularization,  # Forces outputs to be pure bivectors
    HermitianGradeRegularization,  # Regularizes grade energy distribution
    ChamferDistance,     # Symmetric point cloud distance
    ConservativeLoss,    # Enforces F = -grad(E) for molecular dynamics
    PhysicsInformedLoss, # MSE + conservation penalty for weather forecasting
)

# SubspaceLoss: keep only vector components
vec_indices = [1, 2, 4]  # e1, e2, e3 in Cl(3,0)
loss_fn = SubspaceLoss(algebra, target_indices=vec_indices)

# IsometryLoss: input and output should have same norm
iso_loss = IsometryLoss(algebra)
loss = iso_loss(output, input)

# HermitianGradeRegularization: target grade energy distribution
grade_reg = HermitianGradeRegularization(
    algebra,
    target_spectrum=[0.4, 0.4, 0.15, 0.05]  # Scalar+vector dominant
)
loss = grade_reg(multivector_features)
```

## 9. Automatic Metric Search

Don't know the right $(p, q)$? Let Versor find it:

```python
from core.search import MetricSearch

data = torch.randn(100, 6)  # 6D data
searcher = MetricSearch(device='cpu')
best_p, best_q = searcher.search(data)
print(f"Optimal signature: Cl({best_p}, {best_q})")
```

This brute-forces all $(p, q)$ with $p + q = D$ and picks the signature that minimizes a geometric stress metric.

## 10. New Tasks

### MD17 — Molecular Dynamics

Predicts energy and forces for molecular configurations using $Cl(3,0)$. Multi-task learning with conservative force constraint ($F = -\nabla E$) and Hermitian grade regularization.

```bash
uv run main.py task=md17 dataset.molecule=aspirin
bash scripts/download_md17.sh  # Download real data
```

### PDBbind — Protein-Ligand Binding Affinity

Dual-graph encoding with GeometricCrossAttention for binding affinity prediction ($pK_d$) using $Cl(3,0)$.

```bash
uv run main.py task=pdbbind
bash scripts/download_pdbbind.sh  # Download instructions
```

### WeatherBench — Global Weather Forecasting

Spacetime forecasting with TemporalRotorLayer using $Cl(2,1)$ for causal temporal structure. PhysicsInformedLoss enforces energy conservation.

```bash
uv run main.py task=weatherbench
bash scripts/download_weatherbench.sh  # Download ERA5 data
```

### ABC — CAD Point Cloud Reconstruction

Conformal geometric algebra autoencoder using $Cl(4,1)$ for unified primitive representation. Embeds points into CGA null cone.

```bash
uv run main.py task=abc
bash scripts/download_abc.sh  # Download instructions
```

All tasks use RiemannianAdam, bivector decomposition, RotorGadget backend, and Hermitian grade regularization by default. Synthetic data generators are included for development without external data.
