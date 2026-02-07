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

linear = CliffordLinear(algebra, in_channels=4, out_channels=8)

x = torch.randn(32, 4, 8)
y = linear(x)  # [32, 8, 8] — channels mixed, blades preserved
```

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

## 6. Creating a Task

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

## 7. Losses

```python
from functional.loss import (
    GeometricMSELoss,    # Standard MSE on multivector coefficients
    SubspaceLoss,        # Penalizes energy outside target blades
    IsometryLoss,        # Enforces norm preservation
    BivectorRegularization,  # Forces outputs to be pure bivectors
)

# SubspaceLoss: keep only vector components
vec_indices = [1, 2, 4]  # e1, e2, e3 in Cl(3,0)
loss_fn = SubspaceLoss(algebra, target_indices=vec_indices)

# IsometryLoss: input and output should have same norm
iso_loss = IsometryLoss(algebra)
loss = iso_loss(output, input)
```

## 8. Automatic Metric Search

Don't know the right $(p, q)$? Let Versor find it:

```python
from core.search import MetricSearch

data = torch.randn(100, 6)  # 6D data
searcher = MetricSearch(device='cpu')
best_p, best_q = searcher.search(data)
print(f"Optimal signature: Cl({best_p}, {best_q})")
```

This brute-forces all $(p, q)$ with $p + q = D$ and picks the signature that minimizes a geometric stress metric.
