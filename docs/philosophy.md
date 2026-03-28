# Philosophy: Why Geometric Algebra?

## The Linear Algebra Ceiling

Standard deep learning has achieved extraordinary results, but it operates under an invisible constraint. Every layer applies a weight matrix $W$ to a vector $x$:

$$x' = Wx + b$$

This operation is unrestricted. A general matrix $W$ can stretch, shear, reflect, and rotate all at once. It has no awareness of the geometry of your data. When your data lives on a manifold (and it almost always does), matrix multiplication can inadvertently deform that manifold — and the network must spend additional capacity to compensate.

The result: parameters that could be learning structure are instead compensating for geometric side effects introduced by unconstrained linear maps.

This concern applies specifically to layers that process *geometrically structured* data — spatial coordinates, orientations, phase relationships — where the transformation is expected to respect a known group action. For layers that mix channels without a geometric interpretation (readout projections, attention gates, scalar aggregation), an unconstrained linear map remains appropriate. Versor's design premise is that the two regimes coexist in the same model: rotor operations for geometric transformation, standard linear algebra for the rest.

## The Unbending Paradigm

Versor takes a different approach. Instead of general linear transformations, we use **Rotors** — elements of Clifford Algebra that perform pure geometric rotations:

$$x' = R x \tilde{R}$$

where $R = \exp(-B/2)$ is generated from a bivector $B$ (a rotation plane).

This sandwich product is an **isometry**: it preserves lengths, angles, and the origin. It aligns the manifold purely through rotation — no stretching, no shearing, no distortion.

We call this **unbending**: the network learns the minimal rotation that maps the input manifold to the output manifold, dedicating all capacity to geometric structure rather than compensating for non-geometric side effects.

In code, this is the `RotorLayer` forward pass (`layers/primitives/rotor.py`):

```python
def _compute_rotors(self, device, dtype):
    B = torch.zeros(self.channels, self.algebra.dim, device=device, dtype=dtype)
    indices = self.bivector_indices.unsqueeze(0).expand(self.channels, -1)
    B.scatter_(1, indices, self.bivector_weights)
    R = self.algebra.exp(-0.5 * B)       # Bivector -> Rotor
    R_rev = self.algebra.reverse(R)       # Clifford conjugate
    return R, R_rev

def forward(self, x):
    R, R_rev = self._compute_rotors(x.device, x.dtype)
    Rx = self.algebra.geometric_product(R.unsqueeze(0), x)
    return self.algebra.geometric_product(Rx, R_rev.unsqueeze(0))
```

Every learnable parameter is a bivector component — a specific rotation plane. The `exp(-B/2)` maps it to the Spin group; the sandwich product applies the isometry.

Note that `RotorLayer` handles only the geometric rotation step. Channel mixing — redistributing features across the channel dimension — is done by `CliffordLinear`, which by default uses a standard scalar weight matrix (`torch.einsum('oi,bid->bod', self.weight, x)`). Both layers are necessary in a GBN stack; a rotor alone cannot change the number of feature channels or learn arbitrary cross-channel relationships.

### Rotors, Reflections, and the Versor Group

A **versor** is any product of invertible vectors in a Clifford algebra. Versors fall into two cosets by grade parity:

- **Even versors (rotors)** — products of an even number of unit vectors. They belong to the Spin group: $R\tilde{R} = 1$, $R = \exp(-B/2)$. A rotor performs a pure rotation with no parity change. This is the primary building block in Versor, because rotations cover the majority of geometric tasks.
- **Odd versors (reflections)** — products of an odd number of unit vectors. They belong to the Pin group and apply a hyperplane reflection: $x' = -nxn^{-1}$. Two reflections compose to a rotation. `ReflectionLayer` (`layers/primitives/reflection.py`) implements this as a learnable layer.

The framework is named **Versor** because it supports both cosets. In practice, most GBN architectures use only `RotorLayer`; `ReflectionLayer` is available for tasks with explicit reflection symmetry or for constructing Pin group actions by composition.

## Why This Matters

### Geometric Inductive Bias
Rotors encode the correct inductive bias for spatial data. A rotation in 3D is parameterized by 3 bivector components ($e_{12}, e_{13}, e_{23}$), not 9 matrix entries. The model **cannot** learn a shear because the parameterization doesn't allow it.

### Metric Awareness
Clifford Algebra naturally handles different geometries through the metric signature $(p, q)$:

- **Euclidean** $Cl(3, 0)$: Standard 3D rotations (molecules, point clouds)
- **Minkowski** $Cl(1, 1)$: Lorentz boosts (spacetime, hyperbolic geometry)
- **Conformal** $Cl(4, 1)$: Translations become rotations (computer vision)

The same `RotorLayer` code works across all of these — the algebra handles the sign conventions automatically.

### Composability
Rotors compose by multiplication: $R_{total} = R_2 R_1$. This means stacking RotorLayers is equivalent to learning a single, more complex rotation. No information is lost between layers; there is no "activation bottleneck."

### Multi-Rotor Decomposition
A single rotor rotates in one plane. The `MultiRotorLayer` learns $K$ overlapping rotors and mixes them:

$$x' = \sum_k w_k R_k x \tilde{R_k}$$

This is analogous to a Fourier decomposition — complex transformations are expressed as a superposition of simple rotations, each parameterized by $O(n^2)$ bivector components instead of $O(2^n)$ general multivector entries.

## The GBN Architecture

The **Geometric Blade Network (GBN)** is a sequential architecture built from:

1. **CliffordLinear** — channel mixing (scalar weights, no geometric product)
2. **RotorLayer** / **MultiRotorLayer** — geometric rotation (sandwich product)
3. **CliffordLayerNorm** — magnitude normalization (preserves direction)
4. **GeometricGELU** — magnitude-based non-linearity (preserves direction)
5. **BladeSelector** — soft attention over basis blades (grade filtering)

Each component respects the multivector structure. The model learns **what to rotate** (CliffordLinear), **how to rotate** (RotorLayer), and **what to keep** (BladeSelector).

The key non-linearity, `GeometricGELU`, is five lines (`functional/activation.py`):

```python
def forward(self, x):
    norm = x.norm(dim=-1, keepdim=True)
    scale = F.gelu(norm + self.bias.view(1, -1, 1)) / (norm + 1e-6)
    return x * scale
```

It scales the magnitude via GELU while preserving the direction `x / ||x||`. Standard activations (ReLU, GELU) applied coefficient-wise destroy this directional information.

## CliffordLinear: Two Backends, Two Philosophies

`CliffordLinear` is the channel-mixing layer of the GBN stack. It now supports two backends that embody fundamentally different design philosophies.

### Traditional Backend (default)

```python
linear = CliffordLinear(algebra, in_channels=8, out_channels=16)
# backend='traditional' — O(in × out) scalar weights
```

A standard learned weight matrix $W \in \mathbb{R}^{C_{out} \times C_{in}}$ applied across channels. Each output channel is an arbitrary linear combination of all input channels. This is maximally expressive: the network can learn any channel-to-channel mapping, including ones with no geometric meaning. The cost is that these weights can introduce unconstrained stretching and scaling — the same geometric side effects that Rotors are designed to avoid. Use this when full cross-channel expressivity matters more than geometric purity, or when the task does not have strong manifold structure in the channel dimension.

### Rotor Backend (RotorGadget)

```python
linear = CliffordLinear(algebra, in_channels=8, out_channels=16, backend='rotor')
# O(num_rotor_pairs × n(n-1)/2) bivector parameters — ~63% reduction
```

Replaces the weight matrix with a **RotorGadget** (`layers/primitives/rotor_gadget.py`): an asymmetric rotor sandwich

$$\psi(x) = r \cdot x \cdot s^\dagger$$

where $r$ and $s$ are independently trained rotors (each parameterized by bivector coefficients alone). Unlike the symmetric sandwich $R x \tilde{R}$ of a `RotorLayer` — which is a pure isometry — the asymmetric product can mix grades and change channel dimensions, making it a geometrically constrained replacement for a linear map.

Channel routing is block-diagonal: each of the $K$ rotor pairs operates on a disjoint partition of the input channels, then aggregated (mean, sum, or learned weights). An optional channel shuffle (`shuffle='fixed'` or `'random'`) breaks the block structure when global interaction is needed.

**When to prefer the rotor backend:** tasks with strong geometric structure in the channel dimension, when parameter efficiency matters (~63% fewer parameters), or when you want every learnable weight to live on the Bivector Manifold — so that Riemannian Adam's Lie algebra updates apply uniformly to the entire network.

**When to keep the traditional backend:** tasks where channels do not carry geometric meaning, where arbitrary cross-channel mixing is essential (e.g., learned attention projections), or where the added inductive bias of rotation-constrained mixing actively hurts convergence.

The two backends are drop-in replacements. The right choice depends on the geometry of the problem, not on a blanket preference for one paradigm over the other.

## Towards the Unified Geometric Engine

### The Fragmentation of Matrices
The ubiquity of matrix operations has led to a fragmentation of our mathematical intuition. In the standard paradigm, a rotation, a projection, and a complex number are treated as disparate mathematical objects, each requiring its own unique data structure and rules. We represent rotations with $3 \times 3$ orthogonal matrices, translations with vectors, and logical states with bits. This fragmentation means models must re-learn the relationships between these concepts from scratch, spending compute and data to rediscover fundamental geometric truths that could be axiomatic.

Matrices are general-purpose containers that carry no geometric semantics by default. The remarkable achievements of modern deep learning were accomplished despite this — imagine what becomes possible when the representation itself encodes geometric meaning.

### The Bridge to True Geometric Intelligence
The long-term goal is a unified geometric computing engine where geometric primitives — vectors, bivectors, rotors — are first-class citizens of the model's representation. Current implementations demonstrate that this approach is feasible and competitive; the theoretical machinery is in place for further extension.

### Beyond the GPU Transition
Our current reliance on matrix operations reflects the architecture of modern GPUs, which are optimized for dense linear algebra. This is an implementation reality, not a fundamental limit.

The true essence of Geometric Algebra lies deeper, offering efficiency at the level of bit manipulation. In a native GA hardware architecture, geometric products would be atomic operations, and the sparsity and symmetry of the algebra would be exploited not just for mathematical elegance, but for computational speed beyond current matrix multiplication. We are building the software paradigm for this future today, preparing for the shift from dense linear algebra to elegant, bit-efficient geometric computation.

### From Rotors to Formulas

The ultimate proof of interpretability: trained bivector weights can be read back as symbolic formulas. Each rotation plane maps to a closed-form trigonometric or hyperbolic term (`models/sr/translator.py`):

```python
def _plane_to_action(self, plane: SimplePlane) -> sympy.Expr:
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

A standard neural network is a black box of millions of uninterpretable scalars. A trained GBN is a composition of named rotation planes with exact symbolic correspondences. This is not post-hoc explanation — it is direct readout from the algebra.

---

## Author's Vision

*The following is the author's personal perspective on where this work is headed — speculative, not descriptive of the current implementation.*

Versor is not just a library; it is a bridge to a Unified Geometric Engine. Our current models are the first steps toward an architecture that does not merely process numbers but thinks geometrically. By treating geometric primitives — vectors, bivectors, rotors — as first-class citizens, we move from manipulating arrays to manipulating concepts.

In this future, a "rotation" is not a function learned by a matrix but a fundamental operation of the model's mind, as natural as addition is to a calculator. The remarkable achievements of modern deep learning were accomplished despite the geometric blindness of unconstrained matrices — imagine what becomes possible when the representation itself encodes geometric meaning.