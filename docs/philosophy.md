# Philosophy: Why Geometric Algebra?

## The Linear Algebra Cage

Standard deep learning operates inside what we call the **Linear Algebra Cage**. Every layer applies a weight matrix $W$ to a vector $x$:

$$x' = Wx + b$$

This operation is unrestricted. A general matrix $W$ can stretch, shear, reflect, and rotate all at once. It doesn't know or care about the geometry of your data. When your data lives on a manifold (and it almost always does), matrix multiplication **warps** that manifold — introducing distortions that the network must spend capacity to undo.

The result: millions of parameters fighting against each other, some warping the space, others trying to un-warp it.

## The Unbending Paradigm

Versor takes a different approach. Instead of general linear transformations, we use **Rotors** — elements of Clifford Algebra that perform pure geometric rotations:

$$x' = RxR\tilde{}$$

where $R = \exp(-B/2)$ is generated from a bivector $B$ (a rotation plane).

This sandwich product is an **isometry**: it preserves lengths, angles, and the origin. It cannot warp the manifold. It can only *rotate* it — aligning it with the target structure without distortion.

We call this **unbending**: the network learns the minimal rotation that maps the input manifold to the output manifold, with zero wasted capacity on non-geometric transformations.

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

$$x' = \sum_k w_k R_k x R_k\tilde{}$$

This is analogous to a Fourier decomposition — complex transformations are expressed as a superposition of simple rotations, each parameterized by $O(n^2)$ bivector components instead of $O(2^n)$ general multivector entries.

## The GBN Architecture

The **Geometric Blade Network (GBN)** is a sequential architecture built from:

1. **CliffordLinear** — channel mixing (scalar weights, no geometric product)
2. **RotorLayer** / **MultiRotorLayer** — geometric rotation (sandwich product)
3. **CliffordLayerNorm** — magnitude normalization (preserves direction)
4. **GeometricGELU** — magnitude-based non-linearity (preserves direction)
5. **BladeSelector** — soft attention over basis blades (grade filtering)

Each component respects the multivector structure. The model learns **what to rotate** (CliffordLinear), **how to rotate** (RotorLayer), and **what to keep** (BladeSelector).

## The Remaining Challenge: CliffordLinear

While RotorLayers provide pure isometric transformations, the current `CliffordLinear` layer still uses standard scalar weight matrices for channel mixing. This is the last vestige of the Linear Algebra Cage — these weights can introduce unconstrained scaling.

Active development is replacing CliffordLinear with compositions of irreducible rotors, moving all weight updates to the Bivector Manifold (Lie Algebra). The goal: a fully geometric network where every parameter has a clear geometric meaning.
