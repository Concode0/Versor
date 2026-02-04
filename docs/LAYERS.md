# Geometric Neural Layers

Versor provides specialized PyTorch modules that respect the algebraic structure of Multivectors.

## 1. RotorLayer (`layers/rotor.py`)
Learns an optimal rotation (or Lorentz boost) to align data.
*   **Parameters**: Learns a Bivector $B$.
*   **Operation**: $x \mapsto e^{-B/2} x e^{B/2}$.
*   **Use Case**: Manifold alignment, pose estimation, view synthesis.

## 2. CliffordLinear (`layers/linear.py`)
A fully connected layer in the geometric domain.
*   **Parameters**: Weights $W_{ij}$ (Scalars or Multivectors).
*   **Operation**: Standard linear map over the multivector components.
*   **Use Case**: Feature transformation, dimension projection.

## 3. CliffordGraphConv (`layers/gnn.py`)
Graph Convolution for geometric signals.
*   **Input**: Node features (Multivectors), Adjacency Matrix.
*   **Operation**: Aggregates neighbor multivectors (geometric sum) and applies a `CliffordLinear` transform.
*   **Use Case**: Physics simulations, molecular graphs.

## 4. BladeSelector (`layers/projection.py`)
A soft attention mechanism for geometric grades.
*   **Parameters**: Scalar weights for each basis blade.
*   **Operation**: Element-wise multiplication.
*   **Use Case**: Filtering noise (e.g., suppressing non-vector grades), dimensionality reduction.

## 5. CliffordLayerNorm (`layers/normalization.py`)
Normalization that preserves directional information.
*   **Operation**: Normalizes magnitude $||x|| \to 1$, then scales.
*   **Use Case**: Stabilizing deep geometric networks.

## 6. Activations (`functional/activation.py`)
*   **GeometricGELU**: Scales magnitude non-linearly (`GELU(|x|)`), preserves direction.
*   **GradeSwish**: Gates specific grades based on their energy.
