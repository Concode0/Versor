# Versor: A PyTorch Framework for Geometric Algebra Deep Learning

> **"Standard Deep Learning warps the manifold. Versor unbends it."**

**Versor** is a PyTorch framework for **Geometric Algebra Deep Learning**. It provides the building blocks for the **Geometric Blade Network (GBN)** and **Multi-Rotor GBN** — model architectures that replace distorted linear transformations with pure, manifold-aligned geometric rotations using Clifford Algebra and Rotors.

## Core Idea

Rotors ($R = \exp(-B/2)$) perform pure geometric rotations via the sandwich product ($x \to RxR\tilde{}$), preserving manifold structure instead of distorting it like standard weight matrices.

## Key Features

*   **Metric-Agnostic Kernel**: Supports Euclidean $Cl(p, 0)$, Minkowski/Hyperbolic $Cl(p, q)$, and Projective algebras out of the box.
*   **Geometric Layers**: `RotorLayer`, `MultiRotorLayer`, `CliffordLinear`, `CliffordGraphConv`, `CliffordLayerNorm`.
*   **Model Architectures**: `GeometricBladeNetwork`, `MultiRotorModel`, `MoleculeGNN`, `MotionManifoldNetwork`.
*   **Novel Activations**: `GeometricGELU` (magnitude-based), `GradeSwish` (per-grade gating).
*   **Automatic Metric Search**: Finds optimal $(p, q)$ signature based on data topology.
*   **Geometric Sparsity**: `prune_bivectors` for compression of geometric layers.

## Installation

Versor requires Python 3.9+ and PyTorch.

```bash
# Clone the repository
git clone https://github.com/Concode0/Versor.git
cd Versor

# Install core dependencies
uv sync

# Install with optional dependency groups
uv sync --extra viz          # matplotlib, seaborn, scikit-learn, plotly, imageio
uv sync --extra examples     # transformers, pillow, scikit-learn, matplotlib
uv sync --extra graph        # torch-geometric (for molecular GNN tasks)
uv sync --extra demo         # streamlit, plotly
uv sync --extra all          # everything
```

## Quick Start

### Using Versor Layers in Your Own Model

```python
import torch
from core.algebra import CliffordAlgebra
from layers.rotor import RotorLayer
from layers.linear import CliffordLinear
from functional.activation import GeometricGELU

# Create a 3D Euclidean Clifford Algebra
algebra = CliffordAlgebra(p=3, q=0)

# Build a model with geometric layers
rotor = RotorLayer(algebra, channels=4)
linear = CliffordLinear(algebra, in_channels=4, out_channels=8)
activation = GeometricGELU(algebra, channels=8)

# Input: [Batch, Channels, 2^n] multivectors
x = torch.randn(32, 4, algebra.dim)
out = activation(linear(rotor(x)))
```

### Running Tasks via CLI

Versor uses **Hydra** for configuration management:

```bash
# Run a real-data task
uv run main.py task=qm9 training.epochs=100
uv run main.py task=motion training.epochs=100

# Override parameters
uv run main.py task=qm9 algebra.device=cuda training.lr=0.001
```

### Interactive Demo (Streamlit)

![DEMO](demo_manifold_comp.gif)

```bash
streamlit run examples/demo.py
```

## Benchmarks

### QM9 (Molecular Property Prediction)
**Task**: Predict the internal energy ($U_0$) of small molecules using the Multi-Rotor (Geometric FFT) architecture.

| Metric | Value |
|--------|-------|
| **Algebra** | $Cl(3, 0)$ (3D Euclidean) |
| **Network** | MultiRotorQuantumNet |
| **Num Rotors** | 12 |
| **Validation MAE** | **7.3505** |
| **Avg Inference Time** | **7.0942 ms / molecule** |

![QM9 predictions](assets/multi_rotor_qm9_prediction.png)

```bash
uv run main.py task=multi_rotor_qm9 training.epochs=100
```

### Motion Alignment (UCI-HAR)
**Task**: Align high-dimensional motion data into a linearly separable latent space using geometric rotation.

| Metric | Value |
|--------|-------|
| **Algebra** | $Cl(4, 0)$ (Optimized via MetricSearch) |
| **Network** | MotionManifoldNetwork (Rotor Alignment) |
| **Latent Accuracy** | **~100%** |
| **Latent Grade Purity** | 0.9957 |

![Motion latent space](assets/motion_latent_space.png)

```bash
uv run main.py task=motion training.epochs=100
```

## Examples (Synthetic/Demo Tasks)

Synthetic experiments demonstrating GA concepts are in the `examples/` directory:

```bash
# Run synthetic tasks
uv run python -m examples.main task=manifold training.epochs=500
uv run python -m examples.main task=crossmodal training.epochs=200
uv run python -m examples.main task=hyperbolic training.epochs=500
uv run python -m examples.main task=semantic training.epochs=200
uv run python -m examples.main task=sanity
```

| Example | Algebra | Description |
|---------|---------|-------------|
| **Manifold** | $Cl(3,0)$ | Flatten a figure-8 manifold (100% topology restoration) |
| **Cross-Modal** | $Cl(6,0)$ | Align BERT + synthetic modalities (98% alignment improvement) |
| **Hyperbolic** | $Cl(1,1)$ | Reverse a Lorentz boost in Minkowski spacetime |
| **Semantic** | $Cl(6,0)$ | Disentangle BERT embeddings by grade (88.7% purity) |
| **Sanity** | $Cl(3,0)$ | Verify algebra correctness (identity learning) |

## Configuration

Configuration files are in `conf/` (real tasks) and `examples/conf/` (synthetic tasks).

```bash
# Override any parameter from CLI
uv run main.py task=qm9 algebra.p=4 training.lr=0.001
```

## Project Structure

```
Versor/
├── core/               # Math kernel (CliffordAlgebra, metric, visualizer)
├── layers/             # Neural layers (Rotor, MultiRotor, Linear, GNN, Norm)
├── functional/         # Activations (GeometricGELU, GradeSwish) & losses
├── models/             # Model architectures (GBN, MultiRotor, Molecule, Motion)
├── tasks/              # Real-data task runners (QM9, Motion)
├── datasets/           # Real data loaders (QM9, HAR, ModelNet)
├── conf/               # Hydra configs for real tasks
├── examples/           # Synthetic demos and interactive Streamlit app
│   ├── tasks/          # Manifold, CrossModal, Hyperbolic, Semantic, Sanity
│   ├── datasets/       # Synthetic data generators
│   └── conf/           # Hydra configs for example tasks
├── tests/              # Unit & property tests
└── main.py             # CLI entry point
```

## Roadmap

- [ ] Native CUDA Kernels for $Cl(3,0)$ and $Cl(1,3)$
- [ ] JIT Compilation with metric-aware operation graph optimization
- [ ] Geometric Transformer (GAT): Fully geometric attention mechanism
- [ ] Multi-head & Dynamic Rotors: Input-dependent rotation axes
- [x] **Automatic Metric Search**: Self-optimizing signature ($p, q$)
- [x] **Automatic Bivector Pruning**: Geometric sparsity-driven compression
- [x] **CliffordGraphConv**: Geometric signal processing on graphs (QM9)
- [x] **Multi-Rotor GBN**: Spectral decomposition with overlapping rotors

## Documentation

*   [**Mathematics**](docs/MATHEMATICS.md): Primer on Clifford Algebra, Rotors, and Metric Signatures.
*   [**Layers**](docs/LAYERS.md): Guide to `RotorLayer`, `CliffordLinear`, and `CliffordGraphConv`.
*   [**Tasks**](docs/TASKS.md): Explanation of task architectures.
*   [**API Reference**](docs/API.md): Overview of core classes and functions.

## License & Intellectual Property

This project is licensed under the **Apache License 2.0**.

**Notice on Patents**:
The core GBN architecture is covered by **KR Patent Application 10-2026-0023023**.
By releasing this under Apache 2.0, we provide a **perpetual, royalty-free patent license** to any individual or entity using this software.

## Citation

```bibtex
@software{kim2026versor,
  author = {Kim, Eunkyum},
  title = {Versor: Universal Geometric Algebra Neural Network},
  url = {https://github.com/Concode0/versor},
  year = {2026},
  month = {2},
  note = {ROK Patent Application 10-2026-0023023}
}
```

## ⚠️ Technical Note: Current State of CliffordLinear
While Versor achieves SOTA-level efficiency, the current implementation of CliffordLinear still utilizes standard scalar weight matrices for channel mixing. We identify this as a lingering vestige of the "Linear Algebra Cage." > Current Limitation: These linear mappings can introduce unconstrained scaling and manifold warping, which slightly deviates from the pure isometric unbending provided by our RotorLayers.

Active Development: We are currently transitioning to a Pure Geometric Update paradigm. This involves:

Replacing matrix-based mixing with a Composition of Irreducible Rotors.

Moving all weight updates from Euclidean space to the Bivector Manifold (Lie Algebra).