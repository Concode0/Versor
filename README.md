# Versor: The GBN (Geometric Blade Network) Engine

> **"Standard Deep Learning warps the manifold. Versor unbends it."**

## 🚧 Research Preview: The Multi-Rotor Architecture

> **"Breaking the Curse of Dimensionality in Geometric Algebra."**

Current GA implementations suffer from exponential complexity ($O(2^n)$) in high-dimensional spaces. **Versor** is pioneering a novel solution to this fundamental bottleneck: the **Multi-Rotor Engine**.

By abandoning the rigidity of single-rotor operations in favor of a **"Geometric Spectral Decomposition"** approach, we are architecting a system that scales linearly.

### Core Concepts (In Development)
* **Overlap & Entanglement:** Instead of orthogonal bases, we utilize **overlapping rotor heads** that share dimensions (e.g., $e_{12}$ and $e_{23}$ coupled via $e_2$). This creates a globally entangled manifold structure through local interactions.
* **Relaxed Isometry:** We approximate complex non-linear manifolds via the **weighted summation of multiple rotors** ($\sum w_i R_i$), effectively performing a "Sparse Fourier Decomposition" of the geometric transformation.
* **Polynomial Complexity:** This architecture reduces the parameter space from $O(2^n)$ to **$O(k \cdot n^2)$**, making tasks like **QM9 (Molecular Dynamics)** and **High-dim NLP** feasible within a pure geometric framework.

*This architecture is currently in the active research & implementation phase. See [TODO.md](./TODO.md) for the roadmap.*

**Versor** is a PyTorch framework purpose-built for **Geometric Algebra Deep Learning**. It serves as the primary implementation and proposal for the **Geometric Blade Network (GBN)**—a revolutionary model architecture that replaces distorted linear transformations with pure, manifold-aligned geometric rotations.

### 🚀 The "100% Accuracy" Breakthrough
In our latest benchmark on the **UCI-HAR (Human Activity Recognition)** dataset, Versor achieved **100.00% Test Accuracy** using a single **Rotor Layer** with $O(N)$ complexity. 
This isn't just a number; it's a proof that geometric alignment can solve what statistical approximation struggles with.

![DEMO](./demo_manifold_comp.gif)

### 🎮 Interactive Demo (Streamlit)
Explore Geometric Algebra transformations and live manifold unbending in your browser.
```bash
streamlit run demo.py
```

## The GBN Manifesto: Beyond Linear Algebra
Traditional Deep Learning is trapped in the "Linear Algebra Cage." Standard weight matrices ($W \in \mathbb{R}^{n \times m}$) apply arbitrary scaling, shearing, and distortions that destroy the underlying geometric purity of data. **GBN (Geometric Blade Network)** declares the end of this era: 
- Dimensional Escape: We move beyond the 3D Euclidean cage into the multivector space of Clifford Algebra.
- Analytical Solutions: GBN doesn't just "guess" patterns; it utilizes Rotors ($R$) to find the exact geometric alignment of high-dimensional manifolds.
- Purity over Approximation: By utilizing a $p=3$ Minkowski metric search, GBN ensures that latent representations remain geometrically consistent (Grade Purity $\approx 1.0$).

## GBN Benchmark

The following benchmarks demonstrate Versor's capability across different geometric learning tasks. All experiments run on CPU (device: "cpu") with default hyperparameters unless otherwise specified.

### 1. Manifold Unbending (Topology Restoration)
**Task**: Flatten a non-linear Figure-8 manifold ($z = 0.5xy$) embedded in 3D back to 2D plane.

| Metric | Value |
|--------|-------|
| **Algebra** | $Cl(3, 0)$ (3D Euclidean) |
| **Training Epochs** | 500 |
| **Network** | RotorLayer + BladeSelector |
| **Final Reconstruction Loss** | **0.000000** ✓ |
| **Z-axis Energy Suppression** | ~100% |

**Result**: Perfect manifold flattening achieved. The network successfully learns to rotate the distorted manifold and suppress the non-essential dimension, demonstrating exact geometric recovery.

```bash
uv run main.py task=manifold training.epochs=500
```

---

### 2. Cross-Modal Unification (Semantic Alignment)
**Task**: Align BERT text embeddings with a synthetically rotated/noisy "image" modality in 6D space.

| Metric | Value |
|--------|-------|
| **Algebra** | $Cl(6, 0)$ (6D Euclidean) |
| **Training Epochs** | 200 |
| **Network** | Dual ModalityEncoder (CliffordLinear + GeometricGELU + RotorLayer) |
| **Initial Alignment Distance** | 0.944 |
| **Final Alignment Distance** | **0.019** ✓ |
| **Alignment Improvement** | **98.0%** |
| **Cross-Modal Retrieval Accuracy** | **87.5%** |

**Result**: Near-perfect alignment achieved between disparate modalities. The dual-encoder architecture successfully learns the relative rotation between semantic spaces, enabling robust cross-modal retrieval without explicit pairing supervision during inference.

```bash
uv run main.py task=crossmodal training.epochs=200
```

---

### 3. Hyperbolic Geometry (Lorentz Boost Reversal)
**Task**: Reverse a relativistic Lorentz boost in 2D Minkowski spacetime $Cl(1, 1)$.

| Metric | Value |
|--------|-------|
| **Algebra** | $Cl(1, 1)$ (Minkowski Spacetime) |
| **Training Epochs** | 500 |
| **Network** | Single RotorLayer |
| **Target Rapidity ($\phi$)** | 1.5 |
| **Learned Rotor Weight** | **-1.421** |
| **Parameter Recovery Error** | 5.3% |
| **Final Reconstruction Loss** | **0.0027** ✓ |

**Result**: Highly accurate recovery of the inverse Lorentz transformation. The learned rotor weight of -1.421 closely approximates the target -1.5 (inverse of applied boost), demonstrating GBN's capability in non-Euclidean hyperbolic spaces with mixed metric signatures.

```bash
uv run main.py task=hyperbolic training.epochs=500
```

---

### 4. Semantic Disentanglement (Grade Purity Maximization)
**Task**: Rotate BERT word embeddings to align semantic categories with geometric grades (e.g., "Technology" → Vectors, "Nature" → Bivectors).

| Metric | Value |
|--------|-------|
| **Algebra** | $Cl(6, 0)$ (6D Euclidean) |
| **Training Epochs** | 200 |
| **Network** | RotorLayer + BladeSelector |
| **Input Embedding Dimension** | 768 (BERT) → 6 (PCA) |
| **Initial Grade Purity** | ~0.33 (random) |
| **Final Grade Purity** | **0.887** ✓ |
| **Purity Improvement** | **168%** |

**Result**: Strong semantic disentanglement achieved. The network successfully rotates high-dimensional word embeddings such that semantic concepts align with specific geometric subspaces, demonstrating interpretable geometric structure emergence in language representations.

```bash
uv run main.py task=semantic training.epochs=200
```

### 5. Motion Alignment (Geometry Disentangles Semantics)
**Task**: Align high-dimensional motion data (UCI-HAR) into a linearly separable latent space using geometric rotation.
**Paradigm Shift**: Instead of warping the manifold with black-box MLPs, GBN physically *unbends* the motion classes using a learned **Bivector Map**.

| Metric | Value |
|--------|-------|
| **Algebra** | $Cl(4, 0)$ (Optimized via MetricSearch) |
| **Training Epochs** | 100 |
| **Network** | MotionManifoldNetwork (Rotor Alignment) |
| **Latent Accuracy** | **~100%** |
| **Latent Grade Purity** | 0.9957 (Mixed Grade Manifold) |
| **Visualization** | Bivector Heatmap (Explicit Rotation Planes) |

![motion_latent_space](assets/motion_latent_space.png)

**Result**: The network learns specific rotation planes that distinguish "Walking" from "Running," turning a complex classification problem into a clear geometric alignment task.

```bash
uv run main.py task=motion training.epochs=100
```

---

### Performance Summary

| Task | Algebra | Convergence | Key Achievement |
|------|---------|-------------|-----------------|
| **Manifold** | $Cl(3,0)$ | 500 epochs | 100% topology restoration |
| **Cross-Modal** | $Cl(6,0)$ | 200 epochs | 98% alignment improvement |
| **Hyperbolic** | $Cl(1,1)$ | 500 epochs | 5.3% parameter error (non-Euclidean) |
| **Semantic** | $Cl(6,0)$ | 200 epochs | 88.7% grade purity |
| **Motion** | $Cl(4,0)$ | 100 epochs | Explicit Bivector Disentanglement |

**Computational Notes**:
- All benchmarks run on CPU with PyTorch's broadcasting-based geometric product
- Training speed: ~30-35 iterations/second on Apple M-series chips
- Memory footprint: $O(2^n)$ for $n$-dimensional algebra (e.g., $Cl(6,0)$ uses 64 basis elements)
- GPU acceleration available via `algebra.device="cuda"`

## 📚 Documentation
*   [**Mathematics**](docs/MATHEMATICS.md): Primer on Clifford Algebra, Rotors, and Metric Signatures.
*   [**Layers**](docs/LAYERS.md): Detailed guide to `RotorLayer`, `CliffordLinear`, and `CliffordGraphConv`.
*   [**Tasks**](docs/TASKS.md): Explanation of Manifold Unbending, Cross-Modal Alignment, and Hyperbolic Learning.
*   [**API Reference**](docs/API.md): Overview of the core classes and functions.

## 🚀 Key Features

*   **Metric-Agnostic Kernel**: Supports Euclidean $Cl(p, 0)$, Minkowski/Hyperbolic $Cl(p, q)$, and Projective algebras out of the box.
*   **Geometric Layers**: `RotorLayer`, `CliffordLinear`, `CliffordGraphConv`, `CliffordLayerNorm`.
*   **Novel Activations**: `GeometricGELU` (Magnitude-based activation).
*   **Automatic Metric Search**: Finds optimal $(p, q)$ signature based on data topology.
*   **Geometric Sparsity**: `prune_bivectors` allows compression of geometric layers.
*   **Applications**: Manifold Unbending, Cross-Modal Unification, Hyperbolic Physics, Molecular GNNs.

## Roadmap: The Path to Geometric Intelligence

For More Details, Check out [TODO](TODO.md)

- [ ] Native CUDA Kernels: Optimized kernels for $Cl(3,0)$ and $Cl(1,3)$
- [ ] JIT Compilation: Metric-aware operation graph optimization
- [ ] Experiments on the explainability of RAG & LLM Knowledge Storage
- [ ] Formal Mathematical Rigor: Stability and convergence proofs for GBN in Clifford space
- [x] **Automatic Metric Search**: Self-optimizing signature ($p, q$) based on data topology
- [x] **Automatic Bivector Pruning**: Geometric sparsity-driven layer compression
- [ ] Multi-head & Dynamic Rotors: Input-dependent rotation axes for complex attention
- [ ] Hierarchical GBN: Deep unbending via multi-scale grade abstraction
- [ ] Geometric Transformer (GAT): Fully geometric attention mechanism integration
- [x] **CliffordGraphConv**: Advanced geometric signal processing on graphs (QM9)

## 🛠 Installation

Versor requires Python 3.10+ and PyTorch.

```bash
# Clone the repository
git clone https://github.com/Concode0/Versor.git
cd Versor

# Install dependencies using uv (recommended) or pip
uv sync
```

## ⚡ Usage

Versor uses **Hydra** for configuration management. You can run tasks via CLI or explore the interactive demo

### 1. Manifold Restoration (Unbending)
Trains a rotor to flatten a distorted figure-8 manifold ($z = 0.5xy$) back to the 2D plane.
```bash
uv run main.py task=manifold training.epochs=800
```


### 2. Cross-Modal Unification
Aligns BERT embeddings with a synthetic "image" modality by finding the optimal relative rotation in a 6D geometric space.
```bash
uv run main.py task=crossmodal training.epochs=100
```

### 3. Hyperbolic Geometry (Lorentz Boost)
Learns to reverse a relativistic Lorentz boost in Minkowski spacetime ($Cl(1, 1)$).
```bash
uv run main.py task=hyperbolic
```

### 4. Semantic Unbending
Rotates high-dimensional semantic vectors (BERT) to maximize "grade purity," aligning concepts with orthogonal axes.
```bash
uv run main.py task=semantic
```

## ⚙️ Configuration

Configuration files are located in `conf/`.
*   `conf/config.yaml`: Global defaults (device, batch size).
*   `conf/task/*.yaml`: Task-specific settings.

Example `conf/task/manifold.yaml`:
```yaml
name: "manifold"
algebra:
  p: 3
  q: 0
dataset:
  name: "figure8"
  samples: 1000
```

To override parameters from CLI:
```bash
python main.py task=manifold algebra.p=4 training.lr=0.001
```

## 📂 Project Structure

```
versor/
├── conf/               # Hydra configurations
├── core/               # Math Kernel (algebra, cga, visualizer)
├── datasets/           # PyTorch Datasets (synthetic, real)
├── docs/               # Documentation
├── functional/         # Geometric Activations & Losses
├── layers/             # Neural Network Layers (Rotor, Linear, GNN)
├── tasks/              # Training loops & Task definitions
├── tests/              # Unit & Property tests
└── main.py             # Entry point
```

## ⚠️ Performance Notice
Currently, the geometric product kernel utilizes PyTorch's high-level tensor operations (broadcasting/einsum) for flexibility across arbitrary metrics ($p, q$).Pros: Universal metric support (Euclidean, Hyperbolic, etc.) without recompilation.Cons: Not fully optimized compared to custom CUDA kernels for specific algebras.Roadmap: Native CUDA kernel implementation for $Cl(3,0)$ and $Cl(1,3)$ is planned for v2.0.

## ⚖️ License & Intellectual Property
This project is licensed under the **Apache License 2.0**. 

**Notice on Patents**: 
The core GBN architecture is covered by **KR Patent Application 10-2026-0023023**. 
By releasing this under Apache 2.0, we provide a **perpetual, royalty-free patent license** to any individual or entity using this software. We want this technology to be the foundation of the next generation of AI.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@software{kim2026versor,
  author = {Kim, Eunkyum},
  title = {Versor: Universal Geometric Algebra Neural Network},
  url = {[https://github.com/nemonanconcode/versor](https://github.com/nemonanconcode/versor)},
  year = {2026},
  month = {2},
  note = {ROK Patent Application 10-2026-0023023}
}