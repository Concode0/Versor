# Versor: Universal Geometric Algebra Neural Network

> **Notice:** This technology is protected by intellectual property rights.
> **Patent Pending (Application No. 10-2026-0023023)**

**Versor** is a PyTorch-based framework for **Geometric Deep Learning** using **Clifford Algebra**. Unlike standard neural networks that operate on flat vectors, Versor operates on **Multivectors**, enabling it to natively learn geometric transformations (rotations, reflections, boosts) and align high-dimensional manifolds across different metric signatures (Euclidean, Hyperbolic, etc.).

## 📚 Documentation
*   [**Mathematics**](docs/MATHEMATICS.md): Primer on Clifford Algebra, Rotors, and Metric Signatures.
*   [**Layers**](docs/LAYERS.md): Detailed guide to `RotorLayer`, `CliffordLinear`, and `CliffordGraphConv`.
*   [**Tasks**](docs/TASKS.md): Explanation of Manifold Unbending, Cross-Modal Alignment, and Hyperbolic Learning.
*   [**API Reference**](docs/API.md): Overview of the core classes and functions.

## 🚀 Key Features

*   **Metric-Agnostic Kernel**: Supports Euclidean $Cl(p, 0)$, Minkowski/Hyperbolic $Cl(p, q)$, and Projective algebras out of the box.
*   **Geometric Layers**: `RotorLayer`, `CliffordLinear`, `CliffordGraphConv`, `CliffordLayerNorm`.
*   **Novel Activations**: `GeometricGELU` (Magnitude-based activation).
*   **Applications**: Manifold Unbending, Cross-Modal Unification, Hyperbolic Physics.

## 🛠 Installation

Versor requires Python 3.10+ and PyTorch.

```bash
# Clone the repository
git clone https://github.com/your-username/versor.git
cd versor

# Install dependencies using uv (recommended) or pip
uv sync
# OR
pip install -r requirements.txt
```

## ⚡ Usage

Versor uses **Hydra** for configuration management. You can run tasks via CLI or explore the interactive demo.

### 🎮 Interactive Demo (Streamlit)
Explore Geometric Algebra transformations and live manifold unbending in your browser.
```bash
streamlit run demo.py
```

### 1. Manifold Restoration (Unbending)
Trains a rotor to flatten a distorted figure-8 manifold ($z = 0.5xy$) back to the 2D plane.
```bash
python main.py task=manifold training.epochs=800
```

### 2. Cross-Modal Unification
Aligns BERT embeddings with a synthetic "image" modality by finding the optimal relative rotation in a 6D geometric space.
```bash
python main.py task=crossmodal training.epochs=100
```

### 3. Hyperbolic Geometry (Lorentz Boost)
Learns to reverse a relativistic Lorentz boost in Minkowski spacetime ($Cl(1, 1)$).
```bash
python main.py task=hyperbolic
```

### 4. Semantic Unbending
Rotates high-dimensional semantic vectors (BERT) to maximize "grade purity," aligning concepts with orthogonal axes.
```bash
python main.py task=semantic
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

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.
See the [LICENSE](LICENSE) file for details.

This ensures that any modifications or network services running this code must also be open-sourced.

## Intellectual Property Notice
The core algorithm and architecture of 'Versor' are protected by patent applications.
- KR Patent Application No. 10-2026-0023023

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