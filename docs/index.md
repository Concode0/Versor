# Versor: A PyTorch Framework for Geometric Algebra Deep Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/) [![Docs](https://img.shields.io/badge/docs-MkDocs-brightgreen)](https://concode0.github.io/Versor/) [![DOI](https://zenodo.org/badge/1149480519.svg)](https://doi.org/10.5281/zenodo.18939518)

> **"There is a ceiling above standard Deep Learning that no one saw. Versor opens the door above it."**

## At a Glance

**Versor** provides geometrically structured layers — built on **Geometric Algebra versor** operations — that constrain spatial transformations to the rotation group where manifold structure matters, while composing naturally with standard linear algebra for channel mixing and readout. It supplies the building blocks for the **Geometric Blade Network (GBN)**: a hybrid architecture in which `CliffordLinear` (standard scalar weights) mixes channels, `RotorLayer` rotates multivectors geometrically, and `nn.Linear` handles task-specific projection.

| Task                                       | Algebra             | Key Metric         | Result                                                                                     | Note                               |
| :----------------------------------------- | :------------------ | :----------------- | :----------------------------------------------------------------------------------------- | :--------------------------------- |
| **Symbolic Regression** (First Principles) | $Cl(4,0)$           | Median R²          | 0.9525                                                                                     | Iterative geometric unbending      |
| **MD17** (Molecular Dynamics)              | $Cl(3,0,1)$ PGA     | Energy / Force MAE | 0.476 / 0.077 · benzene, 0.613 / 0.079 · ethanol, 1.229 / 0.125 · malonaldehyde (kcal/mol) | Error distributions peak at 0      |
| **LQA** (Logical Query Answering)          | $Cl(4,1)$ CGA       | Chain / Negation   | 100% @len1–13 / 64.6%                                                                      | Geometric ALU on frozen embeddings |
| **DEAP EEG** (Emotion)                     | $Cl(3,1)$ Minkowski | RMSE               | 0.2576 / 0.2329                                                                            | Cross / Within-subject LOSO        |

## Core Idea

Rotors ($R = \exp(-B/2)$) perform pure geometric rotations via the sandwich product ($x \to R x \tilde{R}$), preserving manifold structure where standard weight matrices may inadvertently deform it.

In practice, GBN models are hybrid: `RotorLayer` / `MultiRotorLayer` handle geometric rotation, while `CliffordLinear` (traditional backend) and `nn.Linear` handle channel mixing and readout. The two approaches are complementary, not mutually exclusive.

## What's Built

- **Cl(p,q,r) kernel** with null dimension support for Projective GA
- **Signature-aware exp map** — closed-form elliptic/hyperbolic/parabolic (no Taylor series)
- **Hermitian metrics** for positive-definite inner products in any signature
- **Multi-Rotor GBN** with K weighted rotors (geometric spectral decomposition)
- **Rotor Gadget** — parameter-efficient linear replacement (~63% param reduction)
- **Automatic Metric Search** via GeodesicFlow + bivector energy analysis → (p,q,r) discovery
- **CGA embedding** Cl(n+1,1) for conformal geometric algebra
- **Riemannian Adam** optimizer — Adam momentum in the Lie algebra (bivector space)
- **Geometric activations** — GeometricGELU, GradeSwish, GeometricSquare
- **Rotor-to-symbolic-formula translation** — direct readout of trained weights as equations
- **Iterative geometric unbending** — 4-phase SR pipeline with blade rejection
- **ReflectionLayer** / Pin group — learns unit vectors, applies the odd versor `x' = -nxn⁻¹`; composable with `RotorLayer` for full Pin(p,q,r) group coverage
- **Algebraic completeness** — `clifford_conjugation`, `norm_sq`, `left_contraction`, `dual`, `versor_product` covering even and odd versor transformations in a unified interface
- **CliffordGraphConv** for molecular graphs
- **Bivector pruning** for geometric sparsity
- **GeometricTransformerBlock** with entropy-gated attention

For code examples of each innovation, see [Innovations](innovations.md).

## Key Features

- **Metric-Agnostic Kernel**: Supports Euclidean $Cl(p, 0)$, Minkowski $Cl(p, q)$, Projective $Cl(p, 0, r)$, and Conformal $Cl(n+1, 1)$ algebras out of the box.
- **Geometric Layers**: `RotorLayer`, `MultiRotorLayer`, `ReflectionLayer`, `CliffordLinear`, `CliffordGraphConv`, `CliffordLayerNorm`, `BladeSelector`, `RotorGadget`.
- **Novel Activations**: `GeometricGELU` (magnitude-based), `GradeSwish` (per-grade gating), `GeometricSquare` (gated self-product).
- **Automatic Metric Search**: Finds optimal $(p, q, r)$ signature based on data topology via GBN probes.
- **Riemannian Optimization**: `RiemannianAdam` and `ExponentialSGD` with manifold retraction.
- **Geometric Sparsity**: `prune_bivectors` for compression of geometric layers.

## Installation

Versor requires Python 3.9+ and PyTorch.

```bash
# Clone the repository
git clone https://github.com/Concode0/Versor.git
cd Versor

# Install core dependencies
uv sync

# Install task-specific dependencies
uv sync --extra sr          # pmlb, scikit-learn, sympy (Symbolic Regression)
uv sync --extra md17        # torch-geometric (Molecular Dynamics)
uv sync --extra lqa         # sentence-transformers, datasets (Logical Query Answering)
uv sync --extra viz         # matplotlib, seaborn (visualization)
uv sync --extra demo        # streamlit, plotly (interactive demo)
uv sync --extra dev         # pytest (testing)
uv sync --extra all_tasks   # all task dependencies (sr + md17 + lqa)
uv sync --extra all         # everything
```

## Quick Start

### Using Versor Layers in Your Own Model

```python
import torch
from core.algebra import CliffordAlgebra
from layers.primitives.rotor import RotorLayer
from layers.linear import CliffordLinear
from functional.activation import GeometricGELU

# Create a 3D Euclidean Clifford Algebra
algebra = CliffordAlgebra(p=3, q=0, device='cpu')

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
uv run main.py task=sr training.epochs=100
uv run main.py task=md17 training.epochs=100
uv run main.py task=lqa probe=chain training.epochs=50
uv run main.py task=deap_eeg training.epochs=100

# Override any parameter
uv run main.py task=sr algebra.p=4 training.lr=0.001
```

### Interactive Demo (Streamlit)

```bash
streamlit run examples/demo.py
```

## Tasks

### Symbolic Regression (SR)

Discovers closed-form symbolic formulas from numerical data using iterative geometric unbending.

| Property     | Value                                                                        |
| :----------- | :--------------------------------------------------------------------------- |
| **Algebra**  | $Cl(4,0)$                                                                    |
| **Pipeline** | probe → train → extract → subtract → refine (4 phases)                       |
| **Datasets** | [SRBench 2.0](https://arxiv.org/abs/2505.03977) (first_principles, blackbox) |
| **Result**   | Median R² = 0.9525 on 15 First Principles equations                          |

**Analysis:** The median R² of 0.9525 on First Principles equations should be understood primarily as a **structural proposal**: a demonstration that iterative geometric unbending is a viable and interpretable framework for symbolic regression. The most important properties are **interpretability** (formulas are read directly from trained rotor weights) and **physically plausible structure** (rotor composition mirrors the composition of physical symmetries). The current implementation has known limitations with numerical instability and high-dimensional inputs, both planned for future improvement.

**Speed:** For 12 first_principles datasets, the entire execution took ~5 minutes (avg. ~23s per dataset). Geometric unbending computes laws deterministically rather than searching stochastically — a structurally different efficiency from Genetic Algorithm-based SR.

```bash
uv run main.py task=sr
```

---

### MD17 (Molecular Dynamics)

Multi-task energy + force prediction with conservative constraint ($F = -\nabla E$), using $Cl(3,0,1)$ PGA — translations are exact rotors, no approximation.

| Molecule      | Atoms | Epochs | VRAM  |  Time   | E MAE (kcal/mol) | F MAE (kcal/mol/Å) |
| :------------ | :---: | :----: | :---: | :-----: | :--------------: | :----------------: |
| benzene       |  12   |  400   | 11 GB | ~62 min |    **0.476**     |     **0.077**      |
| ethanol       |   9   |  500   | 6 GB  | ~52 min |    **0.613**     |     **0.079**      |
| malonaldehyde |   9   |  400   | 6 GB  | ~41 min |    **1.229**     |     **0.125**      |

All runs: rMD17 · 1000 train / 1000 val / 98 000 test · RTX Pro 4500.

**Error distribution:** Prediction errors peak sharply at 0 and follow a Gaussian-like shape across all three molecules — no systematic bias. The residual is purely stochastic noise, consistent with a model that has learned the true underlying potential energy surface geometry.

```bash
uv run main.py task=md17
```

---

### LQA (Logical Query Answering)

A **geometric arithmetic logic device** (~228K params) that operates directly on frozen latent embeddings. Rather than building an end-to-end LLM, LQA isolates the question: *what can geometric algebra do to a latent space that flat linear algebra cannot?*

| Property       | Value                                               |
| :------------- | :-------------------------------------------------- |
| **Algebra**    | $Cl(4,1)$ Conformal GA                              |
| **Probes**     | chain (CLUTRR), entailment (HANS), negation (BoolQ) |
| **Chain**      | 100% accuracy at all lengths 1–13                   |
| **Negation**   | 64.6% — orig 65.2%, neg 63.9% (gap 1.3%)            |
| **Entailment** | 52.6% — ent 81.4%, non-ent 23.8%                    |

**Chain (composition):** Perfect 100% accuracy across all chain lengths 1–13. Rotor composition $R_1 R_2 \cdots R_k$ naturally represents multi-hop relational chains — the algebraic structure matches the task structure exactly.

**Negation & Entailment (encoder ceiling):** These probes deliberately expose the limits of flat embeddings. MiniLM maps "Is X?" and "Isn't X?" to cosine similarity 0.967 — the embedding shifts only 18% of inter-question distance under negation. An MLP baseline on the same embeddings achieves 59.5% (vs GBN 64.6%) with a comparable 1.0% negation gap, confirming the ceiling is in the encoder, not the geometric model.

**Next step:** Replace the frozen MiniLM encoder with a geometric embedding pipeline, removing the flat-space bottleneck entirely.

```bash
uv run main.py task=lqa probe=chain training.epochs=50
uv run main.py task=lqa probe=negation training.epochs=10
uv run main.py task=lqa probe=entailment training.epochs=10
```

---

### DEAP EEG (Emotion Regression)

EEG emotion regression using phase-amplitude representation in Minkowski algebra with mother manifold alignment across subjects.

| Property       | Value                                                              |
| :------------- | :----------------------------------------------------------------- |
| **Algebra**    | $Cl(3,1)$ Minkowski                                                |
| **Input**      | 32-channel EEG + 8 peripheral channels                             |
| **Targets**    | Valence, Arousal, Dominance, Liking                                |
| **Evaluation** | LOSO (cross-subject) and within-subject (80/20 split), 32 subjects |

```bash
uv run main.py task=deap_eeg training.epochs=10                                  # within-subject
uv run main.py task=deap_eeg evaluation.mode=cross_subject training.epochs=10    # cross-subject
```

**Results (32 subjects, 10 epochs, stride-applied windowing):**

| Dimension     | Cross-subject RMSE | Within-subject RMSE |     Δ (cross − within)     |
| :------------ | :----------------: | :-----------------: | :------------------------: |
| **Valence**   |   0.2478 ± 0.055   |   0.2700 ± 0.086    | **−0.0222** (cross better) |
| **Arousal**   |   0.2438 ± 0.060   |   0.2243 ± 0.072    |          +0.0195           |
| **Dominance** |   0.2551 ± 0.073   |   0.1951 ± 0.062    |          +0.0600           |
| **Liking**    |   0.2839 ± 0.070   |   0.2423 ± 0.077    |          +0.0416           |
| **Mean**      |     **0.2576**     |     **0.2329**      |      +0.0247 (+9.6%)       |

The cross/within RMSE gap is only **0.025 units** (9.6% relative), despite cross-subject training predicting a completely held-out subject. The Minkowski rotor representation appears to capture subject-invariant affective structure. **Valence anomaly:** cross-subject RMSE (0.2478) is lower than within-subject (0.2700) — cross-subject training forces the model toward the stable population-level valence manifold rather than overfitting to individual rating biases.

---

## Examples (Synthetic / Demo Tasks)

Synthetic experiments demonstrating GA concepts:

```bash
uv run python -m examples.main task=manifold training.epochs=500
uv run python -m examples.main task=hyperbolic training.epochs=500
uv run python -m examples.main task=sanity
```

| Example        | Algebra   | Description                                             |
| :------------- | :-------- | :------------------------------------------------------ |
| **Manifold**   | $Cl(3,0)$ | Flatten a figure-8 manifold (100% topology restoration) |
| **Hyperbolic** | $Cl(1,1)$ | Reverse a Lorentz boost in Minkowski spacetime          |
| **Sanity**     | $Cl(3,0)$ | Verify algebra correctness (identity learning)          |

### Paper Counterparts (Synthetic Data)

Three well-known GA deep learning papers approached through Versor's composable primitives. These are **not reimplementations** — each paper has its own construction. They demonstrate that Versor's general-purpose layers achieve the same algebraic guarantees through different mechanisms. All run on **synthetic data** to verify structural properties.

```bash
uv run python -m examples.main task=gatr training.epochs=200
uv run python -m examples.main task=cgenn training.epochs=200
uv run python -m examples.main task=clifford_pde training.epochs=300
```

| Example          | Paper                                                              | Algebra         |  Original   |   Versor   | Synthetic Task                   | Property Verified                          |
| :--------------- | :----------------------------------------------------------------- | :-------------- | :---------: | :--------: | :------------------------------- | :----------------------------------------- |
| **GATr**         | [Brehmer et al., NeurIPS 2023](https://arxiv.org/abs/2305.18415)   | $Cl(3,0,1)$ PGA | ~2500 lines | ~80 lines  | N-body spring dynamics           | E(3) equivariance (rotation + translation) |
| **CGENN**        | [Ruhe et al., NeurIPS 2023 Oral](https://arxiv.org/abs/2305.11141) | $Cl(3,0)$       | ~3000 lines | ~90 lines  | Point cloud invariant regression | O(3) invariance (rotation + reflection)    |
| **Clifford PDE** | [Brandstetter et al., ICLR 2023](https://arxiv.org/abs/2209.04934) | $Cl(2,0)$       | ~4000 lines | ~120 lines | 2D Taylor-Green vortex           | Emergent vorticity in grade-2 bivector     |

Each counterpart credits the paper's contribution, describes Versor's different construction path, and is honest about where the architecture diverges. See each task file's docstring for the detailed comparison.

## Project Structure

```
Versor/
├── core/               # Math kernel (CliffordAlgebra, metric, search, decomposition, CGA)
├── layers/             # Neural layers (Rotor, MultiRotor, Linear, GNN, Norm, RotorGadget)
├── functional/         # Activations (GeometricGELU, GradeSwish, GeometricSquare) & losses
├── models/             # Task-specific architectures
│   └── sr/             # SR models (SRGBN, translator, unbender, grouper, estimator)
├── optimizers/         # Riemannian optimizers (RiemannianAdam, ExponentialSGD)
├── tasks/              # Task runners (SR, MD17, LQA, DEAP EEG)
├── datalib/            # Data loaders (PMLB, MD17, CLUTRR/HANS/BoolQ, DEAP)
├── conf/               # Hydra configs for main tasks
├── examples/           # Synthetic demos, paper reimplementations, interactive Streamlit app
│   ├── tasks/          # Manifold, Hyperbolic, Sanity, GATr, CGENN, Clifford PDE
│   ├── datasets/       # Synthetic data generators
│   └── conf/           # Hydra configs for example tasks
├── tests/              # Unit & property tests
└── main.py             # CLI entry point
```

## Contributing

Versor is currently in a **Stabilization Phase** as the lead maintainer focuses on academic milestones. While we are not actively seeking major feature contributions at this moment, we highly value community feedback.

- **Found a Bug?** Please open an [Issue](https://github.com/Concode0/Versor/issues) with a detailed reproduction case.
- **Have an Idea?** Open an Issue to discuss it before submitting a Pull Request.
- **Code of Conduct:** All participants are expected to adhere to our [Code of Conduct](../CODE_OF_CONDUCT.md).


## License & Intellectual Property

This project is licensed under the **Apache License 2.0**.

**Notice on Patents:** The core GBN architecture is covered by **KR Patent Application 10-2026-0023023**. By releasing this under Apache 2.0, we provide a **perpetual, royalty-free patent license** to any individual or entity using this software.

**Notice:** This project is the original, independent work of Eunkyum Kim. We have no affiliation with the paper "Versor: A Geometric Sequence Architecture" (arXiv:2602.10195).

## Citation

```bibtex
@software{kim2026versor,
  author  = {Kim, Eunkyum},
  title   = {Versor: Universal Geometric Algebra Neural Network},
  url     = {https://github.com/Concode0/versor},
  version = {1.0.0},
  year    = {2026},
  month   = {3},
  doi     = {10.5281/zenodo.18939519},
  license = {Apache-2.0},
  note    = {ROK Patent Application 10-2026-0023023 (Geometric Blade Networks)}
```

## Reference

Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers from Irreducibles." arXiv:2507.11688v1 [cs.LG]
