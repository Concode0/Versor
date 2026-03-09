# Roadmap: Versor & GBN

**Versor** and the **Geometric Blade Network (GBN)** grew from recognizing an invisible ceiling in standard Linear Algebra for Deep Learning. By moving from unconstrained projections to isometric rotations via **Clifford Rotors**, we preserve the geometric structure that matrices can inadvertently discard.

While the core engine is operational, the vision for a universal Geometric Computing Engine is vast. We are officially opening this roadmap to the global community.

> For completed work, see the "What's Built" section in the [README](../README.md).

## 1. Core Architecture Evolution

- **Real-time Adaptive Rotors (Online SGD)**: Online learning framework for time-series data (e.g., Finance) where rotors adapt dynamically to streaming inputs.
- **Multi-head & Dynamic Rotors**: Input-dependent rotation axes to enable a fully geometric attention mechanism.
- **Geometric Transformer (GAT)**: Integration of "Geometric Attention" where Q, K, V interactions are defined by Clifford blade products.

> **Partial progress**: `GeometricTransformerBlock` and `GeometricProductAttention` already exist in `layers/adapters/` and are used by the DEAP EEG task.

## 2. Mathematical Rigor & Stability (Open Research)

- **Convergence Proofs**: Establishing UUB (Uniformly Ultimately Bounded) stability using Lyapunov Candidate Functions ($Z$-Energy).
- **Loss Landscape Analysis**: Investigating the convexity of the Spin Group manifold to prove the absence of traditional local minima in GBN.
- **High-Dimensional Lifting & Ambiguity**: Researching the "Flip Ambiguity" and gauge transformations when lifting manifolds to higher dimensions.

## 3. Performance & Hardware Optimization

### Analysis: Why High-IPC CPUs May Outperform GPUs for GBN Inference

GBN's computational profile is fundamentally different from standard deep learning:

- **Heavy branching**: The geometric product requires per-blade sign lookups (Cayley table), grade-conditional projections (`popcount`-based masking), and scaling-and-squaring with dynamic iteration counts. These cause GPU warp divergence.
- **Small, structured computations**: A rotor sandwich product in $Cl(3,0)$ is $8 \times 8 = 64$ multiply-accumulates — far below the threshold where GPU parallelism pays off.
- **Low arithmetic intensity**: GBN layers are memory-bound. The ratio of FLOPs to bytes transferred is low. CPUs with large caches and high IPC handle this more efficiently.
- **Sequential dependencies**: `exp(-B/2)` via scaling-and-squaring is inherently sequential.

### Planned Optimizations

- **Native CUDA Kernels**: Fused kernels for $Cl(3,0)$ and $Cl(1,3)$ that minimize warp divergence by precomputing sign tables in shared memory.
- **JIT Compilation**: Aggressive loop unrolling and metric-aware operation graph optimization — compile-time specialization for specific $(p, q)$ signatures.
- **CPU-Centric Acceleration**: SIMD (AVX-512/AMX on x86, NEON on ARM) implementations of the geometric product. Exploiting the fixed sparsity pattern of the Cayley table for vectorized execution.
- **Formal Benchmarking**: Systematic CPU vs. GPU latency/throughput comparison across signatures, batch sizes, and hardware targets.

## 4. Next-Gen Vector Intelligence (Subspace Search)

- **Geometric Subspace Retrieval**: Moving beyond point-to-point distance (Cosine/L2) to subspace-to-subspace relationship mapping.
- **Explainable Search Axes**: Assigning geometric meaning to individual blades (e.g., temporal change vs. spatial gradient).
- **Orthogonality-based Indexing**: Utilizing geometric orthogonality as a proxy for "non-similarity" for deterministic explainability.

## 5. Geometric Integrity & Invariance

- **Pure Invariant Design**: Models inherently invariant to rotation and scaling without data augmentation.
- **Preservation of Information**: Avoiding information collapse from lossy projections and investigating quantization impact on geometric purity.
- **Metric Autonomy**: Enhancing Automatic Metric Search for complex datasets (Molecular, 3D Shapes, multi-modal data).
