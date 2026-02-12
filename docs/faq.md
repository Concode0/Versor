# Frequently Asked Questions

## General

### What is Geometric Algebra?
Geometric Algebra (Clifford Algebra) is a mathematical framework that unifies vectors, complex numbers, quaternions, and more into a single algebraic system. It provides a coordinate-free way to express rotations, reflections, and projections in any dimension and any metric signature.

### How is this different from regular PyTorch?
Standard PyTorch layers operate on flat vectors with matrix multiplication. Versor layers operate on **multivectors** — structured geometric objects with scalar, vector, bivector, and higher-grade components. The key difference: `RotorLayer` performs pure rotations (isometries) that preserve manifold structure by construction, while `nn.Linear` applies unconstrained linear maps that may inadvertently stretch, shear, or deform the geometry.

### How does Versor compare to e3nn and other geometric DL libraries?
**e3nn** focuses on $SO(3)$/$O(3)$ equivariance for 3D point clouds using spherical harmonics and irreducible representations. Versor takes a different approach:

- **Metric-agnostic**: Versor works with any $(p, q)$ signature — Euclidean, Minkowski, Conformal — using the same `RotorLayer` code. e3nn is specialized for 3D Euclidean symmetry.
- **Rotor-native**: GBN layers learn rotations directly as bivector exponentials, not as Wigner-D matrices or CG tensor products.
- **Unified algebra**: Scalars, vectors, bivectors, and higher-grade elements coexist in one multivector. There's no separate handling of different representation orders.

The trade-off: e3nn has a more mature ecosystem and extensive $SO(3)$-equivariant tooling. Versor is a broader paradigm (any metric, any dimension) in its alpha stage.

### Do I need to understand Clifford Algebra to use Versor?
For basic usage, no. Think of it as: your data gets embedded into a richer representation (multivectors), and the network learns rotations to align it. The algebra handles the math internally. For advanced usage (custom losses, new layers), reading `docs/mathematical.md` will help.

### What domains is Versor suited for?
Any domain where data has geometric structure:

- **Molecular property prediction**: 3D atom positions in $Cl(3,0)$ — QM9 benchmark at 7.64 meV MAE.
- **Motion/sensor data**: High-dimensional time-series aligned into separable latent spaces — UCI-HAR at ~100% accuracy.
- **NLP/semantic embedding**: BERT embeddings disentangled via rotor-based autoencoder — 100% grade purity on 20 Newsgroups.
- **Physics simulation**: Minkowski $Cl(1,1)$ or $Cl(1,3)$ for spacetime-aware models.
- **Any manifold-valued data**: If your data lives on a manifold (and it almost always does), rotors align it without deformation.

## Setup & Environment

### What are the dependencies?
Core: `torch`, `numpy`, `hydra-core`, `tqdm`. Everything else is optional:
- `uv sync --extra viz` for visualization (matplotlib, seaborn, scikit-learn, plotly, imageio)
- `uv sync --extra examples` for example tasks (transformers, scikit-learn, matplotlib)
- `uv sync --extra graph` for molecular GNN tasks (torch-geometric)
- `uv sync --extra all` for everything

### Does Versor support GPU?
Yes. Pass `device='cuda'` when creating the algebra:
```python
algebra = CliffordAlgebra(p=3, q=0, device='cuda')
```
Or via Hydra config: `uv run main.py task=qm9 algebra.device=cuda`

### I get a CUDA error on macOS / CPU-only machine
The default device in some configs may be `'cuda'`. Override it:
```bash
uv run main.py task=qm9 algebra.device=cpu
```
Or pass `device='cpu'` when creating the algebra directly.

## Architecture

### What is a GBN (Geometric Blade Network)?
GBN is the model architecture family built with Versor. A GBN is composed of:

1. **CliffordLinear** — channel mixing (what to rotate)
2. **RotorLayer** / **MultiRotorLayer** — geometric rotation (how to rotate)
3. **CliffordLayerNorm** — magnitude normalization (preserves direction)
4. **GeometricGELU** — non-linear activation (preserves direction)
5. **BladeSelector** — soft attention over basis blades (what to keep)

Versor provides several GBN variants: `GeometricBladeNetwork` (general-purpose), `MultiRotorModel` (multi-rotor), `MoleculeGNN` / `MultiRotorQuantumNet` (graph-based), `MotionManifoldNetwork` (alignment), and `SemanticAutoEncoder` (disentanglement).

### What is a multivector tensor shape?
`[Batch, Channels, 2^n]` where $n = p + q$. For $Cl(3,0)$: `[B, C, 8]`. For $Cl(6,0)$: `[B, C, 64]`.

### How do I embed regular vectors into multivectors?
Use `algebra.embed_vector()`:
```python
vectors = torch.randn(32, 3)      # [Batch, n]
mv = algebra.embed_vector(vectors)  # [Batch, 2^n]
mv = mv.unsqueeze(1)                # [Batch, 1, 2^n] for layers
```

### What does RotorLayer actually learn?
It learns bivector weights $B$ — one scalar per rotation plane. In $Cl(3,0)$, there are 3 planes ($e_{12}, e_{13}, e_{23}$), so each channel learns 3 parameters. The rotor $R = \exp(-B/2)$ is computed from these weights, and the sandwich product $Rx\tilde{R}$ is applied.

### What is MultiRotorLayer?
A single rotor rotates in one plane. `MultiRotorLayer` uses $K$ rotors in parallel with learned mixing weights. Think of it as multi-head attention but for geometric rotations — each "head" rotates in a different plane, and the outputs are combined by weighted superposition.

### What is CliffordLinear?
Channel mixing via scalar weights — essentially `nn.Linear` applied across the channel dimension without mixing blade components. It now supports two backends:

- **`backend='traditional'`** (default): Uses a dense weight matrix. O(in_channels × out_channels) parameters.
- **`backend='rotor'`**: Uses rotor compositions (RotorGadget). Achieves 60-88% parameter reduction.

```python
# Traditional
linear = CliffordLinear(algebra, in_channels=16, out_channels=32)

# Rotor-based (parameter-efficient)
linear = CliffordLinear(algebra, in_channels=16, out_channels=32, backend='rotor')
```

### What is BladeSelector?
A soft gate over basis blades. Each blade gets a learned sigmoid weight. This lets the network suppress noise in unwanted grades (e.g., keep only vectors, suppress bivectors).

### How does GeometricGELU work?
It computes `x * GELU(|x| + bias) / |x|`. The magnitude is scaled non-linearly via GELU, but the **direction** (the unit multivector) is preserved. This ensures the activation doesn't rotate the data — only the RotorLayer does that.

### What is RotorGadget?
RotorGadget is a parameter-efficient alternative to `CliffordLinear` that replaces dense weight matrices with **rotor sandwich products** `r·x·s†`. Based on [Pence et al., 2025](https://arxiv.org/abs/2507.11688), it uses compositions of simple rotors to achieve the same expressiveness with dramatically fewer parameters.

**Key features**:
- **60-88% parameter reduction** compared to traditional linear layers
- Optional **bivector decomposition** via power iteration (Pence et al., Algorithm 2)
- **Input shuffle** modes for regularization (none/fixed/random)
- Multiple **aggregation** strategies (mean/sum/learned)
- Drop-in replacement via `CliffordLinear(backend='rotor')`

**When to use**:
- Large channel counts (savings scale with size)
- Memory-constrained environments
- Transfer learning (fewer parameters to fine-tune)
- When strict geometric structure is critical

See `docs/tutorial.md` (Section 4: RotorGadget) for detailed examples and usage.

### What are the shuffle modes in RotorGadget?
RotorGadget supports three input channel shuffle modes:

1. **`shuffle='none'`** (default): Sequential block assignment. No overhead, deterministic.
2. **`shuffle='fixed'`**: Random permutation at initialization. Prevents reliance on specific channel orderings.
3. **`shuffle='random'`**: Different permutation each forward pass. Built-in regularization (like dropout for channels).

Use `'fixed'` when channels have no natural order. Use `'random'` for regularization in small-data regimes.

### How many parameters does a GBN have compared to a standard MLP?
Significantly fewer. A RotorLayer in $Cl(3,0)$ with $C$ channels learns $C \times 3$ bivector parameters (one per rotation plane). A standard `nn.Linear` mapping $C$ features would need $C^2$ parameters.

With RotorGadget, the reduction is even more dramatic:
- **Traditional** `CliffordLinear(16, 32)`: 512 parameters
- **RotorGadget** `CliffordLinear(16, 32, backend='rotor')`: ~150 parameters (70% reduction)
- **RotorGadget** `CliffordLinear(64, 64, backend='rotor')`: ~500 parameters vs 4,096 traditional (88% reduction)

For the QM9 task, the full `MultiRotorQuantumNet` achieves 7.64 meV MAE with a lightweight parameter count, because rotors parameterize only the geometrically meaningful degrees of freedom.

### Which metric signature $(p, q)$ should I use?
- **$Cl(3,0)$** — 3D spatial data (molecules, point clouds, robotics)
- **$Cl(1,1)$** — Hyperbolic/relativistic data (Lorentz boosts, hyperbolic embeddings)
- **$Cl(4,0)$ to $Cl(6,0)$** — Higher-dimensional feature alignment (motion, NLP, embedding spaces)
- **Not sure?** Use `MetricSearch` to brute-force the optimal signature for your dataset.

## Tasks

### What tasks are included?
**Main tasks** (via `main.py`):
- `qm9` / `multi_rotor_qm9` — Molecular property prediction (graph-based, requires `--extra graph`)
- `motion` — UCI-HAR motion alignment with rotor-based latent space
- `semantic` — Semantic disentanglement autoencoder (BERT → grade purity, requires `--extra examples`)
- `md17` — Molecular dynamics energy+force prediction in $Cl(3,0)$ (requires `--extra graph`)
- `pdbbind` — Protein-ligand binding affinity in $Cl(3,0)$ (requires `--extra pdbbind`)
- `weatherbench` — Global weather forecasting in $Cl(2,1)$ (requires `--extra weather`)
- `abc` — CAD point cloud reconstruction in $Cl(4,1)$ (requires `--extra cad`)

**Example tasks** (via `examples/main.py`):
- `manifold` — Flatten a figure-8 manifold
- `hyperbolic` — Reverse a Lorentz boost in $Cl(1,1)$
- `sanity` — Identity learning (algebra correctness check)

### How do I add a new task?
1. Create `tasks/mytask.py` inheriting from `BaseTask`
2. Implement: `setup_algebra`, `setup_model`, `setup_criterion`, `get_data`, `train_step`, `evaluate`, `visualize`
3. Create `conf/task/mytask.yaml` with algebra and training config
4. Add to `task_map` in `main.py`
5. Run: `uv run main.py task=mytask`

See `docs/tutorial.md` for a complete example.

### What does `get_data()` return?
Either a `DataLoader` (for batched training) or a raw tensor (for in-memory training). The `BaseTask.run()` method auto-detects which and handles both.

## Performance

### Why is inference fast on CPU?
GBN's computational profile differs fundamentally from standard deep learning. The core operation (rotor sandwich product in $Cl(3,0)$) is only $8 \times 8 = 64$ multiply-accumulates — far below the threshold where GPU parallelism pays off. Additionally, GBN involves heavy branching (Cayley table lookups, grade-conditional projections) that causes GPU warp divergence. High-IPC CPUs with large caches (Apple Silicon, modern x86) handle these small, branchy, memory-bound workloads more efficiently. Result: **5.8 ms/molecule** on an M4 CPU for QM9 inference. See `docs/milestone.md` for detailed analysis.

### Why is training slow?
The geometric product has complexity $O(2^{2n})$ — it's a full bilinear product over all basis blade pairs. For $Cl(3,0)$ this is $8 \times 8 = 64$ operations per product. For $Cl(6,0)$ it's $64 \times 64 = 4096$. Strategies to mitigate:
- Use `MultiRotorLayer` instead of general geometric products (reduces to $O(K \cdot n^2)$)
- Use `prune_bivectors()` to zero out unused rotation planes
- Prefer smaller signatures when possible
- Use GPU for batch training (`algebra.device='cuda'`)

### How does the Cayley table caching work?
`CliffordAlgebra` caches multiplication tables by `(p, q, device)`. The first instantiation for a given signature computes the table; subsequent instantiations reuse it. This means creating multiple layers with the same algebra is free.

### What is MetricSearch?
It brute-forces all $(p, q)$ signatures with $p + q = D$ and evaluates a geometric stress metric on your data. Useful when you don't know whether your data is better modeled by Euclidean, hyperbolic, or mixed geometry.

### What does "Lipschitz by construction" mean?
The rotor sandwich product $Rx\tilde{R}$ is an isometry — it preserves norms exactly ($\|x'\| = \|x\|$), giving a Lipschitz constant of exactly 1. Standard networks must use spectral normalization or gradient penalties to approximate this property. In a GBN, it's guaranteed by the algebra. This directly improves robustness to input perturbations: the motion task shows only a small accuracy drop under noise (see README benchmarks).

## Troubleshooting

### `RuntimeError: Shapes ... are not broadcastable`
Check your tensor shapes. Multivectors should be `[Batch, Channels, 2^n]`. Common mistakes:
- Missing channel dimension (use `x.unsqueeze(1)`)
- Wrong algebra dimension (check `algebra.dim`)

### `qm9` or `multi_rotor_qm9` fails with ImportError
These tasks require `torch-geometric`. Install it:
```bash
uv sync --extra graph
```

### `semantic` task fails with ImportError
This task requires optional dependencies (`transformers`, `scikit-learn`). Install them:
```bash
uv sync --extra examples
```

### First run of `semantic` task is slow
The first run downloads the BERT model (~440 MB) and encodes the 20 Newsgroups corpus. Embeddings are cached to `data/newsgroups/` — subsequent runs load from cache and start immediately.

### Checkpoint loading fails with PyTorch version mismatch
Versor uses `weights_only=False` with a fallback for older PyTorch versions. If you still get errors, ensure your PyTorch version is >= 2.0.

### How do I download training data?
Download scripts are provided in `scripts/`:
```bash
bash scripts/download_md17.sh          # MD17 (SGDML, free)
bash scripts/download_pdbbind.sh       # PDBbind (requires registration)
bash scripts/download_weatherbench.sh  # WeatherBench2 (ERA5, free)
bash scripts/download_abc.sh           # ABC CAD (instructions)
```
All tasks include synthetic data generators, so you can develop and test without external data.

### What are Hermitian metrics?
In mixed-signature algebras $Cl(p,q)$ with $q > 0$, the standard norm $\langle \tilde{A}A \rangle_0$ can be negative. Hermitian metrics use the Clifford conjugation (bar involution) to produce the algebraically proper signed inner product $\langle \bar{A}B \rangle_0$, which respects the algebra structure while enabling stable optimization. For Euclidean algebras $Cl(p,0)$, this reduces to the simple coefficient inner product. See `docs/mathematical.md` Section 12 for details.
