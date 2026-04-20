# Frequently Asked Questions

## General

### What is Geometric Algebra?
Geometric Algebra (Clifford Algebra) is a mathematical framework that unifies vectors, complex numbers, quaternions, and more into a single algebraic system. It provides a coordinate-free way to express rotations, reflections, and projections in any dimension and any metric signature.

### How is this different from regular PyTorch?
Standard PyTorch layers operate on flat vectors with matrix multiplication. Versor layers operate on **multivectors** — structured geometric objects with scalar, vector, bivector, and higher-grade components. The key difference: `RotorLayer` performs pure rotations (isometries) that preserve manifold structure by construction, while `nn.Linear` applies unconstrained linear maps that may inadvertently stretch, shear, or deform the geometry.

### How does Versor compare to e3nn and other geometric DL libraries?
**e3nn** focuses on $SO(3)$/$O(3)$ equivariance for 3D point clouds using spherical harmonics and irreducible representations. Versor takes a different approach:

- **Metric-agnostic**: Versor works with any $(p, q, r)$ signature — Euclidean, Minkowski, Conformal, Degenerate — using the same `RotorLayer` code. e3nn is specialized for 3D Euclidean symmetry.
- **Rotor-native**: GBN layers learn rotations directly as bivector exponentials, not as Wigner-D matrices or CG tensor products.
- **Unified algebra**: Scalars, vectors, bivectors, and higher-grade elements coexist in one multivector. There's no separate handling of different representation orders.

The trade-off: e3nn has a more mature ecosystem and extensive $SO(3)$-equivariant tooling. Versor is a broader paradigm (any metric, any dimension) in its alpha stage.

### Do I need to understand Clifford Algebra to use Versor?
For basic usage, no. Think of it as: your data gets embedded into a richer representation (multivectors), and the network learns rotations to align it. The algebra handles the math internally. For advanced usage (custom losses, new layers), reading `docs/mathematical.md` will help.

### What domains is Versor suited for?
Any domain where data has geometric structure:

- **Symbolic regression**: Cl(4,0) — iterative geometric unbending discovers closed-form formulas from data. Median R² = 0.9984 on Feynman equations.
- **Molecular dynamics**: Cl(3,0,1) PGA — energy + force prediction with conservative constraints.
- **Logical query answering**: Cl(4,1) CGA — geometric reasoning on frozen LLM embeddings with ~300K params.
- **EEG emotion classification**: Cl(3,1) Minkowski — phase-amplitude representation with mother manifold alignment.
- **Any manifold-valued data**: If your data lives on a manifold (and it almost always does), rotors align it without deformation.

## Setup & Environment

### What are the dependencies?
Core: `torch`, `numpy`, `hydra-core`, `tqdm`. Everything else is optional:
- `uv sync --extra viz` for visualization (matplotlib, seaborn)
- `uv sync --extra all_tasks` for all task dependencies (sr, md17, lqa)
- `uv sync --extra all` for everything

### Does Versor support GPU?
Yes. Pass `device='cuda'` when creating the algebra:
```python
algebra = CliffordAlgebra(p=3, q=0, device='cuda')
```
Or via Hydra config: `uv run main.py task=sr algebra.device=cuda`

### I get a CUDA error on macOS / CPU-only machine
The default device in some configs may be `'cuda'`. Override it:
```bash
uv run main.py task=sr algebra.device=cpu
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

Versor provides several GBN variants: `SRGBN` (symbolic regression), `MD17ForceNet` (molecular dynamics), `GLRNet` (logical query answering), and `EEGNet` (emotion classification).

### What is a multivector tensor shape?
`[Batch, Channels, 2^n]` where $n = p + q + r$. For $Cl(3,0)$: `[B, C, 8]`. For $Cl(4,1)$: `[B, C, 32]`.

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
Channel mixing via scalar weights — essentially `nn.Linear` applied across the channel dimension without mixing blade components. It's the only non-geometric layer: it can scale channels independently but doesn't perform geometric products. Replacing it with pure rotor compositions is on the roadmap (see `docs/milestone.md`). The `RotorGadget` backend (`CliffordLinear(algebra, in_ch, out_ch, backend='rotor')`) already achieves ~63% parameter reduction via rotor pairs.

### What is BladeSelector?
A soft gate over basis blades. Each blade gets a learned sigmoid weight. This lets the network suppress noise in unwanted grades (e.g., keep only vectors, suppress bivectors).

### How does GeometricGELU work?
It computes `x * GELU(|x| + bias) / |x|`. The magnitude is scaled non-linearly via GELU, but the **direction** (the unit multivector) is preserved. This ensures the activation doesn't rotate the data — only the RotorLayer does that.

### How many parameters does a GBN have compared to a standard MLP?
Significantly fewer. A RotorLayer in $Cl(3,0)$ with $C$ channels learns $C \times 3$ bivector parameters (one per rotation plane). A standard `nn.Linear` mapping $C$ features would need $C^2$ parameters. Rotors parameterize only the geometrically meaningful degrees of freedom.

### Which metric signature $(p, q, r)$ should I use?
- **$Cl(3,0)$** — 3D spatial data (molecules, point clouds, robotics)
- **$Cl(3,0,1)$** — PGA: SE(3) rigid-body motions (molecular dynamics with translation)
- **$Cl(3,1)$** — Minkowski: spacetime or phase-amplitude data (EEG, physics)
- **$Cl(4,0)$ to $Cl(6,0)$** — Higher-dimensional feature alignment (symbolic regression, embedding spaces)
- **$Cl(4,1)$** — Conformal: translations become rotations (logical reasoning, CAD)
- **Not sure?** Use `MetricSearch` to discover the optimal signature for your dataset.

### What is MetricSearch?
It lifts data into a conformal algebra `Cl(X+1, 1)`, trains GBN probes with biased initialization (euclidean, minkowski, projective), then analyzes the learned bivector energy distribution to classify each base vector as elliptic, hyperbolic, or null — returning an optimal `(p, q, r)` 3-tuple. See `core/analysis/signature.py`.

### How does the Cayley table caching work?
`CliffordAlgebra` caches multiplication tables by `(p, q, r, device)`. The first instantiation for a given signature computes the table; subsequent instantiations reuse it. This means creating multiple layers with the same algebra is free.

## Tasks

### What tasks are included?
**Main tasks** (via `main.py`):
- `sr` — Symbolic regression via iterative geometric unbending (PMLB datasets)
- `md17` — Molecular dynamics: energy + force prediction (requires `--extra md17`)
- `lqa` — Logical query answering with geometric reasoning probes (requires `--extra lqa`)
- `deap_eeg` — EEG emotion classification with mother manifold alignment

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
GBN's computational profile differs fundamentally from standard deep learning. The core operation (rotor sandwich product in $Cl(3,0)$) is only $8 \times 8 = 64$ multiply-accumulates — far below the threshold where GPU parallelism pays off. Additionally, GBN involves heavy branching (Cayley table lookups, grade-conditional projections) that causes GPU warp divergence. High-IPC CPUs with large caches (Apple Silicon, modern x86) handle these small, branchy, memory-bound workloads more efficiently. See `docs/milestone.md` for detailed analysis.

### Why is training slow?
The geometric product has complexity $O(2^{2n})$ — it's a full bilinear product over all basis blade pairs. For $Cl(3,0)$ this is $8 \times 8 = 64$ operations per product. For $Cl(6,0)$ it's $64 \times 64 = 4096$. Strategies to mitigate:
- Use `MultiRotorLayer` instead of general geometric products (reduces to $O(K \cdot n^2)$)
- Use `prune_bivectors()` to zero out unused rotation planes
- Prefer smaller signatures when possible
- Use GPU for batch training (`algebra.device='cuda'`)

### What does "Lipschitz by construction" mean?
The rotor sandwich product $Rx\tilde{R}$ is an isometry — it preserves norms exactly ($\|x'\| = \|x\|$), giving a Lipschitz constant of exactly 1. Standard networks must use spectral normalization or gradient penalties to approximate this property. In a GBN, it's guaranteed by the algebra.

## Troubleshooting

### `RuntimeError: Shapes ... are not broadcastable`
Check your tensor shapes. Multivectors should be `[Batch, Channels, 2^n]`. Common mistakes:
- Missing channel dimension (use `x.unsqueeze(1)`)
- Wrong algebra dimension (check `algebra.dim`)

### SR task runs out of memory during symbolic translation
The symbolic trace through the model can produce large sympy expressions. Solutions:
- Reduce the number of channels in the model
- The translator automatically caps symbolic channels at 4 (`MAX_SYM_CHANNELS`)
- Use `translate()` (indirect) instead of `translate_direct()` for simpler expressions

### LQA task fails with ImportError
This task requires optional dependencies (`sentence-transformers`, `datasets`). Install them:
```bash
uv sync --extra lqa
```

### DEAP task can't find data files
The DEAP dataset requires manual download from the official source. Place preprocessed `.dat` files in `data/DEAP/data_preprocessed_python/`. See `docs/tasks/deap_eeg.md` for details.

### MD17 task fails with ImportError
This task requires `torch-geometric`. Install it:
```bash
uv sync --extra md17
```

### Checkpoint loading fails with PyTorch version mismatch
Versor uses `weights_only=False` with a fallback for older PyTorch versions. If you still get errors, ensure your PyTorch version is >= 2.0.
