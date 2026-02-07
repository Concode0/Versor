# Frequently Asked Questions

## General

### What is Geometric Algebra?
Geometric Algebra (Clifford Algebra) is a mathematical framework that unifies vectors, complex numbers, quaternions, and more into a single algebraic system. It provides a coordinate-free way to express rotations, reflections, and projections in any dimension and any metric signature.

### How is this different from regular PyTorch?
Standard PyTorch layers operate on flat vectors with matrix multiplication. Versor layers operate on **multivectors** — structured geometric objects with scalar, vector, bivector, and higher-grade components. The key difference: `RotorLayer` performs pure rotations (isometries) that cannot warp the data manifold, while `nn.Linear` applies unconstrained linear maps that can stretch, shear, and distort.

### Do I need to understand Clifford Algebra to use Versor?
For basic usage, no. Think of it as: your data gets embedded into a richer representation (multivectors), and the network learns rotations to align it. The algebra handles the math internally. For advanced usage (custom losses, new layers), reading `docs/mathematical.md` will help.

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
It learns bivector weights $B$ — one scalar per rotation plane. In $Cl(3,0)$, there are 3 planes ($e_{12}, e_{13}, e_{23}$), so each channel learns 3 parameters. The rotor $R = \exp(-B/2)$ is computed from these weights, and the sandwich product $RxR\tilde{}$ is applied.

### What is MultiRotorLayer?
A single rotor rotates in one plane. `MultiRotorLayer` uses $K$ rotors in parallel with learned mixing weights. Think of it as multi-head attention but for geometric rotations — each "head" rotates in a different plane, and the outputs are combined by weighted superposition.

### What is CliffordLinear?
Channel mixing via scalar weights — essentially `nn.Linear` applied across the channel dimension without mixing blade components. It's the only non-geometric layer: it can scale channels independently but doesn't perform geometric products.

### What is BladeSelector?
A soft gate over basis blades. Each blade gets a learned sigmoid weight. This lets the network suppress noise in unwanted grades (e.g., keep only vectors, suppress bivectors).

### How does GeometricGELU work?
It computes `x * GELU(|x| + bias) / |x|`. The magnitude is scaled non-linearly via GELU, but the **direction** (the unit multivector) is preserved. This ensures the activation doesn't rotate the data — only the RotorLayer does that.

## Tasks

### What tasks are included?
**Main tasks** (via `main.py`):
- `qm9` / `multi_rotor_qm9` — Molecular property prediction (graph-based)
- `motion` — UCI-HAR motion alignment with rotor-based latent space
- `crossmodal` — Dual-encoder cross-modal alignment (BERT embeddings)
- `semantic` — Semantic disentanglement by grade (BERT word embeddings)

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

### Why is training slow?
The geometric product has complexity $O(2^{2n})$ — it's a full bilinear product over all basis blade pairs. For $Cl(3,0)$ this is $8 \times 8 = 64$ operations per product. For $Cl(6,0)$ it's $64 \times 64 = 4096$. Strategies to mitigate:
- Use `MultiRotorLayer` instead of general geometric products (reduces to $O(K \cdot n^2)$)
- Use `prune_bivectors()` to zero out unused rotation planes
- Prefer smaller signatures when possible
- Use GPU (`algebra.device='cuda'`)

### How does the Cayley table caching work?
`CliffordAlgebra` caches multiplication tables by `(p, q, device)`. The first instantiation for a given signature computes the table; subsequent instantiations reuse it. This means creating multiple layers with the same algebra is free.

### What is MetricSearch?
It brute-forces all $(p, q)$ signatures with $p + q = D$ and evaluates a geometric stress metric on your data. Useful when you don't know whether your data is better modeled by Euclidean, hyperbolic, or mixed geometry.

## Troubleshooting

### `RuntimeError: Shapes ... are not broadcastable`
Check your tensor shapes. Multivectors should be `[Batch, Channels, 2^n]`. Common mistakes:
- Missing channel dimension (use `x.unsqueeze(1)`)
- Wrong algebra dimension (check `algebra.dim`)

### `crossmodal` or `semantic` task fails with ImportError
These tasks require optional dependencies (`transformers`, `scikit-learn`). Install them:
```bash
uv sync --extra examples
```

### Checkpoint loading fails with PyTorch version mismatch
Versor uses `weights_only=False` with a fallback for older PyTorch versions. If you still get errors, ensure your PyTorch version is >= 2.0.
