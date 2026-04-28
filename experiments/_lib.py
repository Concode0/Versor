# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Shared harness for Versor experiments.

Every helper here is intentionally tiny and composable. Experiments import
what they need and assemble their own train loop, model, loss, and plots.
No opinionated ``Trainer`` class, no forced flag names, no prescribed block
topology — real experiments diverge on all three (ortho annealing, PINN
collocation, energy-weighted loss, per-block gauge projection, etc.), and
a one-shape-fits-all helper would bleed.

Inclusion rule
--------------
A helper belongs here iff:
  1. It appears in at least two experiments.
  2. It has no domain coupling (no model/block/loss/dataset assumptions).
  3. Its body is under ~20 lines.
Otherwise, inline it in the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from core.metric import hermitian_grade_spectrum
from functional.activation import GeometricGELU
from layers import CliffordLayerNorm, CliffordLinear, RotorLayer

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed ``torch``, ``numpy``, and ``random``; optionally force determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Algebra factory
# ---------------------------------------------------------------------------


def setup_algebra(p: int, q: int = 0, r: int = 0, device: str = "cpu") -> CliffordAlgebra:
    """One-line ``CliffordAlgebra(p, q, r, device=device).to(device)`` wrapper."""
    return CliffordAlgebra(p=p, q=q, r=r, device=device).to(device)


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------


def ensure_output_dir(path: str) -> str:
    """``os.makedirs(path, exist_ok=True)`` and return ``path``."""
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_plot_token(value: Any) -> str:
    """Normalize arbitrary metadata into lowercase ASCII-ish snake_case."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def signature_metadata(p: int, q: int, r: int = 0) -> str:
    """Compact Clifford signature token suitable for filenames."""
    parts = [str(p), str(q)]
    if r != 0:
        parts.append(str(r))
    return "cl" + "_".join(parts)


def build_visualization_metadata(*parts: Any, **named_parts: Any) -> str:
    """Compose a stable metadata token from positional and named parts."""
    tokens: List[str] = []

    for part in parts:
        if part is None or part == "":
            continue
        if isinstance(part, (list, tuple, set)):
            joined = "_".join(sanitize_plot_token(item) for item in part if item is not None and item != "")
            if joined:
                tokens.append(joined)
            continue
        tokens.append(sanitize_plot_token(part))

    for key, value in named_parts.items():
        if value is None or value == "":
            continue
        label = sanitize_plot_token(key)
        if isinstance(value, bool):
            tokens.append(f"{label}_{int(value)}")
            continue
        if isinstance(value, float):
            value_token = sanitize_plot_token(f"{value:g}")
        elif isinstance(value, (list, tuple, set)):
            value_token = "_".join(sanitize_plot_token(item) for item in value if item is not None and item != "")
        else:
            value_token = sanitize_plot_token(value)
        if value_token:
            tokens.append(f"{label}_{value_token}")

    return "_".join(tokens) or "default"


def _json_safe(value: Any) -> Any:
    """Convert common experiment metadata into JSON-serializable values."""
    if isinstance(value, argparse.Namespace):
        return _json_safe(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_experiment_figure(
    fig: Any,
    *,
    output_dir: str,
    experiment_name: str,
    metadata: str,
    plot_name: str,
    args: Any = None,
    module: Optional[str] = None,
    dpi: int = 150,
    close: bool = True,
    manifest_name: str = "visualization_manifest.jsonl",
) -> str:
    """Save a figure with standardized naming and append a verification record."""
    out_dir = Path(ensure_output_dir(output_dir))
    exp_token = sanitize_plot_token(experiment_name)
    meta_token = sanitize_plot_token(metadata)
    plot_token = sanitize_plot_token(plot_name)
    filename = f"{exp_token}_{meta_token}_{plot_token}.png"
    path = out_dir / filename

    fig.savefig(path, dpi=dpi, bbox_inches="tight")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = out_dir / manifest_name
    entry = {
        "experiment_name": exp_token,
        "metadata": meta_token,
        "plot_name": plot_token,
        "filename": filename,
        "path": str(path.resolve()),
        "sha256": digest,
        "module": module or exp_token,
        "args": _json_safe(args),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Grade-1 embedding / extraction
# ---------------------------------------------------------------------------


def grade1_indices(algebra: CliffordAlgebra) -> List[int]:
    """Multivector indices of grade-1 basis elements ``[e1, e2, ..., e_n]``."""
    return [1 << i for i in range(algebra.n)]


def extract_grade1(mv: torch.Tensor, algebra: CliffordAlgebra, n: Optional[int] = None) -> torch.Tensor:
    """Slice grade-1 components from a multivector ``[..., dim] → [..., n]``.

    ``n`` defaults to ``algebra.n`` (all grade-1 slots). Inverse of
    :meth:`CliffordAlgebra.embed_vector`.
    """
    g1 = grade1_indices(algebra)[: (n if n is not None else algebra.n)]
    return mv[..., g1]


# ---------------------------------------------------------------------------
# Canonical GBN residual block
# ---------------------------------------------------------------------------


def gbn_residual_block(algebra: CliffordAlgebra, channels: int) -> nn.ModuleDict:
    """The four-step block every GBN experiment shares.

    Returns ``{'norm', 'rotor', 'act', 'linear'}`` — no skip, no outer module
    list. Callers assemble their own ``nn.ModuleList`` of these and apply
    :func:`apply_residual_block` per block. Keeps the per-experiment
    composition local while removing the boilerplate.
    """
    return nn.ModuleDict(
        {
            "norm": CliffordLayerNorm(algebra, channels),
            "rotor": RotorLayer(algebra, channels),
            "act": GeometricGELU(algebra, channels=channels),
            "linear": CliffordLinear(algebra, channels, channels),
        }
    )


def apply_residual_block(block: nn.ModuleDict, h: torch.Tensor) -> torch.Tensor:
    """Apply a :func:`gbn_residual_block` with an outer skip connection."""
    residual = h
    h = block["norm"](h)
    h = block["rotor"](h)
    h = block["act"](h)
    h = block["linear"](h)
    return residual + h


# ---------------------------------------------------------------------------
# Grade-energy aggregation
# ---------------------------------------------------------------------------


@torch.no_grad()
def mean_grade_spectrum(mv_iter: Iterable[torch.Tensor], algebra: CliffordAlgebra) -> np.ndarray:
    """Mean Hermitian grade spectrum across an iterable of multivectors.

    Each element may be any shape ending in ``algebra.dim``; it is flattened
    to ``[*, dim]`` before :func:`hermitian_grade_spectrum`. Returns a
    ``[n+1]`` numpy array of per-grade mean energies (empty iterable → zeros).
    """
    totals = torch.zeros(algebra.n + 1, dtype=torch.float64)
    count = 0
    for mv in mv_iter:
        flat = mv.reshape(-1, algebra.dim)
        spec = hermitian_grade_spectrum(algebra, flat)  # [N, n+1]
        totals += spec.sum(dim=0).double().cpu()
        count += spec.shape[0]
    if count == 0:
        return np.zeros(algebra.n + 1)
    return (totals / count).numpy()


# ---------------------------------------------------------------------------
# Non-coercive argparse — opt-in standard flags
# ---------------------------------------------------------------------------

_STANDARD_ARG_SPECS: Mapping[str, dict] = {
    "seed": {"flags": ("--seed",), "type": int, "default": 42, "help": "Random seed."},
    "device": {"flags": ("--device",), "type": str, "default": "cpu", "help": "Torch device (cpu/cuda/mps)."},
    "epochs": {"flags": ("--epochs",), "type": int, "default": 200, "help": "Number of training epochs."},
    "lr": {"flags": ("--lr",), "type": float, "default": 1e-3, "help": "Learning rate."},
    "batch_size": {"flags": ("--batch-size",), "type": int, "default": 128, "help": "Mini-batch size."},
    "output_dir": {
        "flags": ("--output-dir", "--out-dir"),
        "type": str,
        "default": "experiment_plots",
        "help": "Directory for saved artefacts.",
    },
    "diag_interval": {
        "flags": ("--diag-interval",),
        "type": int,
        "default": 20,
        "help": "Epoch stride for diagnostic logging.",
    },
    "p": {"flags": ("--p",), "type": int, "default": 3, "help": "Positive signature dimensions."},
    "q": {"flags": ("--q",), "type": int, "default": 0, "help": "Negative signature dimensions."},
    "r": {"flags": ("--r",), "type": int, "default": 0, "help": "Degenerate (null) dimensions."},
}


def add_standard_args(
    parser: argparse.ArgumentParser,
    *,
    include: Sequence[str] = ("seed", "device", "epochs", "lr", "batch_size", "output_dir", "diag_interval"),
    defaults: Optional[Mapping[str, Any]] = None,
) -> argparse.ArgumentParser:
    """Additively attach common flags to ``parser``.

    Each entry of ``include`` names a flag from ``_STANDARD_ARG_SPECS``;
    the caller chooses the subset. ``defaults`` overrides per-flag defaults
    (e.g. ``defaults={'device': 'mps', 'output_dir': 'lorentz_plots'}``).
    Returns the same parser for chaining.
    """
    overrides = dict(defaults or {})
    for name in include:
        spec = dict(_STANDARD_ARG_SPECS[name])
        flags = spec.pop("flags")
        if name in overrides:
            spec["default"] = overrides[name]
        parser.add_argument(*flags, **spec)
    return parser


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve multi-line descriptions while still showing defaults."""


def make_experiment_parser(
    description: str,
    *,
    include: Sequence[str] = ("seed", "device", "epochs", "lr", "batch_size", "output_dir", "diag_interval"),
    defaults: Optional[Mapping[str, Any]] = None,
    formatter_class: type[argparse.HelpFormatter] = argparse.ArgumentDefaultsHelpFormatter,
) -> argparse.ArgumentParser:
    """Create a parser and attach the standard experiment flags."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=formatter_class,
    )
    if include:
        add_standard_args(parser, include=include, defaults=defaults)
    return parser


def parse_clifford_signature(value: str) -> tuple[int, int, int]:
    """Parse ``p,q`` or ``p,q,r`` into a Clifford signature tuple."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (2, 3) or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(f"signature must be 'p,q' or 'p,q,r', got {value!r}")
    try:
        ints = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"signature must be 'p,q' or 'p,q,r', got {value!r}") from exc
    if len(ints) == 2:
        return ints[0], ints[1], 0
    return ints  # type: ignore[return-value]


def add_signature_arg(
    parser: argparse.ArgumentParser,
    *,
    default: tuple[int, int, int] = (3, 0, 0),
    flag: str = "--signature",
) -> argparse.ArgumentParser:
    """Attach a reusable ``Cl(p,q,r)`` signature argument."""
    parser.add_argument(
        flag,
        type=parse_clifford_signature,
        default=default,
        help="Comma-separated p,q[,r]. Default: 3,0,0.",
    )
    return parser


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def section_header(title: str, char: str = "=", width: int = 60) -> str:
    """Return the three-line banner: ``char*width / title / char*width``."""
    bar = char * width
    return f"{bar}\n{title}\n{bar}"


def print_banner(title: str, **kv: Any) -> None:
    """Print a titled banner followed by key=value lines."""
    print(section_header(title))
    for key, value in kv.items():
        print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Plotting (lazy matplotlib import)
# ---------------------------------------------------------------------------


def save_training_curve(
    history: Mapping[str, Sequence[float]],
    output_path: Optional[str] = None,
    *,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    metadata: Optional[str] = None,
    plot_name: str = "training_curve",
    args: Any = None,
    module: Optional[str] = None,
    x_key: str = "epochs",
    y_keys: Optional[Sequence[str]] = None,
    y_log: bool = True,
    title: str = "Training curves",
) -> str:
    """Plot each ``y_key`` in ``history`` against ``history[x_key]`` and save.

    ``y_keys=None`` auto-selects every key of ``history`` that is not ``x_key``.
    Uses a non-interactive backend and lazy-imports matplotlib so headless CI
    doesn't break when plots aren't requested. Returns the absolute saved path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x_key not in history:
        raise KeyError(f"history missing x_key {x_key!r}; keys: {list(history)}")
    xs = list(history[x_key])
    if y_keys is None:
        y_keys = [k for k in history.keys() if k != x_key]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_fn = ax.semilogy if y_log else ax.plot
    for key in y_keys:
        series = history.get(key)
        if series is None or len(series) == 0:
            continue
        plot_fn(xs, series, label=key)
    ax.set_xlabel(x_key)
    ax.set_ylabel("value (log)" if y_log else "value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_dir is not None or experiment_name is not None or metadata is not None:
        if output_dir is None or experiment_name is None or metadata is None:
            raise ValueError("output_dir, experiment_name, and metadata must be provided together")
        return save_experiment_figure(
            fig,
            output_dir=output_dir,
            experiment_name=experiment_name,
            metadata=metadata,
            plot_name=plot_name,
            args=args,
            module=module,
            dpi=150,
        )

    if output_path is None:
        raise ValueError("output_path is required when standardized plot saving is not used")

    saved = os.path.abspath(output_path)
    fig.savefig(saved, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Supervised training loop (single-loss, natural-expression style)
# ---------------------------------------------------------------------------


def run_supervised_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[nn.Module, Any], torch.Tensor],
    data: Any,
    *,
    epochs: int,
    diag_interval: int = 20,
    diag_fn: Optional[Callable[[nn.Module, int], Dict[str, float]]] = None,
    grad_clip: Optional[float] = 1.0,
    log: bool = True,
    history_extra_keys: Sequence[str] = (),
) -> Dict[str, List[float]]:
    """Minimal single-loss training loop.

    ``loss_fn(model, batch) -> scalar`` — returns one scalar. Any term that
    is *not* the natural loss must NOT enter this scalar; put it in
    ``diag_fn`` instead (runs under ``no_grad`` every ``diag_interval``
    epochs). ``data`` is either a DataLoader-like iterable or a single
    batch — experiments pick what fits their domain. Returns a history
    dict compatible with :func:`save_training_curve`.
    """
    history: Dict[str, List[float]] = {"epochs": [], "train_loss": []}
    for key in history_extra_keys:
        history[key] = []
    is_loader = hasattr(data, "__iter__") and not torch.is_tensor(data) and not isinstance(data, dict)
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        batches: Iterable = data if is_loader else [data]
        for batch in batches:
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item())
            n += 1
        avg = total / max(n, 1)
        if epoch == 1 or epoch == epochs or epoch % diag_interval == 0:
            history["epochs"].append(epoch)
            history["train_loss"].append(avg)
            extras: Dict[str, float] = {}
            if diag_fn is not None:
                with torch.no_grad():
                    model.eval()
                    extras = diag_fn(model, epoch) or {}
                for k, v in extras.items():
                    history.setdefault(k, []).append(float(v))
            if log:
                extra_str = "  ".join(f"{k}={v:.4e}" for k, v in extras.items())
                print(f"  epoch {epoch:>5d}/{epochs}  loss={avg:.6e}  {extra_str}")
    return history


# ---------------------------------------------------------------------------
# Post-training diagnostic table
# ---------------------------------------------------------------------------


def report_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    title: str = "Post-training diagnostics",
    tolerance: Optional[Mapping[str, float]] = None,
) -> str:
    """Format a flat diagnostic dict as a three-column table.

    Every value must reduce to ``float``. If ``tolerance`` is provided, each
    row ends in ``OK``/``FAIL`` based on per-metric thresholds; otherwise
    only the numeric value is shown. Mirrors the style of
    :mod:`experiments._templates.dbg_template.format_report`.
    """
    tolerance = dict(tolerance or {})
    name_w = max((len(k) for k in diagnostics), default=10)
    lines = [section_header(title)]
    header = f"  {'metric':<{name_w}}  {'value':>14}"
    if tolerance:
        header += f"  {'tol':>10}  {'status':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    all_ok = True
    for name, raw in diagnostics.items():
        try:
            val = float(raw)
        except (TypeError, ValueError):
            lines.append(f"  {name:<{name_w}}  {str(raw):>14}")
            continue
        row = f"  {name:<{name_w}}  {val:>14.6e}"
        if name in tolerance:
            tol = float(tolerance[name])
            ok = not math.isnan(val) and abs(val) <= tol
            all_ok = all_ok and ok
            row += f"  {tol:>10.1e}  {('OK' if ok else 'FAIL'):>8}"
        lines.append(row)
    if tolerance:
        lines.append("  " + "-" * (len(header) - 2))
        lines.append(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Docstring header constants
# ---------------------------------------------------------------------------

DBG_HEADER = """\
==============================================================================
VERSOR EXPERIMENT: MATHEMATICAL DEBUGGER
==============================================================================

This script is designed to validate topological and algebraic phenomena rather
than to achieve State-of-the-Art (SOTA) on traditional benchmarks. Its focus
is to confirm that the Clifford Algebra framework computes known identities
and physical laws correctly, and to surface regressions when they do not.

Please kindly note that as an experimental module, formal mathematical proofs
and exhaustive literature reviews may still be in progress. Contributions that
tighten the validation suite — additional check_* methods, sharper tolerances,
cross-references to the literature — are warmly welcomed.
"""

INC_HEADER = """\
==============================================================================
VERSOR EXPERIMENT: IDEA INCUBATOR (SPIN-OFF CONCEPT)
==============================================================================

This script serves as an early-stage proof-of-concept for radical, non-Euclidean
architectures. The concepts demonstrated here are strongly driven by geometric
intuition and may currently reside ahead of established academic literature.

Please understand that rigorous mathematical proofs or comprehensive citations
might be incomplete at this stage. If this geometric hypothesis proves
structurally sound, it is planned to be spun off into a dedicated, independent
repository for detailed research.
"""


__all__ = [
    "set_seed",
    "setup_algebra",
    "ensure_output_dir",
    "sanitize_plot_token",
    "signature_metadata",
    "build_visualization_metadata",
    "save_experiment_figure",
    "count_parameters",
    "grade1_indices",
    "extract_grade1",
    "gbn_residual_block",
    "apply_residual_block",
    "mean_grade_spectrum",
    "add_standard_args",
    "make_experiment_parser",
    "RawDefaultsHelpFormatter",
    "parse_clifford_signature",
    "add_signature_arg",
    "section_header",
    "print_banner",
    "save_training_curve",
    "run_supervised_loop",
    "report_diagnostics",
    "INC_HEADER",
    "DBG_HEADER",
]
