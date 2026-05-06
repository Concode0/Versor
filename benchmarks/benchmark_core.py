#!/usr/bin/env python3
"""Standalone benchmark suite for ``core/``.

This benchmark intentionally depends only on the local ``core`` package. It
measures:

1. Scaling over algebra dimension, batch size, dtype, operator, and compile
   mode.
2. Compile correctness relative to eager for the same dtype.
3. Non-simple bivector exponential error for n >= 4.
4. Precision-dependent cumulative sandwich drift.
5. Precision-dependent convergence of the compiled-safe decomposed exp path.
6. Algebraic stability/correctness invariants across representative signatures.
7. Forward+backward latency and gradients for differentiable core operators.
8. PyTorch profiler kernel/op counts, fusion proxies, and peak allocation.
9. High-resolution line plots and heatmaps.

Artifacts are written to ``benchmarks/results/benchmark_core_<timestamp>/``.

Example:
    uv run python benchmarks/benchmark_core.py
    # Adjust the parameters as you want to benchmark.


"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib
import torch
import torch.nn as nn
from torch.utils import benchmark as torch_bench

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import PartitionConfig, make_algebra
from core.decomposition import ExpPolicy, compiled_safe_decomposed_exp  # noqa: E402
from core.device import FLOAT_DTYPES, dtype_name as _format_dtype_name, optional_dtype, resolve_device
from core.module import AlgebraLike

DTYPES: dict[str, torch.dtype] = FLOAT_DTYPES


@dataclass(frozen=True)
class TimerStats:
    first_call_ms: float
    median_ms: float
    mean_ms: float
    std_ms: float
    p10_ms: float
    p90_ms: float
    runs: int
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class SignatureSpec:
    p: int
    q: int
    r: int = 0

    @property
    def n(self) -> int:
        return self.p + self.q + self.r

    @property
    def label(self) -> str:
        return f"Cl({self.p},{self.q},{self.r})"


class CoreOpModule(nn.Module):
    """Small wrapper so core operators can be passed to torch.compile."""

    def __init__(self, algebra: AlgebraLike, op: str):
        super().__init__()
        self.algebra = algebra
        self.op = op

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if self.op == "gp":
            return self.algebra.geometric_product(args[0], args[1])
        if self.op == "wedge":
            return self.algebra.wedge(args[0], args[1])
        if self.op == "inner":
            return self.algebra.inner_product(args[0], args[1])
        if self.op == "commutator":
            return self.algebra.commutator(args[0], args[1])
        if self.op == "grade2":
            return self.algebra.grade_projection(args[0], 2)
        if self.op == "reverse":
            return self.algebra.reverse(args[0])
        if self.op == "norm_sq":
            return self.algebra.norm_sq(args[0])
        if self.op in {"exp", "exp_precise"}:
            return self.algebra.exp(args[0])
        if self.op == "sandwich":
            return self.algebra.sandwich_product(args[0], args[1])
        raise RuntimeError(f"unknown op: {self.op}")


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def _parse_optional_int_csv(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    return _parse_int_csv(str(value))


def _parse_signature_csv(value: str) -> list[SignatureSpec]:
    specs: list[SignatureSpec] = []
    for raw in _parse_csv(value):
        cleaned = (
            raw.lower()
            .removeprefix("cl")
            .replace("(", "")
            .replace(")", "")
            .replace("/", ":")
        )
        parts = [part for part in cleaned.split(":") if part]
        if len(parts) == 1:
            p, q, r = int(parts[0]), 0, 0
        elif len(parts) == 2:
            p, q = (int(part) for part in parts)
            r = 0
        elif len(parts) == 3:
            p, q, r = (int(part) for part in parts)
        else:
            raise ValueError(
                f"invalid signature {raw!r}; use n, p:q, or p:q:r entries"
            )
        if p < 0 or q < 0 or r < 0 or p + q + r < 1 or p + q + r > 16:
            raise ValueError(
                f"invalid signature {raw!r}; dimensions must be non-negative "
                "and sum to 1..16"
            )
        specs.append(SignatureSpec(p, q, r))
    return specs


def _dtype_name(dtype: torch.dtype) -> str:
    return _format_dtype_name(dtype)


def setup_algebra(
    p: int,
    q: int = 0,
    r: int = 0,
    *,
    device: str,
    dtype: torch.dtype,
    exp_policy: str | ExpPolicy = "balanced",
    fixed_iterations: int | None = None,
    args: argparse.Namespace | None = None,
) -> AlgebraLike:
    """Construct benchmark algebras through the shared core factory."""
    partition = PartitionConfig(
        leaf_n=getattr(args, "partition_leaf_n", 4),
        product_chunk_size=getattr(args, "partition_product_chunk_size", None),
        tree=getattr(args, "partition_tree", None),
        accumulation_dtype=optional_dtype(getattr(args, "partition_accumulation_dtype", None)),
    )
    return make_algebra(
        p=p,
        q=q,
        r=r,
        kernel=getattr(args, "algebra_kernel", "auto"),
        partition_threshold=getattr(args, "partition_threshold", 8),
        partition=partition,
        device=device,
        dtype=dtype,
        exp_policy=exp_policy,
        fixed_iterations=fixed_iterations,
    )


def _ordered_modes(value: str) -> list[str]:
    modes = _parse_csv(value)
    if "eager" in modes:
        return ["eager"] + [mode for mode in modes if mode != "eager"]
    return modes


def _resolve_compile_modes(value: str, device: str) -> str:
    if value != "auto":
        modes = _ordered_modes(value)
    elif device.startswith("cuda"):
        modes = ["eager", "aot_eager", "compile", "reduce-overhead"]
    else:
        modes = ["eager", "aot_eager", "compile"]

    valid = {"eager", "aot_eager", "compile", "reduce-overhead", "max-autotune"}
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"unknown compile mode(s): {unknown}; valid: {sorted(valid)}")
    return ",".join(_ordered_modes(",".join(modes)))


def _tf32_modes_for_dtype(args: argparse.Namespace, dtype: torch.dtype) -> list[str]:
    requested = _parse_csv(args.tf32_modes)
    valid = {"auto", "default", "strict", "tf32"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"unknown TF32 mode(s): {unknown}; valid: {sorted(valid)}")
    if "auto" in requested:
        return ["strict", "tf32"] if args.device.startswith("cuda") and dtype == torch.float32 else ["default"]
    if not args.device.startswith("cuda") or dtype != torch.float32:
        return ["default"]
    return requested or ["default"]


def _preferred_tf32_mode(args: argparse.Namespace, dtype_name: str) -> str:
    if dtype_name != "float32" or not args.device.startswith("cuda"):
        return "default"
    modes = _parse_csv(args.tf32_modes)
    if "auto" in modes:
        return "tf32"
    if "tf32" in modes:
        return "tf32"
    return modes[0] if modes else "default"


@contextmanager
def _tf32_context(device: str, mode: str) -> Iterable[None]:
    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    old_matmul = getattr(cuda_matmul, "allow_tf32", None)
    old_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)
    has_precision = hasattr(torch, "get_float32_matmul_precision") and hasattr(
        torch, "set_float32_matmul_precision"
    )
    old_precision = torch.get_float32_matmul_precision() if has_precision else None

    try:
        if device.startswith("cuda") and mode != "default":
            enabled = mode == "tf32"
            if cuda_matmul is not None and old_matmul is not None:
                cuda_matmul.allow_tf32 = enabled
            if old_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = enabled
            if has_precision:
                torch.set_float32_matmul_precision("high" if enabled else "highest")
        yield
    finally:
        if cuda_matmul is not None and old_matmul is not None:
            cuda_matmul.allow_tf32 = old_matmul
        if old_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = old_cudnn
        if has_precision and old_precision is not None:
            torch.set_float32_matmul_precision(old_precision)


def _current_tf32_flags(device: str, mode: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return {
            "tf32_mode": "default",
            "cuda_matmul_allow_tf32": "",
            "cudnn_allow_tf32": "",
            "float32_matmul_precision": "",
        }
    precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else ""
    )
    return {
        "tf32_mode": mode,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": precision,
    }


def _channels_for_op(args: argparse.Namespace, op: str) -> list[int]:
    if op != "sandwich":
        return [0]
    values = getattr(args, "sandwich_channel_values", None) or [args.channels]
    return sorted({int(v) for v in values if int(v) > 0})


def _plot_channel_value(args: argparse.Namespace) -> int:
    values = getattr(args, "sandwich_channel_values", None) or [args.channels]
    return max(int(v) for v in values)


def _matches_plot_axes(args: argparse.Namespace, row: dict[str, Any], dtype: str) -> bool:
    if row["dtype"] != dtype:
        return False
    if row.get("tf32_mode", "default") != _preferred_tf32_mode(args, dtype):
        return False
    if row.get("op") == "sandwich":
        return int(row.get("channels", 0)) == _plot_channel_value(args)
    return int(row.get("channels", 0)) == 0


def _op_exp_policy(op: str) -> ExpPolicy:
    return ExpPolicy.PRECISE if op == "exp_precise" else ExpPolicy.BALANCED


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device.startswith("cuda"):
        return torch.cuda.is_available()
    return True


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _release_memory(device: str) -> None:
    gc.collect()
    gc.collect()
    _sync(device)
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _reset_peak_memory(device: str) -> None:
    _sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _allocated_bytes(device: str) -> float:
    if device.startswith("cuda"):
        return float(torch.cuda.memory_allocated())
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory())
    return float("nan")


def _peak_allocated_bytes(device: str) -> float:
    if device.startswith("cuda"):
        return float(torch.cuda.max_memory_allocated())
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        # PyTorch MPS exposes current allocation, but not a resettable peak.
        return float(torch.mps.current_allocated_memory())
    return float("nan")


def _bytes_to_mb(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return value / (1024.0 * 1024.0)


def _seed_all(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def _supported_dtypes(
    args: argparse.Namespace,
    requested: str,
    probe_n_values: Iterable[int],
) -> list[torch.dtype]:
    if requested == "auto":
        candidates = ["float64", "float32"]
        if device.startswith("cuda") or device == "mps":
            candidates += ["bfloat16", "float16"]
    else:
        candidates = _parse_csv(requested)

    supported: list[torch.dtype] = []
    probe_ns = sorted({int(n) for n in probe_n_values if int(n) >= 1})
    if not probe_ns:
        probe_ns = [2]
    for name in candidates:
        if name not in DTYPES:
            raise ValueError(f"unknown dtype {name!r}; valid: {sorted(DTYPES)}")
        dtype = DTYPES[name]
        algebra: AlgebraLike | None = None
        x: torch.Tensor | None = None
        y: torch.Tensor | None = None
        try:
            for n in probe_ns:
                algebra = setup_algebra(n, 0, device=args.device, dtype=dtype, args=args)
                x = torch.randn(2, algebra.dim, device=args.device, dtype=dtype)
                y = algebra.geometric_product(x, x)
                _sync(args.device)
                if not torch.isfinite(y.float()).all().item():
                    raise RuntimeError(f"n={n} probe produced non-finite values")
                algebra = None
                x = None
                y = None
            supported.append(dtype)
        except Exception as exc:
            print(f"Skipping dtype {name} on {args.device}: {exc}")
        finally:
            algebra = None
            x = None
            y = None
            _release_memory(args.device)
    if not supported:
        raise SystemExit(f"No requested dtypes are usable on {args.device}.")
    return supported


def _sample_steps(max_steps: int, samples: int) -> list[int]:
    if max_steps <= 1:
        return [1]
    values = {1, max_steps}
    for i in range(samples):
        t = i / max(samples - 1, 1)
        values.add(max(1, int(round(math.exp(t * math.log(max_steps))))))
    return sorted(values)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _measurement_to_stats(
    measurement: torch_bench.Measurement,
    first_call_ms: float,
) -> TimerStats:
    """Project a torch.utils.benchmark Measurement onto TimerStats (per-call ms).

    ``measurement.times`` is already per-call seconds (block time / number_per_run),
    so the only conversion is seconds -> milliseconds.
    """
    per_call_ms = [t * 1000.0 for t in measurement.times]
    if not per_call_ms:
        per_call_ms = [float(measurement.mean) * 1000.0]
    sorted_ms = sorted(per_call_ms)
    std_ms = statistics.stdev(per_call_ms) if len(per_call_ms) > 1 else 0.0
    return TimerStats(
        first_call_ms=first_call_ms,
        median_ms=statistics.median(per_call_ms),
        mean_ms=statistics.fmean(per_call_ms),
        std_ms=std_ms,
        p10_ms=_percentile(sorted_ms, 0.10),
        p90_ms=_percentile(sorted_ms, 0.90),
        runs=len(per_call_ms),
        samples_ms=tuple(per_call_ms),
    )


def _bootstrap_speedup_ci(
    eager_ms: Iterable[float],
    candidate_ms: Iterable[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    eager = np.array([float(v) for v in eager_ms if float(v) > 0], dtype=float)
    candidate = np.array([float(v) for v in candidate_ms if float(v) > 0], dtype=float)
    if samples <= 0 or eager.size == 0 or candidate.size == 0:
        return float("nan"), float("nan")
    if eager.size == 1 and candidate.size == 1:
        ratio = float(eager[0] / candidate[0])
        return ratio, ratio
    rng = np.random.default_rng(seed)
    ratios = np.empty(samples, dtype=float)
    for idx in range(samples):
        eager_sample = eager[rng.integers(0, eager.size, eager.size)]
        candidate_sample = candidate[rng.integers(0, candidate.size, candidate.size)]
        denominator = float(np.median(candidate_sample))
        ratios[idx] = (
            float(np.median(eager_sample)) / denominator
            if denominator > 0
            else float("nan")
        )
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _speedup_fields(
    baseline: TimerStats | None,
    stats: TimerStats,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if baseline is None or stats.median_ms <= 0:
        return {
            "speedup_vs_eager": float("nan"),
            "speedup_vs_eager_ci_low": float("nan"),
            "speedup_vs_eager_ci_high": float("nan"),
            "speedup_vs_eager_ci_excludes_1": False,
            "speedup_vs_eager_significant": False,
        }
    speedup = baseline.median_ms / stats.median_ms
    low, high = _bootstrap_speedup_ci(
        baseline.samples_ms,
        stats.samples_ms,
        samples,
        seed,
    )
    ci_excludes_1 = bool(math.isfinite(low) and math.isfinite(high) and (low > 1.0 or high < 1.0))
    significant = bool(math.isfinite(low) and low > 1.0)
    return {
        "speedup_vs_eager": speedup,
        "speedup_vs_eager_ci_low": low,
        "speedup_vs_eager_ci_high": high,
        "speedup_vs_eager_ci_excludes_1": ci_excludes_1,
        "speedup_vs_eager_significant": significant,
    }


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _nonzero_terms(tensor: torch.Tensor) -> int:
    return int(torch.count_nonzero(tensor.detach()).item())


def _analytic_forward_metrics(
    algebra: AlgebraLike,
    op: str,
    batch: int,
    channels: int,
) -> dict[str, float]:
    dim = algebra.dim
    dtype_bytes = _dtype_bytes(algebra.dtype)
    logical_items = batch * channels if op == "sandwich" else batch
    flops = float("nan")
    io_elements = 0

    if op == "gp":
        signs = getattr(algebra, "gp_signs", None)
        flops = 2.0 * batch * _nonzero_terms(signs) if signs is not None else float("nan")
        io_elements = 3 * batch * dim
    elif op == "wedge":
        signs = getattr(algebra, "wedge_gp_signs", None)
        flops = 2.0 * batch * _nonzero_terms(signs) if signs is not None else float("nan")
        io_elements = 3 * batch * dim
    elif op == "inner":
        signs = getattr(algebra, "inner_gp_signs", None)
        flops = 2.0 * batch * _nonzero_terms(signs) if signs is not None else float("nan")
        io_elements = 3 * batch * dim
    elif op == "commutator":
        signs = getattr(algebra, "comm_gp_signs", None)
        flops = 2.0 * batch * _nonzero_terms(signs) if signs is not None else float("nan")
        io_elements = 3 * batch * dim
    elif op in {"grade2", "reverse"}:
        flops = float(batch * dim)
        io_elements = 2 * batch * dim
    elif op == "norm_sq":
        flops = float(3 * batch * dim)
        io_elements = batch * dim + batch
    elif op == "sandwich":
        c = max(channels, 1)
        flops = 2.0 * batch * dim * dim * dim + 2.0 * batch * c * dim * dim
        io_elements = batch * dim + 2 * batch * c * dim + batch * dim * dim
    elif op in {"exp", "exp_precise"}:
        io_elements = 2 * batch * dim

    return {
        "logical_items_per_call": float(logical_items),
        "analytic_forward_flops": flops,
        "analytic_io_bytes": float(io_elements * dtype_bytes),
    }


def _add_rate_metrics(row: dict[str, Any], median_ms: float) -> None:
    logical_items = float(row.get("logical_items_per_call", 0.0))
    flops = float(row.get("analytic_forward_flops", float("nan")))
    io_bytes = float(row.get("analytic_io_bytes", float("nan")))
    row["items_per_sec"] = (
        logical_items * 1000.0 / median_ms if median_ms > 0 else float("inf")
    )
    row["achieved_forward_gflops"] = (
        flops / (median_ms * 1e6)
        if median_ms > 0 and math.isfinite(flops)
        else float("nan")
    )
    row["achieved_io_gbps"] = (
        io_bytes / (median_ms * 1e6)
        if median_ms > 0 and math.isfinite(io_bytes)
        else float("nan")
    )


def _time_callable(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    device: str,
    warmup: int,
    runs: int,
    min_time_ms: float,
    max_runs: int,
) -> tuple[TimerStats, torch.Tensor]:
    """Time a callable using ``torch.utils.benchmark.Timer``.

    Timer picks an inner-loop multiplier large enough that block measurements
    dwarf Python overhead, uses CUDA events on CUDA, and freezes GC during the
    timed loop -- all of which a hand-rolled ``perf_counter`` loop would have
    to replicate.
    """
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter_ns()
        out = fn(*args)
        _sync(device)
        first_call_ms = (time.perf_counter_ns() - t0) / 1e6

        for _ in range(warmup):
            out = fn(*args)
        _sync(device)

        timer = torch_bench.Timer(
            stmt="_fn(*_args)",
            globals={"_fn": fn, "_args": args},
            num_threads=torch.get_num_threads(),
        )
        # Drive run count via min_run_time; cap roughly via the wall-time budget.
        # ``runs`` and ``max_runs`` shape Timer's behavior only through the
        # min_run_time hint -- Timer chooses block count adaptively.
        min_run_time_s = max(min_time_ms / 1000.0, 1e-3)
        measurement = timer.blocked_autorange(min_run_time=min_run_time_s)

        # Re-evaluate fn once for the caller (Timer's last output is discarded).
        out = fn(*args)
        _sync(device)

    return _measurement_to_stats(measurement, first_call_ms), out.detach()


def _clone_inputs_for_backward(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    cloned = []
    for tensor in inputs:
        leaf = tensor.detach().clone()
        if leaf.is_floating_point():
            leaf.requires_grad_(True)
        cloned.append(leaf)
    return tuple(cloned)


def _zero_input_grads(inputs: tuple[torch.Tensor, ...]) -> None:
    for tensor in inputs:
        if tensor.grad is not None:
            tensor.grad = None


def _loss_from_output(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _time_forward_backward(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    device: str,
    warmup: int,
    runs: int,
    min_time_ms: float,
    max_runs: int,
) -> tuple[TimerStats, torch.Tensor, float]:
    """Time forward+backward via Timer; return stats, last output, grad norm."""
    args = _clone_inputs_for_backward(inputs)

    def _step() -> torch.Tensor:
        _zero_input_grads(args)
        out_inner = fn(*args)
        _loss_from_output(out_inner).backward()
        return out_inner

    _sync(device)
    t0 = time.perf_counter_ns()
    out = _step()
    _sync(device)
    first_call_ms = (time.perf_counter_ns() - t0) / 1e6

    for _ in range(warmup):
        out = _step()
    _sync(device)

    timer = torch_bench.Timer(
        stmt="_step()",
        globals={"_step": _step},
        num_threads=torch.get_num_threads(),
    )
    min_run_time_s = max(min_time_ms / 1000.0, 1e-3)
    measurement = timer.blocked_autorange(min_run_time=min_run_time_s)

    # Final pass to capture grad norm + output for the caller.
    _zero_input_grads(args)
    out = fn(*args)
    _loss_from_output(out).backward()
    grad_norm_sq = 0.0
    for tensor in args:
        if tensor.grad is not None:
            grad = tensor.grad.detach().float()
            grad_norm_sq += float((grad * grad).sum().item())
    grad_norm = math.sqrt(grad_norm_sq)
    _zero_input_grads(args)

    return _measurement_to_stats(measurement, first_call_ms), out.detach(), grad_norm


def _profiler_activities(device: str) -> list[Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _event_attr(event: Any, names: Iterable[str]) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except TypeError:
                continue
    return 0.0


def _is_cuda_runtime_event(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("cuda") or lowered.startswith("hip")


def _is_device_kernel_event(event: Any) -> bool:
    name = str(getattr(event, "name", ""))
    lowered = name.lower()
    if _is_cuda_runtime_event(name):
        return False
    device_type = str(getattr(event, "device_type", "")).lower()
    if "cuda" in device_type or "hip" in device_type:
        return True
    if _event_attr(event, ["self_device_time_total", "self_cuda_time_total"]) > 0:
        return True
    # Older profiler builds sometimes lose the CUDA device_type on Triton
    # kernels, but their names remain stable enough to classify separately
    # from cudaLaunchKernel runtime API entries.
    return "triton" in lowered or "cutlass" in lowered


def _profiler_summary(prof: Any, steps: int) -> dict[str, float]:
    key_averages = prof.key_averages()
    total_events = 0.0
    aten_events = 0.0
    compiled_events = 0.0
    cpu_time_us = 0.0
    device_time_us = 0.0
    self_cpu_mem = 0.0
    self_device_mem = 0.0

    for event in key_averages:
        count = float(getattr(event, "count", 1.0))
        key = str(getattr(event, "key", ""))
        total_events += count
        if key.startswith("aten::"):
            aten_events += count
        if (
            "Torch-Compiled Region" in key
            or "CompiledFunction" in key
            or "inductor" in key.lower()
            or "triton" in key.lower()
        ):
            compiled_events += count
        cpu_time_us += _event_attr(event, ["self_cpu_time_total"])
        device_time_us += _event_attr(
            event,
            ["self_device_time_total", "self_cuda_time_total"],
        )
        self_cpu_mem += max(_event_attr(event, ["self_cpu_memory_usage"]), 0.0)
        self_device_mem += max(
            _event_attr(
                event,
                [
                    "self_device_memory_usage",
                    "self_cuda_memory_usage",
                    "self_privateuse1_memory_usage",
                ],
            ),
            0.0,
        )

    kernel_events = 0.0
    triton_kernel_events = 0.0
    cuda_runtime_events = 0.0
    try:
        for event in prof.events():
            name = str(getattr(event, "name", ""))
            if _is_cuda_runtime_event(name):
                cuda_runtime_events += 1.0
                continue
            if _is_device_kernel_event(event):
                kernel_events += 1.0
                if "triton" in name.lower():
                    triton_kernel_events += 1.0
    except Exception:
        kernel_events = 0.0
        triton_kernel_events = 0.0
        cuda_runtime_events = 0.0

    denom = max(float(steps), 1.0)
    return {
        "profiler_total_events": total_events,
        "profiler_total_events_per_step": total_events / denom,
        "aten_ops": aten_events,
        "aten_ops_per_step": aten_events / denom,
        "compiled_region_events": compiled_events,
        "compiled_region_events_per_step": compiled_events / denom,
        "kernel_events": kernel_events,
        "kernel_events_per_step": kernel_events / denom,
        "triton_kernel_events": triton_kernel_events,
        "triton_kernel_events_per_step": triton_kernel_events / denom,
        "cuda_runtime_events": cuda_runtime_events,
        "cuda_runtime_events_per_step": cuda_runtime_events / denom,
        "profiler_self_cpu_time_ms": cpu_time_us / 1000.0,
        "profiler_self_cpu_time_ms_per_step": cpu_time_us / (1000.0 * denom),
        "profiler_self_device_time_ms": device_time_us / 1000.0,
        "profiler_self_device_time_ms_per_step": device_time_us / (1000.0 * denom),
        "profiler_positive_self_cpu_memory_mb": _bytes_to_mb(self_cpu_mem),
        "profiler_positive_self_device_memory_mb": _bytes_to_mb(self_device_mem),
    }


def _run_profile_step(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    phase: str,
) -> torch.Tensor:
    if phase == "forward":
        with torch.no_grad():
            return module(*inputs)
    out = module(*inputs)
    _loss_from_output(out).backward()
    _zero_input_grads(inputs)
    return out.detach()


def _profile_module(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    phase: str,
    device: str,
    warmup: int,
    steps: int,
    trace_path: Path | None = None,
) -> dict[str, float]:
    profile_inputs = (
        _clone_inputs_for_backward(inputs) if phase == "backward" else inputs
    )

    _run_profile_step(module, profile_inputs, phase)
    for _ in range(warmup):
        _run_profile_step(module, profile_inputs, phase)
    _sync(device)

    _reset_peak_memory(device)
    before_bytes = _allocated_bytes(device)

    with torch.profiler.profile(
        activities=_profiler_activities(device),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as prof:
        for step in range(steps):
            with torch.profiler.record_function(f"{phase}_step_{step}"):
                _run_profile_step(module, profile_inputs, phase)

    _sync(device)
    after_bytes = _allocated_bytes(device)
    peak_bytes = _peak_allocated_bytes(device)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    summary = _profiler_summary(prof, steps)
    summary.update(
        {
            "allocated_before_mb": _bytes_to_mb(before_bytes),
            "allocated_after_mb": _bytes_to_mb(after_bytes),
            "peak_allocated_mb": _bytes_to_mb(peak_bytes),
            "peak_delta_mb": _bytes_to_mb(max(peak_bytes - before_bytes, 0.0))
            if math.isfinite(peak_bytes) and math.isfinite(before_bytes)
            else float("nan"),
            "allocated_delta_mb": _bytes_to_mb(after_bytes - before_bytes)
            if math.isfinite(after_bytes) and math.isfinite(before_bytes)
            else float("nan"),
        }
    )
    return summary


def _error_stats(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual64 = actual.detach().cpu().to(torch.float64)
    ref64 = reference.detach().cpu().to(torch.float64)
    diff = actual64 - ref64
    abs_diff = diff.abs()
    ref_scale = ref64.abs().max().clamp(min=1e-30)
    return {
        "max_abs_error": float(abs_diff.max().item()),
        "rms_error": float(torch.sqrt((diff * diff).mean()).item()),
        "max_rel_error": float((abs_diff.max() / ref_scale).item()),
    }


def _max_abs_diff(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual64 = actual.detach().cpu().to(torch.float64)
    reference64 = reference.detach().cpu().to(torch.float64)
    diff = actual64 - reference64
    return float(diff.abs().max().item())


def _tensor_max_abs(value: torch.Tensor) -> float:
    return float(
        value.detach().cpu().to(torch.float64).abs().max().item()
    )


def _stability_tolerance(
    dtype: torch.dtype,
    dim: int,
    multiplier: float = 1.0,
) -> float:
    if dtype == torch.float64:
        base = 1e-10
    elif dtype == torch.float32:
        base = 5e-5
    elif dtype == torch.bfloat16:
        base = 2e-2
    elif dtype == torch.float16:
        base = 2e-2
    else:
        base = float(torch.finfo(dtype).eps) * 100.0
    return base * max(1.0, math.sqrt(float(dim))) * multiplier


def _finite_grad_residual(tensors: Iterable[torch.Tensor]) -> float:
    for tensor in tensors:
        if tensor.grad is None:
            return float("inf")
        if not torch.isfinite(tensor.grad.detach().float()).all().item():
            return float("inf")
    return 0.0


def _record_stability(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    dtype_name: str,
    signature: SignatureSpec,
    dim: int,
    case: str,
    residual: float,
    tolerance: float,
    elapsed_ms: float,
    note: str = "",
) -> None:
    passed = math.isfinite(residual) and residual <= tolerance
    rows.append(
        {
            "suite": "stability",
            "device": args.device,
            "dtype": dtype_name,
            "n": signature.n,
            "dim": dim,
            "p": signature.p,
            "q": signature.q,
            "r": signature.r,
            "signature": signature.label,
            "case": case,
            "residual": residual,
            "tolerance": tolerance,
            "residual_to_tolerance": (
                residual / tolerance
                if tolerance > 0
                else (0.0 if residual == 0.0 else float("inf"))
            ),
            "passed": passed,
            "sample_batch": args.stability_batch,
            "elapsed_ms": elapsed_ms,
            "status": "ok",
            "error": "",
            "note": note,
        }
    )


def _record_stability_failure(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    dtype_name: str,
    signature: SignatureSpec,
    case: str,
    exc: Exception,
) -> None:
    rows.append(
        {
            "suite": "stability",
            "device": args.device,
            "dtype": dtype_name,
            "n": signature.n,
            "dim": 2**signature.n,
            "p": signature.p,
            "q": signature.q,
            "r": signature.r,
            "signature": signature.label,
            "case": case,
            "residual": float("nan"),
            "tolerance": float("nan"),
            "residual_to_tolerance": float("nan"),
            "passed": False,
            "sample_batch": args.stability_batch,
            "elapsed_ms": float("nan"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
            "note": "",
        }
    )


def _basis_bivector_index(i: int, j: int) -> int:
    return (1 << i) | (1 << j)


def _commuting_pairs(n: int) -> list[tuple[int, int]]:
    pairs = [(0, 1), (2, 3)]
    if n >= 6:
        pairs.append((4, 5))
    if n >= 8:
        pairs.append((6, 7))
    return pairs


def _make_commuting_nonsimple_bivector(
    algebra: AlgebraLike,
    batch: int,
    scale: float,
) -> torch.Tensor:
    if algebra.n < 4:
        raise ValueError("non-simple bivectors require n >= 4")
    base_coeffs = [0.37, -0.23, 0.13, -0.07]
    pairs = _commuting_pairs(algebra.n)
    multipliers = torch.linspace(
        0.85,
        1.15,
        batch,
        device=algebra.device,
        dtype=algebra.dtype,
    )
    b = torch.zeros(batch, algebra.dim, device=algebra.device, dtype=algebra.dtype)
    for coeff, (i, j) in zip(base_coeffs, pairs):
        b[:, _basis_bivector_index(i, j)] = scale * coeff * multipliers
    return b


def _exact_commuting_exp(algebra: AlgebraLike, b: torch.Tensor) -> torch.Tensor:
    """Exact exp for the controlled non-simple bivector used here.

    The bivector is a sum of disjoint coordinate-plane bivectors. Those
    components commute, so exp(sum B_i) = product exp(B_i).
    """
    result = torch.zeros_like(b)
    result[..., 0] = 1.0
    for i, j in _commuting_pairs(algebra.n):
        idx = _basis_bivector_index(i, j)
        component = torch.zeros_like(b)
        component[..., idx] = b[..., idx]
        result = algebra.geometric_product(result, algebra._exp_bivector_closed(component))
    return result


def _grade_leak(algebra: AlgebraLike, x: torch.Tensor, grade: int) -> float:
    mask = algebra.grade_masks_float[grade]
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    leak = x * (1.0 - mask)
    return float(leak.detach().float().norm().item())


def _make_speed_inputs(
    algebra: AlgebraLike,
    op: str,
    batch: int,
    channels: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, ...]:
    _seed_all(seed, device)
    dim = algebra.dim
    if op in {"gp", "wedge", "inner", "commutator"}:
        a = torch.randn(batch, dim, device=device, dtype=algebra.dtype)
        b = torch.randn(batch, dim, device=device, dtype=algebra.dtype)
        return (a, b)
    if op in {"grade2", "reverse", "norm_sq"}:
        a = torch.randn(batch, dim, device=device, dtype=algebra.dtype)
        return (a,)
    if op in {"exp", "exp_precise"}:
        b = torch.randn(batch, dim, device=device, dtype=algebra.dtype)
        b = algebra.grade_projection(b, 2) * 0.1
        return (b,)
    if op == "sandwich":
        b = torch.randn(batch, dim, device=device, dtype=algebra.dtype)
        b = algebra.grade_projection(b, 2) * 0.05
        rotor = algebra._exp_bivector_closed(-0.5 * b).detach()
        x = torch.randn(batch, channels, dim, device=device, dtype=algebra.dtype)
        return (rotor, x)
    raise ValueError(f"unsupported op: {op}")


def _random_multivector(
    algebra: AlgebraLike,
    shape: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    return (
        torch.randn(*shape, algebra.dim, device=algebra.device, dtype=algebra.dtype)
        * scale
    )


def _random_grade(
    algebra: AlgebraLike,
    shape: tuple[int, ...],
    grade: int,
    scale: float,
) -> torch.Tensor:
    return algebra.grade_projection(_random_multivector(algebra, shape, scale), grade)


def run_stability_suite(args: argparse.Namespace, dtypes: list[torch.dtype]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signatures = _parse_signature_csv(args.stability_signatures)
    cases = _parse_csv(args.stability_cases)
    valid_cases = {
        "associativity",
        "reverse_involution",
        "grade_projection",
        "rotor_unitarity",
        "sandwich_consistency",
        "per_channel_sandwich",
        "exp_policy_consistency",
        "large_angle_unitarity",
        "grad_finite",
        "compile_parity",
    }
    unknown = sorted(set(cases) - valid_cases)
    if unknown:
        raise ValueError(f"unknown stability case(s): {unknown}; valid: {sorted(valid_cases)}")

    for dtype in dtypes:
        dtype_name = _dtype_name(dtype)
        for sig_index, signature in enumerate(signatures):
            _release_memory(args.device)
            try:
                algebra = setup_algebra(
                    signature.p,
                    signature.q,
                    signature.r,
                    device=args.device,
                    dtype=dtype,
                    exp_policy=ExpPolicy.PRECISE,
                    args=args,
                )
            except Exception as exc:
                _record_stability_failure(rows, args, dtype_name, signature, "setup", exc)
                continue

            dim = algebra.dim
            seed_base = (
                args.seed
                + signature.p * 1009
                + signature.q * 211
                + signature.r * 37
                + sig_index * 17
            )

            def run_case(case: str, fn: Callable[[], tuple[float, float, str]]) -> None:
                if case not in cases:
                    return
                _seed_all(seed_base + 13 * (cases.index(case) + 1), args.device)
                _sync(args.device)
                start = time.perf_counter_ns()
                try:
                    residual, tolerance, note = fn()
                    _sync(args.device)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    _record_stability(
                        rows,
                        args,
                        dtype_name,
                        signature,
                        dim,
                        case,
                        residual,
                        tolerance,
                        elapsed_ms,
                        note,
                    )
                    mark = "PASS" if rows[-1]["passed"] else "FAIL"
                    print(
                        f"stability {mark:<4s} {args.device:>4s} {dtype_name:>8s} "
                        f"{signature.label:<10s} {case:<24s} "
                        f"res={residual:.2e} tol={tolerance:.2e}"
                    )
                except Exception as exc:
                    _record_stability_failure(rows, args, dtype_name, signature, case, exc)
                    print(
                        f"stability failed {args.device} {dtype_name} "
                        f"{signature.label} {case}: {exc}"
                    )

            def associativity() -> tuple[float, float, str]:
                with torch.no_grad():
                    a = _random_multivector(
                        algebra,
                        (args.stability_batch,),
                        args.stability_input_scale,
                    )
                    b = _random_multivector(
                        algebra,
                        (args.stability_batch,),
                        args.stability_input_scale,
                    )
                    c = _random_multivector(
                        algebra,
                        (args.stability_batch,),
                        args.stability_input_scale,
                    )
                    left = algebra.geometric_product(algebra.geometric_product(a, b), c)
                    right = algebra.geometric_product(a, algebra.geometric_product(b, c))
                return (
                    _max_abs_diff(left, right),
                    _stability_tolerance(dtype, dim, 50.0),
                    "dense multivector associativity",
                )

            def reverse_involution() -> tuple[float, float, str]:
                with torch.no_grad():
                    a = _random_multivector(
                        algebra,
                        (args.stability_batch,),
                        args.stability_input_scale,
                    )
                    residual = _max_abs_diff(algebra.reverse(algebra.reverse(a)), a)
                return (
                    residual,
                    _stability_tolerance(dtype, dim, 0.25),
                    "reverse(reverse(x)) equals x",
                )

            def grade_projection() -> tuple[float, float, str]:
                with torch.no_grad():
                    a = _random_multivector(
                        algebra,
                        (args.stability_batch,),
                        args.stability_input_scale,
                    )
                    residual = 0.0
                    for grade in range(algebra.num_grades):
                        projected = algebra.grade_projection(a, grade)
                        reprojection = algebra.grade_projection(projected, grade)
                        mask = algebra.grade_masks_float[grade]
                        if mask.dtype != projected.dtype:
                            mask = mask.to(dtype=projected.dtype)
                        residual = max(
                            residual,
                            _max_abs_diff(projected, reprojection),
                            _tensor_max_abs(projected * (1.0 - mask)),
                        )
                return (
                    residual,
                    _stability_tolerance(dtype, dim, 0.25),
                    "projection idempotence and off-grade leak",
                )

            def rotor_unitarity() -> tuple[float, float, str]:
                if algebra.n < 2:
                    return 0.0, 0.0, "skipped: no bivectors"
                with torch.no_grad():
                    b = _random_grade(
                        algebra,
                        (args.stability_batch,),
                        2,
                        args.stability_bivector_scale,
                    )
                    rotor = algebra.exp(-0.5 * b)
                    unit = algebra.geometric_product(rotor, algebra.reverse(rotor))
                    identity = torch.zeros_like(unit)
                    identity[..., 0] = 1.0
                return (
                    _max_abs_diff(unit, identity),
                    _stability_tolerance(dtype, dim, 200.0),
                    "exp(bivector) rotor unitarity",
                )

            def sandwich_consistency() -> tuple[float, float, str]:
                if algebra.n < 2:
                    return 0.0, 0.0, "skipped: no bivectors"
                with torch.no_grad():
                    b = _random_grade(
                        algebra,
                        (args.stability_batch,),
                        2,
                        args.stability_bivector_scale,
                    )
                    rotor = algebra.exp(-0.5 * b).detach()
                    x = _random_multivector(
                        algebra,
                        (args.stability_batch, args.channels),
                        args.stability_input_scale,
                    )
                    sandwich = algebra.sandwich_product(rotor, x)
                    naive = algebra.geometric_product(
                        algebra.geometric_product(rotor.unsqueeze(1), x),
                        algebra.reverse(rotor).unsqueeze(1),
                    )
                return (
                    _max_abs_diff(sandwich, naive),
                    _stability_tolerance(dtype, dim, 50.0),
                    "optimized sandwich equals two geometric products",
                )

            def per_channel_sandwich() -> tuple[float, float, str]:
                if algebra.n < 2:
                    return 0.0, 0.0, "skipped: no bivectors"
                channels = max(args.channels, 1)
                with torch.no_grad():
                    b = _random_grade(
                        algebra,
                        (channels,),
                        2,
                        args.stability_bivector_scale,
                    )
                    rotors = algebra.exp(-0.5 * b).detach()
                    x = _random_multivector(
                        algebra,
                        (args.stability_batch, channels),
                        args.stability_input_scale,
                    )
                    per_channel = algebra.per_channel_sandwich(rotors, x)
                    loop = torch.zeros_like(per_channel)
                    for channel in range(channels):
                        rotor = rotors[channel : channel + 1].expand(
                            args.stability_batch,
                            -1,
                        )
                        loop[:, channel : channel + 1, :] = algebra.sandwich_product(
                            rotor,
                            x[:, channel : channel + 1, :],
                        )
                return (
                    _max_abs_diff(per_channel, loop),
                    _stability_tolerance(dtype, dim, 50.0),
                    "per-channel fast path equals channel loop",
                )

            def exp_policy_consistency() -> tuple[float, float, str]:
                if algebra.n < 2:
                    return 0.0, 0.0, "skipped: no bivectors"
                balanced = setup_algebra(
                    signature.p,
                    signature.q,
                    signature.r,
                    device=args.device,
                    dtype=dtype,
                    exp_policy=ExpPolicy.BALANCED,
                    args=args,
                )
                with torch.no_grad():
                    b = torch.zeros(
                        args.stability_batch,
                        dim,
                        device=args.device,
                        dtype=dtype,
                    )
                    values = torch.linspace(
                        -0.7,
                        0.7,
                        args.stability_batch,
                        device=args.device,
                        dtype=dtype,
                    )
                    b[:, _basis_bivector_index(0, 1)] = values
                    balanced_out = balanced.exp(b)
                    precise_out = algebra.exp(b)
                return (
                    _max_abs_diff(balanced_out, precise_out),
                    _stability_tolerance(dtype, dim, 20.0),
                    "balanced and precise policies agree on simple bivectors",
                )

            def large_angle_unitarity() -> tuple[float, float, str]:
                if algebra.n < 2:
                    return 0.0, 0.0, "skipped: no bivectors"
                with torch.no_grad():
                    b = torch.zeros(
                        args.stability_batch,
                        dim,
                        device=args.device,
                        dtype=dtype,
                    )
                    b[:, _basis_bivector_index(0, 1)] = args.stability_large_angle
                    rotor = algebra.exp(-0.5 * b)
                    unit = algebra.geometric_product(rotor, algebra.reverse(rotor))
                    identity = torch.zeros_like(unit)
                    identity[..., 0] = 1.0
                return (
                    _max_abs_diff(unit, identity),
                    _stability_tolerance(dtype, dim, 500.0),
                    f"simple bivector angle={args.stability_large_angle:g}",
                )

            def grad_finite() -> tuple[float, float, str]:
                a = _random_multivector(
                    algebra,
                    (args.stability_batch,),
                    args.stability_input_scale,
                ).requires_grad_(True)
                b = _random_multivector(
                    algebra,
                    (args.stability_batch,),
                    args.stability_input_scale,
                ).requires_grad_(True)
                out = algebra.geometric_product(a, b)
                _loss_from_output(out).backward()
                residual = _finite_grad_residual((a, b))
                _zero_input_grads((a, b))
                return residual, 0.0, "geometric product gradient finiteness"

            def compile_parity() -> tuple[float, float, str]:
                if not hasattr(torch, "compile"):
                    return 0.0, 0.0, "skipped: torch.compile unavailable"
                with torch.no_grad():
                    module = CoreOpModule(algebra, "gp").eval()
                    compiled = _compile_module(module, args.stability_compile_mode)
                    inputs = _make_speed_inputs(
                        algebra,
                        "gp",
                        args.stability_batch,
                        args.channels,
                        seed_base + 97,
                        args.device,
                    )
                    eager_out = module(*inputs)
                    compiled_out = compiled(*inputs)
                return (
                    _max_abs_diff(compiled_out, eager_out),
                    _stability_tolerance(dtype, dim, 20.0),
                    f"torch.compile parity via {args.stability_compile_mode}",
                )

            case_fns: dict[str, Callable[[], tuple[float, float, str]]] = {
                "associativity": associativity,
                "reverse_involution": reverse_involution,
                "grade_projection": grade_projection,
                "rotor_unitarity": rotor_unitarity,
                "sandwich_consistency": sandwich_consistency,
                "per_channel_sandwich": per_channel_sandwich,
                "exp_policy_consistency": exp_policy_consistency,
                "large_angle_unitarity": large_angle_unitarity,
                "grad_finite": grad_finite,
                "compile_parity": compile_parity,
            }
            for case in cases:
                run_case(case, case_fns[case])

            algebra = None
            _release_memory(args.device)
    return rows


def _compile_module(module: nn.Module, mode: str) -> nn.Module:
    if mode == "eager":
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available")
    if mode == "aot_eager":
        return torch.compile(module, backend="aot_eager")
    if mode == "compile":
        return torch.compile(module)
    if mode == "reduce-overhead":
        return torch.compile(module, mode="reduce-overhead")
    if mode == "max-autotune":
        return torch.compile(module, mode="max-autotune")
    raise ValueError(f"unknown compile mode: {mode}")


def _count_graph_breaks(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> int:
    """Return the dynamo graph-break count for ``module(*inputs)``.

    ``torch.compile`` silently falls back to eager on graph break, so a
    "compile" timing without this check can be just an eager run with a
    different name. Returns -1 if dynamo doesn't expose the count.
    """
    try:
        import torch._dynamo as dynamo
    except ImportError:
        return -1
    try:
        explained = dynamo.explain(module)(*inputs)
    except TypeError:
        try:
            explained = dynamo.explain(module, *inputs)
        except Exception:
            return -1
    except Exception:
        return -1
    for attr in ("graph_break_count", "break_reasons"):
        value = getattr(explained, attr, None)
        if value is None:
            continue
        if attr == "graph_break_count":
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        if attr == "break_reasons":
            try:
                return int(len(value))
            except TypeError:
                continue
    return -1


def run_speed_suite(args: argparse.Namespace, dtypes: list[torch.dtype]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = _ordered_modes(args.compile_modes)
    ops = _parse_csv(args.ops)

    for dtype in dtypes:
        dtype_name = _dtype_name(dtype)
        tf32_modes = _tf32_modes_for_dtype(args, dtype)
        for n in args.n_values:
            if n < 2:
                print(f"Skipping n={n}: benchmark ops require n >= 2")
                continue
            for batch in args.batch_sizes:
                for op in ops:
                    for channels in _channels_for_op(args, op):
                        for tf32_mode in tf32_modes:
                            baseline_output: torch.Tensor | None = None
                            baseline_stats: TimerStats | None = None
                            for mode in modes:
                                _release_memory(args.device)
                                row: dict[str, Any] = {
                                    "suite": "speed",
                                    "device": args.device,
                                    "dtype": dtype_name,
                                    "n": n,
                                    "dim": 2**n,
                                    "batch": batch,
                                    "channels": channels,
                                    "op": op,
                                    "compile_mode": mode,
                                    "tf32_mode": tf32_mode,
                                    "status": "ok",
                                    "error": "",
                                }
                                try:
                                    with _tf32_context(args.device, tf32_mode):
                                        row.update(_current_tf32_flags(args.device, tf32_mode))
                                        algebra = setup_algebra(
                                            n,
                                            0,
                                            device=args.device,
                                            dtype=dtype,
                                            exp_policy=_op_exp_policy(op),
                                            args=args,
                                        )
                                        row.update(
                                            _analytic_forward_metrics(
                                                algebra,
                                                op,
                                                batch,
                                                channels,
                                            )
                                        )
                                        module = CoreOpModule(algebra, op).eval()
                                        inputs = _make_speed_inputs(
                                            algebra,
                                            op,
                                            batch,
                                            channels,
                                            args.seed + n * 1009 + batch * 17 + channels,
                                            args.device,
                                        )
                                        compiled = _compile_module(module, mode)
                                        graph_breaks = (
                                            -1 if mode == "eager"
                                            else _count_graph_breaks(module, inputs)
                                        )
                                        stats, out = _time_callable(
                                            compiled,
                                            inputs,
                                            args.device,
                                            args.warmup,
                                            args.runs,
                                            args.min_time_ms,
                                            args.max_runs,
                                        )
                                    if mode == "eager":
                                        baseline_output = out.detach().cpu()
                                        baseline_stats = stats
                                        err = {
                                            "max_abs_error": 0.0,
                                            "rms_error": 0.0,
                                            "max_rel_error": 0.0,
                                        }
                                    elif baseline_output is not None:
                                        err = _error_stats(out, baseline_output)
                                    else:
                                        err = {
                                            "max_abs_error": float("nan"),
                                            "rms_error": float("nan"),
                                            "max_rel_error": float("nan"),
                                        }
                                    row.update(
                                        {
                                            "first_call_ms": stats.first_call_ms,
                                            "median_ms": stats.median_ms,
                                            "mean_ms": stats.mean_ms,
                                            "std_ms": stats.std_ms,
                                            "p10_ms": stats.p10_ms,
                                            "p90_ms": stats.p90_ms,
                                            "timed_runs": stats.runs,
                                            "graph_breaks": graph_breaks,
                                            **_speedup_fields(
                                                None if mode == "eager" else baseline_stats,
                                                stats,
                                                args.bootstrap_samples,
                                                args.seed + n * 4099 + batch * 97 + channels * 13,
                                            ),
                                            **err,
                                        }
                                    )
                                    _add_rate_metrics(row, stats.median_ms)
                                    channel_suffix = (
                                        f" c={channels:<4d}" if op == "sandwich" else ""
                                    )
                                    tf32_suffix = (
                                        f" tf32={row['tf32_mode']:<7s}"
                                        if row["tf32_mode"] != "default"
                                        else ""
                                    )
                                    print(
                                        f"speed {args.device:>4s} {dtype_name:>8s} "
                                        f"n={n:<2d} b={batch:<4d}{channel_suffix} "
                                        f"{op:<11s} {mode:<15s}{tf32_suffix} "
                                        f"{stats.median_ms:9.4f} ms "
                                        f"err={row['max_abs_error']:.2e} "
                                        f"breaks={graph_breaks}"
                                    )
                                except Exception as exc:
                                    row.update(
                                        {
                                            "status": "failed",
                                            "error": f"{type(exc).__name__}: {exc}",
                                            "traceback": traceback.format_exc(limit=4),
                                            "first_call_ms": float("nan"),
                                            "median_ms": float("nan"),
                                            "mean_ms": float("nan"),
                                            "std_ms": float("nan"),
                                            "p10_ms": float("nan"),
                                            "p90_ms": float("nan"),
                                            "timed_runs": 0,
                                            "logical_items_per_call": float("nan"),
                                            "items_per_sec": float("nan"),
                                            "analytic_forward_flops": float("nan"),
                                            "achieved_forward_gflops": float("nan"),
                                            "analytic_io_bytes": float("nan"),
                                            "achieved_io_gbps": float("nan"),
                                            "graph_breaks": float("nan"),
                                            "speedup_vs_eager": float("nan"),
                                            "speedup_vs_eager_ci_low": float("nan"),
                                            "speedup_vs_eager_ci_high": float("nan"),
                                            "speedup_vs_eager_ci_excludes_1": False,
                                            "speedup_vs_eager_significant": False,
                                            "max_abs_error": float("nan"),
                                            "rms_error": float("nan"),
                                            "max_rel_error": float("nan"),
                                        }
                                    )
                                    print(
                                        f"speed failed {args.device} {dtype_name} "
                                        f"n={n} b={batch} c={channels} {op} {mode}: {exc}"
                                    )
                                finally:
                                    rows.append(row)
                                    try:
                                        del algebra, module, compiled, inputs, out
                                    except UnboundLocalError:
                                        pass
                                    _release_memory(args.device)
    return rows


def run_backward_suite(args: argparse.Namespace, dtypes: list[torch.dtype]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = _ordered_modes(args.backward_compile_modes)
    ops = _parse_csv(args.backward_ops)

    for dtype in dtypes:
        dtype_name = _dtype_name(dtype)
        tf32_modes = _tf32_modes_for_dtype(args, dtype)
        for n in args.backward_n_values:
            if n < 2:
                print(f"Skipping backward n={n}: benchmark ops require n >= 2")
                continue
            for batch in args.backward_batch_sizes:
                for op in ops:
                    for channels in _channels_for_op(args, op):
                        for tf32_mode in tf32_modes:
                            baseline_output: torch.Tensor | None = None
                            baseline_stats: TimerStats | None = None
                            for mode in modes:
                                _release_memory(args.device)
                                row: dict[str, Any] = {
                                    "suite": "backward",
                                    "device": args.device,
                                    "dtype": dtype_name,
                                    "n": n,
                                    "dim": 2**n,
                                    "batch": batch,
                                    "channels": channels,
                                    "op": op,
                                    "compile_mode": mode,
                                    "tf32_mode": tf32_mode,
                                    "status": "ok",
                                    "error": "",
                                }
                                try:
                                    with _tf32_context(args.device, tf32_mode):
                                        row.update(_current_tf32_flags(args.device, tf32_mode))
                                        algebra = setup_algebra(
                                            n,
                                            0,
                                            device=args.device,
                                            dtype=dtype,
                                            exp_policy=_op_exp_policy(op),
                                            args=args,
                                        )
                                        row.update(
                                            _analytic_forward_metrics(
                                                algebra,
                                                op,
                                                batch,
                                                channels,
                                            )
                                        )
                                        module = CoreOpModule(algebra, op).eval()
                                        inputs = _make_speed_inputs(
                                            algebra,
                                            op,
                                            batch,
                                            channels,
                                            args.seed + n * 1709 + batch * 31 + channels,
                                            args.device,
                                        )
                                        compiled = _compile_module(module, mode)
                                        graph_breaks = (
                                            -1 if mode == "eager"
                                            else _count_graph_breaks(module, inputs)
                                        )
                                        stats, out, grad_norm = _time_forward_backward(
                                            compiled,
                                            inputs,
                                            args.device,
                                            args.backward_warmup,
                                            args.backward_runs,
                                            args.backward_min_time_ms,
                                            args.backward_max_runs,
                                        )
                                    if mode == "eager":
                                        baseline_output = out.detach().cpu()
                                        baseline_stats = stats
                                        err = {
                                            "max_abs_error": 0.0,
                                            "rms_error": 0.0,
                                            "max_rel_error": 0.0,
                                        }
                                    elif baseline_output is not None:
                                        err = _error_stats(out, baseline_output)
                                    else:
                                        err = {
                                            "max_abs_error": float("nan"),
                                            "rms_error": float("nan"),
                                            "max_rel_error": float("nan"),
                                        }
                                    row.update(
                                        {
                                            "first_call_ms": stats.first_call_ms,
                                            "median_ms": stats.median_ms,
                                            "mean_ms": stats.mean_ms,
                                            "std_ms": stats.std_ms,
                                            "p10_ms": stats.p10_ms,
                                            "p90_ms": stats.p90_ms,
                                            "timed_runs": stats.runs,
                                            "graph_breaks": graph_breaks,
                                            "grad_norm": grad_norm,
                                            **_speedup_fields(
                                                None if mode == "eager" else baseline_stats,
                                                stats,
                                                args.bootstrap_samples,
                                                args.seed + n * 5101 + batch * 113 + channels * 17,
                                            ),
                                            **err,
                                        }
                                    )
                                    _add_rate_metrics(row, stats.median_ms)
                                    channel_suffix = (
                                        f" c={channels:<4d}" if op == "sandwich" else ""
                                    )
                                    tf32_suffix = (
                                        f" tf32={row['tf32_mode']:<7s}"
                                        if row["tf32_mode"] != "default"
                                        else ""
                                    )
                                    print(
                                        f"backward {args.device:>4s} {dtype_name:>8s} "
                                        f"n={n:<2d} b={batch:<4d}{channel_suffix} "
                                        f"{op:<11s} {mode:<15s}{tf32_suffix} "
                                        f"{stats.median_ms:9.4f} ms grad={grad_norm:.2e} "
                                        f"breaks={graph_breaks}"
                                    )
                                except Exception as exc:
                                    row.update(
                                        {
                                            "status": "failed",
                                            "error": f"{type(exc).__name__}: {exc}",
                                            "traceback": traceback.format_exc(limit=4),
                                            "first_call_ms": float("nan"),
                                            "median_ms": float("nan"),
                                            "mean_ms": float("nan"),
                                            "std_ms": float("nan"),
                                            "p10_ms": float("nan"),
                                            "p90_ms": float("nan"),
                                            "timed_runs": 0,
                                            "logical_items_per_call": float("nan"),
                                            "items_per_sec": float("nan"),
                                            "analytic_forward_flops": float("nan"),
                                            "achieved_forward_gflops": float("nan"),
                                            "analytic_io_bytes": float("nan"),
                                            "achieved_io_gbps": float("nan"),
                                            "graph_breaks": float("nan"),
                                            "speedup_vs_eager": float("nan"),
                                            "speedup_vs_eager_ci_low": float("nan"),
                                            "speedup_vs_eager_ci_high": float("nan"),
                                            "speedup_vs_eager_ci_excludes_1": False,
                                            "speedup_vs_eager_significant": False,
                                            "grad_norm": float("nan"),
                                            "max_abs_error": float("nan"),
                                            "rms_error": float("nan"),
                                            "max_rel_error": float("nan"),
                                        }
                                    )
                                    print(
                                        f"backward failed {args.device} {dtype_name} "
                                        f"n={n} b={batch} c={channels} {op} {mode}: {exc}"
                                    )
                                finally:
                                    rows.append(row)
                                    try:
                                        del algebra, module, compiled, inputs, out
                                    except UnboundLocalError:
                                        pass
                                    _release_memory(args.device)
    return rows


def run_fusion_suite(
    args: argparse.Namespace,
    dtypes: list[torch.dtype],
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = _ordered_modes(args.profiler_compile_modes)
    ops = _parse_csv(args.profiler_ops)
    phases = _parse_csv(args.profiler_phases)
    baseline: dict[tuple[str, int, int, int, str, str, str], dict[str, float]] = {}

    for dtype in dtypes:
        dtype_name = _dtype_name(dtype)
        tf32_modes = _tf32_modes_for_dtype(args, dtype)
        for n in args.profiler_n_values:
            if n < 2:
                print(f"Skipping profiler n={n}: benchmark ops require n >= 2")
                continue
            batch = args.profiler_batch_size
            for op in ops:
                for channels in _channels_for_op(args, op):
                    for tf32_mode in tf32_modes:
                        for phase in phases:
                            for mode in modes:
                                _release_memory(args.device)
                                row: dict[str, Any] = {
                                    "suite": "fusion",
                                    "device": args.device,
                                    "dtype": dtype_name,
                                    "n": n,
                                    "dim": 2**n,
                                    "batch": batch,
                                    "channels": channels,
                                    "op": op,
                                    "phase": phase,
                                    "compile_mode": mode,
                                    "tf32_mode": tf32_mode,
                                    "profiler_steps": args.profiler_steps,
                                    "status": "ok",
                                    "error": "",
                                    "trace_path": "",
                                }
                                try:
                                    with _tf32_context(args.device, tf32_mode):
                                        row.update(_current_tf32_flags(args.device, tf32_mode))
                                        algebra = setup_algebra(
                                            n,
                                            0,
                                            device=args.device,
                                            dtype=dtype,
                                            exp_policy=_op_exp_policy(op),
                                            args=args,
                                        )
                                        row.update(
                                            _analytic_forward_metrics(
                                                algebra,
                                                op,
                                                batch,
                                                channels,
                                            )
                                        )
                                        module = CoreOpModule(algebra, op).eval()
                                        inputs = _make_speed_inputs(
                                            algebra,
                                            op,
                                            batch,
                                            channels,
                                            args.seed + n * 2029 + batch * 43 + channels,
                                            args.device,
                                        )
                                        compiled = _compile_module(module, mode)
                                        graph_breaks = (
                                            -1 if mode == "eager"
                                            else _count_graph_breaks(module, inputs)
                                        )
                                        trace_path = None
                                        if args.export_profiler_traces:
                                            safe = (
                                                f"{dtype_name}_n{n}_b{batch}_c{channels}_"
                                                f"{op}_{phase}_{tf32_mode}_{mode}.json"
                                            )
                                            trace_path = out_dir / "traces" / safe
                                            row["trace_path"] = str(trace_path)
                                        metrics = _profile_module(
                                            compiled,
                                            inputs,
                                            phase,
                                            args.device,
                                            args.profiler_warmup,
                                            args.profiler_steps,
                                            trace_path,
                                        )
                                    key = (dtype_name, n, batch, channels, op, phase, tf32_mode)
                                    row.update(metrics)
                                    row.update(
                                        {
                                            "graph_breaks": graph_breaks,
                                            "aten_reduction_vs_eager": float("nan"),
                                            "kernel_reduction_vs_eager": float("nan"),
                                            "cpu_time_speedup_vs_eager": float("nan"),
                                            "device_time_speedup_vs_eager": float("nan"),
                                        }
                                    )
                                    if mode == "eager":
                                        baseline[key] = metrics
                                    elif key in baseline:
                                        eager = baseline[key]
                                        aten = float(metrics["aten_ops_per_step"])
                                        kernels = float(metrics["kernel_events_per_step"])
                                        cpu_time = float(metrics["profiler_self_cpu_time_ms_per_step"])
                                        device_time = float(metrics["profiler_self_device_time_ms_per_step"])
                                        eager_aten = float(eager["aten_ops_per_step"])
                                        eager_kernels = float(eager["kernel_events_per_step"])
                                        eager_cpu = float(eager["profiler_self_cpu_time_ms_per_step"])
                                        eager_device = float(eager["profiler_self_device_time_ms_per_step"])
                                        row["aten_reduction_vs_eager"] = (
                                            eager_aten / aten if aten > 0 else float("nan")
                                        )
                                        row["kernel_reduction_vs_eager"] = (
                                            eager_kernels / kernels if kernels > 0 else float("nan")
                                        )
                                        row["cpu_time_speedup_vs_eager"] = (
                                            eager_cpu / cpu_time if cpu_time > 0 else float("nan")
                                        )
                                        row["device_time_speedup_vs_eager"] = (
                                            eager_device / device_time if device_time > 0 else float("nan")
                                        )
                                    print(
                                        f"fusion {args.device:>4s} {dtype_name:>8s} "
                                        f"n={n:<2d} b={batch:<4d} "
                                        f"{'c=' + str(channels):<7s} {op:<11s} "
                                        f"{phase:<8s} {mode:<15s} "
                                        f"tf32={row['tf32_mode']:<7s} "
                                        f"aten/step={row['aten_ops_per_step']:.1f} "
                                        f"kernels/step={row['kernel_events_per_step']:.1f} "
                                        f"peak_delta={row['peak_delta_mb']:.2f} MB"
                                    )
                                except Exception as exc:
                                    row.update(
                                        {
                                            "status": "failed",
                                            "error": f"{type(exc).__name__}: {exc}",
                                            "traceback": traceback.format_exc(limit=4),
                                            "profiler_total_events": float("nan"),
                                            "profiler_total_events_per_step": float("nan"),
                                            "aten_ops": float("nan"),
                                            "aten_ops_per_step": float("nan"),
                                            "compiled_region_events": float("nan"),
                                            "compiled_region_events_per_step": float("nan"),
                                            "kernel_events": float("nan"),
                                            "kernel_events_per_step": float("nan"),
                                            "triton_kernel_events": float("nan"),
                                            "triton_kernel_events_per_step": float("nan"),
                                            "cuda_runtime_events": float("nan"),
                                            "cuda_runtime_events_per_step": float("nan"),
                                            "profiler_self_cpu_time_ms": float("nan"),
                                            "profiler_self_cpu_time_ms_per_step": float("nan"),
                                            "profiler_self_device_time_ms": float("nan"),
                                            "profiler_self_device_time_ms_per_step": float("nan"),
                                            "profiler_positive_self_cpu_memory_mb": float("nan"),
                                            "profiler_positive_self_device_memory_mb": float("nan"),
                                            "allocated_before_mb": float("nan"),
                                            "allocated_after_mb": float("nan"),
                                            "peak_allocated_mb": float("nan"),
                                            "peak_delta_mb": float("nan"),
                                            "allocated_delta_mb": float("nan"),
                                            "logical_items_per_call": float("nan"),
                                            "analytic_forward_flops": float("nan"),
                                            "analytic_io_bytes": float("nan"),
                                            "graph_breaks": float("nan"),
                                            "aten_reduction_vs_eager": float("nan"),
                                            "kernel_reduction_vs_eager": float("nan"),
                                            "cpu_time_speedup_vs_eager": float("nan"),
                                            "device_time_speedup_vs_eager": float("nan"),
                                        }
                                    )
                                    print(
                                        f"fusion failed {args.device} {dtype_name} "
                                        f"n={n} b={batch} c={channels} {op} {phase} {mode}: {exc}"
                                    )
                                finally:
                                    rows.append(row)
                                    try:
                                        del algebra, module, compiled, inputs
                                    except UnboundLocalError:
                                        pass
                                    _release_memory(args.device)
    return rows


def run_nonsimple_suite(args: argparse.Namespace, dtypes: list[torch.dtype]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = ["balanced", "precise"]
    for n in args.nonsimple_n_values:
        if n < 4:
            continue
        ref_alg = setup_algebra(n, 0, device="cpu", dtype=torch.float64, exp_policy="precise", args=args)
        b_ref = _make_commuting_nonsimple_bivector(ref_alg, args.error_batch, args.bivector_scale)
        r_ref = _exact_commuting_exp(ref_alg, b_ref)
        bb_ref = ref_alg.geometric_product(b_ref, b_ref)
        scalar_ref = ref_alg.grade_projection(bb_ref, 0)
        nonscalar_norm = float((bb_ref - scalar_ref).norm(dim=-1).max().item())

        for dtype in dtypes:
            dtype_name = _dtype_name(dtype)
            for policy in policies:
                _release_memory(args.device)
                row: dict[str, Any] = {
                    "suite": "nonsimple_exp",
                    "device": args.device,
                    "dtype": dtype_name,
                    "n": n,
                    "dim": 2**n,
                    "batch": args.error_batch,
                    "policy": policy,
                    "status": "ok",
                    "error": "",
                    "nonscalar_BB_norm": nonscalar_norm,
                }
                try:
                    algebra = setup_algebra(
                        n,
                        0,
                        device=args.device,
                        dtype=dtype,
                        exp_policy=policy,
                        args=args,
                    )
                    b = _make_commuting_nonsimple_bivector(
                        algebra,
                        args.error_batch,
                        args.bivector_scale,
                    )
                    _seed_all(args.seed + n * 211 + len(policy), args.device)
                    stats, r = _time_callable(
                        algebra.exp,
                        (b,),
                        args.device,
                        max(1, args.warmup // 2),
                        max(1, min(args.runs, args.error_runs)),
                        args.min_time_ms,
                        args.max_runs,
                    )
                    unit = algebra.geometric_product(r, algebra.reverse(r))
                    identity = torch.zeros_like(unit)
                    identity[..., 0] = 1.0
                    row.update(
                        {
                            **_error_stats(r, r_ref),
                            "unitarity_error": _error_stats(unit, identity)["max_abs_error"],
                            "median_ms": stats.median_ms,
                            "p90_ms": stats.p90_ms,
                            "timed_runs": stats.runs,
                        }
                    )
                    print(
                        f"nonsimple {args.device:>4s} {dtype_name:>8s} "
                        f"n={n:<2d} {policy:<5s} err={row['max_abs_error']:.2e} "
                        f"unit={row['unitarity_error']:.2e}"
                    )
                except Exception as exc:
                    row.update(
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=4),
                            "max_abs_error": float("nan"),
                            "rms_error": float("nan"),
                            "max_rel_error": float("nan"),
                            "unitarity_error": float("nan"),
                            "median_ms": float("nan"),
                            "p90_ms": float("nan"),
                            "timed_runs": 0,
                        }
                    )
                    print(f"nonsimple failed {args.device} {dtype_name} n={n} {policy}: {exc}")
                finally:
                    rows.append(row)
                    algebra = None
                    b = None
                    r = None
                    unit = None
                    identity = None
                    _release_memory(args.device)
        del ref_alg, b_ref, r_ref, bb_ref, scalar_ref
    return rows


def run_cumulative_suite(args: argparse.Namespace, dtypes: list[torch.dtype]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steps_to_sample = set(_sample_steps(args.chain_steps, args.chain_samples))

    for n in args.cumulative_n_values:
        if n < 4:
            continue
        for batch in args.cumulative_batch_sizes:
            ref_alg = setup_algebra(n, 0, device="cpu", dtype=torch.float64, exp_policy="precise", args=args)
            b_ref = _make_commuting_nonsimple_bivector(
                ref_alg,
                batch,
                args.chain_bivector_scale,
            )
            x0_ref = torch.zeros(batch, 1, ref_alg.dim, dtype=torch.float64)
            x0_ref[..., 1] = 1.0

            for dtype in dtypes:
                dtype_name = _dtype_name(dtype)
                _release_memory(args.device)
                try:
                    algebra = setup_algebra(
                        n,
                        0,
                        device=args.device,
                        dtype=dtype,
                        exp_policy="balanced",
                        args=args,
                    )
                    b = _make_commuting_nonsimple_bivector(
                        algebra,
                        batch,
                        args.chain_bivector_scale,
                    )
                    rotor = _exact_commuting_exp(algebra, -0.5 * b).detach()
                    x = torch.zeros(batch, 1, algebra.dim, device=args.device, dtype=dtype)
                    x[..., 1] = 1.0
                    grade1_mask = algebra.grade_masks_float[1]
                    if grade1_mask.dtype != dtype:
                        grade1_mask = grade1_mask.to(dtype=dtype)

                    _sync(args.device)
                    start_ns = time.perf_counter_ns()
                    last: dict[str, Any] = {}
                    for step in range(1, args.chain_steps + 1):
                        x = algebra.sandwich_product(rotor, x)
                        if step not in steps_to_sample:
                            continue

                        step_ref_rotor = _exact_commuting_exp(ref_alg, -0.5 * float(step) * b_ref)
                        x_ref = ref_alg.sandwich_product(step_ref_rotor, x0_ref)
                        x_cpu = x.detach().cpu().to(torch.float64)
                        err = _error_stats(x_cpu, x_ref)
                        per_sample_norm = x.detach().float().norm(dim=-1)
                        norm = float(per_sample_norm.mean().item())
                        norm_drift = float((per_sample_norm - 1.0).abs().max().item())
                        grade_leak = float((x * (1.0 - grade1_mask)).detach().float().norm().item())
                        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                        row = {
                            "suite": "cumulative",
                            "device": args.device,
                            "dtype": dtype_name,
                            "n": n,
                            "dim": algebra.dim,
                            "batch": batch,
                            "step": step,
                            "chain_steps": args.chain_steps,
                            "norm": norm,
                            "norm_drift": norm_drift,
                            "grade_leak": grade_leak,
                            "elapsed_ms": elapsed_ms,
                            "status": "ok",
                            "error": "",
                            **err,
                        }
                        rows.append(row)
                        last = row
                    print(
                        f"cumulative {args.device:>4s} {dtype_name:>8s} "
                        f"n={n:<2d} b={batch:<4d} steps={args.chain_steps:<5d} "
                        f"err={last.get('max_abs_error', float('nan')):.2e} "
                        f"drift={last.get('norm_drift', float('nan')):.2e}"
                    )
                except Exception as exc:
                    row = {
                        "suite": "cumulative",
                        "device": args.device,
                        "dtype": dtype_name,
                        "n": n,
                        "dim": 2**n,
                        "batch": batch,
                        "step": 0,
                        "chain_steps": args.chain_steps,
                        "norm": float("nan"),
                        "norm_drift": float("nan"),
                        "grade_leak": float("nan"),
                        "elapsed_ms": float("nan"),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=4),
                        "max_abs_error": float("nan"),
                        "rms_error": float("nan"),
                        "max_rel_error": float("nan"),
                    }
                    rows.append(row)
                    print(f"cumulative failed {args.device} {dtype_name} n={n} b={batch}: {exc}")
                finally:
                    try:
                        del algebra, b, rotor, x, grade1_mask
                    except UnboundLocalError:
                        pass
                    _release_memory(args.device)
            del ref_alg, b_ref, x0_ref
    return rows


def run_convergence_suite(args: argparse.Namespace, dtypes: list[torch.dtype]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iterations = args.convergence_iters
    for n in args.convergence_n_values:
        if n < 4:
            continue
        ref_alg = setup_algebra(n, 0, device="cpu", dtype=torch.float64, exp_policy="precise", args=args)
        b_ref = _make_commuting_nonsimple_bivector(ref_alg, args.error_batch, args.bivector_scale)
        r_ref = _exact_commuting_exp(ref_alg, b_ref)

        for dtype in dtypes:
            dtype_name = _dtype_name(dtype)
            for fixed_iterations in iterations:
                _release_memory(args.device)
                row: dict[str, Any] = {
                    "suite": "convergence",
                    "device": args.device,
                    "dtype": dtype_name,
                    "n": n,
                    "dim": 2**n,
                    "batch": args.error_batch,
                    "fixed_iterations": fixed_iterations,
                    "status": "ok",
                    "error": "",
                }
                try:
                    algebra = setup_algebra(
                        n,
                        0,
                        device=args.device,
                        dtype=dtype,
                        exp_policy="precise",
                        args=args,
                    )
                    b = _make_commuting_nonsimple_bivector(
                        algebra,
                        args.error_batch,
                        args.bivector_scale,
                    )

                    _seed_all(args.seed + n * 307 + fixed_iterations, args.device)

                    def exp_fixed(
                        inp: torch.Tensor,
                        alg: AlgebraLike = algebra,
                        iterations: int = fixed_iterations,
                    ) -> torch.Tensor:
                        return compiled_safe_decomposed_exp(
                            alg,
                            inp,
                            fixed_iterations=iterations,
                        )

                    stats, r = _time_callable(
                        exp_fixed,
                        (b,),
                        args.device,
                        max(1, args.warmup // 2),
                        max(1, min(args.runs, args.error_runs)),
                        args.min_time_ms,
                        args.max_runs,
                    )
                    unit = algebra.geometric_product(r, algebra.reverse(r))
                    identity = torch.zeros_like(unit)
                    identity[..., 0] = 1.0
                    row.update(
                        {
                            **_error_stats(r, r_ref),
                            "unitarity_error": _error_stats(unit, identity)["max_abs_error"],
                            "median_ms": stats.median_ms,
                            "p90_ms": stats.p90_ms,
                            "timed_runs": stats.runs,
                        }
                    )
                    print(
                        f"converge {args.device:>4s} {dtype_name:>8s} "
                        f"n={n:<2d} it={fixed_iterations:<3d} "
                        f"err={row['max_abs_error']:.2e}"
                    )
                except Exception as exc:
                    row.update(
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=4),
                            "max_abs_error": float("nan"),
                            "rms_error": float("nan"),
                            "max_rel_error": float("nan"),
                            "unitarity_error": float("nan"),
                            "median_ms": float("nan"),
                            "p90_ms": float("nan"),
                            "timed_runs": 0,
                        }
                    )
                    print(
                        f"convergence failed {args.device} {dtype_name} "
                        f"n={n} it={fixed_iterations}: {exc}"
                    )
                finally:
                    rows.append(row)
                    algebra = None
                    b = None
                    r = None
                    unit = None
                    identity = None
                    _release_memory(args.device)
        del ref_alg, b_ref, r_ref
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _ok_rows(rows: list[dict[str, Any]], suite: str | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "ok" and (suite is None or row.get("suite") == suite)
    ]


def _configure_plots(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
        }
    )


def _save_heatmap(
    matrix: list[list[float]],
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cbar_label: str,
    path: Path,
    log10: bool = False,
) -> None:
    if not matrix:
        return
    data = np.array(matrix, dtype=float)
    if data.size == 0:
        return
    finite_positive = data[np.isfinite(data) & (data > 0)]
    if log10:
        floor = float(finite_positive.min()) if finite_positive.size else 1e-18
        data = np.where(np.isfinite(data), np.maximum(data, floor), np.nan)
        data = np.log10(data)
        cbar_label = f"log10 {cbar_label}"

    height = max(4.5, 0.45 * len(ylabels) + 2.0)
    width = max(6.5, 0.75 * len(xlabels) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    for y in range(len(ylabels)):
        for x in range(len(xlabels)):
            value = data[y, x]
            if not np.isfinite(value):
                continue
            label = f"{value:.2f}" if log10 else f"{value:.2g}"
            ax.text(x, y, label, ha="center", va="center", color="white", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _heatmap_matrix(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str,
) -> tuple[list[list[float]], list[str], list[str]]:
    def sort_key(value: Any) -> tuple[int, float | str]:
        if isinstance(value, (int, float)):
            return (0, float(value))
        return (1, str(value))

    x_values = sorted({row[x_key] for row in rows}, key=sort_key)
    y_values = sorted({row[y_key] for row in rows}, key=sort_key)
    by_key = {(row[y_key], row[x_key]): row for row in rows}
    matrix: list[list[float]] = []
    for y in y_values:
        line = []
        for x in x_values:
            row = by_key.get((y, x))
            if row is None:
                line.append(float("nan"))
            else:
                line.append(float(row.get(value_key, float("nan"))))
        matrix.append(line)
    return matrix, [str(x) for x in x_values], [str(y) for y in y_values]


def plot_speed(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    ok = _ok_rows(rows, "speed")
    if not ok:
        return
    dtype = _dtype_name(args.dtypes_resolved[0])
    batch = max(args.batch_sizes)
    filtered = [
        r
        for r in ok
        if _matches_plot_axes(args, r, dtype)
        and r["batch"] == batch
        and r["compile_mode"] == "eager"
    ]
    if not filtered:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for op in sorted({r["op"] for r in filtered}):
        pts = sorted((int(r["n"]), float(r["median_ms"])) for r in filtered if r["op"] == op)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=op)
    ax.set_yscale("log")
    ax.set_xlabel("algebra dimension n (basis dim = 2^n)")
    ax.set_ylabel("median latency (ms)")
    ax.set_title(f"Core operator scaling, eager, {dtype}, batch={batch}")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "operator_latency_by_n.png")
    plt.close(fig)


def plot_compile_speedup(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    ok = _ok_rows(rows, "speed")
    if not ok:
        return
    dtype = _dtype_name(args.dtypes_resolved[0])
    batch = max(args.batch_sizes)
    n = max(args.n_values)
    filtered = [
        r
        for r in ok
        if _matches_plot_axes(args, r, dtype)
        and r["batch"] == batch
        and r["n"] == n
    ]
    if not filtered:
        return
    eager_by_op = {
        r["op"]: float(r["median_ms"])
        for r in filtered
        if r["compile_mode"] == "eager" and float(r["median_ms"]) > 0
    }
    modes = [m for m in _parse_csv(args.compile_modes) if m != "eager"]
    ops = sorted(eager_by_op)
    if not modes or not ops:
        return

    width = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mode_idx, mode in enumerate(modes):
        heights = []
        for op in ops:
            row = next((r for r in filtered if r["op"] == op and r["compile_mode"] == mode), None)
            if row is None or float(row["median_ms"]) <= 0:
                heights.append(float("nan"))
            else:
                heights.append(eager_by_op[op] / float(row["median_ms"]))
        xs = [i + (mode_idx - (len(modes) - 1) / 2) * width for i in range(len(ops))]
        ax.bar(xs, heights, width=width, label=mode)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(ops, rotation=25, ha="right")
    ax.set_ylabel("speedup vs eager")
    ax.set_title(f"Compile speedup, {dtype}, n={n}, batch={batch}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compile_speedup.png")
    plt.close(fig)


def plot_compile_error(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    ok = _ok_rows(rows, "speed")
    if not ok:
        return
    dtype = _dtype_name(args.dtypes_resolved[0])
    batch = max(args.batch_sizes)
    n = max(args.n_values)
    filtered = [
        r
        for r in ok
        if _matches_plot_axes(args, r, dtype)
        and r["batch"] == batch
        and r["n"] == n
        and r["compile_mode"] != "eager"
    ]
    if not filtered:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    by_mode = sorted({r["compile_mode"] for r in filtered})
    by_op = sorted({r["op"] for r in filtered})
    width = 0.8 / max(len(by_mode), 1)
    for mode_idx, mode in enumerate(by_mode):
        ys = []
        for op in by_op:
            row = next((r for r in filtered if r["op"] == op and r["compile_mode"] == mode), None)
            value = float(row["max_abs_error"]) if row is not None else float("nan")
            ys.append(max(value, 1e-18))
        xs = [i + (mode_idx - (len(by_mode) - 1) / 2) * width for i in range(len(by_op))]
        ax.bar(xs, ys, width=width, label=mode)
    ax.set_yscale("log")
    ax.set_xticks(range(len(by_op)))
    ax.set_xticklabels(by_op, rotation=25, ha="right")
    ax.set_ylabel("max abs error vs eager")
    ax.set_title(f"Compile numerical parity, {dtype}, n={n}, batch={batch}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compile_error_vs_eager.png")
    plt.close(fig)


def plot_nonsimple(rows: list[dict[str, Any]], out_dir: Path) -> None:
    ok = _ok_rows(rows, "nonsimple_exp")
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dtype in sorted({r["dtype"] for r in ok}):
        for policy in sorted({r["policy"] for r in ok}):
            pts = sorted(
                (int(r["n"]), max(float(r["max_abs_error"]), 1e-18))
                for r in ok
                if r["dtype"] == dtype and r["policy"] == policy
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=f"{policy} / {dtype}")
    ax.set_yscale("log")
    ax.set_xlabel("algebra dimension n")
    ax.set_ylabel("max abs error vs commuting-plane reference")
    ax.set_title("Non-simple bivector exp error (n >= 4)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "nonsimple_exp_error.png")
    plt.close(fig)


def plot_cumulative(rows: list[dict[str, Any]], out_dir: Path) -> None:
    ok = _ok_rows(rows, "cumulative")
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dtype in sorted({r["dtype"] for r in ok}):
        for n in sorted({int(r["n"]) for r in ok}):
            for batch in sorted({int(r.get("batch", 1)) for r in ok if r["dtype"] == dtype and int(r["n"]) == n}):
                pts = sorted(
                    (int(r["step"]), max(float(r["max_abs_error"]), 1e-18))
                    for r in ok
                    if r["dtype"] == dtype
                    and int(r["n"]) == n
                    and int(r.get("batch", 1)) == batch
                )
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker=".", label=f"n={n} b={batch} / {dtype}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("chain step")
    ax.set_ylabel("max abs error vs float64 single-exp reference")
    ax.set_title("Cumulative sandwich error by precision")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_error.png")
    plt.close(fig)


def plot_convergence(rows: list[dict[str, Any]], out_dir: Path) -> None:
    ok = _ok_rows(rows, "convergence")
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dtype in sorted({r["dtype"] for r in ok}):
        for n in sorted({int(r["n"]) for r in ok}):
            pts = sorted(
                (
                    int(r["fixed_iterations"]),
                    max(float(r["max_abs_error"]), 1e-18),
                )
                for r in ok
                if r["dtype"] == dtype and int(r["n"]) == n
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=f"n={n} / {dtype}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("fixed power-iteration steps")
    ax.set_ylabel("max abs exp error")
    ax.set_title("Compiled-safe decomposed exp convergence")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "decomposed_exp_convergence.png")
    plt.close(fig)


def plot_stability(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    ok = _ok_rows(rows, "stability")
    if not ok:
        return
    dtype = _dtype_name(args.dtypes_resolved[0])
    filtered: list[dict[str, Any]] = []
    for row in ok:
        if row["dtype"] != dtype:
            continue
        ratio = float(row.get("residual_to_tolerance", float("nan")))
        if not math.isfinite(ratio):
            ratio = 1e6
        plot_row = dict(row)
        plot_row["plot_ratio"] = max(ratio, 1e-18)
        filtered.append(plot_row)
    if not filtered:
        return
    matrix, xlabels, ylabels = _heatmap_matrix(
        filtered,
        "signature",
        "case",
        "plot_ratio",
    )
    _save_heatmap(
        matrix,
        xlabels,
        ylabels,
        f"Stability residual/tolerance, {dtype}",
        "residual / tolerance",
        out_dir / "stability_residual_ratio.png",
        log10=True,
    )


def plot_backward(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    ok = _ok_rows(rows, "backward")
    if not ok:
        return
    dtype = _dtype_name(args.dtypes_resolved[0])
    batch = max(args.backward_batch_sizes)
    filtered = [
        row
        for row in ok
        if _matches_plot_axes(args, row, dtype)
        and row["batch"] == batch
        and row["compile_mode"] == "eager"
    ]
    if not filtered:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for op in sorted({row["op"] for row in filtered}):
        pts = sorted(
            (int(row["n"]), float(row["median_ms"]))
            for row in filtered
            if row["op"] == op
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=op)
    ax.set_yscale("log")
    ax.set_xlabel("algebra dimension n (basis dim = 2^n)")
    ax.set_ylabel("forward+backward median latency (ms)")
    ax.set_title(f"Backward-pass scaling, eager, {dtype}, batch={batch}")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "backward_latency_by_n.png")
    plt.close(fig)


def plot_fusion(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    ok = _ok_rows(rows, "fusion")
    if not ok:
        return
    dtype = _dtype_name(args.dtypes_resolved[0])
    n = max(args.profiler_n_values)
    batch = args.profiler_batch_size
    filtered = [
        row
        for row in ok
        if _matches_plot_axes(args, row, dtype)
        and int(row["n"]) == n
        and int(row["batch"]) == batch
        and row["phase"] == "forward"
    ]
    if not filtered:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    modes = [mode for mode in _parse_csv(args.profiler_compile_modes) if mode != "eager"]
    ops = sorted({row["op"] for row in filtered})
    width = 0.8 / max(len(modes), 1)
    for mode_idx, mode in enumerate(modes):
        ys = []
        for op in ops:
            row = next(
                (
                    item
                    for item in filtered
                    if item["op"] == op and item["compile_mode"] == mode
                ),
                None,
            )
            ys.append(
                float(row["aten_reduction_vs_eager"])
                if row is not None
                else float("nan")
            )
        xs = [i + (mode_idx - (len(modes) - 1) / 2) * width for i in range(len(ops))]
        ax.bar(xs, ys, width=width, label=mode)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(ops, rotation=25, ha="right")
    ax.set_ylabel("aten op-count reduction vs eager")
    ax.set_title(f"Profiler fusion proxy, {dtype}, n={n}, batch={batch}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "profiler_fusion_reduction.png")
    plt.close(fig)


def plot_heatmaps(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    dtype = _dtype_name(args.dtypes_resolved[0])

    speed_rows = [
        row
        for row in _ok_rows(rows, "speed")
        if _matches_plot_axes(args, row, dtype)
        and row["compile_mode"] == "eager"
        and int(row["batch"]) == max(args.batch_sizes)
    ]
    matrix, xlabels, ylabels = _heatmap_matrix(speed_rows, "n", "op", "median_ms")
    _save_heatmap(
        matrix,
        xlabels,
        ylabels,
        f"Forward latency heatmap, eager, {dtype}",
        "median ms",
        out_dir / "heatmap_forward_latency.png",
        log10=True,
    )

    backward_rows = [
        row
        for row in _ok_rows(rows, "backward")
        if _matches_plot_axes(args, row, dtype)
        and row["compile_mode"] == "eager"
        and int(row["batch"]) == max(args.backward_batch_sizes)
    ]
    matrix, xlabels, ylabels = _heatmap_matrix(backward_rows, "n", "op", "median_ms")
    _save_heatmap(
        matrix,
        xlabels,
        ylabels,
        f"Forward+backward latency heatmap, eager, {dtype}",
        "median ms",
        out_dir / "heatmap_backward_latency.png",
        log10=True,
    )

    fusion_rows = [
        row
        for row in _ok_rows(rows, "fusion")
        if _matches_plot_axes(args, row, dtype)
        and row["compile_mode"] == "eager"
        and row["phase"] == "backward"
    ]
    matrix, xlabels, ylabels = _heatmap_matrix(
        fusion_rows,
        "n",
        "op",
        "peak_delta_mb",
    )
    _save_heatmap(
        matrix,
        xlabels,
        ylabels,
        f"Peak allocation heatmap, backward, eager, {dtype}",
        "peak delta MB",
        out_dir / "heatmap_peak_allocation.png",
        log10=False,
    )

    fusion_forward = [
        row
        for row in _ok_rows(rows, "fusion")
        if _matches_plot_axes(args, row, dtype)
        and int(row["n"]) == max(args.profiler_n_values)
        and row["phase"] == "forward"
        and row["compile_mode"] != "eager"
    ]
    matrix, xlabels, ylabels = _heatmap_matrix(
        fusion_forward,
        "compile_mode",
        "op",
        "aten_reduction_vs_eager",
    )
    _save_heatmap(
        matrix,
        xlabels,
        ylabels,
        f"Fusion proxy heatmap, forward, {dtype}",
        "aten reduction vs eager",
        out_dir / "heatmap_fusion_reduction.png",
        log10=False,
    )


def write_summary(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    speed = _ok_rows(rows, "speed")
    backward = _ok_rows(rows, "backward")
    fusion = _ok_rows(rows, "fusion")
    nonsimple = _ok_rows(rows, "nonsimple_exp")
    cumulative = _ok_rows(rows, "cumulative")
    convergence = _ok_rows(rows, "convergence")
    stability = _ok_rows(rows, "stability")
    stability_not_passed = [r for r in stability if not bool(r.get("passed", False))]
    failed = [r for r in rows if r.get("status") != "ok"]

    def _num(value: Any, default: float = float("nan")) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _finite(value: Any) -> bool:
        return math.isfinite(_num(value))

    def _fmt(value: Any, digits: int = 2, style: str = "f") -> str:
        number = _num(value)
        if not math.isfinite(number):
            return ""
        return f"{number:.{digits}{style}}"

    def _fmt_ms(value: Any) -> str:
        return _fmt(value, 4)

    def _fmt_sci(value: Any) -> str:
        return _fmt(value, 2, "e")

    def _fmt_compact(value: Any) -> str:
        number = _num(value)
        if not math.isfinite(number):
            return ""
        abs_number = abs(number)
        for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs_number >= scale:
                return f"{number / scale:.2f}{suffix}"
        return f"{number:.2f}"

    def _fmt_io_kib(value: Any) -> str:
        number = _num(value)
        if not math.isfinite(number):
            return ""
        return _fmt_compact(number / 1024.0)

    def _fmt_speedup(row: dict[str, Any], key: str = "speedup_vs_eager") -> str:
        value = _num(row.get(key))
        if math.isfinite(value):
            return f"{value:.2f}x"
        if row.get("compile_mode") == "eager":
            return "1.00x"
        return ""

    def _fmt_ci(row: dict[str, Any]) -> str:
        low = _num(row.get("speedup_vs_eager_ci_low"))
        high = _num(row.get("speedup_vs_eager_ci_high"))
        if not math.isfinite(low) or not math.isfinite(high):
            return ""
        return f"{low:.2f}-{high:.2f}x"

    def _md(value: Any) -> str:
        return str(value).replace("|", "/")

    def _case_label(row: dict[str, Any]) -> str:
        parts = [str(row.get("dtype", "")), f"n{row.get('n', '')}"]
        if row.get("batch", "") != "":
            parts.append(f"b{row.get('batch')}")
        channels = int(_num(row.get("channels", 0), 0.0))
        if channels:
            parts.append(f"c{channels}")
        for key in ("op", "phase", "policy"):
            if row.get(key, "") != "":
                parts.append(str(row[key]))
        if row.get("fixed_iterations", "") != "":
            parts.append(f"it{row['fixed_iterations']}")
        if row.get("compile_mode", "") != "":
            parts.append(str(row["compile_mode"]))
        tf32 = str(row.get("tf32_mode", "default"))
        if tf32 and tf32 != "default":
            parts.append(f"tf32={tf32}")
        return " ".join(parts)

    def _best_row(
        source: list[dict[str, Any]],
        key: str,
        *,
        highest: bool = True,
    ) -> dict[str, Any] | None:
        candidates = [row for row in source if _finite(row.get(key))]
        if not candidates:
            return None
        selector = max if highest else min
        return selector(candidates, key=lambda r: _num(r[key]))

    lines = [
        "# Core Benchmark Dashboard",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"PyTorch: {torch.__version__}",
        f"Device: {args.device}",
        f"Dtypes: {', '.join(_dtype_name(d) for d in args.dtypes_resolved)}",
        (
            f"Rows: {len(rows)} total, {len(failed)} failed, "
            f"{len(stability_not_passed)} stability non-pass"
        ),
        "",
    ]

    suite_specs = [
        ("speed", speed, "speed.csv"),
        ("backward", backward, "backward.csv"),
        ("fusion", fusion, "fusion.csv"),
        ("nonsimple_exp", nonsimple, "nonsimple_exp.csv"),
        ("cumulative", cumulative, "cumulative.csv"),
        ("convergence", convergence, "convergence.csv"),
        ("stability", stability, "stability.csv"),
    ]
    lines += [
        "## Run Snapshot",
        "",
        "| suite | ok rows | failed rows | detail CSV |",
        "|-------|---------|-------------|------------|",
    ]
    for suite, ok_rows, csv_name in suite_specs:
        total = sum(1 for row in rows if row.get("suite") == suite)
        fail_count = sum(1 for row in rows if row.get("suite") == suite and row.get("status") != "ok")
        detail = f"`{csv_name}`" if total else ""
        lines.append(f"| {suite} | {len(ok_rows)} | {fail_count} | {detail} |")
    lines += [f"| all | {len(rows) - len(failed)} | {len(failed)} | `all_rows.csv` |", ""]

    lines += [
        "## KPI Strip",
        "",
        "| metric | value | case |",
        "|--------|-------|------|",
    ]
    kpi_specs: list[tuple[str, list[dict[str, Any]], str, Callable[[Any], str], bool]] = [
        ("forward min latency", speed, "median_ms", _fmt_ms, False),
        ("forward max throughput", speed, "items_per_sec", _fmt_compact, True),
        ("forward max GFLOP/s", speed, "achieved_forward_gflops", lambda v: _fmt(v, 2), True),
        ("forward max IO GB/s", speed, "achieved_io_gbps", lambda v: _fmt(v, 2), True),
        ("forward max speedup", speed, "speedup_vs_eager", lambda v: f"{_fmt(v, 2)}x" if _fmt(v, 2) else "", True),
        ("backward min latency", backward, "median_ms", _fmt_ms, False),
        ("backward max throughput", backward, "items_per_sec", _fmt_compact, True),
        ("backward max GFLOP/s proxy", backward, "achieved_forward_gflops", lambda v: _fmt(v, 2), True),
        ("backward max speedup", backward, "speedup_vs_eager", lambda v: f"{_fmt(v, 2)}x" if _fmt(v, 2) else "", True),
        ("max aten reduction", fusion, "aten_reduction_vs_eager", lambda v: f"{_fmt(v, 2)}x" if _fmt(v, 2) else "", True),
        ("max kernel reduction", fusion, "kernel_reduction_vs_eager", lambda v: f"{_fmt(v, 2)}x" if _fmt(v, 2) else "", True),
        ("max profiler CPU speedup", fusion, "cpu_time_speedup_vs_eager", lambda v: f"{_fmt(v, 2)}x" if _fmt(v, 2) else "", True),
        ("max profiler device speedup", fusion, "device_time_speedup_vs_eager", lambda v: f"{_fmt(v, 2)}x" if _fmt(v, 2) else "", True),
        ("max peak allocation delta", fusion, "peak_delta_mb", lambda v: f"{_fmt(v, 2)} MB" if _fmt(v, 2) else "", True),
    ]
    for label, source, key, formatter, highest in kpi_specs:
        row = _best_row(source, key, highest=highest)
        if row is not None:
            lines.append(f"| {label} | {formatter(row[key])} | {_md(_case_label(row))} |")
    if stability:
        passed = len(stability) - len(stability_not_passed)
        lines.append(f"| stability pass rate | {passed}/{len(stability)} | {len(stability_not_passed)} non-pass |")
        worst = _best_row(stability, "residual_to_tolerance", highest=True)
        if worst is not None:
            lines.append(
                f"| worst stability residual/tolerance | {_fmt(worst['residual_to_tolerance'], 2)} | "
                f"{_md(worst.get('dtype', ''))} {_md(worst.get('signature', ''))} {_md(worst.get('case', ''))} |"
            )
    lines.append("")

    if not args.skip_plots:
        plot_links: list[tuple[str, str]] = []
        if speed:
            plot_links += [
                ("Operator latency", "plots/operator_latency_by_n.png"),
                ("Compile speedup", "plots/compile_speedup.png"),
                ("Compile parity", "plots/compile_error_vs_eager.png"),
                ("Forward latency heatmap", "plots/heatmap_forward_latency.png"),
            ]
        if backward:
            plot_links += [
                ("Backward latency", "plots/backward_latency_by_n.png"),
                ("Backward latency heatmap", "plots/heatmap_backward_latency.png"),
            ]
        if fusion:
            plot_links += [
                ("Profiler fusion", "plots/profiler_fusion_reduction.png"),
                ("Fusion heatmap", "plots/heatmap_fusion_reduction.png"),
                ("Peak allocation", "plots/heatmap_peak_allocation.png"),
            ]
        if nonsimple:
            plot_links.append(("Non-simple exp error", "plots/nonsimple_exp_error.png"))
        if cumulative:
            plot_links.append(("Cumulative error", "plots/cumulative_error.png"))
        if convergence:
            plot_links.append(("Exp convergence", "plots/decomposed_exp_convergence.png"))
        if stability:
            plot_links.append(("Stability residuals", "plots/stability_residual_ratio.png"))
        available_plots = [(title, path) for title, path in plot_links if (out_dir / path).exists()]
        if available_plots:
            lines += [
                "## Plot Index",
                "",
                "| plot | file |",
                "|------|------|",
            ]
            for title, path in available_plots:
                lines.append(f"| {title} | [{path}]({path}) |")
            lines.append("")

    if stability:
        passed = len(stability) - len(stability_not_passed)
        lines += [
            "## Accuracy And Stability",
            "",
            f"Passed: **{passed}/{len(stability)}**",
            "",
            "| dtype | signature | case | residual | tolerance | residual/tol | passed | note |",
            "|-------|-----------|------|----------|-----------|--------------|--------|------|",
        ]
        sorted_stability = sorted(
            stability,
            key=lambda r: (
                bool(r.get("passed", False)),
                -_num(r.get("residual_to_tolerance"), float("-inf")),
                r["dtype"],
                r["signature"],
                r["case"],
            ),
        )
        for r in sorted_stability[:32]:
            mark = "PASS" if bool(r.get("passed", False)) else "FAIL"
            note = _md(r.get("note", ""))
            lines.append(
                f"| {r['dtype']} | {r['signature']} | {r['case']} | "
                f"{_fmt_sci(r.get('residual'))} | {_fmt_sci(r.get('tolerance'))} | "
                f"{_fmt(r.get('residual_to_tolerance'), 3)} | "
                f"{mark} | {note} |"
            )
        if len(sorted_stability) > 32:
            lines.append(
                f"| ... | ... | ... | ... | ... | ... | ... | "
                f"{len(sorted_stability) - 32} more rows in `stability.csv` |"
            )
        lines.append("")

    if speed:
        lines += [
            "## Forward Performance",
            "",
            "Best row per dtype/tf32/n/batch/channel/operator, selected by median latency.",
            "",
            "| dtype | tf32 | n | dim | batch | ch | op | mode | med ms | p90 ms | items/s | FLOP/call | GF/s | IO KiB | GB/s | speedup | CI | err | breaks |",
            "|-------|------|---|-----|-------|----|----|------|--------|--------|---------|-----------|------|--------|------|---------|----|-----|--------|",
        ]
        keys = sorted(
            {
                (
                    r["dtype"],
                    r.get("tf32_mode", "default"),
                    r["n"],
                    r["batch"],
                    r.get("channels", 0),
                    r["op"],
                )
                for r in speed
            }
        )
        for key in keys[:80]:
            candidates = [
                r
                for r in speed
                if (
                    r["dtype"],
                    r.get("tf32_mode", "default"),
                    r["n"],
                    r["batch"],
                    r.get("channels", 0),
                    r["op"],
                )
                == key
                and math.isfinite(float(r["median_ms"]))
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: float(r["median_ms"]))
            ci = _fmt_ci(best)
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {best.get('dim', '')} | {key[3]} | {key[4]} | "
                f"{key[5]} | {best['compile_mode']} | {_fmt_ms(best['median_ms'])} | "
                f"{_fmt_ms(best.get('p90_ms'))} | {_fmt_compact(best.get('items_per_sec'))} | "
                f"{_fmt_compact(best.get('analytic_forward_flops'))} | "
                f"{_fmt(best.get('achieved_forward_gflops'), 2)} | "
                f"{_fmt_io_kib(best.get('analytic_io_bytes'))} | "
                f"{_fmt(best.get('achieved_io_gbps'), 2)} | {_fmt_speedup(best)} | "
                f"{ci} | {_fmt_sci(best.get('max_abs_error'))} | {_fmt(best.get('graph_breaks'), 0)} |"
            )
        if len(keys) > 80:
            lines += ["", f"Showing 80 of {len(keys)} workloads; complete rows are in `speed.csv`."]
        lines.append("")

    if backward:
        lines += [
            "## Backward Performance",
            "",
            "GFLOP/s and IO columns use the existing analytic forward-work proxy divided by forward+backward latency.",
            "",
            "| dtype | tf32 | n | dim | batch | ch | op | mode | med ms | p90 ms | items/s | fwd FLOP/call | fwd GF/s | IO KiB | GB/s | speedup | CI | grad | err | breaks |",
            "|-------|------|---|-----|-------|----|----|------|--------|--------|---------|---------------|----------|--------|------|---------|----|------|-----|--------|",
        ]
        keys = sorted(
            {
                (
                    r["dtype"],
                    r.get("tf32_mode", "default"),
                    r["n"],
                    r["batch"],
                    r.get("channels", 0),
                    r["op"],
                )
                for r in backward
            }
        )
        for key in keys[:80]:
            candidates = [
                r
                for r in backward
                if (
                    r["dtype"],
                    r.get("tf32_mode", "default"),
                    r["n"],
                    r["batch"],
                    r.get("channels", 0),
                    r["op"],
                )
                == key
                and math.isfinite(float(r["median_ms"]))
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: float(r["median_ms"]))
            ci = _fmt_ci(best)
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {best.get('dim', '')} | {key[3]} | {key[4]} | "
                f"{key[5]} | {best['compile_mode']} | {_fmt_ms(best['median_ms'])} | "
                f"{_fmt_ms(best.get('p90_ms'))} | {_fmt_compact(best.get('items_per_sec'))} | "
                f"{_fmt_compact(best.get('analytic_forward_flops'))} | "
                f"{_fmt(best.get('achieved_forward_gflops'), 2)} | "
                f"{_fmt_io_kib(best.get('analytic_io_bytes'))} | "
                f"{_fmt(best.get('achieved_io_gbps'), 2)} | {_fmt_speedup(best)} | "
                f"{ci} | {_fmt_sci(best.get('grad_norm'))} | "
                f"{_fmt_sci(best.get('max_abs_error'))} | {_fmt(best.get('graph_breaks'), 0)} |"
            )
        if len(keys) > 80:
            lines += ["", f"Showing 80 of {len(keys)} workloads; complete rows are in `backward.csv`."]
        lines.append("")

    if fusion:
        lines += [
            "## Profiler And Fusion",
            "",
            "| dtype | tf32 | n | batch | ch | op | phase | mode | breaks | aten/step | kern/step | triton/step | cpu ms/step | dev ms/step | peak MB | aten x | kern x | CPU x | dev x |",
            "|-------|------|---|-------|----|----|-------|------|--------|-----------|-----------|-------------|-------------|-------------|---------|--------|--------|-------|-------|",
        ]
        sorted_fusion = sorted(
            fusion,
            key=lambda r: (
                r["dtype"],
                r.get("tf32_mode", "default"),
                int(r["n"]),
                int(r.get("batch", 0)),
                int(r.get("channels", 0)),
                r["op"],
                r["phase"],
                r["compile_mode"],
            ),
        )
        for r in sorted_fusion[:120]:
            lines.append(
                f"| {r['dtype']} | {r.get('tf32_mode', 'default')} | {r['n']} | "
                f"{r.get('batch', '')} | {r.get('channels', 0)} | {r['op']} | {r['phase']} | "
                f"{r['compile_mode']} | {_fmt(r.get('graph_breaks'), 0)} | "
                f"{_fmt(r.get('aten_ops_per_step'), 1)} | "
                f"{_fmt(r.get('kernel_events_per_step'), 1)} | "
                f"{_fmt(r.get('triton_kernel_events_per_step'), 1)} | "
                f"{_fmt(r.get('profiler_self_cpu_time_ms_per_step'), 3)} | "
                f"{_fmt(r.get('profiler_self_device_time_ms_per_step'), 3)} | "
                f"{_fmt(r.get('peak_delta_mb'), 2)} | "
                f"{_fmt_speedup(r, 'aten_reduction_vs_eager')} | "
                f"{_fmt_speedup(r, 'kernel_reduction_vs_eager')} | "
                f"{_fmt_speedup(r, 'cpu_time_speedup_vs_eager')} | "
                f"{_fmt_speedup(r, 'device_time_speedup_vs_eager')} |"
            )
        if len(sorted_fusion) > 120:
            lines += ["", f"Showing 120 of {len(sorted_fusion)} profiler rows; complete rows are in `fusion.csv`."]
        lines.append("")

    if nonsimple:
        lines += [
            "## Non-Simple Bivector Exp",
            "",
            "| dtype | n | policy | med ms | p90 ms | max abs error | unitarity error |",
            "|-------|---|--------|--------|--------|---------------|-----------------|",
        ]
        for r in sorted(nonsimple, key=lambda x: (x["dtype"], int(x["n"]), x["policy"])):
            lines.append(
                f"| {r['dtype']} | {r['n']} | {r['policy']} | "
                f"{_fmt_ms(r.get('median_ms'))} | "
                f"{_fmt_ms(r.get('p90_ms'))} | "
                f"{_fmt_sci(r.get('max_abs_error'))} | "
                f"{_fmt_sci(r.get('unitarity_error'))} |"
            )
        lines.append("")

    if cumulative:
        lines += [
            "## Cumulative Error At Final Sample",
            "",
            "| dtype | n | batch | step | max abs error | norm drift | grade leak | elapsed ms |",
            "|-------|---|-------|------|---------------|------------|------------|------------|",
        ]
        by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
        for r in cumulative:
            key = (r["dtype"], int(r["n"]), int(r.get("batch", 1)))
            if key not in by_key or int(r["step"]) > int(by_key[key]["step"]):
                by_key[key] = r
        for (dtype, n, batch), r in sorted(by_key.items()):
            lines.append(
                f"| {dtype} | {n} | {batch} | {r['step']} | "
                f"{_fmt_sci(r.get('max_abs_error'))} | "
                f"{_fmt_sci(r.get('norm_drift'))} | "
                f"{_fmt_sci(r.get('grade_leak'))} | "
                f"{_fmt_ms(r.get('elapsed_ms'))} |"
            )
        lines.append("")

    if convergence:
        lines += [
            "## Best Convergence Row",
            "",
            "| dtype | n | fixed iterations | med ms | p90 ms | max abs error | unitarity error |",
            "|-------|---|------------------|--------|--------|---------------|-----------------|",
        ]
        for dtype in sorted({r["dtype"] for r in convergence}):
            for n in sorted({int(r["n"]) for r in convergence if r["dtype"] == dtype}):
                candidates = [r for r in convergence if r["dtype"] == dtype and int(r["n"]) == n]
                best = min(candidates, key=lambda r: float(r["max_abs_error"]))
                lines.append(
                    f"| {dtype} | {n} | {best['fixed_iterations']} | "
                    f"{_fmt_ms(best.get('median_ms'))} | "
                    f"{_fmt_ms(best.get('p90_ms'))} | "
                    f"{_fmt_sci(best.get('max_abs_error'))} | "
                    f"{_fmt_sci(best.get('unitarity_error'))} |"
                )
        lines.append("")

    if failed:
        lines += [
            "## Failures",
            "",
            "| suite | dtype | n | op/policy | mode | error |",
            "|-------|-------|---|-----------|------|-------|",
        ]
        for r in failed[:80]:
            op = r.get("op", r.get("case", r.get("policy", r.get("fixed_iterations", ""))))
            mode = r.get("compile_mode", "")
            error = str(r.get("error", "")).replace("|", "/")
            lines.append(
                f"| {r.get('suite', '')} | {r.get('dtype', '')} | "
                f"{r.get('n', '')} | {op} | {mode} | {error} |"
            )
        if len(failed) > 80:
            lines.append(f"| ... | ... | ... | ... | ... | {len(failed) - 80} more |")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _collect_runtime_metadata(device: str) -> dict[str, Any]:
    """Capture host/runtime knobs that swing perf 2-3x silently.

    Without these, two runs on the same hardware can produce wildly different
    numbers and the artifact is not reproducible.
    """
    info: dict[str, Any] = {
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "env_omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "env_mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "anomaly_detection_enabled": bool(torch.is_anomaly_enabled()),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else ""
        ),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = int(device.split(":", 1)[1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_mb": float(props.total_memory) / (1024.0 * 1024.0),
                "cuda_multi_processor_count": int(props.multi_processor_count),
                "cuda_runtime_version": torch.version.cuda,
                "cudnn_available": bool(torch.backends.cudnn.is_available()),
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    and torch.backends.cudnn.version() is not None
                    else None
                ),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
                    getattr(
                        torch.backends.cuda.matmul,
                        "allow_fp16_reduced_precision_reduction",
                        False,
                    )
                ),
            }
        )
    elif device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        info["mps_built"] = bool(mps_backend and mps_backend.is_built())
        info["mps_available"] = bool(mps_backend and mps_backend.is_available())
    return info


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the Versor core package.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--algebra-kernel", default="auto", choices=("auto", "dense", "partitioned"))
    parser.add_argument("--partition-threshold", type=int, default=8)
    parser.add_argument("--partition-leaf-n", type=int, default=4)
    parser.add_argument("--partition-product-chunk-size", type=int, default=None)
    parser.add_argument("--partition-tree", default=None)
    parser.add_argument("--partition-accumulation-dtype", default=None)
    parser.add_argument("--out", default="benchmarks/results", help="artifact root")
    parser.add_argument("--sections", default="speed,backward,fusion,nonsimple,cumulative,convergence,stability")
    parser.add_argument("--n-values", type=_parse_int_csv, default=_parse_int_csv("2,3,4,5,6"))
    parser.add_argument("--batch-sizes", type=_parse_int_csv, default=_parse_int_csv("1,32"))
    parser.add_argument("--backward-n-values", type=_parse_int_csv, default=_parse_int_csv("2,3,4,5"))
    parser.add_argument("--backward-batch-sizes", type=_parse_int_csv, default=_parse_int_csv("1,16"))
    parser.add_argument("--nonsimple-n-values", type=_parse_int_csv, default=_parse_int_csv("4,5,6"))
    parser.add_argument("--cumulative-n-values", type=_parse_int_csv, default=_parse_int_csv("4,5,6"))
    parser.add_argument("--cumulative-batch-sizes", type=_parse_int_csv, default=_parse_int_csv("1,32"))
    parser.add_argument("--convergence-n-values", type=_parse_int_csv, default=_parse_int_csv("4,5,6"))
    parser.add_argument("--profiler-n-values", type=_parse_int_csv, default=_parse_int_csv("3,4,5"))
    parser.add_argument("--stability-signatures", default="2:0:0,1:1:0,2:0:1,2:1:0,3:0:0,4:0:0,4:1:0,6:0:0")
    parser.add_argument("--dtypes", default="auto", help="auto or comma list: float64,float32,bfloat16,float16")
    parser.add_argument("--ops", default="gp,wedge,inner,commutator,grade2,reverse,norm_sq,exp,exp_precise,sandwich")
    parser.add_argument("--backward-ops", default="gp,wedge,inner,commutator,grade2,reverse,norm_sq,exp,sandwich")
    parser.add_argument("--profiler-ops", default="gp,exp,sandwich")
    parser.add_argument(
        "--stability-cases",
        default=(
            "associativity,reverse_involution,grade_projection,rotor_unitarity,"
            "sandwich_consistency,per_channel_sandwich,exp_policy_consistency,"
            "large_angle_unitarity,grad_finite"
        ),
    )
    parser.add_argument(
        "--compile-modes",
        default="auto",
        help="auto or comma list: eager,aot_eager,compile,reduce-overhead,max-autotune",
    )
    parser.add_argument(
        "--backward-compile-modes",
        default="auto",
        help="auto or comma list: eager,aot_eager,compile,reduce-overhead,max-autotune",
    )
    parser.add_argument(
        "--profiler-compile-modes",
        default="auto",
        help="auto or comma list: eager,aot_eager,compile,reduce-overhead,max-autotune",
    )
    parser.add_argument("--stability-compile-mode", default="aot_eager")
    parser.add_argument(
        "--tf32-modes",
        default="auto",
        help="auto or comma list: default,strict,tf32. Applies only to CUDA float32 rows.",
    )
    parser.add_argument("--profiler-phases", default="forward,backward")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument(
        "--sandwich-channel-values",
        type=_parse_optional_int_csv,
        default=None,
        help=(
            "Comma list of sandwich channels for speed/backward/profiler. "
            "Defaults to 32,64,128; --channels remains the stability-suite size."
        ),
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--error-runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-time-ms", type=float, default=25.0)
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--backward-runs", type=int, default=10)
    parser.add_argument("--backward-warmup", type=int, default=3)
    parser.add_argument("--backward-min-time-ms", type=float, default=25.0)
    parser.add_argument("--backward-max-runs", type=int, default=100)
    parser.add_argument("--profiler-batch-size", type=int, default=16)
    parser.add_argument("--profiler-steps", type=int, default=5)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--error-batch", type=int, default=8)
    parser.add_argument("--bivector-scale", type=float, default=1.0)
    parser.add_argument("--stability-batch", type=int, default=8)
    parser.add_argument("--stability-input-scale", type=float, default=0.25)
    parser.add_argument("--stability-bivector-scale", type=float, default=0.15)
    parser.add_argument("--stability-large-angle", type=float, default=12.0)
    parser.add_argument("--chain-bivector-scale", type=float, default=0.025)
    parser.add_argument("--chain-steps", type=int, default=512)
    parser.add_argument("--chain-samples", type=int, default=28)
    parser.add_argument("--convergence-iters", type=_parse_int_csv, default=_parse_int_csv("1,2,3,4,5,6,7,8,16,32,64,96,128"))
    parser.add_argument(
        "--dtype-probe-n-values",
        type=_parse_optional_int_csv,
        default=None,
        help="Comma list for dtype support probes. Defaults to the requested benchmark n-values.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help=(
            "If > 0, pin torch.set_num_threads to this value before timing. "
            "Recorded in metadata.json regardless. CPU benchmarks are not "
            "comparable across machines without a fixed thread count."
        ),
    )
    parser.add_argument("--plot-dpi", type=int, default=300)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--export-profiler-traces", action="store_true")
    return parser


def main() -> None:
    parser = make_argparser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    if not _device_available(args.device):
        raise SystemExit(f"Requested device is unavailable: {args.device}")

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    args.compile_modes = _resolve_compile_modes(args.compile_modes, args.device)
    args.backward_compile_modes = _resolve_compile_modes(args.backward_compile_modes, args.device)
    args.profiler_compile_modes = _resolve_compile_modes(args.profiler_compile_modes, args.device)
    _tf32_modes_for_dtype(args, torch.float32)
    if args.sandwich_channel_values is None:
        args.sandwich_channel_values = [32, 64, 128]

    args.n_values = sorted(set(args.n_values))
    args.batch_sizes = sorted(set(args.batch_sizes))
    args.backward_n_values = sorted(set(args.backward_n_values))
    args.backward_batch_sizes = sorted(set(args.backward_batch_sizes))
    args.nonsimple_n_values = sorted(set(args.nonsimple_n_values))
    args.cumulative_n_values = sorted(set(args.cumulative_n_values))
    args.cumulative_batch_sizes = sorted(set(args.cumulative_batch_sizes))
    args.convergence_n_values = sorted(set(args.convergence_n_values))
    args.profiler_n_values = sorted(set(args.profiler_n_values))
    args.sandwich_channel_values = sorted(set(args.sandwich_channel_values))
    args.convergence_iters = sorted(set(args.convergence_iters))
    args.sections = set(_parse_csv(args.sections))

    stability_probe_ns = [spec.n for spec in _parse_signature_csv(args.stability_signatures)]
    if args.dtype_probe_n_values is None:
        probe_sources: list[int] = []
        if "speed" in args.sections:
            probe_sources += args.n_values
        if "backward" in args.sections:
            probe_sources += args.backward_n_values
        if "nonsimple" in args.sections:
            probe_sources += args.nonsimple_n_values
        if "cumulative" in args.sections:
            probe_sources += args.cumulative_n_values
        if "convergence" in args.sections:
            probe_sources += args.convergence_n_values
        if "fusion" in args.sections or "profiler" in args.sections or "memory" in args.sections:
            probe_sources += args.profiler_n_values
        if "stability" in args.sections or "verify" in args.sections:
            probe_sources += stability_probe_ns
        args.dtype_probe_n_values = sorted(set(probe_sources or args.n_values))
    else:
        args.dtype_probe_n_values = sorted(set(args.dtype_probe_n_values))

    dtypes = _supported_dtypes(args, args.dtypes, args.dtype_probe_n_values)
    args.dtypes_resolved = dtypes

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"benchmark_core_{timestamp}"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "torch": torch.__version__,
        "device": args.device,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "runtime": _collect_runtime_metadata(args.device),
        "args": _json_safe(vars(args)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(
        f"PyTorch {torch.__version__} | device={args.device} | "
        f"dtypes={','.join(_dtype_name(d) for d in dtypes)} | out={out_dir}"
    )

    all_rows: list[dict[str, Any]] = []

    if "speed" in args.sections:
        print("\n[1/7] operator scaling and compile modes")
        speed_rows = run_speed_suite(args, dtypes)
        all_rows.extend(speed_rows)
        _write_csv(out_dir / "speed.csv", speed_rows)

    if "backward" in args.sections:
        print("\n[2/7] backward pass")
        backward_rows = run_backward_suite(args, dtypes)
        all_rows.extend(backward_rows)
        _write_csv(out_dir / "backward.csv", backward_rows)

    if "fusion" in args.sections or "profiler" in args.sections or "memory" in args.sections:
        print("\n[3/7] profiler fusion and peak allocation")
        fusion_rows = run_fusion_suite(args, dtypes, out_dir)
        all_rows.extend(fusion_rows)
        _write_csv(out_dir / "fusion.csv", fusion_rows)

    if "nonsimple" in args.sections:
        print("\n[4/7] non-simple bivector exp error")
        nonsimple_rows = run_nonsimple_suite(args, dtypes)
        all_rows.extend(nonsimple_rows)
        _write_csv(out_dir / "nonsimple_exp.csv", nonsimple_rows)

    if "cumulative" in args.sections:
        print("\n[5/7] cumulative sandwich error")
        cumulative_rows = run_cumulative_suite(args, dtypes)
        all_rows.extend(cumulative_rows)
        _write_csv(out_dir / "cumulative.csv", cumulative_rows)

    if "convergence" in args.sections:
        print("\n[6/7] decomposed exp convergence")
        convergence_rows = run_convergence_suite(args, dtypes)
        all_rows.extend(convergence_rows)
        _write_csv(out_dir / "convergence.csv", convergence_rows)

    if "stability" in args.sections or "verify" in args.sections:
        print("\n[7/7] algebraic stability verification")
        stability_rows = run_stability_suite(args, dtypes)
        all_rows.extend(stability_rows)
        _write_csv(out_dir / "stability.csv", stability_rows)

    _write_csv(out_dir / "all_rows.csv", all_rows)
    write_summary(out_dir, all_rows, args)

    if not args.skip_plots:
        _configure_plots(args.plot_dpi)
        plot_speed(all_rows, plots_dir, args)
        plot_compile_speedup(all_rows, plots_dir, args)
        plot_compile_error(all_rows, plots_dir, args)
        plot_nonsimple(all_rows, plots_dir)
        plot_cumulative(all_rows, plots_dir)
        plot_convergence(all_rows, plots_dir)
        plot_stability(all_rows, plots_dir, args)
        plot_backward(all_rows, plots_dir, args)
        plot_fusion(all_rows, plots_dir, args)
        plot_heatmaps(all_rows, plots_dir, args)
        write_summary(out_dir, all_rows, args)

    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
