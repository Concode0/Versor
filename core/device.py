# Versor: Universal Geometric Algebra Neural Network (C) 2026 Eunkyum Kim
# Licensed under the Apache License, Version 2.0

"""Device configuration and backend tuning for Versor.

Centralises device resolution, ``pin_memory``, ``torch.compile``,
``cudnn.benchmark``, and AMP (automatic mixed precision) into a single
:class:`DeviceConfig` dataclass.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Optional

import torch
import torch.nn as nn

FLOAT_DTYPES: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    **FLOAT_DTYPES,
    "fp64": torch.float64,
    "double": torch.float64,
    "fp32": torch.float32,
    "float": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "half": torch.float16,
}


def optional_dtype(value: Any) -> Optional[torch.dtype]:
    """Parse a torch dtype declaration, preserving ``None`` as unset."""
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[normalized]
    raise ValueError(f"Unsupported torch dtype declaration: {value!r}")


def resolve_dtype(value: Any, default: torch.dtype = torch.float32) -> torch.dtype:
    """Parse a torch dtype declaration and fall back to ``default`` when unset."""
    return optional_dtype(value) or default


def dtype_name(dtype: torch.dtype) -> str:
    """Return the canonical short name for a torch dtype."""
    for name, candidate in FLOAT_DTYPES.items():
        if candidate == dtype:
            return name
    return str(dtype).replace("torch.", "")


def resolve_device(device: str = "auto") -> str:
    """Resolve ``'auto'`` to the best available accelerator.

    Priority: cuda > mps > cpu.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DeviceConfig:
    """Immutable bag of device / backend settings.

    Attributes:
        device: Resolved device string (``cuda``, ``mps``, ``cpu``).
        pin_memory: Whether DataLoaders should pin memory.  ``None`` -> auto
            (``True`` for CUDA).
        num_workers: DataLoader worker count.  ``None`` -> auto (4 for CUDA,
            2 otherwise).
        compile_model: Wrap the model with :func:`torch.compile`.
        compile_backend: ``torch.compile`` backend.  ``None`` -> auto
            (``aot_eager`` for MPS, ``inductor`` for CUDA/CPU).
            MPS does not fully support the inductor backend.
        amp: Enable automatic mixed precision (CUDA only).
        amp_dtype: Optional autocast dtype.  ``None`` uses PyTorch's autocast
            default; explicit values should be ``float16`` or ``bfloat16``.
        cudnn_benchmark: Set :attr:`torch.backends.cudnn.benchmark`.
            ``None`` -> auto (``True`` for CUDA).
    """

    device: str = "auto"
    pin_memory: bool | None = None
    num_workers: int | None = None
    compile_model: bool = False
    compile_backend: str | None = None
    amp: bool = False
    amp_dtype: torch.dtype | str | None = None
    cudnn_benchmark: bool | None = None

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        self.amp_dtype = optional_dtype(self.amp_dtype)

        is_cuda = self.device.startswith("cuda")

        if self.pin_memory is None:
            self.pin_memory = is_cuda
        if self.num_workers is None:
            self.num_workers = 4 if is_cuda else 2
        if self.cudnn_benchmark is None:
            self.cudnn_benchmark = is_cuda

        # AMP only makes sense on CUDA
        if self.amp and not is_cuda:
            self.amp = False
        if self.amp_dtype is not None and self.amp_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("amp_dtype must be 'float16', 'bfloat16', or null")

    # Public helpers

    def apply_backend_settings(self) -> None:
        """Apply ``cudnn.benchmark``, TF32 matmul precision, etc."""
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
        # Enable TF32 tensor cores on Ampere+ GPUs (RTX 30xx, 40xx, Ada)
        if self.device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")

    def _resolve_compile_backend(self) -> str:
        """Pick a ``torch.compile`` backend appropriate for :attr:`device`.

        MPS does not fully support the ``inductor`` backend, so we
        default to ``aot_eager`` (graph capture without kernel codegen).
        """
        if self.compile_backend is not None:
            return self.compile_backend
        if self.device == "mps":
            return "aot_eager"
        return "inductor"

    def maybe_compile(self, model: nn.Module) -> nn.Module:
        """Optionally wrap *model* with :func:`torch.compile`."""
        if not self.compile_model:
            return model
        if not hasattr(torch, "compile"):
            return model
        backend = self._resolve_compile_backend()
        try:
            return torch.compile(model, backend=backend)
        except Exception as e:
            import warnings

            warnings.warn(
                f"torch.compile(backend={backend!r}) failed: {e}. Falling back to eager mode.",
                RuntimeWarning,
            )
            return model

    def get_scaler(self) -> torch.amp.GradScaler | None:
        """Return a :class:`GradScaler` when AMP is active, else ``None``."""
        if not self.amp:
            return None
        return torch.amp.GradScaler("cuda")

    def autocast_context(self) -> ContextManager:
        """Return an ``autocast`` context manager or :func:`nullcontext`."""
        if not self.amp:
            return nullcontext()
        if self.amp_dtype is None:
            return torch.amp.autocast("cuda")
        return torch.amp.autocast("cuda", dtype=self.amp_dtype)
