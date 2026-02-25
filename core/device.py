# Versor: Universal Geometric Algebra Neural Network (C) 2026 Eunkyum Kim
# Licensed under the Apache License, Version 2.0 | "Unbending" Paradigm

"""Device configuration and backend tuning for Versor.

Centralises device resolution, ``pin_memory``, ``torch.compile``,
``cudnn.benchmark``, and AMP (automatic mixed precision) into a single
:class:`DeviceConfig` dataclass.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch
import torch.nn as nn


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
        amp: Enable automatic mixed precision (CUDA only).
        cudnn_benchmark: Set :attr:`torch.backends.cudnn.benchmark`.
            ``None`` -> auto (``True`` for CUDA).
    """

    device: str = "auto"
    pin_memory: bool | None = None
    num_workers: int | None = None
    compile_model: bool = False
    amp: bool = False
    cudnn_benchmark: bool | None = None

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)

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

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def apply_backend_settings(self) -> None:
        """Apply ``cudnn.benchmark`` (and future backend knobs)."""
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = self.cudnn_benchmark

    def maybe_compile(self, model: nn.Module) -> nn.Module:
        """Optionally wrap *model* with :func:`torch.compile`."""
        if not self.compile_model:
            return model
        if not hasattr(torch, "compile"):
            return model
        return torch.compile(model)

    def get_scaler(self) -> torch.amp.GradScaler | None:
        """Return a :class:`GradScaler` when AMP is active, else ``None``."""
        if not self.amp:
            return None
        return torch.amp.GradScaler("cuda")

    def autocast_context(self) -> ContextManager:
        """Return an ``autocast`` context manager or :func:`nullcontext`."""
        if not self.amp:
            return nullcontext()
        return torch.amp.autocast("cuda")
