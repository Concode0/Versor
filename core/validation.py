# Versor: Universal Geometric Algebra Neural Network (C) 2026 Eunkyum Kim
# Licensed under the Apache License, Version 2.0 | "Unbending" Paradigm

"""Lightweight input validation for Versor tensors.

All checks use ``assert`` so they are free under ``python -O``.
Set ``VALIDATE = False`` to disable even without the -O flag.
"""

import torch

VALIDATE = True


def check_multivector(x: torch.Tensor, algebra, name: str = "x") -> None:
    """Assert *x* looks like a multivector for *algebra*.

    Checks ``x.ndim >= 1`` and ``x.shape[-1] == algebra.dim``.
    """
    if not VALIDATE:
        return
    assert x.ndim >= 1, (
        f"{name}: expected ndim >= 1, got shape {tuple(x.shape)}"
    )
    assert x.shape[-1] == algebra.dim, (
        f"{name}: last dim should be {algebra.dim} (algebra dim), "
        f"got {x.shape[-1]} (shape {tuple(x.shape)})"
    )


def check_channels(x: torch.Tensor, expected: int, name: str = "x") -> None:
    """Assert the channel dimension of *x* equals *expected*.

    Assumes layout ``[Batch, Channels, Dim]`` (ndim >= 3, channel at -2).
    """
    if not VALIDATE:
        return
    assert x.ndim >= 3, (
        f"{name}: expected ndim >= 3 for channel check, got shape {tuple(x.shape)}"
    )
    assert x.shape[-2] == expected, (
        f"{name}: expected {expected} channels, got {x.shape[-2]} "
        f"(shape {tuple(x.shape)})"
    )
