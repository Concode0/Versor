# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Algebra construction config and dense/partitioned kernel dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import torch

from core.algebra import CliffordAlgebra
from core.device import optional_dtype, resolve_device, resolve_dtype
from core.module import AlgebraLike
from core.partitioned_algebra import DEFAULT_PARTITION_LEAF_N, PartitionedCliffordAlgebra

AlgebraKernel = Literal["auto", "dense", "partitioned"]


@dataclass(frozen=True)
class PartitionConfig:
    """Options specific to :class:`PartitionedCliffordAlgebra`."""

    leaf_n: int = DEFAULT_PARTITION_LEAF_N
    product_chunk_size: Optional[int] = None
    tree: Optional[str] = None
    accumulation_dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "accumulation_dtype", optional_dtype(self.accumulation_dtype))

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "PartitionConfig":
        """Build partition options from a Hydra/OmegaConf-compatible mapping."""
        if config is None:
            return cls()
        leaf_n = _mapping_get(config, "leaf_n", DEFAULT_PARTITION_LEAF_N)
        return cls(
            leaf_n=_int_or_default(leaf_n, DEFAULT_PARTITION_LEAF_N),
            product_chunk_size=_optional_int(_mapping_get(config, "product_chunk_size", None)),
            tree=_optional_str(_mapping_get(config, "tree", None)),
            accumulation_dtype=optional_dtype(_mapping_get(config, "accumulation_dtype", None)),
        )


@dataclass(frozen=True)
class AlgebraConfig:
    """Dense/partitioned algebra declaration."""

    p: int
    q: int = 0
    r: int = 0
    kernel: AlgebraKernel = "auto"
    partition_threshold: int = 8
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    exp_policy: str = "balanced"
    fixed_iterations: Optional[int] = None
    partition: PartitionConfig = field(default_factory=PartitionConfig)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], **overrides) -> "AlgebraConfig":
        """Build an algebra declaration from Hydra/OmegaConf config."""
        partition_mapping = _mapping_get(config, "partition", None)
        if partition_mapping is None:
            partition_mapping = _flat_partition_mapping(config)

        values = {
            "p": int(_mapping_get(config, "p", 0)),
            "q": int(_mapping_get(config, "q", 0)),
            "r": int(_mapping_get(config, "r", 0)),
            "kernel": _mapping_get(config, "kernel", "auto"),
            "partition_threshold": int(_mapping_get(config, "partition_threshold", 8)),
            "device": _mapping_get(config, "device", "cuda"),
            "dtype": resolve_dtype(_mapping_get(config, "dtype", torch.float32)),
            "exp_policy": _mapping_get(config, "exp_policy", "balanced"),
            "fixed_iterations": _optional_int(_mapping_get(config, "fixed_iterations", None)),
            "partition": PartitionConfig.from_mapping(partition_mapping),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        if not isinstance(values["partition"], PartitionConfig):
            values["partition"] = PartitionConfig.from_mapping(values["partition"])
        values["dtype"] = resolve_dtype(values["dtype"])
        return cls(**values)


def make_algebra(
    p: int,
    q: int = 0,
    r: int = 0,
    *,
    kernel: AlgebraKernel = "auto",
    partition_threshold: int = 8,
    partition: Optional[PartitionConfig] = None,
    device="cuda",
    dtype: torch.dtype = torch.float32,
    exp_policy: str = "balanced",
    fixed_iterations: Optional[int] = None,
) -> AlgebraLike:
    """Construct a dense or partitioned algebra according to a kernel policy."""
    kernel = _normalize_kernel(kernel)
    n = p + q + r
    selected_kernel = "partitioned" if kernel == "auto" and n > partition_threshold else kernel
    if selected_kernel == "auto":
        selected_kernel = "dense"

    resolved_device = resolve_device(device) if str(device) == "auto" else device
    resolved_dtype = resolve_dtype(dtype)

    if selected_kernel == "dense":
        return CliffordAlgebra(
            p,
            q,
            r,
            device=resolved_device,
            dtype=resolved_dtype,
            exp_policy=exp_policy,
            fixed_iterations=fixed_iterations,
        )

    partition = PartitionConfig() if partition is None else partition
    return PartitionedCliffordAlgebra(
        p,
        q,
        r,
        device=resolved_device,
        dtype=resolved_dtype,
        leaf_n=partition.leaf_n,
        product_chunk_size=partition.product_chunk_size,
        exp_policy=exp_policy,
        fixed_iterations=fixed_iterations,
        accumulation_dtype=partition.accumulation_dtype,
        partition_tree=partition.tree,
    )


def make_algebra_from_config(config: Mapping[str, Any], **overrides) -> AlgebraLike:
    """Construct an algebra from a Hydra/OmegaConf-compatible config mapping."""
    algebra_config = AlgebraConfig.from_mapping(config, **overrides)
    return make_algebra(
        algebra_config.p,
        algebra_config.q,
        algebra_config.r,
        kernel=algebra_config.kernel,
        partition_threshold=algebra_config.partition_threshold,
        partition=algebra_config.partition,
        device=algebra_config.device,
        dtype=algebra_config.dtype,
        exp_policy=algebra_config.exp_policy,
        fixed_iterations=algebra_config.fixed_iterations,
    )


def _mapping_get(config: Mapping[str, Any], key: str, default):
    """Return a value from plain mappings or OmegaConf DictConfig objects."""
    if config is None:
        return default
    return config.get(key, default)


def _flat_partition_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return partition options from flat ``algebra.*`` aliases."""
    return {
        "leaf_n": _mapping_get(config, "leaf_n", DEFAULT_PARTITION_LEAF_N),
        "product_chunk_size": _mapping_get(config, "product_chunk_size", None),
        "tree": _mapping_get(config, "partition_tree", _mapping_get(config, "tree", None)),
        "accumulation_dtype": _mapping_get(config, "accumulation_dtype", None),
    }


def _normalize_kernel(kernel: str) -> AlgebraKernel:
    """Validate and normalize algebra kernel names."""
    normalized = str(kernel).lower()
    if normalized not in {"auto", "dense", "partitioned"}:
        raise ValueError(f"Unknown algebra kernel {kernel!r}; expected 'auto', 'dense', or 'partitioned'")
    return normalized  # type: ignore[return-value]


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _int_or_default(value, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _optional_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
