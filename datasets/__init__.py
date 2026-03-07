"""Dataset loaders for Versor tasks.

Provides dataset classes and loader functions with three-tier loading:
cached .pt files, raw data processing, and synthetic fallback.
"""

from .symbolic_regression import (
    SRDataset, get_sr_loaders, get_dataset_ids,
    SRBENCH_DATASETS, FIRST_PRINCIPLES_DATASETS, BLACKBOX_DATASETS,
)
from .md17 import get_md17_loaders

__all__ = [
    "SRDataset",
    "get_sr_loaders",
    "get_dataset_ids",
    "SRBENCH_DATASETS",
    "FIRST_PRINCIPLES_DATASETS",
    "BLACKBOX_DATASETS",
    "get_md17_loaders",
]
