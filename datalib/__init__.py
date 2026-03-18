"""Dataset loaders for Versor tasks.

Provides dataset classes and loader functions with three-tier loading:
cached .pt files, raw data processing, and synthetic fallback.
"""

from .symbolic_regression import (
    SRDataset, get_sr_loaders, get_dataset_ids,
    SRBENCH_DATASETS, FIRST_PRINCIPLES_DATASETS, BLACKBOX_DATASETS,
)
from .md17 import get_md17_loaders
from .deap import DEAPDataset, get_deap_loaders, get_group_sizes
from .lqa import CLUTRRDataset, HANSDataset, BoolQNegDataset, get_lqa_loaders
from .arc import ToyARCDataset, ARCDataset, get_arc_loaders

__all__ = [
    "SRDataset",
    "get_sr_loaders",
    "get_dataset_ids",
    "SRBENCH_DATASETS",
    "FIRST_PRINCIPLES_DATASETS",
    "BLACKBOX_DATASETS",
    "get_md17_loaders",
    "DEAPDataset",
    "get_deap_loaders",
    "get_group_sizes",
    "CLUTRRDataset",
    "HANSDataset",
    "BoolQNegDataset",
    "get_lqa_loaders",
    "ToyARCDataset",
    "ARCDataset",
    "get_arc_loaders",
]
