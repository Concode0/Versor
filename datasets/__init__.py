"""Dataset loaders for Versor tasks.

Provides dataset classes and loader functions with three-tier loading:
cached .pt files, raw data processing, and synthetic fallback.
"""

from .symbolic_regression import (
    SRDataset, get_sr_loaders, get_dataset_ids,
    SRBENCH_DATASETS, FIRST_PRINCIPLES_DATASETS, BLACKBOX_DATASETS,
)
from .text import TextDataset, get_text_loaders
from .md17 import get_md17_loaders
from .deeplense import DeepLenseDataset, get_deeplense_loaders

__all__ = [
    "SRDataset",
    "get_sr_loaders",
    "get_dataset_ids",
    "SRBENCH_DATASETS",
    "FIRST_PRINCIPLES_DATASETS",
    "BLACKBOX_DATASETS",
    "TextDataset",
    "get_text_loaders",
    "get_md17_loaders",
    "DeepLenseDataset",
    "get_deeplense_loaders",
]
