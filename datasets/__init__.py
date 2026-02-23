"""Dataset loaders for Versor tasks.

Provides dataset classes and loader functions with three-tier loading:
cached .pt files, raw data processing, and synthetic fallback.
"""

from .feynman import FEYNMAN_EQUATIONS, get_feynman_loaders
from .har import HARDataset
from .newsgroups import NewsgroupsClassificationDataset
from .text import TextDataset, get_text_loaders
from .md17 import get_md17_loaders
from .abc import get_abc_loaders
from .pdbbind import get_pdbbind_loaders
from .weatherbench import get_weatherbench_loaders

# QM9 requires torch_geometric
try:
    from .qm9 import VersorQM9, get_qm9_loaders
except ImportError:
    VersorQM9 = None
    get_qm9_loaders = None

__all__ = [
    "FEYNMAN_EQUATIONS",
    "get_feynman_loaders",
    "HARDataset",
    "NewsgroupsClassificationDataset",
    "TextDataset",
    "get_text_loaders",
    "get_md17_loaders",
    "get_abc_loaders",
    "get_pdbbind_loaders",
    "get_weatherbench_loaders",
    # torch_geometric dependent
    "VersorQM9",
    "get_qm9_loaders",
]
