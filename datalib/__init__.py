"""Dataset loaders for Versor tasks.

Provides dataset classes and loader functions with three-tier loading:
cached .pt files, raw data processing, and synthetic fallback.

Submodules have optional dependencies — import directly from the
relevant submodule rather than from this package:

    from datalib.md17 import get_md17_loaders
    from datalib.symbolic_regression import get_sr_loaders
    from datalib.lqa import get_lqa_loaders
    from datalib.deap import get_deap_loaders
"""
