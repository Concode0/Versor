"""Task-specific model architectures built on the Versor layer stack.

Models combine rotor layers, linear transformations, normalization,
and activation functions into complete architectures.

Submodules have optional dependencies — import directly from the
relevant submodule rather than from this package:

    from models.md17 import MD17ForceNet          # requires --extra md17
    from models.sr import SRGBN                   # requires --extra sr
    from models.lqa import GLRNet                 # requires --extra lqa
    from models.deap import EEGNet
    from models.blocks import GeometricBladeNetwork
"""
