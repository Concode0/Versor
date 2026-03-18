# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Turing Machine (GTM) package — ARC-AGI v4."""

from .grid_codec import GridCodec
from .cpu import GeometricCPU, ColorUnit
from .control_plane import ControlPlane
from .superposition import GeometricSuperpositionSearch
from .turing_step import TuringStep
from .adaptive_halt import AdaptiveHalt
from .turing_vm import TuringVM
from .heads import GridReconstructionHead
from .rule_memory import RuleAggregator
from .gtm_net import GTMNet
from .analysis import GTMAnalyzer

__all__ = [
    "GridCodec",
    "GeometricCPU",
    "ColorUnit",
    "ControlPlane",
    "GeometricSuperpositionSearch",
    "TuringStep",
    "AdaptiveHalt",
    "TuringVM",
    "GridReconstructionHead",
    "RuleAggregator",
    "GTMNet",
    "GTMAnalyzer",
]
