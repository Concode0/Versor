# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Turing Machine (GTM) package — v5 World Model + Search Plane."""

from .grid_codec import GridCodec
from .action_engine import ActionEngine, DiscreteActionHead
from .log_manifold import LogManifoldProjector
from .info_geometry import FIMEvaluator
from .search_plane import AlgebraicProjection, AlgebraicLift, SearchPlane
from .adaptive_halt import FIMAdaptiveHalt
from .world_model import CellAttention, WorldModelStep, WorldModel
from .heads import GridReconstructionHead
from .rule_memory import RuleAggregator
from .gtm_net import GTMNet
from .analysis import GTMAnalyzer

__all__ = [
    "GridCodec",
    "ActionEngine",
    "DiscreteActionHead",
    "LogManifoldProjector",
    "FIMEvaluator",
    "AlgebraicProjection",
    "AlgebraicLift",
    "SearchPlane",
    "FIMAdaptiveHalt",
    "CellAttention",
    "WorldModelStep",
    "WorldModel",
    "GridReconstructionHead",
    "RuleAggregator",
    "GTMNet",
    "GTMAnalyzer",
]
