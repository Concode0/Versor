"""Stateless functional operations for geometric algebra networks.

Includes activation functions, loss functions, and orthogonality enforcement.
"""

from .activation import GeometricGELU, GradeSwish
from .loss import (
    BivectorRegularization,
    ChamferDistance,
    ConservativeLoss,
    GeometricMSELoss,
    HermitianGradeRegularization,
    IsometryLoss,
    PhysicsInformedLoss,
    SubspaceLoss,
)
from .orthogonality import OrthogonalitySettings, StrictOrthogonality

__all__ = [
    # activations
    "GeometricGELU",
    "GradeSwish",
    # losses
    "GeometricMSELoss",
    "SubspaceLoss",
    "IsometryLoss",
    "BivectorRegularization",
    "HermitianGradeRegularization",
    "ChamferDistance",
    "ConservativeLoss",
    "PhysicsInformedLoss",
    # orthogonality
    "StrictOrthogonality",
    "OrthogonalitySettings",
]
