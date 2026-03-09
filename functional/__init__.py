"""Stateless functional operations for geometric algebra networks.

Includes activation functions, loss functions, and orthogonality enforcement.
"""

from .activation import GeometricGELU, GradeSwish

from .loss import (
    GeometricMSELoss,
    SubspaceLoss,
    IsometryLoss,
    BivectorRegularization,
    HermitianGradeRegularization,
    ChamferDistance,
    ConservativeLoss,
    PhysicsInformedLoss,
)

from .orthogonality import StrictOrthogonality, OrthogonalitySettings

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
