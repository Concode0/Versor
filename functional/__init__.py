"""Stateless functional operations for geometric algebra networks.

Includes activation functions and loss functions.
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
]
