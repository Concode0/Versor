"""Riemannian optimizers for geometric algebra neural networks.

Provides optimizers that respect the manifold structure of the Spin group
instead of treating parameter space as flat Euclidean space.
"""

from .riemannian import (
    ExponentialSGD,
    RiemannianAdam,
    project_to_tangent_space,
    exponential_retraction,
)

__all__ = [
    'ExponentialSGD',
    'RiemannianAdam',
    'project_to_tangent_space',
    'exponential_retraction',
]
