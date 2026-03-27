"""Riemannian optimizers for geometric algebra neural networks.

Provides optimizers that respect the manifold structure of parameters:
Spin group (bivectors), unit sphere (vectors), and Euclidean (unconstrained).
"""

from .riemannian import (
    ExponentialSGD,
    RiemannianAdam,
    project_to_tangent_space,
    exponential_retraction,
    tag_manifold,
    group_parameters_by_manifold,
    MANIFOLD_SPIN,
    MANIFOLD_SPHERE,
    MANIFOLD_EUCLIDEAN,
)

__all__ = [
    'ExponentialSGD',
    'RiemannianAdam',
    'project_to_tangent_space',
    'exponential_retraction',
    'tag_manifold',
    'group_parameters_by_manifold',
    'MANIFOLD_SPIN',
    'MANIFOLD_SPHERE',
    'MANIFOLD_EUCLIDEAN',
]
