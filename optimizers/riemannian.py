"""Riemannian optimizers for manifold-valued parameters.

Implements optimization on product manifolds Spin(p,q) x S^{n-1} x R^d
using per-parameter retraction dispatch.

Each parameter can be tagged with a manifold type via ``p._manifold``:
    - ``'spin'``: Lie algebra bivectors — retracted via bivector norm clipping
      (the forward-pass exp map completes the Riemannian update on Spin(n))
    - ``'sphere'``: Unit vectors on S^{n-1} — retracted via L2 normalization
    - ``'euclidean'`` (or untagged): Standard unconstrained parameters

Use ``from_model()`` to auto-group parameters by manifold tag, or pass
``model.parameters()`` directly for legacy behavior (all params get spin
retraction).

References:
    - Absil et al. "Optimization Algorithms on Matrix Manifolds" (2008)
    - Boumal "An Introduction to Optimization on Smooth Manifolds" (2023)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

MANIFOLD_SPIN = "spin"
MANIFOLD_SPHERE = "sphere"
MANIFOLD_EUCLIDEAN = "euclidean"

_VALID_MANIFOLDS = {MANIFOLD_SPIN, MANIFOLD_SPHERE, MANIFOLD_EUCLIDEAN}


def tag_manifold(param: nn.Parameter, manifold: str) -> nn.Parameter:
    """Tag a parameter with its Riemannian manifold type.

    Layers should call this (or set ``param._manifold`` directly) in their
    ``__init__`` so that :meth:`RiemannianAdam.from_model` can auto-group
    parameters for correct per-manifold retraction.

    Args:
        param: The parameter to tag.
        manifold: One of ``'spin'``, ``'sphere'``, ``'euclidean'``.

    Returns:
        The same parameter (for chaining).
    """
    if manifold not in _VALID_MANIFOLDS:
        raise ValueError(f"Unknown manifold '{manifold}'. Must be one of {_VALID_MANIFOLDS}")
    param._manifold = manifold
    return param


def group_parameters_by_manifold(
    model: nn.Module,
) -> Dict[str, List[nn.Parameter]]:
    """Group a model's parameters by their ``_manifold`` tag.

    Parameters without a ``_manifold`` attribute are placed in the
    ``'euclidean'`` group.

    Args:
        model: The model whose parameters to group.

    Returns:
        Dict mapping manifold name to list of parameters.
    """
    groups: Dict[str, List[nn.Parameter]] = {
        MANIFOLD_SPIN: [],
        MANIFOLD_SPHERE: [],
        MANIFOLD_EUCLIDEAN: [],
    }
    for p in model.parameters():
        manifold = getattr(p, "_manifold", MANIFOLD_EUCLIDEAN)
        groups[manifold].append(p)
    return groups


class ExponentialSGD(Optimizer):
    """SGD with exponential map retraction for rotor parameters.

    Instead of Euclidean update: theta <- theta - lr * grad_theta
    Uses manifold update: R <- R . exp(lr * grad_B)

    where grad_B is the gradient in the Lie algebra (bivector space).

    Since Versor parameterizes rotors via bivectors (the Lie algebra),
    Euclidean gradient updates in bivector space ARE geometrically meaningful.
    The exponential map in the forward pass (R = exp(-B/2)) completes the
    Riemannian update on the Spin(n) manifold.

    Args:
        params (Iterable): Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        algebra (CliffordAlgebra): CliffordAlgebra instance for exponential map
        max_bivector_norm: Maximum allowed bivector norm for numerical stability.
            If not None, clips bivector norms after each update. (default: 10.0)

    Example:
        >>> algebra = CliffordAlgebra(p=3, q=0, device='cpu')
        >>> model = RotorLayer(algebra, channels=4)
        >>> optimizer = ExponentialSGD(
        ...     model.parameters(), lr=0.01, algebra=algebra
        ... )
        >>>
        >>> # Training loop
        >>> for data in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(data), target)
        ...     loss.backward()
        ...     optimizer.step()  # Uses exponential map!
    """

    def __init__(
        self, params, lr: float = 0.01, momentum: float = 0, algebra=None, max_bivector_norm: Optional[float] = 10.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if algebra is None:
            raise ValueError("Must provide CliffordAlgebra instance")
        if max_bivector_norm is not None and max_bivector_norm <= 0.0:
            raise ValueError(f"Invalid max_bivector_norm: {max_bivector_norm}")

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.algebra = algebra
        self.max_bivector_norm = max_bivector_norm

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0,
        algebra=None,
        max_bivector_norm: Optional[float] = 10.0,
    ):
        """Create optimizer with auto-detected manifold parameter groups.

        Inspects ``p._manifold`` tags on each parameter and creates separate
        groups for spin, sphere, and euclidean parameters so that each group
        receives the correct retraction in :meth:`step`.

        Args:
            model: The model to optimize.
            lr: Learning rate.
            momentum: Momentum factor.
            algebra: CliffordAlgebra instance (required).
            max_bivector_norm: Clip threshold for spin params.

        Returns:
            ExponentialSGD instance with per-manifold parameter groups.
        """
        grouped = group_parameters_by_manifold(model)
        param_groups = []
        for manifold in (MANIFOLD_SPIN, MANIFOLD_SPHERE, MANIFOLD_EUCLIDEAN):
            params = grouped[manifold]
            if params:
                param_groups.append({"params": params, "manifold": manifold})
        if not param_groups:
            raise ValueError("Model has no parameters")
        return cls(param_groups, lr=lr, momentum=momentum, algebra=algebra, max_bivector_norm=max_bivector_norm)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """Performs a single optimization step using exponential retraction.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[torch.Tensor]: The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            manifold = group.get("manifold", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply momentum (if enabled)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(grad)
                    else:
                        buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Update parameters
                p.add_(grad, alpha=-lr)

                # Per-manifold retraction
                if manifold == MANIFOLD_SPHERE:
                    # Project back to unit sphere: v / ||v||
                    p_norm = p.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    p.div_(p_norm)
                elif manifold == MANIFOLD_EUCLIDEAN:
                    pass  # No retraction for Euclidean parameters
                else:
                    # 'spin' or None (legacy): bivector norm clipping
                    if self.max_bivector_norm is not None:
                        p_norm = p.norm(dim=-1, keepdim=True)
                        scale = torch.clamp(p_norm / self.max_bivector_norm, min=1.0)
                        p.div_(scale)

        return loss


class RiemannianAdam(Optimizer):
    """Adam optimizer with exponential map retraction for rotor parameters.

    Implements Adam momentum in the Lie algebra (bivector space) with
    exponential map updates on the manifold.

    Since Versor parameterizes rotors via bivectors (the Lie algebra), Adam
    momentum naturally lives in the tangent space. The exponential map in the
    forward pass (R = exp(-B/2)) completes the Riemannian update on Spin(n).

    Args:
        params (Iterable): Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        algebra (CliffordAlgebra): CliffordAlgebra instance for exponential map
        max_bivector_norm: Maximum allowed bivector norm for numerical stability.
            If not None, clips bivector norms after each update. (default: 10.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        algebra=None,
        max_bivector_norm: Optional[float] = 10.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if algebra is None:
            raise ValueError("Must provide CliffordAlgebra instance")
        if max_bivector_norm is not None and max_bivector_norm <= 0.0:
            raise ValueError(f"Invalid max_bivector_norm: {max_bivector_norm}")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.algebra = algebra
        self.max_bivector_norm = max_bivector_norm

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        algebra=None,
        max_bivector_norm: Optional[float] = 10.0,
    ):
        """Create optimizer with auto-detected manifold parameter groups.

        Inspects ``p._manifold`` tags on each parameter and creates separate
        groups for spin, sphere, and euclidean parameters so that each group
        receives the correct retraction in :meth:`step`.

        Args:
            model: The model to optimize.
            lr: Learning rate.
            betas: Coefficients for running averages.
            eps: Numerical stability term.
            algebra: CliffordAlgebra instance (required).
            max_bivector_norm: Clip threshold for spin params.

        Returns:
            RiemannianAdam instance with per-manifold parameter groups.
        """
        grouped = group_parameters_by_manifold(model)
        param_groups = []
        for manifold in (MANIFOLD_SPIN, MANIFOLD_SPHERE, MANIFOLD_EUCLIDEAN):
            params = grouped[manifold]
            if params:
                param_groups.append({"params": params, "manifold": manifold})
        if not param_groups:
            raise ValueError("Model has no parameters")
        return cls(param_groups, lr=lr, betas=betas, eps=eps, algebra=algebra, max_bivector_norm=max_bivector_norm)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.

        Applies Adam momentum updates to all parameters, then dispatches
        per-manifold retraction:

        - **spin** (or legacy/untagged): bivector norm clipping
        - **sphere**: L2 normalization to unit sphere
        - **euclidean**: no retraction

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[torch.Tensor]: The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            manifold = group.get("manifold", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute step size
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2**0.5

                # Adam update in parameter space
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Per-manifold retraction
                if manifold == MANIFOLD_SPHERE:
                    # Project back to unit sphere: v / ||v||
                    p_norm = p.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    p.div_(p_norm)
                elif manifold == MANIFOLD_EUCLIDEAN:
                    pass  # No retraction for Euclidean parameters
                else:
                    # 'spin' or None (legacy): bivector norm clipping
                    if self.max_bivector_norm is not None:
                        p_norm = p.norm(dim=-1, keepdim=True)
                        scale = torch.clamp(p_norm / self.max_bivector_norm, min=1.0)
                        p.div_(scale)

        return loss


def project_to_tangent_space(point, vector, algebra):
    """Project a vector to the tangent space at a point on Spin(n).

    For rotors R in Spin(n), the tangent space at R is:
        T_R Spin(n) = { R . B | B is a bivector }

    Args:
        point: Current point on manifold (rotor) [..., dim]
        vector: Vector to project [..., dim]
        algebra: CliffordAlgebra instance

    Returns:
        Projected vector in tangent space [..., dim]
    """
    # Compute ~R . vector
    R_rev = algebra.reverse(point)
    tangent = algebra.geometric_product(R_rev, vector)

    # Project to bivector part (Lie algebra)
    # This extracts only the grade-2 (bivector) components
    bivector = algebra.grade_projection(tangent, grade=2)

    # Map back to tangent space: R . bivector
    return algebra.geometric_product(point, bivector)


def exponential_retraction(point, tangent_vector, algebra):
    """Exponential map: move from point along tangent vector on manifold.

    For Spin(n), the exponential map is:
        Exp_R(R.B) = R . exp(B)

    where B is a bivector in the Lie algebra.

    Args:
        point: Current point on manifold (rotor) [..., dim]
        tangent_vector: Tangent vector (direction to move) [..., dim]
        algebra: CliffordAlgebra instance

    Returns:
        New point on manifold [..., dim]
    """
    # Extract bivector from tangent vector
    R_rev = algebra.reverse(point)
    bivector = algebra.geometric_product(R_rev, tangent_vector)
    bivector = algebra.grade_projection(bivector, grade=2)

    # Exponential map
    update = algebra.exp(bivector)

    # Apply update: R_new = R_old . exp(B)
    return algebra.geometric_product(point, update)
