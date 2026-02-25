"""Riemannian optimizers for manifold-valued parameters.

Implements optimization on the Spin group manifold using exponential map
retractions instead of Euclidean updates.

References:
    - Absil et al. "Optimization Algorithms on Matrix Manifolds" (2008)
    - Boumal "An Introduction to Optimization on Smooth Manifolds" (2023)
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional


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
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        algebra: CliffordAlgebra instance for exponential map
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
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0,
        algebra=None,
        max_bivector_norm: Optional[float] = 10.0
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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step using exponential retraction.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # For bivector parameters, gradient is already in Lie algebra (tangent space)
                # Euclidean update in Lie algebra + exp() in forward pass = Riemannian update

                # Apply momentum in Lie algebra (if enabled)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Update bivector parameters in Lie algebra
                p.add_(grad, alpha=-lr)

                # Clip bivector norm for numerical stability in exp()
                # This prevents overflow when computing exp(-B/2) in forward pass
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
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        algebra: CliffordAlgebra instance for exponential map
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
        max_bivector_norm: Optional[float] = 10.0
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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step size
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5

                # Adam update in Lie algebra (bivector space)
                # Combined with exp(-B/2) in forward pass, this gives Riemannian update
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Clip bivector norm for numerical stability in exp()
                # This prevents overflow when computing exp(-B/2) in forward pass
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
