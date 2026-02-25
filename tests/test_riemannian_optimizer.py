"""Tests for Riemannian optimizers.

Verifies:
1. Gradient flow and parameter updates
2. Numerical stability via bivector norm clipping
3. Convergence on synthetic tasks
4. Geometric properties (rotor manifold membership)
5. Integration with layers
"""

import pytest
import torch
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from optimizers.riemannian import ExponentialSGD, RiemannianAdam
from layers.rotor import RotorLayer
from layers.multi_rotor import MultiRotorLayer
from layers.rotor_gadget import RotorGadget


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def algebra():
    """Standard 3D Clifford algebra."""
    return CliffordAlgebra(p=3, q=0, device='cpu')


@pytest.fixture
def rotor_layer(algebra):
    """Simple rotor layer for testing."""
    return RotorLayer(algebra, channels=4)


@pytest.fixture
def multi_rotor_layer(algebra):
    """Multi-rotor layer for testing."""
    return MultiRotorLayer(algebra, channels=4, num_rotors=2)


@pytest.fixture
def rotor_gadget(algebra):
    """RotorGadget layer for testing."""
    return RotorGadget(algebra, in_channels=4, out_channels=8)


# ============================================================================
# Unit Tests: Gradient Flow
# ============================================================================

def test_exponential_sgd_gradient_flow(algebra, rotor_layer):
    """Verify gradients exist and flow correctly."""
    optimizer = ExponentialSGD(
        rotor_layer.parameters(), lr=0.01, algebra=algebra
    )

    x = torch.randn(8, 4, 8, requires_grad=True)
    y = rotor_layer(x)
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    assert rotor_layer.bivector_weights.grad is not None
    assert not torch.isnan(rotor_layer.bivector_weights.grad).any()
    assert not torch.isinf(rotor_layer.bivector_weights.grad).any()

    # Take step
    optimizer.step()

    # Check parameters remain finite
    assert not torch.isnan(rotor_layer.bivector_weights).any()
    assert not torch.isinf(rotor_layer.bivector_weights).any()


def test_riemannian_adam_gradient_flow(algebra, rotor_layer):
    """Verify gradients exist and flow correctly with Adam."""
    optimizer = RiemannianAdam(
        rotor_layer.parameters(), lr=0.001, algebra=algebra
    )

    x = torch.randn(8, 4, 8, requires_grad=True)
    y = rotor_layer(x)
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    assert rotor_layer.bivector_weights.grad is not None
    assert not torch.isnan(rotor_layer.bivector_weights.grad).any()
    assert not torch.isinf(rotor_layer.bivector_weights.grad).any()

    # Take step
    optimizer.step()

    # Check parameters remain finite
    assert not torch.isnan(rotor_layer.bivector_weights).any()
    assert not torch.isinf(rotor_layer.bivector_weights).any()


def test_exponential_sgd_momentum(algebra, rotor_layer):
    """Verify momentum accumulation works."""
    optimizer = ExponentialSGD(
        rotor_layer.parameters(), lr=0.01, momentum=0.9, algebra=algebra
    )

    x = torch.randn(8, 4, 8)

    # First step
    y1 = rotor_layer(x)
    loss1 = y1.sum()
    loss1.backward()
    optimizer.step()

    # Check momentum buffer created
    param = rotor_layer.bivector_weights
    assert 'momentum_buffer' in optimizer.state[param]

    optimizer.zero_grad()

    # Second step
    y2 = rotor_layer(x)
    loss2 = y2.sum()
    loss2.backward()

    # Check momentum buffer updated
    buf = optimizer.state[param]['momentum_buffer']
    assert not torch.allclose(buf, torch.zeros_like(buf))


def test_riemannian_adam_momentum(algebra, rotor_layer):
    """Verify Adam momentum accumulation works."""
    optimizer = RiemannianAdam(
        rotor_layer.parameters(), lr=0.001, algebra=algebra
    )

    x = torch.randn(8, 4, 8)

    # First step
    y1 = rotor_layer(x)
    loss1 = y1.sum()
    loss1.backward()
    optimizer.step()

    # Check state created
    param = rotor_layer.bivector_weights
    state = optimizer.state[param]
    assert 'exp_avg' in state
    assert 'exp_avg_sq' in state
    assert state['step'] == 1

    optimizer.zero_grad()

    # Second step
    y2 = rotor_layer(x)
    loss2 = y2.sum()
    loss2.backward()
    optimizer.step()

    # Check state updated
    assert state['step'] == 2
    assert not torch.allclose(state['exp_avg'], torch.zeros_like(state['exp_avg']))


# ============================================================================
# Unit Tests: Parameter Updates
# ============================================================================

def test_bivector_norm_clipping_sgd(algebra, rotor_layer):
    """Verify bivector norm clipping prevents overflow."""
    max_norm = 5.0
    optimizer = ExponentialSGD(
        rotor_layer.parameters(),
        lr=0.1,
        algebra=algebra,
        max_bivector_norm=max_norm
    )

    # Intentionally set large bivector norms
    with torch.no_grad():
        rotor_layer.bivector_weights.fill_(10.0)

    x = torch.randn(4, 4, 8)
    y = rotor_layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # After clipping, norms should be bounded
    norms = rotor_layer.bivector_weights.norm(dim=-1)
    assert norms.max() <= max_norm + 0.1  # Small tolerance for numerical precision


def test_bivector_norm_clipping_adam(algebra, rotor_layer):
    """Verify bivector norm clipping prevents overflow with Adam."""
    max_norm = 5.0
    optimizer = RiemannianAdam(
        rotor_layer.parameters(),
        lr=0.1,
        algebra=algebra,
        max_bivector_norm=max_norm
    )

    # Intentionally set large bivector norms
    with torch.no_grad():
        rotor_layer.bivector_weights.fill_(10.0)

    x = torch.randn(4, 4, 8)
    y = rotor_layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # After clipping, norms should be bounded
    norms = rotor_layer.bivector_weights.norm(dim=-1)
    assert norms.max() <= max_norm + 0.1


def test_no_clipping_when_disabled(algebra, rotor_layer):
    """Verify clipping can be disabled."""
    optimizer = ExponentialSGD(
        rotor_layer.parameters(),
        lr=0.01,
        algebra=algebra,
        max_bivector_norm=None  # Disable clipping
    )

    # Set moderate norms
    with torch.no_grad():
        rotor_layer.bivector_weights.fill_(2.0)

    initial_norms = rotor_layer.bivector_weights.norm(dim=-1).clone()

    x = torch.randn(4, 4, 8)
    y = rotor_layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Norms can change freely without clipping
    final_norms = rotor_layer.bivector_weights.norm(dim=-1)
    # At least one should differ (update occurred)
    assert not torch.allclose(initial_norms, final_norms)


# ============================================================================
# Convergence Tests
# ============================================================================

def test_convergence_synthetic_rotation_sgd(algebra):
    """Fit a known rotor transformation using ExponentialSGD."""
    # Ground truth: rotation in e12 plane (index 3 is e12 bivector)
    B_true = torch.zeros(1, 1, algebra.dim)
    B_true[0, 0, 3] = 0.5  # Small rotation in e12 plane
    R_true = algebra.exp(-0.5 * B_true)

    # Generate data: y = R_true . x . ~R_true
    x = torch.randn(100, 1, 8)
    R_true_rev = algebra.reverse(R_true)
    y_true = algebra.geometric_product(
        algebra.geometric_product(R_true, x), R_true_rev
    )

    # Train
    layer = RotorLayer(algebra, channels=1)
    optimizer = ExponentialSGD(layer.parameters(), lr=0.1, algebra=algebra)

    for _ in range(200):
        optimizer.zero_grad()
        y_pred = layer(x)
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()

    # Should converge to near-zero loss
    with torch.no_grad():
        y_pred = layer(x)
        final_loss = F.mse_loss(y_pred, y_true).item()

    assert final_loss < 1e-3, f"Loss too high: {final_loss}"


def test_convergence_synthetic_rotation_adam(algebra):
    """Fit a known rotor transformation using RiemannianAdam."""
    # Ground truth: rotation in e12 plane (index 3 is e12 bivector)
    B_true = torch.zeros(1, 1, algebra.dim)
    B_true[0, 0, 3] = 0.5  # Small rotation in e12 plane
    R_true = algebra.exp(-0.5 * B_true)

    # Generate data
    x = torch.randn(100, 1, 8)
    R_true_rev = algebra.reverse(R_true)
    y_true = algebra.geometric_product(
        algebra.geometric_product(R_true, x), R_true_rev
    )

    # Train
    layer = RotorLayer(algebra, channels=1)
    optimizer = RiemannianAdam(layer.parameters(), lr=0.01, algebra=algebra)

    for _ in range(200):
        optimizer.zero_grad()
        y_pred = layer(x)
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()

    # Should converge
    with torch.no_grad():
        y_pred = layer(x)
        final_loss = F.mse_loss(y_pred, y_true).item()

    assert final_loss < 1e-3, f"Loss too high: {final_loss}"


def test_compare_sgd_convergence(algebra):
    """Compare ExponentialSGD vs standard SGD on simple task."""
    # Target: learn a specific transformation (not mapping to zero, which rotors can't do)
    x = torch.randn(50, 4, 8)
    # Generate target by applying a random rotor transformation
    B_target = torch.randn(1, 1, algebra.dim) * 0.1
    R_target = algebra.exp(-0.5 * B_target)
    R_target_rev = algebra.reverse(R_target)
    y_target = algebra.geometric_product(
        algebra.geometric_product(R_target, x), R_target_rev
    )

    # Train with ExponentialSGD
    layer1 = RotorLayer(algebra, channels=4)
    opt1 = ExponentialSGD(layer1.parameters(), lr=0.01, algebra=algebra)

    losses1 = []
    for _ in range(100):
        opt1.zero_grad()
        y = layer1(x)
        loss = F.mse_loss(y, y_target)
        loss.backward()
        opt1.step()
        losses1.append(loss.item())

    # Train with standard SGD
    layer2 = RotorLayer(algebra, channels=4)
    opt2 = torch.optim.SGD(layer2.parameters(), lr=0.01)

    losses2 = []
    for _ in range(100):
        opt2.zero_grad()
        y = layer2(x)
        loss = F.mse_loss(y, y_target)
        loss.backward()
        opt2.step()
        losses2.append(loss.item())

    # Both should show improvement (not testing superiority, just that both work)
    assert losses1[-1] < losses1[0], "ExponentialSGD should reduce loss"
    assert losses2[-1] < losses2[0], "Standard SGD should reduce loss"


# ============================================================================
# Geometric Validation
# ============================================================================

def test_rotor_manifold_membership_after_optimization(algebra):
    """Verify rotors remain on manifold: ~RR ~= 1 after optimization."""
    layer = RotorLayer(algebra, channels=4)
    optimizer = RiemannianAdam(layer.parameters(), lr=0.01, algebra=algebra)

    x = torch.randn(20, 4, 8)
    y_target = torch.randn(20, 4, 8)

    # Train for a few steps
    for _ in range(50):
        optimizer.zero_grad()
        y = layer(x)
        loss = F.mse_loss(y, y_target)
        loss.backward()
        optimizer.step()

    # Extract rotors using the same logic as RotorLayer.forward()
    # RotorLayer stores only bivector components, need to embed in full space
    B_full = torch.zeros(layer.channels, algebra.dim)
    bivector_indices = layer.bivector_indices.unsqueeze(0).expand(layer.channels, -1)
    B_full.scatter_(1, bivector_indices, layer.bivector_weights)

    R = algebra.exp(-0.5 * B_full)
    R_rev = algebra.reverse(R)
    RR_rev = algebra.geometric_product(R, R_rev)

    # Should be identity (scalar = 1, other grades = 0)
    identity = torch.zeros_like(RR_rev)
    identity[..., 0] = 1.0

    assert torch.allclose(RR_rev, identity, atol=1e-3)


def test_isometry_preservation(algebra):
    """Verify rotors preserve norms: ||R.x.~R|| = ||x||."""
    layer = RotorLayer(algebra, channels=4)
    optimizer = ExponentialSGD(layer.parameters(), lr=0.01, algebra=algebra)

    x = torch.randn(20, 4, 8)
    y_target = torch.randn(20, 4, 8)

    # Train
    for _ in range(50):
        optimizer.zero_grad()
        y = layer(x)
        loss = F.mse_loss(y, y_target)
        loss.backward()
        optimizer.step()

    # Check isometry
    with torch.no_grad():
        y = layer(x)
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        # Rotors should preserve norms (up to numerical precision)
        assert torch.allclose(x_norms, y_norms, rtol=1e-4, atol=1e-4)


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_with_multi_rotor_layer(algebra, multi_rotor_layer):
    """Verify optimizers work with MultiRotorLayer."""
    optimizer = RiemannianAdam(
        multi_rotor_layer.parameters(), lr=0.001, algebra=algebra
    )

    x = torch.randn(8, 4, 8)
    y_target = torch.randn(8, 4, 8)

    # Train for a few steps
    for _ in range(20):
        optimizer.zero_grad()
        y = multi_rotor_layer(x)
        loss = F.mse_loss(y, y_target)
        loss.backward()
        optimizer.step()

    # Should complete without errors and have finite parameters
    # MultiRotorLayer uses 'rotor_bivectors', not 'bivector_weights'
    assert not torch.isnan(multi_rotor_layer.rotor_bivectors).any()
    assert not torch.isinf(multi_rotor_layer.rotor_bivectors).any()


def test_integration_with_rotor_gadget(algebra, rotor_gadget):
    """Verify optimizers work with RotorGadget."""
    optimizer = ExponentialSGD(
        rotor_gadget.parameters(), lr=0.01, algebra=algebra
    )

    x = torch.randn(8, 4, 8)
    y_target = torch.randn(8, 8, 8)

    # Train for a few steps
    for _ in range(20):
        optimizer.zero_grad()
        y = rotor_gadget(x)
        loss = F.mse_loss(y, y_target)
        loss.backward()
        optimizer.step()

    # Should complete without errors
    for param in rotor_gadget.parameters():
        assert not torch.isnan(param).any()
        assert not torch.isinf(param).any()


def test_backward_compatibility_with_standard_params(algebra):
    """Verify optimizers work with non-bivector parameters (e.g., scalars)."""
    # Model with both bivector and scalar parameters
    class MixedModel(torch.nn.Module):
        def __init__(self, algebra):
            super().__init__()
            self.rotor = RotorLayer(algebra, channels=2)
            self.scalar = torch.nn.Parameter(torch.randn(2, 8))

        def forward(self, x):
            return self.rotor(x) + self.scalar

    model = MixedModel(algebra)
    optimizer = RiemannianAdam(model.parameters(), lr=0.001, algebra=algebra)

    x = torch.randn(4, 2, 8)
    y_target = torch.randn(4, 2, 8)

    # Should handle both parameter types
    for _ in range(10):
        optimizer.zero_grad()
        y = model(x)
        loss = F.mse_loss(y, y_target)
        loss.backward()
        optimizer.step()

    # Check all parameters finite
    for param in model.parameters():
        assert not torch.isnan(param).any()
        assert not torch.isinf(param).any()


def test_optimizer_state_dict(algebra, rotor_layer):
    """Verify optimizer state can be saved and loaded."""
    optimizer = RiemannianAdam(
        rotor_layer.parameters(), lr=0.001, algebra=algebra
    )

    x = torch.randn(4, 4, 8)

    # Take a step to initialize state
    y = rotor_layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Save state
    state_dict = optimizer.state_dict()

    # Create new optimizer and load state
    new_layer = RotorLayer(algebra, channels=4)
    new_optimizer = RiemannianAdam(
        new_layer.parameters(), lr=0.001, algebra=algebra
    )
    new_optimizer.load_state_dict(state_dict)

    # Check state transferred
    for param, new_param in zip(rotor_layer.parameters(), new_layer.parameters()):
        if param in optimizer.state and new_param in new_optimizer.state:
            assert 'step' in new_optimizer.state[new_param]


# ============================================================================
# Edge Cases
# ============================================================================

def test_zero_learning_rate(algebra, rotor_layer):
    """Verify optimizer handles zero learning rate gracefully."""
    optimizer = ExponentialSGD(
        rotor_layer.parameters(), lr=0.0, algebra=algebra
    )

    x = torch.randn(4, 4, 8)
    initial_params = rotor_layer.bivector_weights.clone()

    y = rotor_layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Parameters should not change with lr=0
    assert torch.allclose(rotor_layer.bivector_weights, initial_params)


def test_invalid_parameters():
    """Verify optimizers validate input parameters."""
    algebra = CliffordAlgebra(p=3, q=0, device='cpu')
    layer = RotorLayer(algebra, channels=2)

    # Invalid learning rate
    with pytest.raises(ValueError):
        ExponentialSGD(layer.parameters(), lr=-0.01, algebra=algebra)

    # Invalid momentum
    with pytest.raises(ValueError):
        ExponentialSGD(layer.parameters(), lr=0.01, momentum=-0.5, algebra=algebra)

    # Missing algebra
    with pytest.raises(ValueError):
        ExponentialSGD(layer.parameters(), lr=0.01, algebra=None)

    # Invalid max_bivector_norm
    with pytest.raises(ValueError):
        RiemannianAdam(layer.parameters(), lr=0.01, algebra=algebra, max_bivector_norm=-1.0)


def test_empty_parameters(algebra):
    """Verify optimizers reject empty parameter list with proper error."""
    # PyTorch's Optimizer base class raises ValueError for empty params
    with pytest.raises(ValueError, match="empty parameter list"):
        ExponentialSGD([], lr=0.01, algebra=algebra)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
