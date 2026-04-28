# Tests for MD17 task with PGA motors, dynamic rotors, and RBF

import pytest
import torch

from core.algebra import CliffordAlgebra

pytestmark = pytest.mark.unit
from core.decomposition import ExpPolicy
from core.metric import hermitian_grade_spectrum, hermitian_norm
from functional.loss import ConservativeLoss, HermitianGradeRegularization
from models.md17 import DynamicRotorGenerator, GaussianRBF, MD17ForceNet, MD17InteractionBlock


@pytest.fixture
def algebra():
    return CliffordAlgebra(p=3, q=0, r=1, device="cpu")


class TestGaussianRBF:
    def test_output_shape(self):
        rbf = GaussianRBF(num_rbf=20, cutoff=5.0)
        distances = torch.rand(50)
        out = rbf(distances)
        assert out.shape == (50, 20)

    def test_center_peak(self):
        rbf = GaussianRBF(num_rbf=10, cutoff=5.0)
        # Distance at center 0 should have max response at first RBF
        d = torch.tensor([0.0])
        out = rbf(d)
        assert out[0, 0] > out[0, -1]

    def test_positive_output(self):
        rbf = GaussianRBF(num_rbf=20, cutoff=5.0)
        distances = torch.rand(100) * 5.0
        out = rbf(distances)
        assert (out >= 0).all()


class TestDynamicRotorGenerator:
    def test_output_shape(self, algebra):
        gen = DynamicRotorGenerator(algebra, input_dim=64, num_dynamic_rotors=4)
        inv_feat = torch.randn(10, 64)
        R, R_rev = gen(inv_feat)
        assert R.shape == (10, 4, algebra.dim)
        assert R_rev.shape == (10, 4, algebra.dim)

    def test_identity_init(self, algebra):
        gen = DynamicRotorGenerator(algebra, input_dim=32, num_dynamic_rotors=2)
        inv_feat = torch.zeros(5, 32)
        R, R_rev = gen(inv_feat)
        # With zero input and zero-init, output should be identity rotors
        # Identity rotor has 1 in scalar component, 0 elsewhere
        expected_scalar = torch.ones(5, 2)
        assert torch.allclose(R[..., 0], expected_scalar, atol=1e-5)

    def test_differentiable(self, algebra):
        gen = DynamicRotorGenerator(algebra, input_dim=32, num_dynamic_rotors=2)
        inv_feat = torch.randn(5, 32, requires_grad=True)
        R, R_rev = gen(inv_feat)
        loss = R.sum()
        loss.backward()
        assert inv_feat.grad is not None


class TestMD17InteractionBlock:
    def test_with_all_features(self, algebra):
        block = MD17InteractionBlock(
            algebra,
            hidden_dim=16,
            num_static_rotors=4,
            num_dynamic_rotors=2,
            num_rbf=10,
            rbf_cutoff=5.0,
            use_geo_square=True,
        )
        h = torch.randn(10, 16, algebra.dim)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = block(h, pos, edge_index)
        assert out.shape == h.shape

    def test_with_rotor_backend(self, algebra):
        block = MD17InteractionBlock(
            algebra, hidden_dim=16, num_static_rotors=4, num_dynamic_rotors=2, use_rotor_backend=True
        )
        h = torch.randn(10, 16, algebra.dim)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = block(h, pos, edge_index)
        assert out.shape == h.shape

    def test_without_geo_square(self, algebra):
        block = MD17InteractionBlock(
            algebra, hidden_dim=16, num_static_rotors=4, num_dynamic_rotors=2, use_geo_square=False
        )
        h = torch.randn(10, 16, algebra.dim)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = block(h, pos, edge_index)
        assert out.shape == h.shape


class TestMD17ForceNet:
    def test_forward_default(self, algebra):
        model = MD17ForceNet(
            algebra, hidden_dim=16, num_layers=2, num_static_rotors=4, num_dynamic_rotors=2, num_rbf=10
        )
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        assert energy.shape == (2,)
        assert force.shape == (8, 3)

    def test_forward_with_exact_policy(self, algebra):
        from core.decomposition import ExpPolicy

        algebra.exp_policy = ExpPolicy.PRECISE
        model = MD17ForceNet(
            algebra, hidden_dim=16, num_layers=2, num_static_rotors=4, num_dynamic_rotors=2, num_rbf=10
        )
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        assert energy.shape == (2,)
        assert force.shape == (8, 3)
        algebra.exp_policy = ExpPolicy.BALANCED

    def test_forward_with_rotor_backend(self, algebra):
        model = MD17ForceNet(
            algebra,
            hidden_dim=16,
            num_layers=2,
            num_static_rotors=4,
            num_dynamic_rotors=2,
            use_rotor_backend=True,
            num_rbf=10,
        )
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        assert energy.shape == (2,)

    def test_sparsity_loss(self, algebra):
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_static_rotors=4, num_dynamic_rotors=2)
        loss = model.total_sparsity_loss()
        assert loss >= 0

    def test_all_features(self, algebra):
        model = MD17ForceNet(
            algebra,
            hidden_dim=16,
            num_layers=2,
            num_static_rotors=4,
            num_dynamic_rotors=2,
            num_rbf=10,
            rbf_cutoff=5.0,
            use_rotor_backend=True,
            use_geo_square=True,
        )
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        sparsity = model.total_sparsity_loss()
        assert energy.shape == (2,)
        assert force.shape == (8, 3)
        assert sparsity >= 0

    def test_pga_dim(self, algebra):
        """Verify PGA algebra dimensions."""
        assert algebra.dim == 16  # 2^4
        assert algebra.num_grades == 5  # grades 0-4


class TestConservativeLoss:
    def test_conservative_loss(self, algebra):
        loss_fn = ConservativeLoss()
        pos = torch.randn(8, 3, requires_grad=True)
        energy = (pos**2).sum()
        force_pred = torch.randn(8, 3)
        loss = loss_fn(energy.unsqueeze(0), force_pred, pos)
        assert loss.shape == ()
        assert loss >= 0

    def test_grad_flow(self, algebra):
        loss_fn = ConservativeLoss()
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_static_rotors=4, num_dynamic_rotors=2)
        z = torch.randint(1, 10, (4,))
        pos = torch.randn(4, 3, requires_grad=True)
        batch = torch.tensor([0, 0, 1, 1])
        edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        loss = loss_fn(energy, force, pos)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestMD17GradeRegularization:
    def test_grade_reg_loss(self, algebra):
        grade_reg = HermitianGradeRegularization(algebra, target_spectrum=[0.35, 0.30, 0.20, 0.10, 0.05])
        features = torch.randn(8, 16, algebra.dim)
        loss = grade_reg(features)
        assert loss.shape == ()
        assert loss >= 0

    def test_get_latent_features(self, algebra):
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_static_rotors=4, num_dynamic_rotors=2)
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]], dtype=torch.long)
        model(z, pos, batch, edge_index)
        latent = model.get_latent_features()
        assert latent is not None
        assert latent.shape == (8, 16, algebra.dim)

    def test_hermitian_norm_of_features(self, algebra):
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_static_rotors=4, num_dynamic_rotors=2)
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]], dtype=torch.long)
        model(z, pos, batch, edge_index)
        latent = model.get_latent_features()
        h_norm = hermitian_norm(algebra, latent)
        assert h_norm.shape == (8, 16, 1)
        assert (h_norm >= 0).all()
