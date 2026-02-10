# Tests for MD17 task with advanced framework features

import torch
import pytest
from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm, hermitian_grade_spectrum
from models.md17_forcenet import MD17ForceNet, MD17InteractionBlock
from functional.loss import ConservativeLoss, HermitianGradeRegularization


@pytest.fixture
def algebra():
    return CliffordAlgebra(p=3, q=0, device='cpu')


class TestMD17InteractionBlock:
    def test_with_decomposition(self, algebra):
        block = MD17InteractionBlock(
            algebra, hidden_dim=16, num_rotors=4,
            use_decomposition=True, decomp_k=5
        )
        h = torch.randn(10, 16, algebra.dim)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)
        out = block(h, pos, edge_index)
        assert out.shape == h.shape

    def test_with_rotor_backend(self, algebra):
        block = MD17InteractionBlock(
            algebra, hidden_dim=16, num_rotors=4,
            use_rotor_backend=True
        )
        h = torch.randn(10, 16, algebra.dim)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)
        out = block(h, pos, edge_index)
        assert out.shape == h.shape


class TestMD17ForceNet:
    def test_forward_default(self, algebra):
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_rotors=4)
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0,0,0,0,1,1,1,1])
        edge_index = torch.tensor([[0,1,2,4,5,6], [1,2,3,5,6,7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        assert energy.shape == (2,)
        assert force.shape == (8, 3)

    def test_forward_with_decomposition(self, algebra):
        model = MD17ForceNet(
            algebra, hidden_dim=16, num_layers=2, num_rotors=4,
            use_decomposition=True, decomp_k=5
        )
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0,0,0,0,1,1,1,1])
        edge_index = torch.tensor([[0,1,2,4,5,6], [1,2,3,5,6,7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        assert energy.shape == (2,)
        assert force.shape == (8, 3)

    def test_forward_with_rotor_backend(self, algebra):
        model = MD17ForceNet(
            algebra, hidden_dim=16, num_layers=2, num_rotors=4,
            use_rotor_backend=True
        )
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0,0,0,0,1,1,1,1])
        edge_index = torch.tensor([[0,1,2,4,5,6], [1,2,3,5,6,7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        assert energy.shape == (2,)

    def test_sparsity_loss(self, algebra):
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_rotors=4)
        loss = model.total_sparsity_loss()
        assert loss >= 0

    def test_all_features(self, algebra):
        model = MD17ForceNet(
            algebra, hidden_dim=16, num_layers=2, num_rotors=4,
            use_decomposition=True, decomp_k=5, use_rotor_backend=True
        )
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0,0,0,0,1,1,1,1])
        edge_index = torch.tensor([[0,1,2,4,5,6], [1,2,3,5,6,7]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        sparsity = model.total_sparsity_loss()
        assert energy.shape == (2,)
        assert force.shape == (8, 3)
        assert sparsity >= 0


class TestConservativeLoss:
    def test_conservative_loss(self, algebra):
        loss_fn = ConservativeLoss()
        pos = torch.randn(8, 3, requires_grad=True)
        # Simple energy = sum of squared positions
        energy = (pos ** 2).sum()
        force_pred = torch.randn(8, 3)
        loss = loss_fn(energy.unsqueeze(0), force_pred, pos)
        assert loss.shape == ()
        assert loss >= 0

    def test_grad_flow(self, algebra):
        loss_fn = ConservativeLoss()
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_rotors=4)
        z = torch.randint(1, 10, (4,))
        pos = torch.randn(4, 3, requires_grad=True)
        batch = torch.tensor([0,0,1,1])
        edge_index = torch.tensor([[0,1,2], [1,0,3]], dtype=torch.long)
        energy, force = model(z, pos, batch, edge_index)
        loss = loss_fn(energy, force, pos)
        loss.backward()
        # Check gradients flow
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestMD17GradeRegularization:
    def test_grade_reg_loss(self, algebra):
        grade_reg = HermitianGradeRegularization(
            algebra, target_spectrum=[0.4, 0.4, 0.15, 0.05]
        )
        features = torch.randn(8, 16, algebra.dim)
        loss = grade_reg(features)
        assert loss.shape == ()
        assert loss >= 0

    def test_get_latent_features(self, algebra):
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_rotors=4)
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0,0,0,0,1,1,1,1])
        edge_index = torch.tensor([[0,1,2,4,5,6], [1,2,3,5,6,7]], dtype=torch.long)
        model(z, pos, batch, edge_index)
        latent = model.get_latent_features()
        assert latent is not None
        assert latent.shape == (8, 16, algebra.dim)

    def test_hermitian_norm_of_features(self, algebra):
        model = MD17ForceNet(algebra, hidden_dim=16, num_layers=2, num_rotors=4)
        z = torch.randint(1, 10, (8,))
        pos = torch.randn(8, 3)
        batch = torch.tensor([0,0,0,0,1,1,1,1])
        edge_index = torch.tensor([[0,1,2,4,5,6], [1,2,3,5,6,7]], dtype=torch.long)
        model(z, pos, batch, edge_index)
        latent = model.get_latent_features()
        h_norm = hermitian_norm(algebra, latent)
        assert h_norm.shape == (8, 16, 1)
        assert (h_norm >= 0).all()
