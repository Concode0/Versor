# Tests for ABC Dataset task

import torch
import pytest
from core.algebra import CliffordAlgebra
from core.cga import ConformalAlgebra
from core.metric import hermitian_norm, hermitian_grade_spectrum
from models.cad_net import (
    ConformalPointNetEncoder, PointCloudDecoder, PrimitiveDecoder, CADAutoEncoder
)
from functional.loss import ChamferDistance, HermitianGradeRegularization
from datasets.abc import ABCDataset, collate_abc


@pytest.fixture
def algebra():
    return CliffordAlgebra(p=4, q=1, device='cpu')


@pytest.fixture
def cga():
    return ConformalAlgebra(euclidean_dim=3, device='cpu')


class TestConformalAlgebra:
    def test_roundtrip(self, cga):
        """CGA embedding and projection should be approximately inverse."""
        points = torch.randn(10, 3)
        conf = cga.to_cga(points)
        recovered = cga.from_cga(conf)
        assert torch.allclose(points, recovered, atol=1e-4)

    def test_null_cone(self, cga):
        """Conformal points should lie on the null cone (P^2 = 0)."""
        points = torch.randn(5, 3)
        conf = cga.to_cga(points)
        # Inner product with itself should be ~0
        for i in range(5):
            p = conf[i:i+1]
            pp = cga.algebra.geometric_product(p, p)
            # Scalar part should be approximately 0
            assert abs(pp[0, 0].item()) < 0.1


class TestChamferDistance:
    def test_zero_distance(self):
        cd = ChamferDistance()
        pts = torch.randn(2, 100, 3)
        loss = cd(pts, pts)
        assert loss.item() < 1e-6

    def test_positive_distance(self):
        cd = ChamferDistance()
        pts1 = torch.randn(2, 100, 3)
        pts2 = torch.randn(2, 100, 3) + 5.0
        loss = cd(pts1, pts2)
        assert loss.item() > 0

    def test_differentiable(self):
        cd = ChamferDistance()
        pts1 = torch.randn(2, 50, 3, requires_grad=True)
        pts2 = torch.randn(2, 50, 3)
        loss = cd(pts1, pts2)
        loss.backward()
        assert pts1.grad is not None


class TestABCDataset:
    def test_synthetic(self, tmp_path):
        ds = ABCDataset(str(tmp_path), task='reconstruction', num_points=256, split='train')
        assert len(ds) > 0
        sample = ds[0]
        assert sample['points'].shape == (256, 3)
        assert sample['normals'].shape == (256, 3)

    def test_collate(self, tmp_path):
        ds = ABCDataset(str(tmp_path), task='reconstruction', num_points=128, split='train')
        batch = collate_abc([ds[0], ds[1]])
        assert batch['points'].shape[0] == 2
        assert batch['points'].shape[1] == 128
        assert batch['points'].shape[2] == 3

    def test_primitive_mode(self, tmp_path):
        ds = ABCDataset(str(tmp_path), task='primitive', num_points=128, split='train')
        sample = ds[0]
        assert 'primitive_params' in sample


class TestConformalPointNetEncoder:
    def test_forward(self, algebra, cga):
        enc = ConformalPointNetEncoder(
            algebra, cga, latent_dim=32, num_layers=2, num_rotors=4
        )
        points = torch.randn(2, 64, 3)
        latent = enc(points)
        assert latent.shape == (2, 32, algebra.dim)

    def test_grad_flow(self, algebra, cga):
        enc = ConformalPointNetEncoder(
            algebra, cga, latent_dim=32, num_layers=2, num_rotors=4
        )
        points = torch.randn(2, 32, 3)
        latent = enc(points)
        latent.sum().backward()
        assert any(p.grad is not None for p in enc.parameters() if p.requires_grad)


class TestPointCloudDecoder:
    def test_forward(self, algebra, cga):
        dec = PointCloudDecoder(
            algebra, cga, latent_dim=32, output_points=64
        )
        latent = torch.randn(2, 32, algebra.dim)
        points = dec(latent)
        assert points.shape == (2, 64, 3)


class TestCADAutoEncoder:
    def test_reconstruction(self, algebra, cga):
        model = CADAutoEncoder(
            algebra, cga, latent_dim=32, num_rotors=4,
            output_points=64, decoder_type='reconstruction'
        )
        points = torch.randn(2, 64, 3)
        recon = model(points)
        assert recon.shape == (2, 64, 3)

    def test_primitive(self, algebra, cga):
        model = CADAutoEncoder(
            algebra, cga, latent_dim=32, num_rotors=4,
            output_points=64, decoder_type='primitive'
        )
        points = torch.randn(2, 64, 3)
        types, params = model(points)
        assert types.shape[0] == 2
        assert params.shape[0] == 2

    def test_encode_decode(self, algebra, cga):
        model = CADAutoEncoder(
            algebra, cga, latent_dim=32, num_rotors=4,
            output_points=64, decoder_type='reconstruction'
        )
        points = torch.randn(2, 64, 3)
        latent = model.encode(points)
        recon = model.decode(latent)
        assert latent.shape == (2, 32, algebra.dim)
        assert recon.shape == (2, 64, 3)

    def test_grad_flow(self, algebra, cga):
        model = CADAutoEncoder(
            algebra, cga, latent_dim=32, num_rotors=4,
            output_points=64, decoder_type='reconstruction'
        )
        points = torch.randn(2, 32, 3)
        recon = model(points)
        loss = recon.sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_sparsity_loss(self, algebra, cga):
        model = CADAutoEncoder(
            algebra, cga, latent_dim=32, num_rotors=4,
            output_points=64, decoder_type='reconstruction'
        )
        loss = model.total_sparsity_loss()
        assert loss >= 0


class TestABCHermitianMetrics:
    def test_grade_spectrum_cl41(self, algebra):
        """Test Hermitian grade spectrum in Cl(4,1) conformal algebra."""
        mv = torch.randn(5, algebra.dim)
        spec = hermitian_grade_spectrum(algebra, mv)
        assert spec.shape == (5, algebra.n + 1)
        assert (spec >= -1e-6).all()
        # Cl(4,1): n=5, so 6 grades
        assert spec.shape[-1] == 6

    def test_get_latent_features(self, algebra, cga):
        model = CADAutoEncoder(
            algebra, cga, latent_dim=32, num_rotors=4,
            output_points=64, decoder_type='reconstruction'
        )
        points = torch.randn(2, 64, 3)
        model(points)
        latent = model.get_latent_features()
        assert latent is not None
        assert latent.shape == (2, 32, algebra.dim)

        h_norm = hermitian_norm(algebra, latent)
        assert (h_norm >= 0).all()

    def test_grade_reg_cl41(self, algebra):
        grade_reg = HermitianGradeRegularization(
            algebra, target_spectrum=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05]
        )
        features = torch.randn(10, 32, algebra.dim)
        loss = grade_reg(features)
        assert loss.shape == ()
        assert loss >= 0
