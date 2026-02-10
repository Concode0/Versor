# Tests for WeatherBench task

import torch
import pytest
from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm, hermitian_grade_spectrum
from models.weather_gbn import (
    SphericalGraphConv, TemporalRotorLayer, WeatherGBN
)
from functional.loss import PhysicsInformedLoss, HermitianGradeRegularization
from datasets.weatherbench import (
    WeatherBenchDataset, _build_spherical_graph, _lat_weights, _num_nodes,
    collate_weatherbench
)


@pytest.fixture
def algebra():
    return CliffordAlgebra(p=2, q=1, device='cpu')


class TestSphericalGraph:
    def test_build_graph(self):
        H, W = 8, 16
        edges = _build_spherical_graph(H, W)
        assert edges.dim() == 2
        assert edges.size(0) == 2
        # Total nodes = H*W + 2 (grid + poles)
        assert edges.max() == H * W + 1  # south pole index

    def test_periodic_boundary(self):
        """Longitude wraps around: node (i, W-1) connects to (i, 0)."""
        H, W = 4, 8
        edges = _build_spherical_graph(H, W)
        src, dst = edges[0].tolist(), edges[1].tolist()
        # Node (0, W-1)=7 should connect to (0, 0)=0
        assert (7, 0) in list(zip(src, dst))
        # Node (0, 0)=0 should connect to (0, W-1)=7
        assert (0, 7) in list(zip(src, dst))

    def test_north_pole_connectivity(self):
        """North pole connects to all nodes in top row (row 0)."""
        H, W = 4, 8
        edges = _build_spherical_graph(H, W)
        north_pole = H * W
        src, dst = edges[0].tolist(), edges[1].tolist()
        # North pole -> each node in row 0
        for j in range(W):
            assert (north_pole, j) in list(zip(src, dst))
            assert (j, north_pole) in list(zip(src, dst))

    def test_south_pole_connectivity(self):
        """South pole connects to all nodes in bottom row (row H-1)."""
        H, W = 4, 8
        edges = _build_spherical_graph(H, W)
        south_pole = H * W + 1
        src, dst = edges[0].tolist(), edges[1].tolist()
        # South pole -> each node in bottom row
        for j in range(W):
            node = (H - 1) * W + j
            assert (south_pole, node) in list(zip(src, dst))
            assert (node, south_pole) in list(zip(src, dst))

    def test_num_nodes(self):
        assert _num_nodes(8, 16) == 8 * 16 + 2
        assert _num_nodes(32, 64) == 32 * 64 + 2

    def test_pole_edge_count(self):
        """Each pole has W bidirectional edges = 2*W pole edges per pole."""
        H, W = 8, 16
        edges = _build_spherical_graph(H, W)
        north_pole = H * W
        south_pole = H * W + 1
        src = edges[0]
        # Count edges involving north pole
        north_count = ((src == north_pole) | (edges[1] == north_pole)).sum().item()
        # Each grid node in row 0 has 2 edges (to/from pole) = 2*W
        assert north_count == 2 * W
        # Same for south pole
        south_count = ((src == south_pole) | (edges[1] == south_pole)).sum().item()
        assert south_count == 2 * W

    def test_lat_weights(self):
        H = 32
        w = _lat_weights(H)
        assert w.shape == (H,)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
        # Equator should have higher weight
        assert w[H // 2] > w[0]


class TestWeatherBenchDataset:
    def test_synthetic(self, tmp_path):
        ds = WeatherBenchDataset(str(tmp_path), resolution='5.625deg', split='train')
        assert len(ds) > 0
        sample = ds[0]
        assert 'state_t' in sample
        assert 'state_t_plus' in sample
        assert sample['state_t'].shape == (32, 64, 2)
        assert sample['num_nodes'] == 32 * 64 + 2

    def test_collate(self, tmp_path):
        ds = WeatherBenchDataset(str(tmp_path), resolution='5.625deg', split='train')
        batch = collate_weatherbench([ds[0], ds[1]])
        assert batch['state_t'].shape[0] == 2
        assert batch['state_t'].shape == (2, 32, 64, 2)
        assert batch['num_nodes'] == 32 * 64 + 2


class TestSphericalGraphConv:
    def test_forward(self, algebra):
        conv = SphericalGraphConv(algebra, hidden_dim=16, num_rotors=4)
        N = 32
        h = torch.randn(N, 16, algebra.dim)
        edges = torch.tensor([[0,1,2,3,4], [1,2,3,4,0]], dtype=torch.long)
        out = conv(h, edges)
        assert out.shape == h.shape

    def test_with_decomposition(self, algebra):
        conv = SphericalGraphConv(
            algebra, hidden_dim=16, num_rotors=4,
            use_decomposition=True, decomp_k=5
        )
        N = 16
        h = torch.randn(N, 16, algebra.dim)
        edges = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)
        out = conv(h, edges)
        assert out.shape == h.shape


class TestTemporalRotorLayer:
    def test_forward(self, algebra):
        temporal = TemporalRotorLayer(algebra, channels=16, num_steps=4)
        h = torch.randn(32, 16, algebra.dim)
        out = temporal(h, num_steps=4)
        assert out.shape == h.shape

    def test_partial_steps(self, algebra):
        temporal = TemporalRotorLayer(algebra, channels=16, num_steps=8)
        h = torch.randn(32, 16, algebra.dim)
        out = temporal(h, num_steps=3)
        assert out.shape == h.shape


class TestWeatherGBN:
    def test_forward(self, algebra):
        model = WeatherGBN(
            algebra, num_variables=2, spatial_hidden_dim=16,
            num_spatial_layers=2, num_temporal_steps=4, num_rotors=4
        )
        state = torch.randn(2, 8, 16, 2)  # [B, H, W, C]
        edges = _build_spherical_graph(8, 16)
        forecast = model(state, edges, num_steps=2)
        # Output should match input grid shape (poles stripped)
        assert forecast.shape == state.shape

    def test_grad_flow(self, algebra):
        model = WeatherGBN(
            algebra, num_variables=2, spatial_hidden_dim=16,
            num_spatial_layers=2, num_temporal_steps=4, num_rotors=4
        )
        state = torch.randn(1, 8, 16, 2)
        edges = _build_spherical_graph(8, 16)
        forecast = model(state, edges, num_steps=2)
        loss = forecast.sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_pole_embeddings_have_grad(self, algebra):
        """Pole embeddings should receive gradients through message passing."""
        model = WeatherGBN(
            algebra, num_variables=2, spatial_hidden_dim=16,
            num_spatial_layers=2, num_temporal_steps=4, num_rotors=4
        )
        state = torch.randn(1, 4, 8, 2)
        edges = _build_spherical_graph(4, 8)
        forecast = model(state, edges, num_steps=2)
        loss = forecast.sum()
        loss.backward()
        assert model.north_pole_embed.grad is not None
        assert model.south_pole_embed.grad is not None

    def test_sparsity_loss(self, algebra):
        model = WeatherGBN(
            algebra, num_variables=2, spatial_hidden_dim=16,
            num_spatial_layers=2, num_temporal_steps=4, num_rotors=4
        )
        loss = model.total_sparsity_loss()
        assert loss >= 0


class TestPhysicsInformedLoss:
    def test_loss(self):
        loss_fn = PhysicsInformedLoss(physics_weight=0.1)
        forecast = torch.randn(2, 8, 16, 2)
        target = torch.randn(2, 8, 16, 2)
        lat_weights = _lat_weights(8)
        loss = loss_fn(forecast, target, lat_weights)
        assert loss.shape == ()
        assert loss >= 0

    def test_conservation(self):
        loss_fn = PhysicsInformedLoss(physics_weight=1.0)
        # If forecast matches target, conservation loss should be small
        target = torch.randn(2, 8, 16, 2)
        loss = loss_fn(target, target)
        assert loss.item() < 1e-5


class TestWeatherBenchHermitianMetrics:
    def test_grade_spectrum_cl21(self, algebra):
        """Test Hermitian grade spectrum in Cl(2,1) spacetime algebra."""
        mv = torch.randn(10, algebra.dim)
        spec = hermitian_grade_spectrum(algebra, mv)
        assert spec.shape == (10, algebra.n + 1)
        assert (spec >= -1e-6).all()

    def test_get_latent_features(self, algebra):
        model = WeatherGBN(
            algebra, num_variables=2, spatial_hidden_dim=16,
            num_spatial_layers=2, num_temporal_steps=4, num_rotors=4
        )
        state = torch.randn(2, 8, 16, 2)
        edges = _build_spherical_graph(8, 16)
        model(state, edges, num_steps=2)
        latent = model.get_latent_features()
        assert latent is not None

        h_norm = hermitian_norm(algebra, latent)
        assert (h_norm >= 0).all()

    def test_grade_reg_cl21(self, algebra):
        grade_reg = HermitianGradeRegularization(
            algebra, target_spectrum=[0.15, 0.35, 0.35, 0.15]
        )
        features = torch.randn(32, 16, algebra.dim)
        loss = grade_reg(features)
        assert loss.shape == ()
        assert loss >= 0
