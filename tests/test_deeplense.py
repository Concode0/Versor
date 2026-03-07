"""Tests for the DeepLense gravitational lensing task."""

import pytest
import torch
from core.algebra import CliffordAlgebra


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestSyntheticGeneration:
    """Tests for synthetic SIS lens data generation."""

    def test_generate_single_sample(self):
        from datasets.deeplense import generate_synthetic_sample
        sample = generate_synthetic_sample(image_size=32)
        assert sample['lensed'].shape == (1, 32, 32)
        assert sample['source'].shape == (1, 32, 32)
        assert sample['kappa'].shape == (1, 32, 32)
        assert sample['label'] in (0, 1, 2)

    def test_generate_dataset(self):
        from datasets.deeplense import generate_synthetic_dataset
        samples = generate_synthetic_dataset(n_samples=10, image_size=32, seed=42)
        assert len(samples) == 10
        labels = {s['label'] for s in samples}
        assert len(labels) >= 1

    def test_dataset_class(self):
        from datasets.deeplense import generate_synthetic_dataset, DeepLenseDataset
        samples = generate_synthetic_dataset(n_samples=5, image_size=32)
        ds = DeepLenseDataset(samples)
        assert len(ds) == 5
        item = ds[0]
        assert 'lensed' in item
        assert 'source' in item
        assert 'kappa' in item
        assert 'label' in item
        assert item['lensed'].dtype == torch.float32

    def test_loaders(self, tmp_path):
        from datasets.deeplense import get_deeplense_loaders
        train, val, test = get_deeplense_loaders(
            root=str(tmp_path / "data"),
            variant="synthetic",
            image_size=32,
            n_samples=30,
            batch_size=4,
            num_workers=0,
        )
        batch = next(iter(train))
        assert batch['lensed'].shape[0] <= 4
        assert batch['lensed'].shape[1] == 1
        assert batch['lensed'].shape[2] == 32
        assert batch['lensed'].shape[3] == 32

    def test_sis_deflection(self):
        from datasets.deeplense import _sis_deflection
        theta_x = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        theta_y = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        alpha_x, alpha_y = _sis_deflection(theta_x, theta_y, theta_E=0.5)
        assert alpha_x.shape == (2, 2)
        assert alpha_y.shape == (2, 2)

    def test_sis_convergence(self):
        from datasets.deeplense import _sis_convergence
        theta_x = torch.tensor([[1.0, 0.0]])
        theta_y = torch.tensor([[0.0, 1.0]])
        kappa = _sis_convergence(theta_x, theta_y, theta_E=0.5)
        assert kappa.shape == (1, 2)
        assert (kappa > 0).all()

    def test_substructure_labels(self):
        from datasets.deeplense import _add_substructure
        import numpy as np
        kappa = torch.ones(32, 32) * 0.5
        rng = np.random.default_rng(42)
        k0 = _add_substructure(kappa.clone(), 0, 32, 32, rng=rng)
        assert torch.allclose(k0, kappa)
        k1 = _add_substructure(kappa.clone(), 1, 32, 32, rng=rng)
        assert not torch.allclose(k1, kappa)
        k2 = _add_substructure(kappa.clone(), 2, 32, 32, rng=rng)
        assert not torch.allclose(k2, kappa)

    def test_cached_loading(self, tmp_path):
        from datasets.deeplense import get_deeplense_loaders
        root = str(tmp_path / "data")
        get_deeplense_loaders(root=root, variant="synthetic", image_size=32,
                             n_samples=20, batch_size=4, num_workers=0)
        train, _, _ = get_deeplense_loaders(
            root=root, variant="synthetic", image_size=32,
            n_samples=20, batch_size=4, num_workers=0,
        )
        batch = next(iter(train))
        assert batch['lensed'].shape[2] == 32


# ---------------------------------------------------------------------------
# Rotary 2D PE tests
# ---------------------------------------------------------------------------

class TestRotary2DBivectorPE:
    """Tests for 2D rotary positional encoding."""

    @pytest.fixture
    def algebra(self):
        return CliffordAlgebra(p=1, q=3, device='cpu')

    def test_output_shape(self, algebra):
        from models.lensing_net import Rotary2DBivectorPE
        pe = Rotary2DBivectorPE(algebra, channels=4, max_h=4, max_w=4)
        x = torch.randn(2, 16, 4, 16)  # [B, 4*4, C, D]
        out = pe(x, 4, 4)
        assert out.shape == x.shape

    def test_position_dependent(self, algebra):
        from models.lensing_net import Rotary2DBivectorPE
        pe = Rotary2DBivectorPE(algebra, channels=4, max_h=4, max_w=4)
        x = torch.ones(1, 16, 4, 16)
        out = pe(x, 4, 4)
        # Different positions should produce different outputs
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)

    def test_learnable_params(self, algebra):
        from models.lensing_net import Rotary2DBivectorPE
        pe = Rotary2DBivectorPE(algebra, channels=4, max_h=4, max_w=4, learnable=True)
        params = list(pe.parameters())
        assert len(params) == 2  # row_weights, col_weights


# ---------------------------------------------------------------------------
# Patch Merging / Expanding tests
# ---------------------------------------------------------------------------

class TestGeometricPatchOps:
    """Tests for patch merging and expanding."""

    @pytest.fixture
    def algebra(self):
        return CliffordAlgebra(p=1, q=3, device='cpu')

    def test_patch_merging_shape(self, algebra):
        from models.lensing_net import GeometricPatchMerging
        merger = GeometricPatchMerging(algebra, channels=4)
        x = torch.randn(2, 16, 4, 16)  # [B, 4*4, C=4, D=16]
        out = merger(x, nH=4, nW=4)
        # 4x4 -> 2x2, channels 4 -> 8
        assert out.shape == (2, 4, 8, 16)

    def test_patch_expanding_shape(self, algebra):
        from models.lensing_net import GeometricPatchExpanding
        expander = GeometricPatchExpanding(algebra, channels=8)
        x = torch.randn(2, 4, 8, 16)  # [B, 2*2, C=8, D=16]
        out = expander(x, nH=2, nW=2)
        # 2x2 -> 4x4, channels 8 -> 4
        assert out.shape == (2, 16, 4, 16)

    def test_merge_expand_roundtrip_shape(self, algebra):
        from models.lensing_net import GeometricPatchMerging, GeometricPatchExpanding
        merger = GeometricPatchMerging(algebra, channels=4)
        expander = GeometricPatchExpanding(algebra, channels=8)
        x = torch.randn(2, 16, 4, 16)
        merged = merger(x, 4, 4)
        assert merged.shape == (2, 4, 8, 16)
        expanded = expander(merged, 2, 2)
        assert expanded.shape == (2, 16, 4, 16)

    def test_patch_merging_gradient(self, algebra):
        from models.lensing_net import GeometricPatchMerging
        merger = GeometricPatchMerging(algebra, channels=4)
        x = torch.randn(2, 16, 4, 16, requires_grad=True)
        out = merger(x, 4, 4)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestLensingGBN:
    """Tests for the LensingGBN model."""

    @pytest.fixture
    def algebra(self):
        return CliffordAlgebra(p=1, q=3, device='cpu')

    def _make_model(self, algebra, mode="full", **kwargs):
        from models.lensing_net import LensingGBN
        defaults = dict(
            image_size=32, patch_size=8,
            hidden_channels=4, num_encoder_layers=4, num_decoder_layers=4,
            num_rotors=2, num_attention_heads=2,
            use_decomposition=False, use_rotor_backend=False,
        )
        defaults.update(kwargs)
        return LensingGBN(algebra, mode=mode, **defaults)

    def test_model_creation(self, algebra):
        model = self._make_model(algebra)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_forward_full_mode(self, algebra):
        model = self._make_model(algebra, mode="full")
        x = torch.randn(2, 1, 32, 32)
        outputs = model(x)
        assert 'source' in outputs
        assert 'kappa' in outputs
        assert 'shear' in outputs
        assert 'logits' in outputs
        assert outputs['source'].shape == (2, 1, 32, 32)
        assert outputs['kappa'].shape == (2, 1, 32, 32)
        assert outputs['shear'].shape == (2, 2, 32, 32)
        assert outputs['logits'].shape == (2, 3)

    def test_forward_reconstruct_mode(self, algebra):
        model = self._make_model(algebra, mode="reconstruct")
        x = torch.randn(2, 1, 32, 32)
        outputs = model(x)
        assert 'source' in outputs
        assert 'kappa' not in outputs
        assert 'shear' not in outputs

    def test_forward_convergence_mode(self, algebra):
        model = self._make_model(algebra, mode="convergence")
        x = torch.randn(2, 1, 32, 32)
        outputs = model(x)
        assert 'kappa' in outputs
        assert 'source' not in outputs

    def test_forward_classify_mode(self, algebra):
        model = self._make_model(algebra, mode="classify")
        x = torch.randn(2, 1, 32, 32)
        outputs = model(x)
        assert 'logits' in outputs
        assert 'source' not in outputs

    def test_time_conditioning(self, algebra):
        model = self._make_model(algebra, mode="full")
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([[0.0], [1.0]])
        outputs = model(x, t=t)
        assert outputs['source'].shape == (2, 1, 32, 32)

    def test_time_embed_targets_e0(self, algebra):
        """Time embedding should inject into e_0 (index 1) only."""
        model = self._make_model(algebra, mode="full")
        assert model.e0_index == 1  # temporal direction in Cl(1,3)
        # time_embed outputs per-channel scalars, not full multivectors
        assert model.time_embed.out_features == model.hidden_channels

    def test_sparsity_loss(self, algebra):
        model = self._make_model(algebra)
        spar = model.total_sparsity_loss()
        assert spar.shape == ()
        assert spar >= 0

    def test_gradient_flow(self, algebra):
        model = self._make_model(algebra, mode="full")
        x = torch.randn(2, 1, 32, 32)
        outputs = model(x)
        loss = outputs['source'].sum() + outputs['logits'].sum()
        loss.backward()
        assert model.img_to_mv.proj.weight.grad is not None

    def test_image_to_multivector(self, algebra):
        from models.lensing_net import ImageToMultivector
        embed = ImageToMultivector(algebra, in_channels=1, channels=4,
                                   patch_size=8, image_size=32)
        x = torch.randn(2, 1, 32, 32)
        mv = embed(x)
        assert mv.shape == (2, 16, 4, 16)  # 16 patches = (32/8)^2

    def test_image_to_multivector_grade_sparsity(self, algebra):
        """ImageToMultivector should only populate grade-0 and grade-1."""
        from models.lensing_net import ImageToMultivector
        embed = ImageToMultivector(algebra, in_channels=1, channels=4,
                                   patch_size=8, image_size=32)
        x = torch.randn(2, 1, 32, 32)
        mv = embed(x)
        # grade-2 indices: [3, 5, 6, 9, 10, 12]
        g2_idx = [i for i in range(16) if bin(i).count('1') == 2]
        g3_idx = [i for i in range(16) if bin(i).count('1') == 3]
        g4_idx = [i for i in range(16) if bin(i).count('1') == 4]
        higher = g2_idx + g3_idx + g4_idx
        assert (mv[:, :, :, higher] == 0).all(), \
            "Grades 2+ should be zero in ImageToMultivector output"
        # grade-0 and grade-1 should be non-zero (with high probability)
        g0_idx = [0]
        g1_idx = [1, 2, 4, 8]
        low = g0_idx + g1_idx
        assert mv[:, :, :, low].abs().sum() > 0, \
            "Grade-0 and grade-1 should be populated"

    def test_multivector_to_image(self, algebra):
        from models.lensing_net import MultivectorToImage
        head = MultivectorToImage(algebra, channels=4, out_channels=1,
                                  patch_size=8, image_size=32)
        mv = torch.randn(2, 16, 4, 16)
        img = head(mv)
        assert img.shape == (2, 1, 32, 32)

    def test_grade_slice_to_image_kappa(self, algebra):
        """Kappa head should slice grade-0 (scalar = convergence)."""
        from models.lensing_net import GradeSliceToImage
        head = GradeSliceToImage(algebra, channels=4, out_channels=1,
                                 patch_size=8, image_size=32, grade=0)
        mv = torch.randn(2, 16, 4, 16)
        img = head(mv)
        assert img.shape == (2, 1, 32, 32)
        # Verify it only uses grade-0 component
        assert len(head.grade_indices) == 1  # grade-0 has 1 blade

    def test_grade_slice_to_image_shear(self, algebra):
        """Shear head should slice grade-2 (bivector = shear gamma)."""
        from models.lensing_net import GradeSliceToImage
        head = GradeSliceToImage(algebra, channels=4, out_channels=2,
                                 patch_size=8, image_size=32, grade=2)
        mv = torch.randn(2, 16, 4, 16)
        img = head(mv)
        assert img.shape == (2, 2, 32, 32)
        # Verify it uses all 6 grade-2 bivector blades
        assert len(head.grade_indices) == 6

    def test_with_rotor_backend(self, algebra):
        model = self._make_model(algebra, mode="full", use_rotor_backend=True)
        x = torch.randn(2, 1, 32, 32)
        outputs = model(x)
        assert outputs['source'].shape == (2, 1, 32, 32)

    def test_blade_selector_only_at_output(self, algebra):
        """Verify BladeSelector appears only in output, not in encoder/decoder."""
        from layers.projection import BladeSelector
        model = self._make_model(algebra)
        blade_count = 0
        for name, module in model.named_modules():
            if isinstance(module, BladeSelector):
                blade_count += 1
                # Should only be output_blade
                assert 'output_blade' in name, \
                    f"Found BladeSelector at unexpected location: {name}"
        assert blade_count == 1, f"Expected 1 BladeSelector, found {blade_count}"

    def test_has_rotary2d_pe(self, algebra):
        """Verify model uses Rotary2DBivectorPE."""
        from models.lensing_net import Rotary2DBivectorPE
        model = self._make_model(algebra)
        pe_count = sum(1 for m in model.modules() if isinstance(m, Rotary2DBivectorPE))
        assert pe_count == 1

    def test_has_patch_merging_expanding(self, algebra):
        """Verify model has downsamplers and upsamplers."""
        from models.lensing_net import GeometricPatchMerging, GeometricPatchExpanding
        model = self._make_model(algebra)
        mergers = [m for m in model.modules() if isinstance(m, GeometricPatchMerging)]
        expanders = [m for m in model.modules() if isinstance(m, GeometricPatchExpanding)]
        # With num_encoder_layers=4 => 2 stages => 1 downsample/upsample
        assert len(mergers) >= 1
        assert len(expanders) >= 1


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLensingLoss:
    """Tests for the composite lensing loss."""

    def test_full_loss(self):
        from tasks.deeplense import LensingLoss
        loss_fn = LensingLoss(
            weights={'source': 1.0, 'kappa': 0.5, 'classify': 0.3, 'physics': 0.1},
            mode="full",
        )
        outputs = {
            'source': torch.randn(2, 1, 32, 32),
            'kappa': torch.randn(2, 1, 32, 32),
            'shear': torch.randn(2, 2, 32, 32),
            'logits': torch.randn(2, 3),
        }
        targets = {
            'source': torch.randn(2, 1, 32, 32),
            'kappa': torch.randn(2, 1, 32, 32),
            'label': torch.randint(0, 3, (2,)),
        }
        total, logs = loss_fn(outputs, targets)
        assert total.shape == ()
        assert total > 0
        assert 'src' in logs
        assert 'kappa' in logs
        assert 'cls' in logs
        assert 'phys' in logs

    def test_reconstruct_mode_loss(self):
        from tasks.deeplense import LensingLoss
        loss_fn = LensingLoss(weights={'source': 1.0}, mode="reconstruct")
        outputs = {'source': torch.randn(2, 1, 32, 32)}
        targets = {'source': torch.randn(2, 1, 32, 32)}
        total, logs = loss_fn(outputs, targets)
        assert 'src' in logs
        assert 'cls' not in logs

    def test_physics_loss(self):
        from tasks.deeplense import LensingLoss
        loss_fn = LensingLoss(weights={'kappa': 1.0, 'physics': 1.0}, mode="convergence")
        outputs = {'kappa': torch.randn(2, 1, 32, 32)}
        targets = {'kappa': torch.randn(2, 1, 32, 32)}
        total, logs = loss_fn(outputs, targets)
        assert 'phys' in logs


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests."""

    def test_train_step_smoke(self):
        """Smoke test: one forward + backward pass."""
        algebra = CliffordAlgebra(p=1, q=3, device='cpu')
        from models.lensing_net import LensingGBN
        from tasks.deeplense import LensingLoss

        model = LensingGBN(
            algebra, image_size=32, patch_size=8,
            hidden_channels=4, num_encoder_layers=4, num_decoder_layers=4,
            num_rotors=2, num_attention_heads=2,
            use_decomposition=False, use_rotor_backend=False,
            mode="full",
        )
        loss_fn = LensingLoss(
            weights={'source': 1.0, 'kappa': 0.5, 'classify': 0.3},
            mode="full",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(2, 1, 32, 32)
        targets = {
            'source': torch.randn(2, 1, 32, 32),
            'kappa': torch.randn(2, 1, 32, 32),
            'label': torch.randint(0, 3, (2,)),
        }

        optimizer.zero_grad()
        outputs = model(x)
        loss, logs = loss_fn(outputs, targets, model=model)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_dataset_to_model_pipeline(self, tmp_path):
        """Test that dataset output feeds correctly into model."""
        from datasets.deeplense import get_deeplense_loaders
        from models.lensing_net import LensingGBN

        algebra = CliffordAlgebra(p=1, q=3, device='cpu')

        train, _, _ = get_deeplense_loaders(
            root=str(tmp_path / "data"),
            variant="synthetic",
            image_size=32,
            n_samples=10,
            batch_size=2,
            num_workers=0,
        )

        model = LensingGBN(
            algebra, image_size=32, patch_size=8,
            hidden_channels=4, num_encoder_layers=4, num_decoder_layers=4,
            num_rotors=2, num_attention_heads=2,
            use_decomposition=False, use_rotor_backend=False,
            mode="full",
        )

        batch = next(iter(train))
        outputs = model(batch['lensed'])
        assert outputs['source'].shape[2:] == batch['source'].shape[2:]
        assert outputs['kappa'].shape[2:] == batch['kappa'].shape[2:]
        assert outputs['logits'].shape == (batch['lensed'].shape[0], 3)
