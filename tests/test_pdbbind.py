# Tests for PDBbind task

import torch
import pytest
from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm, hermitian_grade_spectrum
from models.pdbbind_net import (
    ProteinEncoder, LigandEncoder, GeometricCrossAttention, PDBBindNet
)
from functional.loss import HermitianGradeRegularization
from datasets.pdbbind import PDBBindDataset, collate_pdbbind, _build_radius_graph


@pytest.fixture
def algebra():
    return CliffordAlgebra(p=3, q=0, device='cpu')


class TestPDBBindDataset:
    def test_synthetic_dataset(self, tmp_path):
        ds = PDBBindDataset(str(tmp_path), split='train')
        assert len(ds) > 0
        sample = ds[0]
        assert 'protein_pos' in sample
        assert 'ligand_pos' in sample
        assert 'affinity' in sample
        assert sample['protein_pos'].dim() == 2
        assert sample['protein_pos'].size(1) == 3

    def test_collate(self, tmp_path):
        ds = PDBBindDataset(str(tmp_path), split='train')
        batch = collate_pdbbind([ds[0], ds[1]])
        assert 'protein_batch' in batch
        assert 'ligand_batch' in batch
        assert batch['affinity'].shape == (2,)

    def test_radius_graph(self):
        pos = torch.randn(10, 3)
        edges = _build_radius_graph(pos, cutoff=5.0)
        assert edges.dim() == 2
        assert edges.size(0) == 2


class TestProteinEncoder:
    def test_forward(self, algebra):
        enc = ProteinEncoder(algebra, hidden_dim=16, num_layers=2, num_rotors=4)
        pos = torch.randn(10, 3)
        z = torch.randint(1, 10, (10,))
        aa = torch.randint(0, 20, (10,))
        edges = _build_radius_graph(pos, cutoff=3.0)
        batch = torch.zeros(10, dtype=torch.long)
        h = enc(pos, z, aa, edges, batch)
        assert h.shape == (10, 16, algebra.dim)

    def test_with_decomposition(self, algebra):
        enc = ProteinEncoder(
            algebra, hidden_dim=16, num_layers=2, num_rotors=4,
            use_decomposition=True, decomp_k=5
        )
        pos = torch.randn(10, 3)
        z = torch.randint(1, 10, (10,))
        aa = torch.randint(0, 20, (10,))
        edges = _build_radius_graph(pos, cutoff=3.0)
        batch = torch.zeros(10, dtype=torch.long)
        h = enc(pos, z, aa, edges, batch)
        assert h.shape == (10, 16, algebra.dim)


class TestLigandEncoder:
    def test_forward(self, algebra):
        enc = LigandEncoder(algebra, hidden_dim=16, num_layers=2, num_rotors=4)
        pos = torch.randn(8, 3)
        z = torch.randint(1, 10, (8,))
        edges = _build_radius_graph(pos, cutoff=3.0)
        batch = torch.zeros(8, dtype=torch.long)
        h = enc(pos, z, edges, batch)
        assert h.shape == (8, 16, algebra.dim)


class TestGeometricCrossAttention:
    def test_forward(self, algebra):
        cross = GeometricCrossAttention(
            algebra, protein_dim=16, ligand_dim=16, interaction_dim=32, num_rotors=4
        )
        protein_h = torch.randn(10, 16, algebra.dim)
        ligand_h = torch.randn(5, 16, algebra.dim)
        protein_pos = torch.randn(10, 3)
        ligand_pos = torch.randn(5, 3)
        protein_batch = torch.zeros(10, dtype=torch.long)
        ligand_batch = torch.zeros(5, dtype=torch.long)

        out = cross(protein_h, ligand_h, protein_pos, ligand_pos,
                    protein_batch, ligand_batch, batch_size=1)
        assert out.shape == (1, 32)


class TestPDBBindNet:
    def test_forward(self, algebra):
        model = PDBBindNet(
            algebra, protein_hidden_dim=16, ligand_hidden_dim=16,
            interaction_dim=32, num_protein_layers=2, num_ligand_layers=2,
            num_rotors=4
        )
        protein_pos = torch.randn(10, 3)
        protein_z = torch.randint(1, 10, (10,))
        protein_aa = torch.randint(0, 20, (10,))
        protein_edges = _build_radius_graph(protein_pos, cutoff=3.0)
        protein_batch = torch.zeros(10, dtype=torch.long)

        ligand_pos = torch.randn(5, 3)
        ligand_z = torch.randint(1, 10, (5,))
        ligand_edges = _build_radius_graph(ligand_pos, cutoff=3.0)
        ligand_batch = torch.zeros(5, dtype=torch.long)

        aff = model(protein_pos, protein_z, protein_aa, protein_edges,
                     protein_batch, ligand_pos, ligand_z, ligand_edges,
                     ligand_batch, batch_size=1)
        assert aff.shape == (1,)

    def test_sparsity_loss(self, algebra):
        model = PDBBindNet(algebra, protein_hidden_dim=16, ligand_hidden_dim=16,
                           interaction_dim=32, num_protein_layers=2,
                           num_ligand_layers=2, num_rotors=4)
        loss = model.total_sparsity_loss()
        assert loss >= 0

    def test_grad_flow(self, algebra):
        model = PDBBindNet(
            algebra, protein_hidden_dim=16, ligand_hidden_dim=16,
            interaction_dim=32, num_protein_layers=2, num_ligand_layers=2,
            num_rotors=4
        )
        protein_pos = torch.randn(6, 3)
        protein_z = torch.randint(1, 10, (6,))
        protein_aa = torch.randint(0, 20, (6,))
        protein_edges = _build_radius_graph(protein_pos, cutoff=3.0)
        protein_batch = torch.zeros(6, dtype=torch.long)

        ligand_pos = torch.randn(4, 3)
        ligand_z = torch.randint(1, 10, (4,))
        ligand_edges = _build_radius_graph(ligand_pos, cutoff=3.0)
        ligand_batch = torch.zeros(4, dtype=torch.long)

        aff = model(protein_pos, protein_z, protein_aa, protein_edges,
                     protein_batch, ligand_pos, ligand_z, ligand_edges,
                     ligand_batch, batch_size=1)
        loss = aff.sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestPDBBindHermitianMetrics:
    def test_hermitian_norm_of_features(self, algebra):
        model = PDBBindNet(
            algebra, protein_hidden_dim=16, ligand_hidden_dim=16,
            interaction_dim=32, num_protein_layers=2, num_ligand_layers=2,
            num_rotors=4
        )
        protein_pos = torch.randn(10, 3)
        protein_z = torch.randint(1, 10, (10,))
        protein_aa = torch.randint(0, 20, (10,))
        protein_edges = _build_radius_graph(protein_pos, cutoff=3.0)
        protein_batch = torch.zeros(10, dtype=torch.long)

        ligand_pos = torch.randn(5, 3)
        ligand_z = torch.randint(1, 10, (5,))
        ligand_edges = _build_radius_graph(ligand_pos, cutoff=3.0)
        ligand_batch = torch.zeros(5, dtype=torch.long)

        model(protein_pos, protein_z, protein_aa, protein_edges,
              protein_batch, ligand_pos, ligand_z, ligand_edges,
              ligand_batch, batch_size=1)

        prot_feat, lig_feat = model.get_latent_features()
        assert prot_feat is not None
        assert lig_feat is not None

        h_norm_prot = hermitian_norm(algebra, prot_feat)
        h_norm_lig = hermitian_norm(algebra, lig_feat)
        assert (h_norm_prot >= 0).all()
        assert (h_norm_lig >= 0).all()

    def test_grade_reg(self, algebra):
        grade_reg = HermitianGradeRegularization(
            algebra, target_spectrum=[0.25, 0.25, 0.25, 0.25]
        )
        features = torch.randn(10, 16, algebra.dim)
        loss = grade_reg(features)
        assert loss.shape == ()
        assert loss >= 0
