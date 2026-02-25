"""Tests for the GA-native Transformer language model.

Covers:
- MultivectorEmbedding: shape, grade-1 init, gradient flow
- RotaryBivectorPE: shape, positions differ, gradient flow
- GeometricProductAttention: shape, causal mask, gradient flow
- MultiRotorFFN: shape, channels preserved, gradient flow
- GATransformerBlock: shape, causality, residual connections
- GALanguageModel: shape, causal, return_hidden, full backward
- TextDataset: synthetic fallback, shapes, vocab size
"""

import math
import pytest
import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from layers.embedding import MultivectorEmbedding, RotaryBivectorPE
from layers.attention import GeometricProductAttention
from layers.multi_rotor_ffn import MultiRotorFFN
from models.ga_transformer import GATransformerBlock, GALanguageModel
from datasets.text import TextDataset, get_text_loaders


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def algebra():
    """Cl(3,1) - same algebra used by the LM task."""
    return CliffordAlgebra(p=3, q=1, device='cpu')


@pytest.fixture(scope="module")
def small_algebra():
    """Cl(2,0) - tiny algebra for cheap tests."""
    return CliffordAlgebra(p=2, q=0, device='cpu')


# ---------------------------------------------------------------------------
# TestMultivectorEmbedding
# ---------------------------------------------------------------------------

class TestMultivectorEmbedding:
    """Tests for MultivectorEmbedding."""

    def test_output_shape(self, algebra):
        vocab, channels, B, L = 32, 4, 2, 8
        emb = MultivectorEmbedding(algebra, vocab, channels)
        ids = torch.randint(0, vocab, (B, L))
        out = emb(ids)
        assert out.shape == (B, L, channels, algebra.dim)

    def test_grade1_init(self, algebra):
        """Non-grade-1 components should be zero after init."""
        vocab, channels = 16, 4
        emb = MultivectorEmbedding(algebra, vocab, channels)
        # Flatten the embedding weight: [vocab, channels * dim]
        w = emb.embedding.weight.detach()
        dim = algebra.dim
        for d in range(dim):
            if bin(d).count('1') != 1:
                # All entries for this non-grade-1 blade should be zero
                # Column indices: ch*dim + d for each ch
                cols = [ch * dim + d for ch in range(channels)]
                assert w[:, cols].abs().max().item() == 0.0, \
                    f"Blade {d} (grade {bin(d).count('1')}) should be zeroed"

    def test_gradient_flow(self, algebra):
        vocab, channels, B, L = 16, 4, 2, 5
        emb = MultivectorEmbedding(algebra, vocab, channels)
        ids = torch.randint(0, vocab, (B, L))
        out = emb(ids)
        loss = out.sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None
        assert emb.embedding.weight.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# TestRotaryBivectorPE
# ---------------------------------------------------------------------------

class TestRotaryBivectorPE:
    """Tests for RotaryBivectorPE."""

    def test_output_shape(self, algebra):
        channels, max_len, B, L = 4, 32, 2, 8
        pe = RotaryBivectorPE(algebra, channels, max_len)
        x = torch.randn(B, L, channels, algebra.dim)
        out = pe(x)
        assert out.shape == (B, L, channels, algebra.dim)

    def test_positions_differ(self, algebra):
        """Different positions should produce different encodings."""
        channels, max_len = 4, 32
        pe = RotaryBivectorPE(algebra, channels, max_len, learnable=False)
        x = torch.ones(1, 8, channels, algebra.dim)
        out = pe(x)
        # Two neighbouring positions should not be identical
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-5)

    def test_gradient_flow(self, algebra):
        channels, max_len, B, L = 4, 32, 2, 6
        pe = RotaryBivectorPE(algebra, channels, max_len, learnable=True)
        x = torch.randn(B, L, channels, algebra.dim)
        out = pe(x)
        out.sum().backward()
        assert pe.bivector_weights.grad is not None


# ---------------------------------------------------------------------------
# TestGeometricProductAttention
# ---------------------------------------------------------------------------

class TestGeometricProductAttention:
    """Tests for GeometricProductAttention."""

    def test_output_shape(self, algebra):
        channels, heads, B, L = 8, 2, 2, 6
        attn = GeometricProductAttention(algebra, channels, heads, causal=True)
        x = torch.randn(B, L, channels, algebra.dim)
        out = attn(x)
        assert out.shape == (B, L, channels, algebra.dim)

    def test_causal_mask_no_future_leak(self, algebra):
        """Changing future tokens must not change current-position output."""
        channels, heads, B, L = 8, 2, 1, 6
        attn = GeometricProductAttention(algebra, channels, heads, causal=True)
        attn.eval()

        x = torch.randn(B, L, channels, algebra.dim)
        x_mod = x.clone()
        # Corrupt all tokens after position 2
        x_mod[0, 3:] = torch.randn_like(x_mod[0, 3:])

        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x_mod)

        # Position 0..2 output should be identical (causal isolation)
        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-4), \
            "Future tokens leaked into past positions - causal mask broken"

    def test_gradient_flow(self, algebra):
        channels, heads, B, L = 8, 2, 2, 4
        attn = GeometricProductAttention(algebra, channels, heads, causal=True)
        x = torch.randn(B, L, channels, algebra.dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# TestMultiRotorFFN
# ---------------------------------------------------------------------------

class TestMultiRotorFFN:
    """Tests for MultiRotorFFN (the Geometric Toolbox layer)."""

    def test_output_shape(self, algebra):
        channels, B = 8, 3
        ffn = MultiRotorFFN(algebra, channels, ffn_mult=2, num_rotors=4)
        x = torch.randn(B, channels, algebra.dim)
        out = ffn(x)
        assert out.shape == (B, channels, algebra.dim), \
            "FFN must preserve (B, channels, dim) shape"

    def test_channels_preserved(self, algebra):
        """Input and output channel count must match."""
        for ch in [4, 8, 16]:
            ffn = MultiRotorFFN(algebra, ch, ffn_mult=2, num_rotors=4)
            x = torch.randn(2, ch, algebra.dim)
            out = ffn(x)
            assert out.shape[1] == ch

    def test_gradient_flow(self, algebra):
        channels, B = 8, 2
        ffn = MultiRotorFFN(algebra, channels, ffn_mult=2, num_rotors=4)
        x = torch.randn(B, channels, algebra.dim, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_small_algebra(self, small_algebra):
        """Works on a minimal algebra."""
        ffn = MultiRotorFFN(small_algebra, channels=4, ffn_mult=2, num_rotors=2)
        x = torch.randn(2, 4, small_algebra.dim)
        out = ffn(x)
        assert out.shape == (2, 4, small_algebra.dim)


# ---------------------------------------------------------------------------
# TestGATransformerBlock
# ---------------------------------------------------------------------------

class TestGATransformerBlock:
    """Tests for GATransformerBlock."""

    def test_output_shape(self, algebra):
        channels, heads, B, L = 8, 2, 2, 6
        block = GATransformerBlock(algebra, channels, heads,
                                   ffn_mult=2, num_rotors=4, causal=True)
        x = torch.randn(B, L, channels, algebra.dim)
        out = block(x)
        assert out.shape == (B, L, channels, algebra.dim)

    def test_causality_preserved(self, algebra):
        """Changing future tokens must not affect past-position block output."""
        channels, heads, B, L = 8, 2, 1, 8
        block = GATransformerBlock(algebra, channels, heads,
                                   ffn_mult=2, num_rotors=4, causal=True)
        block.eval()

        x = torch.randn(B, L, channels, algebra.dim)
        x_mod = x.clone()
        x_mod[0, 5:] = torch.randn_like(x_mod[0, 5:])

        with torch.no_grad():
            out1 = block(x)
            out2 = block(x_mod)

        assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-4), \
            "Block causality broken"

    def test_residual_connections(self, algebra):
        """With zero weights, output should still resemble input (residual path)."""
        channels, heads = 4, 2
        block = GATransformerBlock(algebra, channels, heads,
                                   ffn_mult=2, num_rotors=2, causal=True)
        # Zero out all learnable parameters
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        x = torch.randn(1, 4, channels, algebra.dim)
        out = block(x)
        # Residual path means output is related to input (not all zeros)
        assert out.abs().sum().item() > 0.0

    def test_gradient_flow(self, algebra):
        channels, heads, B, L = 8, 2, 2, 4
        block = GATransformerBlock(algebra, channels, heads,
                                   ffn_mult=2, num_rotors=4, causal=True)
        x = torch.randn(B, L, channels, algebra.dim, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# TestGALanguageModel
# ---------------------------------------------------------------------------

class TestGALanguageModel:
    """Tests for GALanguageModel."""

    @pytest.fixture
    def small_lm(self, algebra):
        return GALanguageModel(
            algebra=algebra,
            vocab_size=32,
            channels=8,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
            ffn_mult=2,
            num_rotors=4,
            causal=True,
        )

    def test_logits_shape(self, small_lm, algebra):
        B, L, vocab = 2, 8, 32
        ids = torch.randint(0, vocab, (B, L))
        logits = small_lm(ids)
        assert logits.shape == (B * L, vocab)

    def test_return_hidden(self, small_lm, algebra):
        B, L, vocab, channels = 2, 8, 32, 8
        ids = torch.randint(0, vocab, (B, L))
        logits, hidden = small_lm(ids, return_hidden=True)
        assert logits.shape == (B * L, vocab)
        assert hidden.shape == (B, L, channels, algebra.dim)

    def test_causal_property(self, algebra):
        """Identical inputs at positions 0..3 with different future -> same logits."""
        vocab, channels = 16, 8
        lm = GALanguageModel(
            algebra=algebra,
            vocab_size=vocab,
            channels=channels,
            num_layers=1,
            num_heads=2,
            max_seq_len=16,
            ffn_mult=2,
            num_rotors=4,
            causal=True,
        )
        lm.eval()
        B, L = 1, 8
        ids = torch.randint(0, vocab, (B, L))
        ids_mod = ids.clone()
        ids_mod[0, 5:] = torch.randint(0, vocab, (3,))

        with torch.no_grad():
            logits1 = lm(ids).reshape(B, L, vocab)
            logits2 = lm(ids_mod).reshape(B, L, vocab)

        assert torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-4), \
            "Model is not causal - future tokens leaked into past logits"

    def test_full_backward(self, small_lm):
        B, L, vocab = 2, 8, 32
        ids = torch.randint(0, vocab, (B, L))
        targets = torch.randint(0, vocab, (B, L))
        logits = small_lm(ids)
        loss = nn.CrossEntropyLoss()(logits, targets.flatten())
        loss.backward()
        # Check at least some parameters have gradients
        grads = [p.grad for p in small_lm.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed - backward pass broken"


# ---------------------------------------------------------------------------
# TestTextDataset
# ---------------------------------------------------------------------------

class TestTextDataset:
    """Tests for TextDataset with synthetic fallback."""

    def test_synthetic_fallback(self, tmp_path):
        """Dataset works without any data files (synthetic corpus)."""
        ds = TextDataset(str(tmp_path), seq_len=64, split='train')
        assert len(ds) > 0
        assert ds.vocab_size > 0

    def test_sample_shapes(self, tmp_path):
        seq_len = 64
        ds = TextDataset(str(tmp_path), seq_len=seq_len, split='train')
        x, y = ds[0]
        assert x.shape == (seq_len,)
        assert y.shape == (seq_len,)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_target_is_shifted_input(self, tmp_path):
        """y should be x shifted by one position."""
        ds = TextDataset(str(tmp_path), seq_len=32, split='train')
        x, y = ds[0]
        x2, _ = ds[1]
        # y[0] == x[1], y[1] == x[2], ...
        assert torch.all(y[:-1] == x[1:]).item(), \
            "y is not a one-step shift of x"

    def test_vocab_size(self, tmp_path):
        ds = TextDataset(str(tmp_path), seq_len=32, split='train')
        assert ds.vocab_size >= 26, "Synthetic corpus should contain at least 26 chars"

    def test_get_text_loaders(self, tmp_path):
        train_loader, val_loader, vocab_size = get_text_loaders(
            data_path=str(tmp_path),
            seq_len=32,
            batch_size=4,
        )
        assert vocab_size > 0
        x, y = next(iter(train_loader))
        assert x.shape == (4, 32)
        assert y.shape == (4, 32)
