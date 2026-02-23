# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


# Synthetic fallback text (~200KB of structured English)
_SYNTHETIC_PARAGRAPHS = [
    "The geometric algebra provides a unified framework for representing and transforming geometric "
    "objects. Rotors encode rotations as exponentials of bivectors, preserving the manifold topology "
    "of the special orthogonal group. Unlike matrix representations, rotor composition is numerically "
    "stable and avoids gimbal lock entirely. The sandwich product R x R-tilde maps any grade-k "
    "multivector to another grade-k multivector, making rotors grade-preserving transformations.",

    "Language models learn to predict the next token given a sequence of previous tokens. The "
    "transformer architecture uses scaled dot-product attention to compute weighted sums of value "
    "vectors. Each attention head learns a different projection of queries, keys, and values, "
    "allowing the model to attend to information from different representation subspaces simultaneously.",

    "Clifford algebras generalize complex numbers and quaternions to arbitrary dimensions and "
    "signatures. The geometric product combines the inner product and wedge product into a single "
    "associative operation. Multivectors span all grades from scalars to the pseudoscalar. The "
    "grade-0 part of any product is a scalar invariant, forming a natural attention score.",

    "Neural networks learn hierarchical representations of data through composition of nonlinear "
    "transformations. Each layer extracts progressively more abstract features. Normalization layers "
    "stabilize training by controlling the scale of activations. Residual connections allow gradients "
    "to flow freely through deep networks, enabling training of very deep models.",

    "The exponential map sends a bivector to the group of rotors via the power series expansion. "
    "Truncating the series at order twelve provides sufficient accuracy for practical applications. "
    "The scaling-and-squaring method improves numerical stability when the input norm is large. "
    "Riemannian optimization respects the manifold structure of the Lie group during gradient descent.",

    "Attention mechanisms allow models to focus on relevant parts of the input sequence. Causal "
    "masking prevents information from future tokens leaking into past positions during autoregressive "
    "generation. The softmax function converts raw attention scores into a probability distribution "
    "over source positions, weighting the value vectors accordingly.",

    "Point clouds represent three-dimensional shapes as collections of unordered points in Euclidean "
    "space. Graph neural networks aggregate information from neighboring points to build local "
    "geometric features. The Chamfer distance measures reconstruction quality as the average nearest "
    "neighbor distance between predicted and target point clouds.",

    "Molecular property prediction requires models that respect the symmetries of physical space. "
    "Rotational equivariance ensures that rotating a molecule produces a correspondingly rotated "
    "output. The geometric algebra naturally encodes such equivariance through the rotor sandwich "
    "product, making it ideal for molecular machine learning applications.",

    "Weather forecasting on the sphere requires models that handle the periodic boundary conditions "
    "and the convergence of meridians at the poles. Graph neural networks on spherical grids can "
    "represent these topology-aware connections. Physics-informed losses encourage the model to "
    "respect conservation laws such as mass and energy balance.",

    "The bivector decomposition theorem states that any bivector in a finite-dimensional vector "
    "space can be written as a sum of simple bivectors, each representing a two-dimensional "
    "rotation plane. Power iteration finds the dominant simple component, enabling a hierarchical "
    "decomposition of complex rotations into elementary ones.",

    "Conformal geometric algebra embeds Euclidean space into a higher-dimensional space where "
    "translations become rotations. The null cone representation allows spheres, planes, and "
    "other geometric primitives to be represented uniformly as multivectors. Intersections and "
    "transformations become simple geometric products in this extended algebra.",

    "The Riemann zeta function encodes deep information about the distribution of prime numbers. "
    "Its analytic continuation to the complex plane reveals a rich structure of zeros. The "
    "Riemann hypothesis conjectures that all non-trivial zeros lie on the critical line where "
    "the real part equals one-half.",

    "Protein-ligand binding affinity prediction is a key challenge in computational drug discovery. "
    "The three-dimensional structure of the binding pocket determines which ligands bind strongly. "
    "Graph neural networks can process the atomic connectivity of both protein and ligand, "
    "learning geometric features relevant to binding.",

    "Positional encoding injects information about the position of tokens in a sequence. Sinusoidal "
    "encodings use frequencies at different scales to represent position uniquely. Learned positional "
    "embeddings allow the model to discover the most useful position representation for the task. "
    "Rotary positional encodings apply position-dependent rotations in the embedding space.",

    "The grade spectrum of a multivector measures how much energy is contained in each grade. "
    "Regularizing the grade spectrum toward a target distribution encourages the model to use "
    "all geometric grades meaningfully. The Hermitian norm provides a positive-definite measure "
    "of multivector magnitude that works for any algebra signature.",

    "Deep learning has revolutionized natural language processing, computer vision, and scientific "
    "computing. Large language models trained on vast corpora of text demonstrate emergent capabilities "
    "in reasoning, coding, and creative writing. The transformer architecture has proven remarkably "
    "flexible, scaling to billions of parameters across diverse modalities.",

    "The Spin group is the double cover of the special orthogonal group. Rotors form a smooth "
    "manifold with the structure of a Lie group. Riemannian optimization on this manifold uses "
    "the exponential map to retract gradient updates back onto the group. This ensures that learned "
    "transformations remain valid rotations throughout training.",

    "Feature normalization is crucial for stable training of deep networks. Layer normalization "
    "computes statistics over the feature dimension rather than the batch dimension, making it "
    "suitable for variable-length sequences. In the geometric setting, normalization should "
    "preserve the direction of multivectors while controlling their magnitude.",

    "The wedge product of two vectors produces a bivector representing the oriented area of the "
    "parallelogram they span. The magnitude equals the area and the orientation encodes the "
    "rotation direction. Higher-grade outer products represent oriented volumes in higher dimensions. "
    "These geometric objects transform naturally under rotor conjugation.",

    "Autoregressive language models generate text by repeatedly sampling the next token from the "
    "conditional distribution given all previous tokens. Temperature scaling controls the sharpness "
    "of the distribution, trading coherence for diversity. Beam search explores multiple hypotheses "
    "simultaneously to find high-probability sequences.",
] * 30  # Repeat to reach ~200KB


def _build_synthetic_text() -> str:
    """Builds the synthetic fallback corpus."""
    return "\n\n".join(_SYNTHETIC_PARAGRAPHS)


class TextDataset(Dataset):
    """Character-level text dataset for language modeling.

    Three-tier loading:
    1. Cached .pt file (fast)
    2. Raw input.txt (processed once)
    3. Synthetic fallback (always works)

    Args:
        data_path (str): Directory containing input.txt or cached .pt.
        seq_len (int): Sequence length for each sample.
        tokenizer (str): 'char' (only option currently).
        split (str): 'train' or 'val'.
        train_ratio (float): Fraction of data used for training.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 256,
        tokenizer: str = 'char',
        split: str = 'train',
        train_ratio: float = 0.9,
    ):
        self.seq_len = seq_len
        self.split = split

        cache_path = os.path.join(data_path, f"text_cache_{tokenizer}.pt")
        raw_path = os.path.join(data_path, "input.txt")

        # Tier 1: Load from cache
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=False)
            data = cache['data']
            self.char_to_idx = cache['char_to_idx']
            self.idx_to_char = cache['idx_to_char']

        # Tier 2: Load from raw file
        elif os.path.exists(raw_path):
            with open(raw_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            data, self.char_to_idx, self.idx_to_char = self._encode(text)
            os.makedirs(data_path, exist_ok=True)
            torch.save({
                'data': data,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
            }, cache_path)

        # Tier 3: Synthetic fallback
        else:
            text = _build_synthetic_text()
            data, self.char_to_idx, self.idx_to_char = self._encode(text)
            # Don't cache synthetic data to disk

        # Train/val split
        split_idx = int(len(data) * train_ratio)
        if split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

    def _encode(self, text: str):
        """Encodes text to integer token ids."""
        chars = sorted(set(text))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for i, c in enumerate(chars)}
        data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
        return data, char_to_idx, idx_to_char

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens."""
        return len(self.char_to_idx)

    def decode(self, ids) -> str:
        """Converts token ids back to string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join(self.idx_to_char.get(i, '?') for i in ids)

    def __len__(self) -> int:
        # Number of non-overlapping windows
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def get_text_loaders(
    data_path: str,
    seq_len: int = 256,
    batch_size: int = 16,
    tokenizer: str = 'char',
    train_ratio: float = 0.9,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int]:
    """Creates train and validation DataLoaders for text.

    Args:
        data_path: Path to data directory.
        seq_len: Sequence length.
        batch_size: Batch size.
        tokenizer: Tokenization scheme ('char').
        train_ratio: Train split fraction.
        num_workers: DataLoader workers.

    Returns:
        (train_loader, val_loader, vocab_size)
    """
    train_ds = TextDataset(data_path, seq_len, tokenizer, 'train', train_ratio)
    val_ds = TextDataset(data_path, seq_len, tokenizer, 'val', train_ratio)

    # Share vocab from train set to val set (they should be identical for char tokenizer)
    val_ds.char_to_idx = train_ds.char_to_idx
    val_ds.idx_to_char = train_ds.idx_to_char

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,    )

    return train_loader, val_loader, train_ds.vocab_size
