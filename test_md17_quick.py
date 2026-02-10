#!/usr/bin/env python
"""Quick test script for MD17 task implementation."""

import torch
from datasets.md17 import get_md17_loaders
from core.algebra import CliffordAlgebra
from models.md17_forcenet import MD17ForceNet

def test_dataset():
    """Test dataset loading."""
    print("=" * 60)
    print("Testing MD17 Dataset Loading...")
    print("=" * 60)

    train_loader, val_loader, test_loader, e_mean, e_std, f_mean, f_std = get_md17_loaders(
        root="./data/MD17",
        molecule="aspirin",
        batch_size=4,
        max_samples=100  # Small subset for testing
    )

    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch structure:")
    print(f"  z (atomic numbers): {batch.z.shape}")
    print(f"  pos (positions): {batch.pos.shape}")
    print(f"  energy: {batch.energy.shape}")
    print(f"  force: {batch.force.shape}")
    print(f"  batch: {batch.batch.shape}")
    print(f"  edge_index: {batch.edge_index.shape}")

    print(f"\nNormalization stats:")
    print(f"  Energy: mean={e_mean:.2f}, std={e_std:.2f}")
    print(f"  Force: mean={f_mean:.4f}, std={f_std:.4f}")

    return train_loader, val_loader, test_loader

def test_model():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Testing MD17ForceNet Model...")
    print("=" * 60)

    # Create algebra and model
    algebra = CliffordAlgebra(p=3, q=0, device='cpu')
    model = MD17ForceNet(
        algebra=algebra,
        hidden_dim=16,
        num_layers=2,
        num_rotors=4
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load a batch
    train_loader, _, _ = get_md17_loaders(
        root="./data/MD17",
        molecule="aspirin",
        batch_size=2,
        max_samples=10
    )
    batch = next(iter(train_loader))

    # Forward pass
    print("\nRunning forward pass...")
    energy_pred, force_pred = model(batch.z, batch.pos, batch.batch, batch.edge_index)

    print(f"  Energy prediction shape: {energy_pred.shape}")
    print(f"  Force prediction shape: {force_pred.shape}")
    print(f"  Energy values: {energy_pred}")
    print(f"  Force values (first atom): {force_pred[0]}")

    # Test backward pass
    print("\nTesting backward pass...")
    loss = energy_pred.sum() + force_pred.sum()
    loss.backward()
    print("  Backward pass successful!")

    return model

def test_task():
    """Test task execution for 2 epochs."""
    print("\n" + "=" * 60)
    print("Testing MD17Task Execution (2 epochs)...")
    print("=" * 60)

    from omegaconf import OmegaConf
    from tasks.md17 import MD17Task

    # Create minimal config
    cfg = OmegaConf.create({
        'name': 'md17',
        'algebra': {'p': 3, 'q': 0, 'device': 'cpu'},
        'dataset': {
            'name': 'md17',
            'molecule': 'aspirin',
            'samples': 100  # Very small for quick test
        },
        'model': {
            'hidden_dim': 16,
            'layers': 2,
            'num_rotors': 4,
            'max_z': 100
        },
        'training': {
            'epochs': 2,
            'lr': 0.001,
            'batch_size': 4,
            'optimizer_type': 'riemannian_adam',
            'loss_weights': {'energy': 1.0, 'force': 10.0},
            'max_bivector_norm': 10.0
        }
    })

    # Create and run task
    task = MD17Task(cfg)
    task.run()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    # Run tests sequentially
    test_dataset()
    test_model()
    test_task()
