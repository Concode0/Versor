# Layers

## Primitives
::: layers.primitives.rotor.RotorLayer
::: layers.primitives.multi_rotor.MultiRotorLayer
::: layers.primitives.linear.CliffordLinear
::: layers.primitives.rotor_gadget.RotorGadget
::: layers.primitives.normalization.CliffordLayerNorm
::: layers.primitives.projection.BladeSelector

## Blocks
::: layers.blocks.attention.GeometricProductAttention
::: layers.blocks.multi_rotor_ffn.MultiRotorFFN
::: layers.blocks.transformer.GeometricTransformerBlock

## Adapters
::: layers.adapters.embedding.MultivectorEmbedding
::: layers.adapters.mother.MotherEmbedding
::: layers.adapters.mother.EntropyGatedAttention

!!! note "Optional dependency"
    `CliffordGraphConv` requires `torch-geometric`. Install with `uv sync --extra md17`.

::: layers.adapters.gnn.CliffordGraphConv
