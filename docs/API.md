# API Reference

## Core
*   **`CliffordAlgebra(p, q)`**: Main kernel.
    *   `geometric_product(A, B)`
    *   `exp(A)`
    *   `reverse(A)`
*   **`Multivector`**: Object-oriented wrapper.
    *   `A * B`, `~A`, `A + B`

## Layers
*   **`RotorLayer(algebra, channels)`**
*   **`CliffordLinear(algebra, in, out)`**
*   **`CliffordGraphConv(algebra, in, out)`**
*   **`CliffordLayerNorm(algebra, channels)`**

## Functional
*   **`GeometricGELU`**: Magnitude-based activation.
*   **`GeometricMSELoss`**: Euclidean distance in embedding space.
*   **`SubspaceLoss`**: Penalizes energy in specific basis blades.
*   **`IsometryLoss`**: Enforces preservation of metric norm.
