# Tasks and Experiments

Versor includes several pre-defined tasks to demonstrate its capabilities.

## 1. Manifold Restoration (Unbending)
*   **Goal**: Flatten a distorted 3D manifold (e.g., a "Figure-8" saddle $z=0.5xy$) back to a 2D plane.
*   **Method**: Learns a Rotor to align the principal components and a BladeSelector to suppress the distortion axis.
*   **Run**: `python main.py task=manifold`

## 2. Cross-Modal Unification
*   **Goal**: Align embeddings from two different modalities (e.g., Text and Image) into a shared semantic space.
*   **Dataset**: Synthetic pair of (BERT Embeddings, Rotated/Noisy Embeddings).
*   **Method**: Two `RotorLayer` encoders trained with Contrastive Geometric Loss.
*   **Run**: `python main.py task=crossmodal`

## 3. Hyperbolic Geometry (Lorentz Boost)
*   **Goal**: Reverse a relativistic Lorentz boost in Minkowski Spacetime ($Cl(1, 1)$).
*   **Method**: The network learns a rotor that corresponds to the inverse boost parameter (rapidity).
*   **Run**: `python main.py task=hyperbolic`

## 4. Semantic Disentanglement
*   **Goal**: Rotate high-dimensional word embeddings so that semantic concepts align with orthogonal geometric axes (basis blades).
*   **Method**: Minimizes "Grade Purity Loss", forcing topics into specific grades (e.g., "Tech" -> Vectors, "Nature" -> Bivectors).
*   **Run**: `python main.py task=semantic`
