# Geometric Algebra Fundamentals

Versor is built upon **Clifford Algebra** (Geometric Algebra), a mathematical framework that unifies geometry and algebra.

## 1. The Geometric Product
The core operation is the **Geometric Product** ($ab$), which decomposes into:
$$ ab = a \cdot b + a \wedge b $$
*   **Inner Product ($a \cdot b$)**: Symmetric, scalar-valued (like dot product). Measures projection.
*   **Outer Product ($a \wedge b$)**: Anti-symmetric, bivector-valued (like cross product but generalized). Measures area/volume.

## 2. Multivectors
A **Multivector** is a sum of different **grades** (scalars, vectors, bivectors, trivectors...).
For an $n$-dimensional space, there are $2^n$ basis blades.

Example in 3D ($Cl(3,0)$):
*   **Grade 0 (Scalar)**: $1$
*   **Grade 1 (Vector)**: $e_1, e_2, e_3$ (X, Y, Z axes)
*   **Grade 2 (Bivector)**: $e_{12}, e_{13}, e_{23}$ (XY, XZ, YZ planes)
*   **Grade 3 (Trivector)**: $e_{123}$ (Volume)

## 3. Rotors and Transformations
In standard linear algebra, rotations are matrices. In GA, rotations are represented by **Rotors** ($R$), which are even-grade multivectors satisfying $R\tilde{R}=1$.

A rotation of vector $x$ by angle $\theta$ in plane $B$ is given by the **Sandwich Product**:
$$ x' = R x \tilde{R} $$
$$ R = \exp(-B/2) = \cos(\theta/2) - I \sin(\theta/2) $$
where $I$ is the unit bivector of the plane.

### Advantages of Rotors
1.  **Metric Agnostic**: Works in Euclidean, Hyperbolic, and Projective spaces without changing the formula.
2.  **Interpolation**: Rotors can be interpolated (SLERP) smoothly.
3.  **Composition**: $R_{total} = R_2 R_1$.
4.  **No Gimbal Lock**: Unlike Euler angles.

## 4. Metric Signatures $(p, q)$
*   **Euclidean**: $Cl(n, 0)$. Basis vectors square to $+1$.
*   **Minkowski (Spacetime)**: $Cl(1, 3)$ or $Cl(1, 1)$. Time squares to $+1$, Space to $-1$ (or vice versa).
*   **Conformal (CGA)**: $Cl(4, 1)$ for 3D Euclidean space. Adds origin ($e_o$) and infinity ($e_\infty$).

Versor's kernel automatically handles the sign flips required for these metrics.