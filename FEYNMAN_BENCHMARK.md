# Feynman Symbolic Regression Benchmark

> Generated: 2026-02-21 12:54

## Setup

| Parameter | Value |
|-----------|-------|
| Algebra | Cl(4, 0) |
| Hidden channels | 16 |
| Layers | 3 residual blocks |
| Rotors / layer | 8 |
| Optimizer | RiemannianAdam (lr = 0.001) |
| Epochs | 30 |
| Samples | 10,000 (70 / 15 / 15 split) |
| Batch size | 128 |
| Sparsity weight | 0.01 |
| Orthogonality | enabled, mode = loss, weight = 0.05 |
| Target grades | [0, 1, 2] (scalar + vector + bivector) |
| Ortho warmup | 5 epochs |

## Results

| # | Equation | Vars | Test MAE | R² | Time (s) | Description |
|---|----------|------|----------|----|----------|-------------|
| 1 | `I.12.1` | 2 | 0.009474 | 0.9999 | 107 | F1 + F2 |
| 2 | `I.26.2` | 2 | 0.001790 | 0.9998 | 104 | arcsin(n*sin(theta2)) |
| 3 | `I.6.20` | 3 | 0.011680 | 0.9583 | 106 | Gaussian distribution |  - In 50 epochs, we achive ~= 0.99
| 4 | `I.24.6` | 3 | 0.108175 | 0.9998 | 108 | m*(omega^2+omega0^2)/4 |
| 5 | `I.34.1` | 3 | 0.169366 | 0.9983 | 101 | omega0/(1-v/c) |
| 6 | `I.37.4` | 3 | 0.037971 | 0.9997 | 100 | I1+I2+2*sqrt(I1*I2)*cos(delta) |
| 7 | `I.43.43` | 3 | 0.046499 | 0.9995 | 99 | mu*T/q |
| 8 | `I.12.2` | 4 | 0.003311 | 0.9962 | 103 | q1*q2/(4*pi*eps*r^2) |
| 9 | `I.15.1` | 4 | 0.040433 | 0.9988 | 97 | (x-u*t)/sqrt(1-(u/c)^2) |
| 10 | `I.50.26` | 4 | 0.110525 | 0.9950 | 92 | x1*(cos(w*t)+alpha*cos^2(w*t)) |
| 11 | `II.2.42` | 4 | 0.237622 | 0.9987 | 98 | kappa*(T2-T1)*A/d |
| 12 | `I.34.27` | 4 | 0.116524 | 0.9951 | 101 | omega0*(1+v*cos(theta)/c)/sqrt(1-(v/c)^2) |
| 13 | `III.4.33` | 4 | 0.059085 | 0.9974 | 99 | h*omega/(exp(h*omega/(k*T))-1) |
| 14 | `I.9.18` | 4 | 0.243835 | 0.9979 | 99 | G*m1*m2/r^2 |
| 15 | `II.11.27` | 5 | 0.031839 | 0.9984 | 101 | n0*exp(-m*u^2/(2*k*T)) |

## Summary

| Metric | Value |
|--------|-------|
| Equations run | 15 |
| Mean test MAE | 0.081875 |
| Median R² | 0.9984 |
| Best R² | 0.9999 (`I.12.1` — F1 + F2) |
| Worst R² | 0.9583 (`I.6.20` — Gaussian distribution) |
| R² ≥ 0.90 | 15/15 equations |
| R² ≥ 0.80 | 15/15 equations |
| Total wall time | 25.3 min |

## Notes

- All inputs / outputs are **z-score normalised** before training; MAE is reported in **original units**.
- Orthogonality loss penalises energy leaking into grades > 2, encouraging the network to stay in the scalar + vector + bivector sub-algebra throughout.
- Checkpoints are saved to the best validation MAE; test metrics are evaluated on the best checkpoint.
- Equations with transcendental functions (arcsin, exp, cos) are generally harder than polynomial ones.

