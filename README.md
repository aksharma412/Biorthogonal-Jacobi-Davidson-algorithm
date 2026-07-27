# Biorthogonal Jacobi-Davidson

A from-scratch implementation of the biorthogonal Jacobi-Davidson (BiJD) method for **non-Hermitian** eigenvalue problems, with a full validation study against a direct reference solver.

## Why

Standard iterative eigensolvers assume a symmetric or Hermitian matrix. Excited-state methods in quantum chemistry, notably equation-of-motion coupled-cluster, do not give you one: the similarity-transformed Hamiltonian is non-Hermitian, its left and right eigenvectors differ, and they are not orthogonal to each other. Applying a symmetric solver produces answers that look plausible and are wrong.

BiJD handles this directly. It propagates left and right eigenvectors simultaneously and enforces the biorthogonality condition **v**<sub>L</sub><sup>H</sup>**v**<sub>R</sub> = 1 at every iteration, which is what keeps the Rayleigh quotient and the residuals meaningful.

## Method

Each iteration:

1. Rescale the left vector to enforce **v**<sub>L</sub><sup>H</sup>**v**<sub>R</sub> = 1
2. Estimate the eigenvalue from the two-sided Rayleigh quotient, λ = **v**<sub>L</sub><sup>H</sup>A**v**<sub>R</sub>
3. Form left and right residuals
4. Solve projected correction equations for both vectors, with a shift to avoid singularity at convergence and Tikhonov regularization for stability
5. Update, renormalize, restore biorthogonality

Convergence is declared when both residual norms fall below tolerance (default 10<sup>-13</sup>).

## Results

Validated against `scipy.linalg.eig` on a dense non-symmetric test matrix (n = 100):

| Quantity | Behaviour |
|---|---|
| Eigenvalue estimate | converges to the closest exact eigenvalue |
| Right eigenvector | converges to the reference right eigenvector |
| Left eigenvector | converges to the reference left eigenvector |
| Residual norms | decrease monotonically to tolerance |

**Cost, compared against a standard Jacobi-Davidson implementation on the same problem:**

| | BiJD | Jacobi-Davidson |
|---|---|---|
| Time | 0.01 s | 0.25 s |
| Peak memory | 2055 kB | 1630 kB |

BiJD reaches the solution roughly 25× faster here, at higher memory cost, because it carries two subspaces rather than one. Profiling identifies the update-and-normalize step for the left and right vectors as the dominant expense.

## Contents

| File | |
|---|---|
| `biorthogonal_jacobi_davidson.ipynb` | Implementation, validation, and convergence plots |
| `report.pdf` | 17-page write-up: survey of iterative eigensolvers (power, Rayleigh quotient, Arnoldi, Lanczos, Davidson, Jacobi-Davidson), derivation of BiJD, and results |

## Usage

```python
lam, vR, vL, residuals, estimates, vR_diffs, vL_diffs, w, vl, vr = \
    bi_orthogonal_jacobi_davidson(A, vR_init, vL_init,
                                  tol=1e-13, max_iter=800,
                                  shift=1e-15, reg=1e-2)
```

Requires `numpy`, `scipy`, `matplotlib`.

## Scope and limitations

This is a research prototype written to study the algorithm's convergence behaviour, not a production solver. Specifically:

- **Dense throughout.** The correction equations are solved with a dense `scipy.linalg.solve` on the full n × n system each iteration, so cost scales as O(n³) per iteration. A production implementation would solve them iteratively and never form the full operator, which is where Jacobi-Davidson-type methods earn their keep on large sparse problems.
- **The test matrix is dense and random**, not representative of the strongly diagonally dominant matrices that arise in quantum chemistry. Diagonal dominance is precisely what Davidson-type methods exploit, so behaviour on a chemistry Hamiltonian would differ.
- **The reference diagonalization is computed inside the solver** for the convergence diagnostics. That is fine for a validation study but must be removed before timing anything seriously.
- Single eigenpair only; no restart, deflation, or blocking.

## References

See `report.pdf` for the full bibliography.
