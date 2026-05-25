//! Dense LU factorization + back-substitution for the LSODE stiff
//! (Newton/chord) corrector's iteration matrix `P = I − h·el0·J`.
//!
//! Faithful port of ODEPACK's `DGEFA` / `DGESL` (JEOD's `gauss_elim_factor`
//! / `linear_solver` in `lsode_first_order_ode_integrator__utility.cc`), with
//! one deliberate divergence: **correct absolute partial-pivot indices**.
//!
//! JEOD's `index_of_max_magnitude` returns the *relative* loop offset for an
//! off-diagonal pivot (`idamax = ii`) rather than the absolute row index
//! (`k + ii`), which would mis-address the pivot row on a swap. That branch
//! is dead in JEOD's use because the iteration matrix `P = I − h·el0·J` is
//! strongly diagonally dominant for the small steps the corrector takes
//! (`|P[k][k]| ≈ 1` dominates the off-diagonals `≈ h·el0·|J|`), so the pivot
//! is essentially always the diagonal and no swap occurs. We implement the
//! mathematically-standard absolute pivot index here: bit-identical to JEOD
//! in the diagonally-dominant regime that actually occurs, and correct in
//! the rare regime JEOD's quirk would corrupt.
//!
//! Fixed to `N_ODES` (the flattened `[pos; vel]` system) — no allocation.

use super::N_ODES;

/// In-place LU factorization with partial pivoting (`DGEFA`). On return,
/// `a` holds the combined L/U factors and the negated multipliers (the
/// `DGESL` convention), and `pivots[k]` is the absolute row swapped to
/// position `k`.
///
/// Returns `Err(col)` naming the first column with a zero pivot if the
/// matrix is singular (the chord corrector treats that as
/// `iteration_matrix_singular` and reduces the step). The corrector's `P`
/// is diagonally dominant, so a singular factorization is a genuine
/// red flag rather than an expected branch.
#[allow(
    clippy::float_cmp,
    reason = "exact zero-pivot test mirrors DGEFA's fpclassify(.)==FP_ZERO singular-column check"
)]
pub(crate) fn lu_factor(
    a: &mut [[f64; N_ODES]; N_ODES],
    pivots: &mut [usize; N_ODES],
) -> Result<(), usize> {
    let n = N_ODES;
    let mut info: Option<usize> = None;
    for k in 0..n - 1 {
        // Absolute row index of the column-k pivot (max |a[i][k]|, i ≥ k).
        let mut l = k;
        let mut maxv = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > maxv {
                maxv = v;
                l = i;
            }
        }
        pivots[k] = l;
        if a[l][k] == 0.0 {
            // Zero pivot column — already triangularized / singular.
            info.get_or_insert(k);
            continue;
        }
        if l != k {
            let t = a[l][k];
            a[l][k] = a[k][k];
            a[k][k] = t;
        }
        // Negated multipliers (DGEFA stores `-a[i][k]/a[k][k]`).
        let t = -1.0 / a[k][k];
        for i in (k + 1)..n {
            a[i][k] *= t;
        }
        // Row elimination with column indexing.
        for j in (k + 1)..n {
            let t = a[l][j];
            if l != k {
                a[l][j] = a[k][j];
                a[k][j] = t;
            }
            for i in (k + 1)..n {
                a[i][j] += t * a[i][k];
            }
        }
    }
    pivots[n - 1] = n - 1;
    if a[n - 1][n - 1] == 0.0 {
        info.get_or_insert(n - 1);
    }
    match info {
        Some(col) => Err(col),
        None => Ok(()),
    }
}

/// Solve `A·x = b` in place (`DGESL`, job 0) using the factors produced by
/// [`lu_factor`]. `y` holds the right-hand side on entry and the solution
/// on return.
pub(crate) fn lu_solve(
    a: &[[f64; N_ODES]; N_ODES],
    pivots: &[usize; N_ODES],
    y: &mut [f64; N_ODES],
) {
    let n = N_ODES;
    // Forward substitution: solve L·z = b (applying the pivot swaps).
    for k in 0..n - 1 {
        let l = pivots[k];
        let t = y[l];
        if l != k {
            y[l] = y[k];
            y[k] = t;
        }
        for i in (k + 1)..n {
            y[i] += t * a[i][k];
        }
    }
    // Back substitution: solve U·x = z.
    for k in (0..n).rev() {
        y[k] /= a[k][k];
        let t = -y[k];
        for i in 0..k {
            y[i] += t * a[i][k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multiply `A·x` for a freshly-built (unfactored) matrix.
    fn matvec(a: &[[f64; N_ODES]; N_ODES], x: &[f64; N_ODES]) -> [f64; N_ODES] {
        let mut out = [0.0; N_ODES];
        for i in 0..N_ODES {
            for j in 0..N_ODES {
                out[i] += a[i][j] * x[j];
            }
        }
        out
    }

    #[test]
    fn lu_solves_diagonally_dominant_system() {
        // A diagonally-dominant matrix in the regime the corrector's
        // P = I − h·el0·J actually occupies (≈ identity + small coupling).
        let mut a = [[0.0_f64; N_ODES]; N_ODES];
        for i in 0..N_ODES {
            for j in 0..N_ODES {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "tiny integer indices i,j ≤ 6 are exact in f64"
                )]
                let v = if i == j {
                    4.0 + i as f64
                } else {
                    0.1 * (i as f64 - j as f64)
                };
                a[i][j] = v;
            }
        }
        let x_true = [1.0, -2.0, 3.0, 0.5, -1.5, 2.25];
        let b = matvec(&a, &x_true);

        let mut lu = a;
        let mut pivots = [0usize; N_ODES];
        lu_factor(&mut lu, &mut pivots).expect("non-singular");
        let mut y = b;
        lu_solve(&lu, &pivots, &mut y);

        for i in 0..N_ODES {
            assert!(
                (y[i] - x_true[i]).abs() < 1e-12,
                "LU solve component {i}: got {}, want {} (residual {:e})",
                y[i],
                x_true[i],
                (y[i] - x_true[i]).abs()
            );
        }
    }

    #[test]
    fn lu_solves_system_requiring_a_row_swap() {
        // First column's largest entry is NOT on the diagonal, forcing a
        // pivot swap — the branch JEOD's relative-index quirk would
        // mis-handle. Exercises the absolute-pivot correctness directly.
        let mut a = [[0.0_f64; N_ODES]; N_ODES];
        for i in 0..N_ODES {
            a[i][i] = 1.0;
        }
        // Make row 3 dominate column 0.
        a[0][0] = 0.01;
        a[3][0] = 7.0;
        a[0][3] = 2.0;
        a[3][3] = 1.0;
        let x_true = [2.0, 1.0, -1.0, 4.0, -3.0, 0.7];
        let b = matvec(&a, &x_true);

        let mut lu = a;
        let mut pivots = [0usize; N_ODES];
        lu_factor(&mut lu, &mut pivots).expect("non-singular");
        let mut y = b;
        lu_solve(&lu, &pivots, &mut y);

        for i in 0..N_ODES {
            assert!(
                (y[i] - x_true[i]).abs() < 1e-12,
                "LU solve (with swap) component {i}: got {}, want {}",
                y[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn lu_factor_flags_singular_matrix() {
        // A zero column ⇒ singular ⇒ Err naming a zero-pivot column.
        let mut a = [[0.0_f64; N_ODES]; N_ODES];
        for i in 1..N_ODES {
            a[i][i] = 1.0;
        }
        // column 0 is entirely zero
        let mut pivots = [0usize; N_ODES];
        assert!(
            lu_factor(&mut a, &mut pivots).is_err(),
            "a matrix with a zero column must be reported singular"
        );
    }
}
