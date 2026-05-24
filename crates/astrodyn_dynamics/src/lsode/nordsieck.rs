//! LSODE Nordsieck history array, predictor, and rescale.
//!
//! The Nordsieck array stores, per ODE component `i` and column `j`, the
//! scaled derivative `history[i][j] = h^j / j! · y^(j)(t)` for
//! `j = 0..=order`. Column 0 is the solution itself.
//!
//! Ports the array operations from
//! `LsodeFirstOrderODEIntegrator::integrator_predict` and the column
//! rescale inside `integrator_reset_yh` (`__integrator.cc`). The
//! step/order *control* that decides when to predict or rescale lives in
//! the (pending) controller; this module is the pure array arithmetic and
//! is unit-tested for polynomial exactness.

/// Nordsieck history array: `num_odes` rows × `max_order + 1` columns.
#[derive(Debug, Clone)]
pub struct Nordsieck {
    /// `history[i][j] = h^j/j! · y_i^(j)(t)`. Row-major `[i][j]`.
    pub history: Vec<Vec<f64>>,
    /// Number of first-order ODE components (6 for one translational body).
    pub num_odes: usize,
    /// Maximum order the array is sized for (columns = `max_order + 1`).
    pub max_order: usize,
}

impl Nordsieck {
    /// Allocate a zeroed Nordsieck array for `num_odes` components at
    /// `max_order`.
    pub fn new(num_odes: usize, max_order: usize) -> Self {
        Self {
            history: vec![vec![0.0; max_order + 1]; num_odes],
            num_odes,
            max_order,
        }
    }

    /// Advance the array one step (Taylor shift by `h`) by multiplying by
    /// Pascal's triangle, in place. `order` is the current method order
    /// (`method_order_current`). Exact for solutions that are polynomials
    /// of degree ≤ `order`.
    ///
    /// Mirrors `integrator_predict`'s triple loop exactly.
    pub fn predict(&mut self, order: usize) {
        debug_assert!(order <= self.max_order, "predict: order exceeds array size");
        for i_iter in (1..=order).rev() {
            for j_hist in (i_iter - 1)..order {
                for k_var in 0..self.num_odes {
                    self.history[k_var][j_hist] += self.history[k_var][j_hist + 1];
                }
            }
        }
    }

    /// Rescale the history columns for a step-size change by `step_ratio`
    /// (`h_new = step_ratio · h_old`): column `j` is multiplied by
    /// `step_ratio^j`, for `j = 1..num_cols`. Column 0 (the solution) is
    /// unchanged.
    ///
    /// Mirrors the `do 180` loop in `integrator_reset_yh` (the step-ratio
    /// clamping that precedes it is controller logic, applied before this
    /// call).
    pub fn rescale_columns(&mut self, step_ratio: f64, num_cols: usize) {
        debug_assert!(
            num_cols <= self.max_order + 1,
            "rescale: num_cols too large"
        );
        let mut r = 1.0;
        for j in 1..num_cols {
            r *= step_ratio;
            for i in 0..self.num_odes {
                self.history[i][j] *= r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the exact Nordsieck array at time `t` for the per-component
    /// quadratic `y_i(t) = a_i + b_i t + c_i t²`, step `h`:
    /// col0 = y, col1 = h·y', col2 = h²/2·y''.
    fn quadratic_nordsieck(a: &[f64], b: &[f64], c: &[f64], t: f64, h: f64) -> Nordsieck {
        let n = a.len();
        let mut nd = Nordsieck::new(n, 2);
        for i in 0..n {
            let y = a[i] + b[i] * t + c[i] * t * t;
            let yp = b[i] + 2.0 * c[i] * t;
            let ypp = 2.0 * c[i];
            nd.history[i][0] = y;
            nd.history[i][1] = h * yp;
            nd.history[i][2] = h * h / 2.0 * ypp;
        }
        nd
    }

    #[test]
    fn predict_is_exact_for_quadratics() {
        let a = [1.0, -3.0];
        let b = [0.5, 2.0];
        let c = [-0.25, 0.1];
        let h = 0.3;
        let mut nd = quadratic_nordsieck(&a, &b, &c, 0.0, h);
        nd.predict(2);
        let want = quadratic_nordsieck(&a, &b, &c, h, h);
        for i in 0..2 {
            for j in 0..=2 {
                assert!(
                    (nd.history[i][j] - want.history[i][j]).abs() < 1e-13,
                    "component {i} col {j}: got {}, want {}",
                    nd.history[i][j],
                    want.history[i][j]
                );
            }
        }
    }

    #[test]
    fn predict_order1_is_linear_taylor_step() {
        // Order 1: col0 += col1 (y(t+h) = y(t) + h·y'), col1 unchanged.
        let mut nd = Nordsieck::new(1, 1);
        nd.history[0][0] = 5.0; // y
        nd.history[0][1] = 0.5; // h·y'
        nd.predict(1);
        assert!((nd.history[0][0] - 5.5).abs() < 1e-15);
        assert!((nd.history[0][1] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn rescale_scales_column_j_by_ratio_pow_j() {
        let mut nd = Nordsieck::new(1, 3);
        nd.history[0] = vec![1.0, 1.0, 1.0, 1.0];
        let ratio = 2.0;
        nd.rescale_columns(ratio, 4);
        // col0 unchanged; col j *= ratio^j.
        assert!((nd.history[0][0] - 1.0).abs() < 1e-15);
        assert!((nd.history[0][1] - 2.0).abs() < 1e-15);
        assert!((nd.history[0][2] - 4.0).abs() < 1e-15);
        assert!((nd.history[0][3] - 8.0).abs() < 1e-15);
    }
}
