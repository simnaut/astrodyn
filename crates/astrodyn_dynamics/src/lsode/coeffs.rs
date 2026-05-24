//! LSODE method/test coefficient generation (`DCFODE`).
//!
//! Faithful port of `LsodeFirstOrderODEIntegrator::calculate_integration_coefficients`
//! (`lsode_first_order_ode_integrator__support.cc`), which itself de-Fortrans
//! ODEPACK's `DCFODE`. Generates, for **all** orders of the selected method
//! family at once:
//!
//! - `method_coeffs` (ELCO): `[13][12]` — column `nq-1` holds the order-`nq`
//!   integration coefficients `el[0..=nq]` in rows `0..=nq`.
//! - `test_coeffs` (TESCO): `[3][12]` — per-order constants used by the
//!   local-error test and the order-change controller.
//!
//! Pure function of the method family; no integrator state. The index-0
//! throwaway slot in the working `poly_coeff` array is preserved exactly as
//! in the JEOD source (its comment explains the Fortran 1-based carry-over).

use super::config::IntegrationMethod;

/// ELCO: `method_coeffs[coefficient_row][order - 1]`.
pub type MethodCoeffs = [[f64; 12]; 13];
/// TESCO: `test_coeffs[row][order - 1]`.
pub type TestCoeffs = [[f64; 12]; 3];

/// Generate the (ELCO, TESCO) coefficient tables for `method`.
///
/// Mirrors `calculate_integration_coefficients` branch-for-branch.
#[allow(
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    reason = "index arithmetic and recurrence assignments mirror the JEOD/Fortran source line-by-line for auditability against DCFODE"
)]
#[allow(
    clippy::cast_precision_loss,
    reason = "all casts are loop counters nq/ii ≤ 12 (Adams) / ≤ 5 (BDF), exactly representable in f64"
)]
pub fn calculate_integration_coefficients(method: IntegrationMethod) -> (MethodCoeffs, TestCoeffs) {
    let mut method_coeffs: MethodCoeffs = [[0.0; 12]; 13];
    let mut test_coeffs: TestCoeffs = [[0.0; 12]; 3];

    // 13-array with a throwaway index 0 so the polynomial recurrence keeps
    // the source's 1-based indices (see the JEOD NOTE).
    let mut poly_coeff = [0.0_f64; 13];

    match method {
        IntegrationMethod::ImplicitAdamsNonStiff => {
            method_coeffs[0][0] = 1.0;
            method_coeffs[1][0] = 1.0;
            test_coeffs[0][0] = 0.0;
            test_coeffs[1][0] = 2.0;
            test_coeffs[0][1] = 1.0;
            test_coeffs[2][11] = 0.0;
            poly_coeff[0] = 1.0;
            let mut rqfac = 1.0_f64;

            for nq in 2..=12 {
                // p(x) = (x+1)(x+2)...(x+nq-1).
                let rq1fac = rqfac;
                rqfac /= nq as f64;

                // Form coefficients of p(x)*(x+nq-1).
                poly_coeff[nq - 1] = 0.0;
                for ii in (1..=nq - 1).rev() {
                    poly_coeff[ii] = poly_coeff[ii - 1] + (nq - 1) as f64 * poly_coeff[ii];
                }
                poly_coeff[0] = (nq - 1) as f64 * poly_coeff[0];

                // Integrals over [-1, 0] of p(x) and x·p(x).
                let mut pint = poly_coeff[0];
                let mut xpin = poly_coeff[0] / 2.0;
                let mut tsign = 1.0_f64;
                for ii in 2..=nq {
                    tsign = -tsign;
                    pint += tsign * poly_coeff[ii - 1] / ii as f64;
                    xpin += tsign * poly_coeff[ii - 1] / (ii + 1) as f64;
                }

                method_coeffs[0][nq - 1] = pint * rq1fac;
                method_coeffs[1][nq - 1] = 1.0;
                for ii in 2..=nq {
                    method_coeffs[ii][nq - 1] = rq1fac * poly_coeff[ii - 1] / ii as f64;
                }
                let agamq = rqfac * xpin;
                let ragq = 1.0 / agamq;
                test_coeffs[1][nq - 1] = ragq;
                if nq < 12 {
                    test_coeffs[0][nq] = ragq * rqfac / (nq + 1) as f64;
                }
                test_coeffs[2][nq - 2] = ragq;
            }
        }
        IntegrationMethod::ImplicitBackDiffStiff => {
            poly_coeff[0] = 1.0;
            let mut rq1fac = 1.0_f64;
            for nq in 1..=5 {
                let nqp1 = nq + 1;
                poly_coeff[nq] = 0.0;
                for ii in (1..=nq).rev() {
                    poly_coeff[ii] = poly_coeff[ii - 1] + nq as f64 * poly_coeff[ii];
                }
                poly_coeff[0] = nq as f64 * poly_coeff[1];
                for ii in 0..=nq {
                    method_coeffs[ii][nq - 1] = poly_coeff[ii] / poly_coeff[1];
                }
                method_coeffs[1][nq - 1] = 1.0;
                test_coeffs[0][nq - 1] = rq1fac;
                test_coeffs[1][nq - 1] = nqp1 as f64 / method_coeffs[0][nq - 1];
                test_coeffs[2][nq - 1] = (nq + 2) as f64 / method_coeffs[0][nq - 1];
                rq1fac /= nq as f64;
            }
        }
    }

    (method_coeffs, test_coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(
        clippy::float_cmp,
        reason = "coefficient generation is exact rational arithmetic in f64; the known Adams-Moulton values are representable exactly"
    )]
    fn assert_exact(got: f64, want: f64, label: &str) {
        assert_eq!(got, want, "{label}: got {got}, want {want}");
    }

    /// Compare against a known rational to machine precision. Needed for
    /// coefficients like 5/12 and 1/6 that aren't exactly representable:
    /// the recurrence's FP path lands within a couple of ULP of the
    /// literal division, which is the correct value, not a bug.
    fn assert_close(got: f64, want: f64, label: &str) {
        let tol = 1e-15 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{label}: got {got}, want {want} (|Δ| > {tol:.3e})"
        );
    }

    #[test]
    fn adams_low_order_coefficients_match_known_adams_moulton() {
        let (el, _) = calculate_integration_coefficients(IntegrationMethod::ImplicitAdamsNonStiff);
        // Order 1 (backward Euler / AM1): el = [1, 1].
        assert_exact(el[0][0], 1.0, "AM1 el0");
        assert_exact(el[1][0], 1.0, "AM1 el1");
        // Order 2 (trapezoidal / AM2): el = [1/2, 1, 1/2].
        assert_exact(el[0][1], 0.5, "AM2 el0");
        assert_exact(el[1][1], 1.0, "AM2 el1");
        assert_exact(el[2][1], 0.5, "AM2 el2");
        // Order 3 (AM3): el = [5/12, 1, 3/4, 1/6]. 5/12 and 1/6 aren't
        // exactly representable, so compare to machine precision.
        assert_close(el[0][2], 5.0 / 12.0, "AM3 el0");
        assert_exact(el[1][2], 1.0, "AM3 el1");
        assert_exact(el[2][2], 3.0 / 4.0, "AM3 el2");
        assert_close(el[3][2], 1.0 / 6.0, "AM3 el3");
    }

    #[test]
    fn bdf_order1_is_backward_euler() {
        let (el, _) = calculate_integration_coefficients(IntegrationMethod::ImplicitBackDiffStiff);
        // BDF1 = backward Euler: el = [1, 1].
        assert_exact(el[0][0], 1.0, "BDF1 el0");
        assert_exact(el[1][0], 1.0, "BDF1 el1");
    }

    #[test]
    fn adams_el1_row_is_unity_for_all_orders() {
        // el[1][nq-1] == 1 for every order is structural in both families
        // (the predicted-derivative coefficient); a cheap guard that the
        // full table was populated, not just the seeded order-1 column.
        let (el, _) = calculate_integration_coefficients(IntegrationMethod::ImplicitAdamsNonStiff);
        for nq in 1..=12 {
            assert_exact(el[1][nq - 1], 1.0, "Adams el1 row");
        }
    }
}
