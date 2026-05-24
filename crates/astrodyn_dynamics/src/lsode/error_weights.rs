//! LSODE error weights (`DEWSET`) and the weighted-RMS norm (`DVNORM`).
//!
//! Ports `LsodeFirstOrderODEIntegrator::load_ew_values`
//! (`__support.cc`) and `magnitude_of_weighted_array` (`__utility.cc`).
//! Both are pure functions over the current state and tolerances.
//!
//! The local-error control compares `‖accumulated_correction‖_ewt` against
//! 1, where the per-component error weight is `ewt[i] = rtol·|y[i]| + atol`
//! and the norm is the weighted root-mean-square `√(Σ (v[i]·ewt[i])² / n)`.

/// Compute the per-component error weights `ewt[i] = rtol·|y[i]| + atol`
/// into `ewt`, for the common-tolerance form (JEOD `CommonAbsCommonRel`,
/// the only error-control mode LSODE uses here).
///
/// `y` is the 0th Nordsieck column (the current solution); JEOD reads
/// `arrays.history[i][0]`.
///
/// # Panics
/// Panics if `ewt` and `y` differ in length.
pub fn load_error_weights(y: &[f64], rel_tol: f64, abs_tol: f64, ewt: &mut [f64]) {
    assert_eq!(
        y.len(),
        ewt.len(),
        "load_error_weights: y ({}) and ewt ({}) length mismatch",
        y.len(),
        ewt.len()
    );
    for (w, &yi) in ewt.iter_mut().zip(y.iter()) {
        *w = rel_tol * yi.abs() + abs_tol;
    }
}

/// Weighted root-mean-square norm `√(Σ (v[i]·ewt[i])² / n)` (`DVNORM`).
///
/// This is the norm LSODE uses for the local-error test and step-ratio
/// selection — it measures the error relative to the per-component
/// tolerance, so a result < 1 means "within tolerance".
///
/// # Panics
/// Panics if `v` and `ewt` differ in length or are empty (the norm
/// divides by `n`).
pub fn weighted_rms_norm(v: &[f64], ewt: &[f64]) -> f64 {
    assert_eq!(
        v.len(),
        ewt.len(),
        "weighted_rms_norm: v ({}) and ewt ({}) length mismatch",
        v.len(),
        ewt.len()
    );
    assert!(
        !v.is_empty(),
        "weighted_rms_norm: empty vector divides by n=0"
    );
    let mut sum = 0.0;
    for (&vi, &wi) in v.iter().zip(ewt.iter()) {
        let m = vi * wi;
        sum += m * m;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "ODE count n is small (6 for one translational body); exactly representable in f64"
    )]
    let n = v.len() as f64;
    (sum / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_weights_are_rtol_times_abs_y_plus_atol() {
        let y = [10.0, -20.0, 0.0];
        let mut ewt = [0.0; 3];
        load_error_weights(&y, 1e-3, 1e-6, &mut ewt);
        // 1e-3*10 + 1e-6, 1e-3*20 + 1e-6, 1e-3*0 + 1e-6
        assert!((ewt[0] - (1e-2 + 1e-6)).abs() < 1e-18);
        assert!((ewt[1] - (2e-2 + 1e-6)).abs() < 1e-18);
        assert!((ewt[2] - 1e-6).abs() < 1e-21);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the weighted RMS of an all-zero vector is exactly 0.0 (sqrt(0/n))"
    )]
    fn weighted_norm_of_zero_is_zero() {
        let ewt = [1.0, 1.0, 1.0];
        assert_eq!(weighted_rms_norm(&[0.0, 0.0, 0.0], &ewt), 0.0);
    }

    #[test]
    fn weighted_norm_matches_hand_computation() {
        // v·ewt = [2, 0, 4]; sum of squares = 4 + 0 + 16 = 20; /3; sqrt.
        let v = [1.0, 0.0, 2.0];
        let ewt = [2.0, 5.0, 2.0];
        let want = (20.0_f64 / 3.0).sqrt();
        assert!((weighted_rms_norm(&v, &ewt) - want).abs() < 1e-15);
    }

    #[test]
    fn weighted_norm_with_unit_weights_is_plain_rms() {
        let v = [3.0, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5)
        let ewt = [1.0, 1.0];
        assert!((weighted_rms_norm(&v, &ewt) - 12.5_f64.sqrt()).abs() < 1e-15);
    }
}
