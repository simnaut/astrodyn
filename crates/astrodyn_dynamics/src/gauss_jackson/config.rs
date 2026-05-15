//! Gauss-Jackson configuration.
//!
//! Port of JEOD's `GaussJacksonConfig` (`gauss_jackson_config.hh/cc`).

/// Configuration for the Gauss-Jackson integrator.
///
/// JEOD: `GaussJacksonConfig` in `gauss_jackson_config.hh`.
/// All fields are public — this is essentially a struct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussJacksonConfig {
    /// Order immediately after priming. Must be even, ≤ 14.
    /// JEOD default: 4.
    pub initial_order: usize,

    /// Operational order. Must be even, ≥ initial_order, ≤ 14.
    /// JEOD default: 12.
    pub final_order: usize,

    /// Number of step-doubling stages between priming and operational.
    /// JEOD default: `(final_order - initial_order) / 2`.
    pub ndoubling_steps: usize,

    /// Maximum correction iterations during bootstrap editing.
    /// 0 = predict-only, 1 = one correction, ≥2 = iterative correction.
    /// JEOD default: 10.
    pub max_correction_iterations: usize,

    /// Relative convergence tolerance.
    /// JEOD default: 1e-14.
    pub relative_tolerance: f64,

    /// Absolute convergence tolerance.
    /// JEOD default: 1e-10.
    pub absolute_tolerance: f64,

    /// Continue when the corrector or a bootstrap edit fails to converge.
    ///
    /// JEOD's `GaussJacksonIntegrationGroup` logs a warning and continues
    /// when the predictor-corrector fails to converge within
    /// [`max_correction_iterations`](Self::max_correction_iterations), or
    /// when a bootstrap edit accepts a non-converged correction. We diverge
    /// from JEOD by default — non-convergence panics — because a degraded
    /// position silently propagating into the rest of a mission trajectory
    /// is the silent-wrong-physics class of failure the fail-loudly rule
    /// exists to prevent (#485 C1).
    ///
    /// Set this to `true` to restore JEOD-faithful behavior: a `log::warn!`
    /// is emitted and integration continues. Use only when matching a JEOD
    /// reference run exactly is worth the silent-degradation risk (typically
    /// short reproduction runs of JEOD verif sims), and document the choice
    /// at the call site.
    pub allow_non_convergence: bool,
}

impl Default for GaussJacksonConfig {
    /// JEOD constructor default: initial=4, final=12, ndoubling=4. The
    /// `allow_non_convergence` flag defaults to `false` — see the field's
    /// rustdoc for the JEOD-faithful opt-in.
    fn default() -> Self {
        Self {
            initial_order: 4,
            final_order: 12,
            ndoubling_steps: 4, // (12 - 4) / 2
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: 1e-10,
            allow_non_convergence: false,
        }
    }
}

impl GaussJacksonConfig {
    /// Create a config with fixed order, no step-doubling.
    /// `initial_order = final_order = order`, `ndoubling_steps = 0`.
    /// Bootstrap editing still runs (controlled by `max_correction_iterations`)
    /// to refine primed data — only step-doubling is skipped.
    pub fn with_order(order: usize) -> Self {
        Self {
            initial_order: order,
            final_order: order,
            ndoubling_steps: 0,
            ..Default::default()
        }
    }

    /// JEOD standard configuration.
    /// JEOD: `GaussJacksonConfig::standard_configuration()`.
    /// initial=8, final=12, ndoubling=2, tolerances=1e-14.
    ///
    /// `allow_non_convergence` defaults to `false` — see the field's
    /// rustdoc for the JEOD-faithful opt-in semantics.
    pub fn standard() -> Self {
        Self {
            initial_order: 8,
            final_order: 12,
            ndoubling_steps: 2,
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: 1e-14,
            allow_non_convergence: false,
        }
    }

    /// Non-panicking validation. Returns a list of error descriptions.
    ///
    /// Used by `Simulation::validate()` to report all issues at once.
    /// JEOD: `validate_config()` in `gauss_jackson_config.cc`.
    pub fn check(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let is_valid_order = |o: usize| (2..=14).contains(&o) && o.is_multiple_of(2);

        // JEOD_INV: IG.04 — initial_order must be even integer in [2, 14]
        if !is_valid_order(self.initial_order) {
            errors.push(format!(
                "initial_order {} must be even, ≥ 2, ≤ 14",
                self.initial_order
            ));
        }
        // JEOD_INV: IG.05 — final_order must be even integer in [initial_order, 14]
        if !is_valid_order(self.final_order) {
            errors.push(format!(
                "final_order {} must be even, ≥ 2, ≤ 14",
                self.final_order
            ));
        } else if self.final_order < self.initial_order {
            errors.push(format!(
                "final_order {} < initial_order {}",
                self.final_order, self.initial_order
            ));
        }
        // JEOD_INV: IG.06 — ndoubling_steps ≤ 20
        if self.ndoubling_steps > 20 {
            errors.push(format!(
                "ndoubling_steps {} must be ≤ 20",
                self.ndoubling_steps
            ));
        }
        // JEOD_INV: IG.07 — relative_tolerance finite and in [0, 1]
        if !self.relative_tolerance.is_finite() || !(0.0..=1.0).contains(&self.relative_tolerance) {
            errors.push(format!(
                "relative_tolerance {} must be finite and in [0, 1]",
                self.relative_tolerance
            ));
        }
        // JEOD_INV: IG.08 — absolute_tolerance finite and ≥ 0.
        // (JEOD's message mentions relative_tolerance here — that's a known
        // message-string bug in `gauss_jackson_config.cc`; the actual variable
        // checked is absolute_tolerance, which is what we validate.)
        if !self.absolute_tolerance.is_finite() || self.absolute_tolerance < 0.0 {
            errors.push(format!(
                "absolute_tolerance {} must be finite and ≥ 0",
                self.absolute_tolerance
            ));
        }
        // JEOD doesn't validate max_correction_iterations, but cap it to
        // prevent overflow in stage-cap arithmetic (order * iterations).
        if self.max_correction_iterations > 1000 {
            errors.push(format!(
                "max_correction_iterations {} must be ≤ 1000",
                self.max_correction_iterations
            ));
        }
        errors
    }

    /// Validate the configuration, panicking on invalid values.
    ///
    /// JEOD: `GaussJacksonConfig::validate_configuration()`.
    pub fn validate(&self) {
        let errors = self.check();
        assert!(
            errors.is_empty(),
            "Invalid GaussJacksonConfig: {}",
            errors.join("; ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IG.04: `initial_order` must be an even integer in [2, 14]. Odd
    /// orders are rejected because Gauss-Jackson's symmetric corrector
    /// coefficients are tabulated only for even orders.
    #[test]
    #[should_panic(expected = "initial_order 3 must be even")]
    fn ig_04_panics_on_odd_initial_order() {
        // JEOD_INV: IG.04 — initial_order must be even integer in [2, 14]
        GaussJacksonConfig {
            initial_order: 3,
            final_order: 4,
            ndoubling_steps: 0,
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: 1e-10,
            allow_non_convergence: false,
        }
        .validate();
    }

    /// IG.05: `final_order` must be ≥ `initial_order`. A final order
    /// below the initial order would require shrinking the corrector
    /// stencil mid-flight, which Gauss-Jackson is not formulated for.
    #[test]
    #[should_panic(expected = "final_order 2 < initial_order 8")]
    fn ig_05_panics_on_final_below_initial() {
        // JEOD_INV: IG.05 — final_order must be even integer in [initial_order, 14]
        GaussJacksonConfig {
            initial_order: 8,
            final_order: 2,
            ndoubling_steps: 0,
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: 1e-10,
            allow_non_convergence: false,
        }
        .validate();
    }

    /// IG.06: `ndoubling_steps` must be ≤ 20. The doubling cap bounds
    /// the tour count `1 << ndoubling_steps`, which otherwise overflows
    /// the stage-cap arithmetic in the integration kernel.
    #[test]
    #[should_panic(expected = "ndoubling_steps 21 must be ≤ 20")]
    fn ig_06_panics_on_excessive_doubling() {
        // JEOD_INV: IG.06 — ndoubling_steps ≤ 20
        GaussJacksonConfig {
            initial_order: 4,
            final_order: 4,
            ndoubling_steps: 21,
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: 1e-10,
            allow_non_convergence: false,
        }
        .validate();
    }

    /// IG.07: `relative_tolerance` must be finite and in [0, 1]. A
    /// tolerance > 1 is meaningless (the corrector would accept any
    /// finite error), and a non-finite tolerance corrupts convergence
    /// arithmetic.
    #[test]
    #[should_panic(expected = "relative_tolerance")]
    fn ig_07_panics_on_relative_tolerance_above_one() {
        // JEOD_INV: IG.07 — relative_tolerance finite and in [0, 1]
        GaussJacksonConfig {
            initial_order: 4,
            final_order: 4,
            ndoubling_steps: 0,
            max_correction_iterations: 10,
            relative_tolerance: 2.0,
            absolute_tolerance: 1e-10,
            allow_non_convergence: false,
        }
        .validate();
    }

    /// IG.08: `absolute_tolerance` must be finite and ≥ 0. A negative
    /// tolerance flips the convergence comparison and lets the corrector
    /// accept arbitrary errors. (JEOD's diagnostic message names the
    /// `relative_tolerance` field at this site — a known bug in
    /// `gauss_jackson_config.cc` — but the variable actually validated
    /// is `absolute_tolerance`; our diagnostic names the right field.)
    #[test]
    #[should_panic(expected = "absolute_tolerance -1 must be finite and ≥ 0")]
    fn ig_08_panics_on_negative_absolute_tolerance() {
        // JEOD_INV: IG.08 — absolute_tolerance finite and ≥ 0
        GaussJacksonConfig {
            initial_order: 4,
            final_order: 4,
            ndoubling_steps: 0,
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: -1.0,
            allow_non_convergence: false,
        }
        .validate();
    }
}
