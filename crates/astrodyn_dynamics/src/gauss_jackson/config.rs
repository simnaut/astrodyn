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

    // =======================================================================
    // Negative tests for the IG.04-IG.08 GaussJacksonConfig validators.
    // The error string is the load-bearing signal: `.validate()` joins the
    // per-row messages from `.check()` into a single panic, so pinning the
    // substring proves the right row's branch fired even though the panic
    // surface is shared. Driving each row independently keeps a refactor
    // that neuters one validator (e.g. by replacing `is_multiple_of(2)`
    // with a typo'd predicate) from sliding past a generic "Invalid
    // GaussJacksonConfig" match.
    // =======================================================================

    // JEOD_INV: IG.04 — initial_order must be even and in [2, 14]; an odd
    // value is the cleanest way to drive the predicate's even-integer
    // half of the check without crossing into IG.05's range-violation
    // path.
    #[test]
    #[should_panic(expected = "initial_order")]
    fn ig_04_panics_on_odd_initial_order() {
        let cfg = GaussJacksonConfig {
            initial_order: 5, // odd; valid range upper bound matches IG.05 separately
            ..Default::default()
        };
        cfg.validate();
    }

    // JEOD_INV: IG.05 — final_order must be even, in [initial_order, 14];
    // 16 is past the upper bound and trips the range arm rather than the
    // "less than initial" arm.
    #[test]
    #[should_panic(expected = "final_order")]
    fn ig_05_panics_on_oversize_final_order() {
        let cfg = GaussJacksonConfig {
            final_order: 16,
            ..Default::default()
        };
        cfg.validate();
    }

    // JEOD_INV: IG.06 — ndoubling_steps must be ≤ 20; one past the cap
    // is enough to fire the guard.
    #[test]
    #[should_panic(expected = "ndoubling_steps")]
    fn ig_06_panics_on_oversize_ndoubling_steps() {
        let cfg = GaussJacksonConfig {
            ndoubling_steps: 21,
            ..Default::default()
        };
        cfg.validate();
    }

    // JEOD_INV: IG.07 — relative_tolerance must be finite and in [0, 1];
    // 2.0 exceeds the upper bound, so the range arm of the predicate
    // fires rather than the finiteness arm.
    #[test]
    #[should_panic(expected = "relative_tolerance")]
    fn ig_07_panics_on_out_of_range_relative_tolerance() {
        let cfg = GaussJacksonConfig {
            relative_tolerance: 2.0,
            ..Default::default()
        };
        cfg.validate();
    }

    // JEOD_INV: IG.08 — absolute_tolerance must be finite and ≥ 0; a
    // negative value trips the sign half of the check. (JEOD's error
    // message string mistakenly references `relative_tolerance` — the
    // variable actually checked is `absolute_tolerance`, which our port
    // names correctly.)
    #[test]
    #[should_panic(expected = "absolute_tolerance")]
    fn ig_08_panics_on_negative_absolute_tolerance() {
        let cfg = GaussJacksonConfig {
            absolute_tolerance: -1.0,
            ..Default::default()
        };
        cfg.validate();
    }
}
