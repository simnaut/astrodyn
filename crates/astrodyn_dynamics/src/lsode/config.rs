//! LSODE integrator configuration.
//!
//! Ports the load-bearing fields of JEOD's `LsodeControlDataInterface`
//! (`models/utils/integration/lsode/include/lsode_control_data_interface.hh`).
//! LSODE is the Livermore Solver: a variable-order, variable-step Nordsieck
//! multistep method with two families — implicit Adams (non-stiff, orders
//! 1–12) and BDF (stiff, orders 1–5) — selected at construction (JEOD does
//! not switch families at runtime; the `internal_state == -1` switch is
//! disabled in the source, so neither do we).
//!
//! Fail-loudly: [`LsodeConfig::check`] panics on a contradictory
//! configuration (naming the broken invariant and the fix) rather than
//! silently producing wrong physics.

/// Method family. `= 1 / = 2` to mirror JEOD's enum values for traceability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrationMethod {
    /// Variable-step, variable-order implicit Adams-Moulton (non-stiff).
    /// Orders 1–12. JEOD's `RUN_lsode` default.
    #[default]
    ImplicitAdamsNonStiff,
    /// Variable-step, variable-order backward-differentiation (stiff).
    /// Orders 1–5.
    ImplicitBackDiffStiff,
}

impl IntegrationMethod {
    /// Maximum order this family supports (Adams 12, BDF 5) — the
    /// `mord` cap in JEOD's `lsode_control_data_interface`.
    pub fn max_method_order(self) -> usize {
        match self {
            IntegrationMethod::ImplicitAdamsNonStiff => 12,
            IntegrationMethod::ImplicitBackDiffStiff => 5,
        }
    }
}

/// Corrector iteration strategy (JEOD `CorrectorMethod`, MITER).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CorrectorMethod {
    /// Functional (fixed-point) iteration — no Jacobian. The orbital /
    /// non-stiff path; JEOD's default.
    #[default]
    FunctionalIteration,
    /// Modified Newton with an internally generated finite-difference
    /// dense Jacobian (MITER = 2). Stiff path.
    NewtonIterInternalJac,
    /// Modified Jacobi-Newton with an internal diagonal Jacobian
    /// approximation (MITER = 3). Stiff path.
    JacobiNewtonInternalJac,
}

/// LSODE configuration. Defaults match JEOD `RUN_lsode`
/// (`integ_option_int = 140`): non-stiff Adams, functional iteration,
/// max order 12.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LsodeConfig {
    /// Method family (Adams vs BDF).
    pub method: IntegrationMethod,
    /// Corrector iteration strategy.
    pub corrector: CorrectorMethod,
    /// Maximum integration order. Clamped to the method family's cap
    /// (Adams 12 / BDF 5) at validation.
    pub max_order: usize,
    /// Relative error tolerance (RTOL, scalar — JEOD's common-tolerance form).
    pub rel_tolerance: f64,
    /// Absolute error tolerance (ATOL, scalar).
    pub abs_tolerance: f64,
    /// Minimum step size (HMIN). 0 = no minimum.
    pub min_step_size: f64,
    /// Maximum step size (HMAX). 0 = unbounded.
    pub max_step_size: f64,
    /// Initial step size (H0). 0 = auto-select.
    pub initial_step_size: f64,
    /// Max internal steps per `integrate` call (MXSTEP).
    pub max_num_steps: usize,
    /// Max corrector iterations per step (MAXCOR).
    pub max_correction_iters: usize,
    /// Max convergence failures before giving up (MXNCF).
    pub max_num_conv_failure: usize,
    /// Fail loudly on non-convergence (default) vs warn-and-continue.
    /// Mirrors the Gauss-Jackson `allow_non_convergence` policy.
    pub allow_non_convergence: bool,
}

impl Default for LsodeConfig {
    fn default() -> Self {
        // JEOD RUN_lsode: integ_option_int = 140 → Adams / functional /
        // max order 12; MXSTEP=500, MAXCOR=3, MXNCF=10.
        Self {
            method: IntegrationMethod::ImplicitAdamsNonStiff,
            corrector: CorrectorMethod::FunctionalIteration,
            max_order: 12,
            rel_tolerance: 1.0e-9,
            abs_tolerance: 1.0e-9,
            min_step_size: 0.0,
            max_step_size: 0.0,
            initial_step_size: 0.0,
            max_num_steps: 500,
            max_correction_iters: 3,
            max_num_conv_failure: 10,
            allow_non_convergence: false,
        }
    }
}

impl LsodeConfig {
    /// Effective maximum order: the configured `max_order` clamped to the
    /// method family's cap.
    pub fn effective_max_order(&self) -> usize {
        self.max_order.min(self.method.max_method_order()).max(1)
    }

    /// Validate the configuration, panicking with a diagnostic naming the
    /// broken invariant and the fix (fail-loudly per CLAUDE.md).
    ///
    /// # Panics
    ///
    /// - `max_order == 0` or above the family cap.
    /// - non-finite or non-positive tolerances.
    /// - functional iteration paired with the stiff (BDF) family — BDF
    ///   requires a Newton corrector with a Jacobian.
    /// - negative or non-finite step-size bounds.
    pub fn check(&self) {
        // JEOD_INV: IG.17 — LSODE max_order within the family cap (we
        // tighten JEOD's `≥ 0` to `[1, cap]`: Adams 12 / BDF 5).
        assert!(
            self.max_order >= 1 && self.max_order <= self.method.max_method_order(),
            "LsodeConfig.max_order = {} out of range [1, {}] for {:?}: set max_order within \
             the method family's cap (Adams 12 / BDF 5).",
            self.max_order,
            self.method.max_method_order(),
            self.method
        );
        // JEOD_INV: IG.20 — LSODE relative/absolute tolerances finite and
        // ≥ 0 (and not both zero, so the error weight rtol·|y|+atol can be
        // positive — the per-step ewt>0 guard is IG.23).
        assert!(
            self.rel_tolerance.is_finite() && self.rel_tolerance >= 0.0,
            "LsodeConfig.rel_tolerance = {} must be finite and ≥ 0.",
            self.rel_tolerance
        );
        assert!(
            self.abs_tolerance.is_finite() && self.abs_tolerance >= 0.0,
            "LsodeConfig.abs_tolerance = {} must be finite and ≥ 0.",
            self.abs_tolerance
        );
        // ewt = rtol·|y| + atol must be able to be positive: at least one
        // tolerance must be > 0 (JEOD allows atol=0 with rtol>0, e.g. its
        // RUN_lsode uses rtol=2.3e-16, atol=0). The per-step ewt>0 check
        // still fires if a component's |y| is zero under atol=0.
        assert!(
            self.rel_tolerance > 0.0 || self.abs_tolerance > 0.0,
            "LsodeConfig: at least one of rel_tolerance / abs_tolerance must be > 0 \
             (both zero makes the error weight identically zero)."
        );
        assert!(
            self.method == IntegrationMethod::ImplicitAdamsNonStiff
                || self.corrector != CorrectorMethod::FunctionalIteration,
            "LsodeConfig: the stiff BDF family requires a Newton corrector (with a Jacobian), \
             not FunctionalIteration. Choose NewtonIterInternalJac, \
             or use the ImplicitAdamsNonStiff family for functional iteration."
        );
        // The diagonal Jacobi-Newton corrector (ODEPACK MITER=3) is not yet
        // ported. Reject it here so the failure is deterministic at
        // configuration time, rather than panicking mid-`dstode_step` after
        // integration work has already been done. Fail-loudly per CLAUDE.md.
        assert!(
            self.corrector != CorrectorMethod::JacobiNewtonInternalJac,
            "LsodeConfig: the diagonal Jacobi-Newton corrector (MITER=3, \
             JacobiNewtonInternalJac) is not yet ported. Use NewtonIterInternalJac (dense \
             finite-difference Newton) for the stiff BDF family, or the ImplicitAdamsNonStiff \
             family with FunctionalIteration."
        );
        for (name, v) in [
            ("min_step_size", self.min_step_size),
            ("max_step_size", self.max_step_size),
            ("initial_step_size", self.initial_step_size),
        ] {
            assert!(
                v.is_finite() && v >= 0.0,
                "LsodeConfig.{name} = {v} must be finite and ≥ 0 (0 means unset)."
            );
        }
        assert!(
            self.max_num_steps >= 1 && self.max_correction_iters >= 1,
            "LsodeConfig: max_num_steps ({}) and max_correction_iters ({}) must be ≥ 1.",
            self.max_num_steps,
            self.max_correction_iters
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // JEOD_INV: IG.17 — max_order out of the family cap is rejected.
    #[test]
    #[should_panic(expected = "max_order")]
    fn ig_17_rejects_out_of_range_order() {
        LsodeConfig {
            max_order: 0,
            ..LsodeConfig::default()
        }
        .check();
    }

    // JEOD_INV: IG.20 — tolerances that cannot yield a positive error
    // weight (both rtol and atol zero) are rejected.
    #[test]
    #[should_panic(expected = "at least one")]
    fn ig_20_rejects_both_tolerances_zero() {
        LsodeConfig {
            rel_tolerance: 0.0,
            abs_tolerance: 0.0,
            ..LsodeConfig::default()
        }
        .check();
    }

    #[test]
    fn run_lsode_config_validates() {
        // JEOD RUN_lsode: rtol=2.3e-16, atol=0 — atol=0 is allowed when
        // rtol>0 (regression guard for the IG.20 relaxation).
        LsodeConfig::default().check();
        LsodeConfig {
            rel_tolerance: 2.3e-16,
            abs_tolerance: 0.0,
            ..LsodeConfig::default()
        }
        .check();
    }
}
