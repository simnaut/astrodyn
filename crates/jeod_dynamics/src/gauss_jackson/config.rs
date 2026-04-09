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
}

impl Default for GaussJacksonConfig {
    /// JEOD constructor default: initial=4, final=12, ndoubling=4.
    fn default() -> Self {
        Self {
            initial_order: 4,
            final_order: 12,
            ndoubling_steps: 4, // (12 - 4) / 2
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: 1e-10,
        }
    }
}

impl GaussJacksonConfig {
    /// Create a config with fixed order (no bootstrap).
    /// `initial_order = final_order = order`, `ndoubling_steps = 0`.
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
    pub fn standard() -> Self {
        Self {
            initial_order: 8,
            final_order: 12,
            ndoubling_steps: 2,
            max_correction_iterations: 10,
            relative_tolerance: 1e-14,
            absolute_tolerance: 1e-14,
        }
    }

    /// Non-panicking validation. Returns a list of error descriptions.
    ///
    /// Used by `Simulation::validate()` to report all issues at once.
    /// JEOD: `validate_config()` in `gauss_jackson_config.cc`.
    pub fn check(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let is_valid_order = |o: usize| (2..=14).contains(&o) && o.is_multiple_of(2);

        if !is_valid_order(self.initial_order) {
            errors.push(format!(
                "initial_order {} must be even, ≥ 2, ≤ 14",
                self.initial_order
            ));
        }
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
        if self.ndoubling_steps > 20 {
            errors.push(format!(
                "ndoubling_steps {} must be ≤ 20",
                self.ndoubling_steps
            ));
        }
        if !(0.0..=1.0).contains(&self.relative_tolerance) {
            errors.push(format!(
                "relative_tolerance {} must be in [0, 1]",
                self.relative_tolerance
            ));
        }
        if self.absolute_tolerance < 0.0 {
            errors.push(format!(
                "absolute_tolerance {} must be ≥ 0",
                self.absolute_tolerance
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
