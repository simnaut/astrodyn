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

    /// Validate the configuration, panicking on invalid values.
    ///
    /// JEOD: `GaussJacksonConfig::validate_configuration()`.
    pub fn validate(&self) {
        assert!(
            self.initial_order >= 2
                && self.initial_order.is_multiple_of(2)
                && self.initial_order <= 14,
            "initial_order must be even, ≥ 2, and ≤ 14, got {}",
            self.initial_order
        );
        assert!(
            self.final_order >= 2
                && self.final_order.is_multiple_of(2)
                && self.final_order >= self.initial_order
                && self.final_order <= 14,
            "final_order must be even, ≥ 2, ≥ initial_order ({}), and ≤ 14, got {}",
            self.initial_order,
            self.final_order
        );
        // JEOD: ndoubling_steps <= 20 in validate_configuration().
        // Also guard against overflow in `1usize << ndoubling_steps`.
        assert!(
            self.ndoubling_steps < usize::BITS as usize,
            "ndoubling_steps must be < {}, got {}",
            usize::BITS,
            self.ndoubling_steps
        );
        assert!(
            self.relative_tolerance >= 0.0 && self.relative_tolerance <= 1.0,
            "relative_tolerance must be in [0, 1], got {}",
            self.relative_tolerance
        );
        assert!(
            self.absolute_tolerance >= 0.0,
            "absolute_tolerance must be ≥ 0, got {}",
            self.absolute_tolerance
        );
    }
}
