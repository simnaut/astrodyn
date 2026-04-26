//! Verification-case scaffolding.
//!
//! [`VerificationCase`] bundles a scenario constructor, a reference
//! CSV path, propagation duration, and per-component tolerances into a
//! single declarative unit. Tier 3 tests in Phase 7/8 collapse to:
//!
//! ```ignore
//! #[test]
//! fn tier3_dyncomp_run2_3dof() {
//!     verification::dyncomp_run2_3dof().run_and_assert();
//! }
//! ```
//!
//! Phase 6 ships only the scaffold: the [`VerificationCase`] /
//! [`Tolerances`] / [`CsvReference`] types and the
//! [`reference_data`] submodule for JEOD-source-dependent loaders
//! (gravity coefficient files, etc.). Concrete `verification::*`
//! constructors live in Phase 7/8.
//!
//! `run_and_assert` itself is implemented by `jeod_runner` via an
//! extension trait, since materializing a [`SimulationBuilder`] into a
//! runtime [`Simulation`] is runner-specific.

pub mod reference_data;

use uom::si::f64::Time;

use crate::SimulationBuilder;

/// A reference-CSV file used by a Tier 3 verification case.
///
/// Phase 6 keeps this as a thin wrapper around a file path; Phase 7
/// will extend it with column-layout descriptors as the Tier 3 wave
/// fills out.
#[derive(Clone, Debug)]
pub struct CsvReference {
    /// Path to the CSV file, relative to the repository's `test_data/`
    /// directory.
    pub path: &'static str,
}

/// Per-component tolerances for trajectory cross-validation.
///
/// Each field corresponds to a `CrossvalReport::assert_*` method —
/// `position_m` per axis, `velocity_m_s` per axis, scalar
/// `quat_angle_rad`, `ang_vel_rad_s` per axis. `extras` lets a Tier 3
/// case attach scenario-specific tolerances (e.g., the GR perihelion-
/// advance arc-second-per-century check on the Mercury case).
#[derive(Clone, Debug)]
pub struct Tolerances {
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
    pub quat_angle_rad: f64,
    pub ang_vel_rad_s: [f64; 3],
    pub extras: &'static [(&'static str, f64)],
}

impl Default for Tolerances {
    /// Default tolerances broad enough to spot a regression while not
    /// rejecting bit-stable runs. Concrete cases tighten these.
    fn default() -> Self {
        Self {
            position_m: [1.0, 1.0, 1.0],
            velocity_m_s: [1.0e-3, 1.0e-3, 1.0e-3],
            quat_angle_rad: 1.0e-6,
            ang_vel_rad_s: [1.0e-9, 1.0e-9, 1.0e-9],
            extras: &[],
        }
    }
}

/// A single Tier 3 verification case.
///
/// Phase 6 ships the type; Phase 7/8 populates `verification::*`
/// constructors that produce one of these per existing Tier 3 test.
/// `run_and_assert` is provided by `jeod_runner::run_verification`
/// because materializing the scenario into a runtime [`Simulation`]
/// is runner-specific.
#[derive(Clone, Debug)]
pub struct VerificationCase {
    /// Unique name used for `target/tier3_crossval/{name}.json` reports.
    pub name: &'static str,
    /// Scenario constructor. The fn pointer stays adapter-neutral so
    /// the runner and (Phase 9) Bevy adapter consume the same scenario.
    pub scenario: fn() -> SimulationBuilder,
    /// Reference CSV produced by the corresponding JEOD verification
    /// simulation.
    pub reference: CsvReference,
    /// Total propagation duration.
    pub duration: Time,
    /// Per-component tolerances for the cross-validation report.
    pub tolerances: Tolerances,
}
