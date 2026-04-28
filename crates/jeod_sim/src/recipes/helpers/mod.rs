//! Tier 3 propagation helpers shared by `recipes` and Tier 3 tests.
//!
//! Phase 8 of #101 hoisted these utilities out of
//! `crates/jeod_runner/tests/sim_test_helpers/mod.rs` so that
//! archetype-B tests (those with bespoke per-step measurement loops:
//! periapsis detection, energy conservation, integrator agreement,
//! attach/detach event scheduling, etc.) can consume them through the
//! same `recipes::*` namespace as their setup.
//!
//! These are **propagation utilities** only — schema-flexible CSV
//! readers (which are reference-data plumbing, not propagation) live
//! in `jeod_test_data::tier3_csv` (the typed Phase 7 catalogue, with
//! `dyncomp_csv` as its current entry). Tests that don't fit a
//! standard scenario but share these mid-loop bookkeeping patterns
//! reach for the corresponding submodule below.
//!
//! Submodules:
//!
//! - [`state_helpers`]: quaternion / matrix angular-error metrics, the
//!   2π-wraparound `angle_diff`, and a `state_from_elements` shortcut
//!   that builds a [`TranslationalState`](crate::TranslationalState)
//!   from raw Keplerian elements (handles e ≥ 1).
//! - [`energy_conservation`]: extract specific orbital energy each
//!   step, accumulate `max |relative error|` against the t = 0 value.
//! - [`periapsis_detection`]: zero-crossing of radial velocity for
//!   Kepler / GR perihelion-advance studies.
//! - [`integrator_agreement`]: propagate the same scenario through two
//!   integrators and report the per-component divergence.
//! - [`force_torque_profiles`]: piecewise step / ramp / sinusoid input
//!   shapes for force–torque response tests.
//! - [`orbinit_case`]: bundle of (orbital elements, tolerance) for
//!   parametric orbital-init roundtrip suites.
//! - [`attach_detach_helpers`]: time-stamped mass-tree event schedule
//!   wrapper for attach / detach tests.
//! - [`custom_csv`]: schema-flexible CSV reader for ad-hoc test
//!   formats that don't yet have a dedicated typed loader in
//!   `jeod_test_data::tier3_csv`.

pub mod attach_detach_helpers;
pub mod custom_csv;
pub mod energy_conservation;
pub mod force_torque_profiles;
pub mod integrator_agreement;
pub mod orbinit_case;
pub mod periapsis_detection;
pub mod state_helpers;

pub use attach_detach_helpers::{AttachDetachEvent, EventSchedule};
pub use custom_csv::read_lines;
pub use energy_conservation::{specific_orbital_energy, KeplerEnergyMonitor};
pub use force_torque_profiles::{step_input, RampInput, SinusoidInput};
pub use integrator_agreement::{integrator_divergence, IntegratorAgreement};
pub use orbinit_case::OrbInitCase;
pub use periapsis_detection::{detect_periapsis_passages, PeriapsisEvent};
pub use state_helpers::{
    angle_diff, angle_diff_restricted, dquat_angle_error, jeodquat_angle_error, max_mat_diff,
    state_from_elements,
};
