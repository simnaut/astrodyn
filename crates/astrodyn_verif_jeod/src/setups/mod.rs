//! Canonical scenario constructors shared by Tier 3 tests, examples,
//! and the `tier3_perf_runner` binary.
//!
//! Each scenario lives in its own file and exposes a builder-returning
//! function whose only knobs are the parameters that legitimately differ
//! between callers (e.g. timestep, initial-condition source). Everything
//! else — source composition, gravity model, ephemeris wiring, vehicle
//! configuration — is fixed at the canonical JEOD-reference values so
//! that the test, the example, and the perf-runner cannot drift out of
//! sync.
//!
//! Issue #447 introduces this module to deduplicate the Earth–Moon
//! Clementine scenario. Other Tier 3 scenarios (`dyncomp_run2`,
//! `apollo_trajectory`, …) move here as the perf-runner adds them to
//! its dispatcher.

pub mod earth_moon_clem;
