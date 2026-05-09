//! JEOD cross-validation parsers, fixtures, and Tier 3 infrastructure.
//!
//! Every JEOD-source parser and Tier 2 / Tier 3 reference loader lives here.
//! The crate ships with the committed JEOD reference fixtures under
//! `test_data/` (trajectory CSVs, Apollo `.out` snapshots,
//! `baselines.{json,md}`, the verbatim NASA mirror under `jeod_inputs/`, the
//! Tier 2 reference vectors under `body_init/` and `jeod_validation/`) and
//! the JEOD-parity ephemeris kernel under `assets/de405.bsp`.
//!
//! ## Top-level modules
//!
//! - [`crossval`] — `CrossvalReport` builder used by Tier 3 trajectory tests
//!   to compute and assert per-component max errors against JEOD CSVs.
//! - [`tier3_csv`] — generic loader for `log_state_ASCII.csv` Trick logs
//!   produced by `verif/SIM_*` runs.
//! - [`dyncomp_csv`] — typed helpers around the `SIM_dyncomp` reference CSVs.
//! - [`apollo_truth`], [`apollo_mass_tree`] — Apollo state / mass-tree fixtures.
//! - [`atmosphere_verif`], [`lvlh_init_data`] — atmosphere and LVLH reference data.
//! - [`euler_test`], [`orbital_data`], [`orbital_init`] — Tier 2 reference vectors.
//! - [`body_init_fixtures`], [`reference_state`], [`mass_data`],
//!   [`time_config`], [`gravity_control`], [`s_define`] — JEOD `Modified_data`
//!   and `S_define` parsers that turn `trick.attach_units(...)` literals into
//!   typed Rust values.
//! - [`jeod_inputs`] — path resolver for the verbatim NASA JEOD source mirror.
//!
//! ## Binaries
//!
//! Helper binaries under `src/bin/`:
//! - `tier3_report` — runs the full Tier 3 suite and emits JSON / Markdown
//!   cross-validation reports.
//! - `tier3_baseline_diff` — diffs a fresh report against `baselines.json`.
//! - `extract_body_init`, `extract_jeod_validation` — regen committed
//!   fixtures from a `$JEOD_HOME` checkout.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod apollo_mass_tree;
pub mod apollo_truth;
pub mod atmosphere_verif;
pub mod body_init_fixtures;
pub mod crossval;
pub mod dyncomp_csv;
pub mod euler_test;
pub mod gravity_control;
pub mod jeod_inputs;
pub mod lvlh_init_data;
pub mod mass_data;
pub mod orbital_data;
pub mod orbital_init;
pub mod reference_state;
pub mod run_verification;
pub mod s_define;
pub mod tier3_csv;
pub mod time_config;
pub mod verification;

pub use run_verification::VerificationCaseExt;
