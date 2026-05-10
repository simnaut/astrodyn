//! Pure parsers and fixture loaders for JEOD reference data.
//!
//! This crate sits at the leaf level of the workspace alongside
//! `astrodyn_quantities` — it has **zero dependency on the `astrodyn`
//! pipeline crate or `astrodyn_runner`**, so any owner crate
//! (`astrodyn_math`, `astrodyn_dynamics`, `astrodyn_atmosphere`, …) can
//! pull it in as a dev-dep without dragging the whole physics tree into
//! its test build.
//!
//! The simulation-driving rigs (`run_verification`, `verification`,
//! `tier3_report`, `tier3_baseline_diff`) live in
//! [`astrodyn_verif_jeod`](../astrodyn_verif_jeod/index.html), which
//! re-exports everything from this crate so existing
//! `astrodyn_verif_jeod::<module>::…` imports keep working from the
//! upper-tree consumers (`astrodyn_verif_parity`, the binaries).
//!
//! Reference data still lives next door at
//! `crates/astrodyn_verif_jeod/test_data/` and `…/assets/`; the path
//! resolver in [`tier3_csv::test_data_path`] walks up to the workspace
//! root and points at that fixed location, so both crates resolve the
//! same files.
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
pub mod s_define;
pub mod tier3_csv;
pub mod time_config;
