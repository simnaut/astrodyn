//! Shared test-only fixtures, parsers, and verification helpers for the
//! `bevy_jeod` workspace.
//!
//! This crate is the single home for code that reads JEOD source files
//! (Modified_data Python, S_define text, gravity coefficient C++ headers,
//! `Leap_Second.dat`, `verif_out.txt`) and Trick CSV/`.bsp` outputs. Every
//! other crate in the workspace consumes those parsed fixtures through this
//! crate so the parsing logic exists in exactly one place. Production code
//! (`jeod_*`, `jeod_sim`, `bevy_jeod`) never depends on this crate — only
//! tests, examples, and the cross-validation report binaries do.
//!
//! ## Top-level modules
//!
//! - [`crossval`] — `CrossvalReport` builder used by Tier 3 trajectory tests
//!   to compute and assert per-component max errors against JEOD CSVs.
//! - [`gravity_verif`] — parser for JEOD's
//!   `grav_geospherical/data/verif_out.txt` (40 acceleration / gradient /
//!   potential test vectors).
//! - [`tier3_csv`] — generic loader for `log_state_ASCII.csv` Trick logs
//!   produced by `verif/SIM_*` runs (time, position, velocity, attitude,
//!   angular velocity columns).
//! - [`dyncomp_csv`] — typed helpers around the `SIM_dyncomp` family of
//!   reference CSVs used by the ISS-orbit Tier 3 tests.
//! - [`apollo_mass_tree`] — fixture wiring for the multi-body Apollo
//!   `MassTree` used by attach/detach staging tests.
//! - [`euler_test`] — six-case rotation-matrix → Euler-angle table lifted
//!   from `euler_derived_state_ut.cc`.
//! - [`leap_second`], [`time_config`] — JEOD `Leap_Second.dat` and
//!   time-scale configuration parsers.
//! - [`mass_data`], [`orbital_data`], [`orbital_init`], [`reference_state`],
//!   [`s_define`], [`gravity_control`] — Modified_data Python and S_define
//!   parsers that turn `trick.attach_units(...)` literals into typed Rust
//!   values.
//!
//! ## Binaries
//!
//! Two helper binaries live in `src/bin/`:
//!
//! - `tier3_report` — runs the full Tier 3 suite, scrapes per-test
//!   tolerance literals from test source, and emits JSON / Markdown
//!   cross-validation reports plus an optional baseline freeze
//!   (`--freeze-baselines`).
//! - `tier3_baseline_diff` — diffs a fresh report against
//!   `test_data/baselines.json` to detect baseline drift on
//!   refactor-only PRs.
//!
//! ## Environment
//!
//! Every parser that resolves JEOD source paths goes through
//! [`jeod_path`], which checks `JEOD_PATH` first and then `JEOD_HOME`
//! (the standard JEOD/Trick convention). [`trick_path`] does the same
//! for `TRICK_HOME`. Tests that need JEOD source data assert (panic)
//! when neither variable is set; they never skip silently.

#![forbid(unsafe_code)]

pub use jeod_quantities::prelude::*;

pub mod apollo_mass_tree;
pub mod atmosphere_verif;
pub mod crossval;
pub mod dyncomp_csv;
pub mod euler_test;
pub mod gravity_control;
pub mod gravity_verif;
pub mod leap_second;
pub mod mass_data;
pub mod orbital_data;
pub mod orbital_init;
pub mod reference_state;
pub mod s_define;
pub mod tier3_csv;
pub mod time_config;

/// Get the JEOD root path from environment variables.
///
/// Checks `JEOD_PATH` first, then `JEOD_HOME` (standard JEOD/Trick convention).
/// Returns a path that may or may not exist — callers should check `.exists()`.
pub fn jeod_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("JEOD_PATH") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("JEOD_HOME") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from("JEOD_PATH_or_JEOD_HOME_not_set")
}

/// Get the Trick root path from the `TRICK_HOME` environment variable.
///
/// Returns a path that may or may not exist — callers should check `.exists()`.
pub fn trick_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("TRICK_HOME") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from("TRICK_HOME_not_set")
}
