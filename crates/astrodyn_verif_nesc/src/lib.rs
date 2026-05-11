//! NESC GN&C Lunar Check Cases — Tier 3 cross-validation track.
//!
//! Parallel to the `astrodyn_verif_jeod` crate but for the NASA NESC
//! "Independent Orbit Propagation Verification" benchmark
//! (NESC-RP-12-00767). Each `tier3_nesc_*` test under `tests/`
//! propagates [`astrodyn_runner::Simulation`] from NESC-published
//! initial conditions and asserts position / velocity (and attitude,
//! when the case specifies it) against the case's reference
//! trajectory at the published checkpoint cadence.
//!
//! The committed reference CSVs under `test_data/` are produced by
//! the regen binary [`extract_nesc`](https://docs.rs/astrodyn_verif_nesc/latest/astrodyn_verif_nesc/bin/extract_nesc/index.html);
//! see `README.md` for the workflow.
//!
//! ## Cases covered
//!
//! - `cc8` — Lunar Check Case 8 (NRHO): 7-day NRHO with Apollo body,
//!   8×8 GRAIL Moon gravity, Earth + Sun third bodies, DE440 ephemeris.
//!   Translation **and** attitude validation.
//!
//! ## Cross-validation infrastructure
//!
//! [`StateLog`] / [`CrossvalReport`] / [`json_escape`] are re-exported
//! from [`astrodyn_verif_jeod_fixtures::crossval`] so NESC tests share
//! the same JSON-report shape as JEOD Tier 3 tests. The
//! `target/tier3_crossval/` output directory is shared too.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod cc8;

// Re-export the shared cross-validation report types so per-case tests
// don't need a direct dep on the JEOD fixtures crate. Same JSON-report
// shape as the JEOD Tier 3 tests, written under
// `target/tier3_crossval/<test_name>.json`.
pub use astrodyn_verif_jeod_fixtures::crossval::{json_escape, CrossvalReport, StateLog};

/// Resolve a fixture under `crates/astrodyn_verif_nesc/test_data/`.
///
/// Parallel to [`astrodyn_verif_jeod_fixtures::tier3_csv::test_data_path`]
/// but rooted in this crate's `test_data/` directory. NESC reference
/// CSVs live here; cross-cutting fixtures (gravity coefficients,
/// leap-second table, ephemerides) are owned by their owner crates.
pub fn test_data_path(filename: &str) -> std::path::PathBuf {
    astrodyn_verif_jeod_fixtures::tier3_csv::workspace_root()
        .join("crates/astrodyn_verif_nesc/test_data")
        .join(filename)
}
