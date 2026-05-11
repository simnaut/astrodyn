//! JEOD cross-validation rigs and Tier 3 simulation-driving infrastructure.
//!
//! Pure parsers and fixture loaders moved out to
//! [`astrodyn_verif_jeod_fixtures`] (a leaf-level crate with no `astrodyn`
//! dependency). They are re-exported below so that
//! `astrodyn_verif_jeod::<module>::…` paths continue to work for upper-tree
//! consumers (`astrodyn_verif_parity`, the `tier3_report` /
//! `tier3_baseline_diff` / `extract_*` binaries, the in-crate Tier 3 tests).
//!
//! What stays here:
//!
//! - [`verification`] — `VerificationCase` framework that drives
//!   `Simulation::step()` end-to-end and asserts against committed JEOD CSVs.
//! - [`run_verification`] — per-sim cross-val runners (`sim_dyncomp`,
//!   `sim_srp`, `sim_planetary`, …) built on top of `verification`.
//! - The committed reference data under `test_data/` and `assets/`.
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

// Re-export every parser/fixture module from the leaf crate so that
// `astrodyn_verif_jeod::tier3_csv::…`, `astrodyn_verif_jeod::crossval::…`,
// etc. keep resolving for in-crate code (`run_verification/`,
// `verification/`, `bin/`, `tests/`) and for upper-tree consumers
// (`astrodyn_verif_parity`, the `tier3_*` binaries).
pub use astrodyn_verif_jeod_fixtures::{
    apollo_mass_tree, apollo_truth, atmosphere_verif, body_init_fixtures, crossval, dyncomp_csv,
    euler_test, gravity_control, jeod_inputs, lvlh_init_data, mass_data, orbital_data,
    orbital_init, reference_state, s_define, tier3_csv, time_config,
};

pub mod run_verification;
pub mod setups;
pub mod verification;

pub use run_verification::VerificationCaseExt;
