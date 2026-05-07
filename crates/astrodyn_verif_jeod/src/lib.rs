//! Tier 3 cross-validation against NASA JEOD reference trajectories.
//!
//! This crate is a test-only home: every meaningful symbol lives under
//! `tests/` as integration tests that drive the full `astrodyn_runner`
//! simulation pipeline and assert against committed JEOD reference CSVs.
//! Per-test fixtures (verbatim NASA JEOD source mirror) live under
//! `crates/astrodyn_verif_jeod/test_data/jeod_inputs/`; the reference CSVs
//! are at workspace-root `test_data/`, resolved via
//! `astrodyn_test_data::tier3_csv::test_data_path`.
