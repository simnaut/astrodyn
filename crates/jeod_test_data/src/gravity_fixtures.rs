//! Committed gravity-coefficient fixtures.
//!
//! Loads spherical-harmonics coefficient sets from the binary fixtures
//! committed under `test_data/gravity/`. These fixtures are produced by
//! parsing JEOD's `.cc` source files into the production
//! [`jeod_gravity::coefficients::save_binary`] format — see the
//! `extract_grav_coeffs` binary in this crate for the regen step.
//!
//! Every test that previously called
//! `jeod_test_data::jeod_cc::load_from_jeod_cc(...)` to read a JEOD
//! checkout at test time should call one of these loaders instead so the
//! test runs with `JEOD_HOME` / `JEOD_PATH` unset.
//!
//! # Regenerate
//!
//! ```bash
//! cargo run -p jeod_test_data --bin extract_grav_coeffs
//! ```
//!
//! Commit the updated `test_data/gravity/{ggm02c,ggm05c}.bin` and the
//! sidecar `*.json` metadata files together.

use jeod_gravity::SphericalHarmonicsData;
use std::path::PathBuf;

/// Resolve a path under the workspace `test_data/gravity/` directory.
///
/// Walks up from `CARGO_MANIFEST_DIR` until it finds `Cargo.lock`, then joins
/// `test_data/gravity/<filename>`. Mirrors the lookup used by
/// [`super::tier3_csv::test_data_path`] so resolution is consistent regardless
/// of whether tests run from a single-crate or workspace root.
fn fixture_path(filename: &str) -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            break;
        }
        if !dir.pop() {
            return PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/gravity")
                .join(filename);
        }
    }
    dir.join("test_data").join("gravity").join(filename)
}

fn load_fixture(label: &str) -> SphericalHarmonicsData {
    let path = fixture_path(&format!("{label}.bin"));
    jeod_gravity::coefficients::load_binary(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to load gravity fixture {label} from {}: {e:?}.\n\
             Regenerate via: cargo run -p jeod_test_data --bin extract_grav_coeffs",
            path.display(),
        )
    })
}

/// Load the GGM02C Earth gravity coefficient set (degree=order=160).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GGM02C.cc`
/// from a JEOD checkout — but reads the committed binary fixture instead, so
/// callers do not need `JEOD_HOME` / `JEOD_PATH` set.
pub fn load_ggm02c() -> SphericalHarmonicsData {
    load_fixture("ggm02c")
}

/// Load the GGM05C Earth gravity coefficient set (degree=order=360).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GGM05C.cc`
/// from a JEOD checkout — but reads the committed binary fixture instead, so
/// callers do not need `JEOD_HOME` / `JEOD_PATH` set.
pub fn load_ggm05c() -> SphericalHarmonicsData {
    load_fixture("ggm05c")
}
