//! Committed planetary gravity-coefficient fixtures.
//!
//! Loads spherical-harmonics coefficient sets (and point-mass `mu`
//! values) from the binary fixtures committed under `test_data/gravity/`.
//! These fixtures are produced by parsing JEOD's `.cc` source files into
//! the production [`astrodyn_gravity::coefficients::save_binary`] format —
//! see the `extract_grav_coeffs` and `extract_mars_data` binaries in
//! this crate for the regen step.
//!
//! Every test that previously called
//! `astrodyn_test_data::jeod_cc::load_from_jeod_cc(...)` to read a JEOD
//! checkout at test time should call one of these loaders instead so the
//! test runs with `JEOD_HOME` unset.
//!
//! ## Coverage
//!
//! - **Earth**: [`load_ggm02c`], [`load_ggm05c`], [`load_gemt1`].
//! - **Moon**: [`load_moon_lp150q`], [`load_moon_grail150`] / [`load_moon_grail150_mu`].
//! - **Mars**: [`load_mars_mro110b2`].
//! - **Sun**: [`load_sun_spherical`] / [`load_sun_spherical_mu`]
//!   (point-mass; encoded as a degree-1 zero-coefficient SH so the
//!   uniform binary loader works — only `mu` and `radius` are
//!   physically meaningful).
//!
//! ## Regenerate
//!
//! Earth coefficients (GGM02C / GGM05C / GEMT1):
//!
//! ```bash
//! cargo run -p astrodyn_test_data --bin extract_grav_coeffs
//! ```
//!
//! Mars / Sun / Moon coefficients:
//!
//! ```bash
//! cargo run -p astrodyn_test_data --bin extract_mars_data
//! ```
//!
//! Commit the updated `test_data/gravity/*.bin` and the sidecar
//! `*.json` metadata files together.

use astrodyn_gravity::SphericalHarmonicsData;
use std::path::PathBuf;

/// Resolve a path under `crates/astrodyn_gravity/test_data/gravity/`.
///
/// Walks up from `CARGO_MANIFEST_DIR` until it finds `Cargo.lock`, then joins
/// the gravity-coefficient fixtures committed in the gravity crate.
fn fixture_path(filename: &str) -> PathBuf {
    crate::tier3_csv::workspace_root()
        .join("crates/astrodyn_gravity/test_data/gravity")
        .join(filename)
}

fn load_fixture(label: &str) -> SphericalHarmonicsData {
    let path = fixture_path(&format!("{label}.bin"));
    astrodyn_gravity::coefficients::load_binary(&path).unwrap_or_else(|e| {
        let regen_bin = match label {
            "ggm02c" | "ggm05c" | "gemt1" => "extract_grav_coeffs",
            "mars_mro110b2" | "moon_lp150q" | "moon_grail150" | "sun_spherical" => {
                "extract_mars_data"
            }
            _ => "extract_grav_coeffs",
        };
        panic!(
            "Failed to load gravity fixture {label} from {}: {e:?}.\n\
             Regenerate via: cargo run -p astrodyn_test_data --bin {regen_bin}",
            path.display(),
        )
    })
}

/// Load the GGM02C Earth gravity coefficient set (degree=order=200).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GGM02C.cc`
/// from a JEOD checkout — but reads the committed binary fixture instead, so
/// callers do not need `JEOD_HOME` set.
pub fn load_ggm02c() -> SphericalHarmonicsData {
    load_fixture("ggm02c")
}

/// Load the GGM05C Earth gravity coefficient set (degree=order=360).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GGM05C.cc`
/// from a JEOD checkout — but reads the committed binary fixture instead, so
/// callers do not need `JEOD_HOME` set.
pub fn load_ggm05c() -> SphericalHarmonicsData {
    load_fixture("ggm05c")
}

/// Load the GEM-T1 Earth gravity coefficient set (degree=order=36).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GEMT1.cc`
/// from a JEOD checkout — but reads the committed binary fixture instead.
/// Used by `SIM_7_time_reversal` Tier 3 tests.
pub fn load_gemt1() -> SphericalHarmonicsData {
    load_fixture("gemt1")
}

/// Load Mars MRO110B2 spherical harmonics coefficients (degree=order=110).
///
/// Equivalent to parsing `models/environment/gravity/data/src/mars_MRO110B2.cc`
/// from a JEOD checkout — but reads the committed binary fixture
/// (regenerable via `extract_mars_data`).
pub fn load_mars_mro110b2() -> SphericalHarmonicsData {
    load_fixture("mars_mro110b2")
}

/// Load Moon LP150Q (Lunar Prospector) spherical harmonics coefficients
/// (degree=order=150).
///
/// Equivalent to parsing `models/environment/gravity/data/src/moon_LP150Q.cc`
/// from a JEOD checkout — but reads the committed binary fixture
/// (regenerable via `extract_mars_data`). Used by `SIM_Earth_Moon`.
pub fn load_moon_lp150q() -> SphericalHarmonicsData {
    load_fixture("moon_lp150q")
}

/// Load Moon GRAIL150 spherical harmonics coefficients (degree=order=150).
///
/// Equivalent to parsing `models/environment/gravity/data/src/moon_GRAIL150.cc`
/// from a JEOD checkout — but reads the committed binary fixture
/// (regenerable via `extract_mars_data`). The GRAIL field is the newer
/// JEOD default for the Moon and is used by SIM_dyncomp's third-body
/// Moon source as well as the gravity-gradient torque rigs
/// (`SIM_torque_compare_simple`, `SIM_tide_verif`).
pub fn load_moon_grail150() -> SphericalHarmonicsData {
    load_fixture("moon_grail150")
}

/// Load the Moon GRAIL150 gravitational parameter (mu, m³/s²).
///
/// Convenience for callers that only need `mu` for a third-body
/// point-mass approximation (most Tier 3 dyncomp scenarios).
pub fn load_moon_grail150_mu() -> f64 {
    load_moon_grail150().mu
}

/// Load the full Sun point-mass record (mu and reference radius).
///
/// `sun_spherical.cc` is a JEOD point-mass entry — it has no Cnm/Snm
/// coefficients. The committed fixture encodes it as a degree-1 SH with
/// all-zero coefficients so the production binary loader works
/// uniformly. Only `mu` and `radius` are physically meaningful; do not
/// evaluate this as a non-spherical model.
///
/// Regenerable via `extract_mars_data`.
pub fn load_sun_spherical() -> SphericalHarmonicsData {
    load_fixture("sun_spherical")
}

/// Load the Sun point-mass gravitational parameter (mu, m³/s²).
///
/// Most callers (Mars, Mercury, Earth–Moon, Venus, etc. Tier 3 tests)
/// model the Sun as a point-mass third-body, so only `mu` is needed.
pub fn load_sun_spherical_mu() -> f64 {
    load_sun_spherical().mu
}
