//! Committed planetary gravity-coefficient fixtures.
//!
//! Loads spherical-harmonics coefficient sets (and point-mass `mu`
//! values) from the binary fixtures committed under `test_data/gravity/`.
//! The bytes are embedded at compile time via the [`crate::data`] module,
//! so these loaders work identically inside the workspace and from a
//! published `.crate` — no filesystem lookups, no `JEOD_HOME`, no
//! `CARGO_MANIFEST_DIR` resolution at runtime.
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
//! cargo run -p astrodyn_gravity --bin extract_grav_coeffs
//! ```
//!
//! Mars / Sun / Moon coefficients:
//!
//! ```bash
//! cargo run -p astrodyn_gravity --bin extract_mars_data
//! ```
//!
//! Commit the updated `test_data/gravity/*.bin` and the sidecar
//! `*.json` metadata files together — `include_bytes!` re-reads them at
//! every compile.

use crate::SphericalHarmonicsData;

fn load_embedded(label: &str, bytes: &[u8]) -> SphericalHarmonicsData {
    crate::coefficients::load_binary_from_bytes(bytes).unwrap_or_else(|e| {
        let regen_bin = match label {
            "ggm02c" | "ggm05c" | "gemt1" => "extract_grav_coeffs",
            _ => "extract_mars_data",
        };
        panic!(
            "Failed to decode embedded gravity fixture {label}: {e:?}.\n\
             The committed binary is corrupt. Regenerate via: \
             cargo run -p astrodyn_gravity --bin {regen_bin}"
        )
    })
}

/// Load the GGM02C Earth gravity coefficient set (degree=order=200).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GGM02C.cc`
/// from a JEOD checkout, but reads the embedded binary fixture instead, so
/// callers do not need `JEOD_HOME` set.
pub fn load_ggm02c() -> SphericalHarmonicsData {
    load_embedded("ggm02c", crate::data::GGM02C_BIN)
}

/// Load the GGM05C Earth gravity coefficient set (degree=order=360).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GGM05C.cc`
/// from a JEOD checkout, but reads the embedded binary fixture instead, so
/// callers do not need `JEOD_HOME` set.
pub fn load_ggm05c() -> SphericalHarmonicsData {
    load_embedded("ggm05c", crate::data::GGM05C_BIN)
}

/// Load the GEM-T1 Earth gravity coefficient set (degree=order=36).
///
/// Equivalent to parsing `models/environment/gravity/data/src/earth_GEMT1.cc`
/// from a JEOD checkout, but reads the embedded binary fixture instead.
/// Used by `SIM_7_time_reversal` Tier 3 tests.
pub fn load_gemt1() -> SphericalHarmonicsData {
    load_embedded("gemt1", crate::data::GEMT1_BIN)
}

/// Load Mars MRO110B2 spherical harmonics coefficients (degree=order=110).
///
/// Equivalent to parsing `models/environment/gravity/data/src/mars_MRO110B2.cc`
/// from a JEOD checkout, but reads the embedded binary fixture
/// (regenerable via `extract_mars_data`).
pub fn load_mars_mro110b2() -> SphericalHarmonicsData {
    load_embedded("mars_mro110b2", crate::data::MARS_MRO110B2_BIN)
}

/// Load Moon LP150Q (Lunar Prospector) spherical harmonics coefficients
/// (degree=order=150).
///
/// Equivalent to parsing `models/environment/gravity/data/src/moon_LP150Q.cc`
/// from a JEOD checkout, but reads the embedded binary fixture
/// (regenerable via `extract_mars_data`). Used by `SIM_Earth_Moon`.
pub fn load_moon_lp150q() -> SphericalHarmonicsData {
    load_embedded("moon_lp150q", crate::data::MOON_LP150Q_BIN)
}

/// Load Moon GRAIL150 spherical harmonics coefficients (degree=order=150).
///
/// Equivalent to parsing `models/environment/gravity/data/src/moon_GRAIL150.cc`
/// from a JEOD checkout, but reads the embedded binary fixture
/// (regenerable via `extract_mars_data`). The GRAIL field is the newer
/// JEOD default for the Moon and is used by SIM_dyncomp's third-body
/// Moon source as well as the gravity-gradient torque rigs
/// (`SIM_torque_compare_simple`, `SIM_tide_verif`).
pub fn load_moon_grail150() -> SphericalHarmonicsData {
    load_embedded("moon_grail150", crate::data::MOON_GRAIL150_BIN)
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
/// coefficients. The embedded fixture encodes it as a degree-1 SH with
/// all-zero coefficients so the production binary loader works
/// uniformly. Only `mu` and `radius` are physically meaningful; do not
/// evaluate this as a non-spherical model.
///
/// Regenerable via `extract_mars_data`.
pub fn load_sun_spherical() -> SphericalHarmonicsData {
    load_embedded("sun_spherical", crate::data::SUN_SPHERICAL_BIN)
}

/// Load the Sun point-mass gravitational parameter (mu, m³/s²).
///
/// Most callers (Mars, Mercury, Earth–Moon, Venus, etc. Tier 3 tests)
/// model the Sun as a point-mass third-body, so only `mu` is needed.
pub fn load_sun_spherical_mu() -> f64 {
    load_sun_spherical().mu
}
