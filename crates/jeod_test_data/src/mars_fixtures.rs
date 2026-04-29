//! Mars, Sun, and Moon gravity fixtures committed under `test_data/gravity/`.
//!
//! Tier 3 tests (e.g. `tier3_sim_mars_orbit`, `tier3_simulation_earth_moon_clem`)
//! read pre-extracted binary coefficient blobs and JSON metadata sidecars
//! instead of parsing JEOD's `.cc` source files at test time. This decouples
//! the test suite from `$JEOD_HOME` / `$JEOD_PATH`, so unit and Tier 3 tests
//! run in CI without a JEOD checkout.
//!
//! The fixtures themselves are produced by the `extract_mars_data` binary
//! from a JEOD source tree (parser lives in [`crate::jeod_cc`]). Despite the
//! binary's historical name, it now also extracts the Moon LP150Q
//! coefficients used by `SIM_Earth_Moon`. Regen after a JEOD upgrade with:
//!
//! ```bash
//! cargo run -p jeod_test_data --bin extract_mars_data
//! ```
//!
//! ## Files
//!
//! - `test_data/gravity/mars_mro110b2.bin` — Mars MRO110B2 spherical
//!   harmonics (degree=order=110) in the production
//!   [`jeod_gravity::coefficients`] binary format.
//! - `test_data/gravity/mars_mro110b2.json` — sidecar with source path,
//!   JEOD git rev, mu, radius, degree, order, tide-free flags.
//! - `test_data/gravity/sun_spherical.bin` — Sun point-mass encoded as a
//!   degree=1 SH with all-zero coefficients (only `mu` and `radius` are
//!   physically meaningful).
//! - `test_data/gravity/sun_spherical.json` — Sun metadata sidecar.
//! - `test_data/gravity/moon_lp150q.bin` — Moon Lunar-Prospector LP150Q
//!   spherical harmonics (degree=order=150) for `SIM_Earth_Moon`.
//! - `test_data/gravity/moon_lp150q.json` — Moon metadata sidecar.

use std::path::{Path, PathBuf};

use jeod_gravity::coefficients::load_binary;
use jeod_gravity::SphericalHarmonicsData;

const REGEN_HINT: &str = "Regenerate with: cargo run -p jeod_test_data --bin extract_mars_data";

/// Workspace-relative path to a fixture under `test_data/gravity/`.
fn fixture_path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .expect("workspace root: CARGO_MANIFEST_DIR has at least two ancestors")
        .join("test_data")
        .join("gravity")
        .join(file)
}

/// Load Mars MRO110B2 spherical harmonics coefficients (degree=order=110).
///
/// Reads `test_data/gravity/mars_mro110b2.bin`, which is produced from
/// JEOD's `models/environment/gravity/data/src/mars_MRO110B2.cc` by the
/// `extract_mars_data` binary.
///
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// corrupt; the message includes the regen command.
pub fn load_mars_mro110b2() -> SphericalHarmonicsData {
    let path = fixture_path("mars_mro110b2.bin");
    load_binary(&path).unwrap_or_else(|err| {
        panic!(
            "Mars MRO110B2 fixture missing or unreadable at {}: {err:?}. {REGEN_HINT}",
            path.display(),
        );
    })
}

/// Load the Sun point-mass gravitational parameter (mu, m³/s²).
///
/// Reads `test_data/gravity/sun_spherical.bin` and returns just `mu`.
/// Most callers (Mars, Mercury, Earth-Moon, Venus, etc. Tier 3 tests)
/// model the Sun as a point-mass third-body, so only `mu` is needed.
///
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// corrupt; the message includes the regen command.
pub fn load_sun_spherical_mu() -> f64 {
    load_sun_spherical().mu
}

/// Load the full Sun point-mass record (mu and reference radius).
///
/// `sun_spherical.cc` is a JEOD point-mass entry — it has no Cnm/Snm
/// coefficients. The committed fixture encodes it as a degree=1 SH with
/// all-zero coefficients so the production binary loader works
/// uniformly. Only `mu` and `radius` are physically meaningful; do not
/// evaluate this as a non-spherical model.
///
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// corrupt; the message includes the regen command.
pub fn load_sun_spherical() -> SphericalHarmonicsData {
    let path = fixture_path("sun_spherical.bin");
    load_binary(&path).unwrap_or_else(|err| {
        panic!(
            "Sun spherical fixture missing or unreadable at {}: {err:?}. {REGEN_HINT}",
            path.display(),
        );
    })
}

/// Load Moon LP150Q (Lunar Prospector) spherical harmonics coefficients
/// (degree=order=150).
///
/// Reads `test_data/gravity/moon_lp150q.bin`, which is produced from
/// JEOD's `models/environment/gravity/data/src/moon_LP150Q.cc` by the
/// `extract_mars_data` binary. Used by `SIM_Earth_Moon RUN_clem` and any
/// other Moon-centred non-spherical-gravity Tier 3 test.
///
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// corrupt; the message includes the regen command.
pub fn load_moon_lp150q() -> SphericalHarmonicsData {
    let path = fixture_path("moon_lp150q.bin");
    load_binary(&path).unwrap_or_else(|err| {
        panic!(
            "Moon LP150Q fixture missing or unreadable at {}: {err:?}. {REGEN_HINT}",
            path.display(),
        );
    })
}
