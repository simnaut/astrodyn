//! High-fidelity gravity-source recipes backed by committed test
//! fixtures **(workspace-internal — not for downstream mission code).**
//!
//! Functions here build [`GravitySourceEntry`] values populated with
//! spherical-harmonics coefficient sets loaded from
//! `test_data/gravity/*.bin` via `jeod_test_data::gravity_fixtures`.
//! Those fixtures live at the workspace root of *this repository* and
//! are not packaged with the crate — calling these recipes from a
//! downstream workspace will panic at runtime when the loader can't
//! find the binaries. The whole `verification` submodule is therefore
//! hidden from rendered rustdoc; the only consumers are `jeod_runner`'s
//! Tier 3 rigs and the in-repo examples that cross-validate against
//! JEOD's reference data.
//!
//! Mission code should use the point-mass building blocks in
//! [`earth`](super::super::earth), [`moon`](super::super::moon),
//! [`mars`](super::super::mars), or supply its own
//! [`SphericalHarmonicsData`](jeod_gravity::SphericalHarmonicsData) if
//! a higher-fidelity field is needed.
//!
//! Fixtures here are regenerable via the `extract_*` binaries under
//! `jeod_test_data` (`cargo run -p jeod_test_data --bin
//! extract_grav_coeffs` for Earth fields, `extract_mars_data` for
//! Moon/Mars/Sun).

use jeod_test_data::gravity_fixtures;

use crate::sources::GravitySourceEntry;
use crate::{EARTH, MARS, MOON};

/// Earth with the GGM05C spherical-harmonics gravity field
/// (degree=order=360).
///
/// Reads `test_data/gravity/ggm05c.bin`, the committed mirror of
/// `models/environment/gravity/data/src/earth_GGM05C.cc` (regenerable
/// via `cargo run -p jeod_test_data --bin extract_grav_coeffs`).
pub fn earth_ggm05c() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&EARTH, gravity_fixtures::load_ggm05c())
}

/// Moon with the LP150Q spherical-harmonics gravity field
/// (degree=order=150).
///
/// Reads `test_data/gravity/moon_lp150q.bin`, the committed mirror of
/// `models/environment/gravity/data/src/moon_LP150Q.cc` (regenerable
/// via `cargo run -p jeod_test_data --bin extract_mars_data`).
pub fn moon_lp150q() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&MOON, gravity_fixtures::load_moon_lp150q())
}

/// Mars with the MRO110B2 spherical-harmonics gravity field
/// (degree=order=110).
///
/// Reads `test_data/gravity/mars_mro110b2.bin`, the committed mirror of
/// `models/environment/gravity/data/src/mars_MRO110B2.cc` (regenerable
/// via `cargo run -p jeod_test_data --bin extract_mars_data`).
pub fn mars_mro110b2() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&MARS, gravity_fixtures::load_mars_mro110b2())
}
