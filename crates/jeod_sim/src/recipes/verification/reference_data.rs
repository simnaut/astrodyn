//! High-fidelity gravity-source recipes backed by committed test
//! fixtures.
//!
//! Functions here build [`GravitySourceEntry`] values populated with
//! spherical-harmonics coefficient sets loaded from
//! `test_data/gravity/*.bin` (regenerable via the `extract_*` binaries
//! under [`jeod_test_data::bin`](../../../../jeod_test_data/index.html)).
//! No JEOD checkout is required at runtime — these recipes work on a
//! fresh clone with `$JEOD_HOME` unset.
//!
//! **Mission code should still prefer the lighter point-mass building
//! blocks** in [`earth`](super::super::earth),
//! [`moon`](super::super::moon), [`mars`](super::super::mars) when
//! mission accuracy doesn't require the SH model. These verification
//! recipes exist to keep examples and Tier 3 rigs that *want*
//! NASA-grade gravity coupled to the same upstream coefficient sets
//! JEOD ships with.

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
