//! Loaders for JEOD reference data used by Tier 3 verification cases.
//!
//! Functions here read files from a JEOD source checkout
//! (`$JEOD_HOME` or `$JEOD_PATH`) and panic with the exact
//! environment / file path if the data is missing — consistent with
//! `feedback_no_graceful_skip.md`. Tests must never silently skip.
//!
//! **Mission code should not call these.** The `bevy_jeod` Rust port
//! is meant to function independently of JEOD source. Use the
//! point-mass building blocks in [`earth`](super::super::earth),
//! [`moon`](super::super::moon), [`mars`](super::super::mars) for
//! mission scenarios. Restoring high-fidelity gravity / ephemeris /
//! rotation data as standalone Rust assets is tracked as a separate
//! follow-up issue.

use std::path::{Path, PathBuf};

use jeod_gravity::SphericalHarmonicsData;
use jeod_test_data::jeod_cc::load_from_jeod_cc;

use crate::sources::GravitySourceEntry;
use crate::{EARTH, MARS, MOON};

/// Earth with the GGM05C spherical-harmonics gravity field.
///
/// Loads `models/environment/gravity/data/src/earth_GGM05C.cc` from
/// `$JEOD_HOME`. Mission code selects the per-vehicle degree/order via
/// [`GravityControl::new_nonspherical`](jeod_gravity::GravityControl::new_nonspherical).
pub fn earth_ggm05c() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&EARTH, load_grav_cc("earth_GGM05C.cc"))
}

/// Moon with the LP150Q spherical-harmonics gravity field.
pub fn moon_lp150q() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&MOON, load_grav_cc("moon_LP150Q.cc"))
}

/// Mars with the MRO110B2 spherical-harmonics gravity field.
pub fn mars_mro110b2() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&MARS, load_grav_cc("mars_MRO110B2.cc"))
}

fn load_grav_cc(file: &str) -> SphericalHarmonicsData {
    let path = jeod_grav_data(file);
    load_from_jeod_cc(&path).unwrap_or_else(|e| {
        panic!(
            "{file}: failed to load {} ({e}). \
             Set JEOD_HOME or JEOD_PATH to the JEOD source checkout.",
            path.display()
        )
    })
}

fn jeod_grav_data(file: &str) -> PathBuf {
    jeod_path()
        .join("models/environment/gravity/data/src")
        .join(file)
}

/// Resolve `$JEOD_HOME` or `$JEOD_PATH`. Panics with a helpful message
/// if neither is set — verification cases require JEOD source data and
/// must not silently skip.
pub fn jeod_path() -> PathBuf {
    if let Ok(p) = std::env::var("JEOD_HOME") {
        Path::new(&p).to_path_buf()
    } else if let Ok(p) = std::env::var("JEOD_PATH") {
        Path::new(&p).to_path_buf()
    } else {
        panic!(
            "JEOD_HOME / JEOD_PATH not set. Verification reference loaders \
             require a JEOD source checkout. Clone https://github.com/nasa/jeod \
             alongside this repo and set JEOD_HOME=../jeod (or copy \
             .cargo/config.toml.example to .cargo/config.toml)."
        )
    }
}
