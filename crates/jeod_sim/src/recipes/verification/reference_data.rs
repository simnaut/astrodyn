//! Loaders for JEOD reference data used by Tier 3 verification cases.
//!
//! Functions here read files from a JEOD source checkout
//! (`$JEOD_HOME`) and panic with the exact
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

use std::path::PathBuf;

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
             Set JEOD_HOME to the JEOD source checkout.",
            path.display()
        )
    })
}

fn jeod_grav_data(file: &str) -> PathBuf {
    // Routes through `jeod_test_data::jeod_path()` (the canonical
    // JEOD-source resolver in this workspace) instead of the previous
    // duplicate copy that lived here. Wave 2 of #232 collapsed the two
    // helpers into one; `jeod_test_data::jeod_path()` reads only
    // `$JEOD_HOME` and returns a sentinel path when it isn't set.
    // Downstream callers (e.g. `load_grav_cc`) surface the resulting
    // I/O error from `load_from_jeod_cc`'s `Result` via
    // `unwrap_or_else`, panicking with a "Set JEOD_HOME" diagnostic
    // identical in spirit to the original behaviour.
    jeod_test_data::jeod_path()
        .join("models/environment/gravity/data/src")
        .join(file)
}
