//! Deprecated thin shims over the mission-facing recipes.
//!
//! Historically this module hosted the only path to high-fidelity SH
//! gravity sources, since the `.cc` coefficient files were not
//! reproducible without a JEOD checkout. After issue #144, the recipes
//! were promoted to the mission-facing
//! [`astrodyn::recipes::{earth, moon, mars}`] modules, backed by
//! `include_bytes!` in `astrodyn_gravity` so they ship in the published
//! `.crate`.
//!
//! Each function here forwards to the new home; new call sites should
//! call the recipe directly.

use astrodyn::GravitySourceEntry;

/// Earth with the GGM05C spherical-harmonics gravity field
/// (degree=order=360).
#[deprecated(
    since = "0.2.0",
    note = "use `astrodyn::recipes::earth::ggm05c` instead"
)]
pub fn earth_ggm05c() -> GravitySourceEntry {
    astrodyn::recipes::earth::ggm05c()
}

/// Moon with the LP150Q spherical-harmonics gravity field
/// (degree=order=150).
#[deprecated(
    since = "0.2.0",
    note = "use `astrodyn::recipes::moon::lp150q` instead"
)]
pub fn moon_lp150q() -> GravitySourceEntry {
    astrodyn::recipes::moon::lp150q()
}

/// Mars with the MRO110B2 spherical-harmonics gravity field
/// (degree=order=110).
#[deprecated(
    since = "0.2.0",
    note = "use `astrodyn::recipes::mars::mro110b2` instead"
)]
pub fn mars_mro110b2() -> GravitySourceEntry {
    astrodyn::recipes::mars::mro110b2()
}
