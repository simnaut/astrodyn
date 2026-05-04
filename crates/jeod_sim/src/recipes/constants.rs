//! Typed physical constants used by recipes.
//!
//! Every constant carries its dimension via `uom`, so mission code that
//! mixes a `GravParam` with a length quantity gets a compile error rather
//! than a unit mismatch at runtime. The gravitational-parameter constants
//! are additionally tagged with the source-body planet phantom — `mu_sun()`
//! returns `GravParam<Sun>`, `mu_ggm05c()` (Earth) returns `GravParam<Earth>`,
//! and so on — so a μ for the wrong central body fails the type check at
//! the consumer rather than passing through silently.
//!
//! # Example
//!
//! ```
//! use jeod_quantities::prelude::*;
//! use jeod_sim::recipes::constants::{mu_ggm05c, r_eq_earth};
//!
//! let mu = mu_ggm05c();         // GravParam<Earth>
//! let r = r_eq_earth();
//!
//! // Circular-orbit speed at the equator: v = sqrt(mu / r).
//! let v_circ_squared = mu.value / r.value;
//! assert!(v_circ_squared > 0.0);
//! ```

use jeod_quantities::dims::GravParam;
use jeod_quantities::ext::F64Ext;
use jeod_quantities::frame::{Earth, Mars, Moon, Sun};
use uom::si::angular_velocity::radian_per_second;
use uom::si::f64::{AngularVelocity, Length};
use uom::si::length::meter;

// ── Earth ───────────────────────────────────────────────────────────────

/// Earth gravitational parameter from `earth_GGM05C.cc` (m³/s²).
pub fn mu_ggm05c() -> GravParam<Earth> {
    3.986_004_415e14_f64.m3_per_s2_for::<Earth>()
}

/// Alias for [`mu_ggm05c`] under the planet-named factory naming
/// convention. `mu_earth()` returns `GravParam<Earth>` from the GGM05C
/// model, matching the per-planet `mu_*()` shape used by `mu_sun()`,
/// `mu_moon()`, `mu_mars()`.
pub fn mu_earth() -> GravParam<Earth> {
    mu_ggm05c()
}

/// Earth equatorial radius (WGS84, m).
pub fn r_eq_earth() -> Length {
    Length::new::<meter>(6_378_137.0)
}

/// Earth polar radius (WGS84, m). `r_eq * (1 - 1/298.257_223_563)`.
pub fn r_pol_earth() -> Length {
    Length::new::<meter>(6_378_137.0 * (1.0 - 1.0 / 298.257_223_563))
}

/// Earth sidereal angular velocity, from JEOD `RNPJ2000_data.cc`.
pub fn omega_earth() -> AngularVelocity {
    AngularVelocity::new::<radian_per_second>(7.292_115_146_706_388e-5)
}

// ── Moon ────────────────────────────────────────────────────────────────

/// Moon gravitational parameter from `moon_GRAIL150.cc` / IAU (m³/s²).
pub fn mu_moon() -> GravParam<Moon> {
    4.902_799_806_931_69e12_f64.m3_per_s2_for::<Moon>()
}

/// Moon mean radius (m).
pub fn r_moon() -> Length {
    Length::new::<meter>(1_738_140.0)
}

// ── Sun ─────────────────────────────────────────────────────────────────

/// Sun gravitational parameter from JEOD `sun_spherical.cc` (m³/s²).
pub fn mu_sun() -> GravParam<Sun> {
    1.327_124_400_18e20_f64.m3_per_s2_for::<Sun>()
}

/// Sun mean radius (m).
pub fn r_sun() -> Length {
    Length::new::<meter>(696_000_000.0)
}

// ── Mars ────────────────────────────────────────────────────────────────

/// Mars gravitational parameter from `mars_MRO110B2.cc` (m³/s²).
pub fn mu_mars() -> GravParam<Mars> {
    4.282_837_452_7e13_f64.m3_per_s2_for::<Mars>()
}

/// Mars sidereal angular velocity (rad/s).
pub fn omega_mars() -> AngularVelocity {
    AngularVelocity::new::<radian_per_second>(7.088_218e-5)
}

/// Mars mean radius (m).
pub fn r_mars() -> Length {
    Length::new::<meter>(3_396_000.0)
}
