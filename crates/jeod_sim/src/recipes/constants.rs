//! Typed physical constants used by recipes.
//!
//! Every constant carries its dimension via `uom`, so mission code that
//! mixes `MU_GGM05C` with a length quantity gets a compile error rather
//! than a unit mismatch at runtime.

use jeod_quantities::dims::GravParam;
use jeod_quantities::ext::F64Ext;
use uom::si::angular_velocity::radian_per_second;
use uom::si::f64::{AngularVelocity, Length};
use uom::si::length::meter;

// ── Earth ───────────────────────────────────────────────────────────────

/// Earth gravitational parameter from `earth_GGM05C.cc` (m³/s²).
pub fn mu_ggm05c() -> GravParam {
    3.986_004_415e14_f64.m3_per_s2()
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
pub fn mu_moon() -> GravParam {
    4.902_799_806_931_69e12_f64.m3_per_s2()
}

/// Moon mean radius (m).
pub fn r_moon() -> Length {
    Length::new::<meter>(1_738_140.0)
}

// ── Sun ─────────────────────────────────────────────────────────────────

/// Sun gravitational parameter from JEOD `sun_spherical.cc` (m³/s²).
pub fn mu_sun() -> GravParam {
    1.327_124_400_18e20_f64.m3_per_s2()
}

/// Sun mean radius (m).
pub fn r_sun() -> Length {
    Length::new::<meter>(696_000_000.0)
}

// ── Mars ────────────────────────────────────────────────────────────────

/// Mars gravitational parameter from `mars_MRO110B2.cc` (m³/s²).
pub fn mu_mars() -> GravParam {
    4.282_837_452_7e13_f64.m3_per_s2()
}

/// Mars sidereal angular velocity (rad/s).
pub fn omega_mars() -> AngularVelocity {
    AngularVelocity::new::<radian_per_second>(7.088_218e-5)
}

/// Mars mean radius (m).
pub fn r_mars() -> Length {
    Length::new::<meter>(3_396_000.0)
}
