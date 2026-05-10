//! Canonical body constants (gravitational parameter, equatorial radius,
//! polar radius, flattening coefficient) for the bodies that astrodyn
//! supports out of the box.
//!
//! These are bare `f64` SI constants — m³/s² for `_MU`, metres for radii,
//! dimensionless for `_FLAT_COEFF` — kept here at the universal-leaf
//! crate so any owner crate (kernel, mission, test) can depend on them
//! without pulling in `astrodyn_planet` or the rest of the physics
//! tree.
//!
//! Higher-level wrappers (`astrodyn_planet::presets::{EARTH, MOON, …}`)
//! compose these constants into typed `PlanetShape` structs; both forms
//! reference the same numbers, so updating the JEOD source mirror means
//! editing one place.
//!
//! Source-line citations live on each constant. All four bodies match
//! [JEOD v5.4.0](https://github.com/nasa/jeod/blob/jeod_v5.4.0)
//! `models/environment/planet/data/src/<body>.cc` plus the corresponding
//! gravity-coefficient file.

// ── Earth ──────────────────────────────────────────────────────────────

/// Earth gravitational parameter (m³/s²) — JEOD `earth_GGM05C.cc:40`.
///
/// Note: GGM05C mu (3.986004415e14) differs from IERS 2010 (3.986004418e14)
/// by 3e6 m³/s². We use the GGM05C value to match JEOD source.
pub const EARTH_MU: f64 = 398_600.441_50e9;

/// Earth equatorial radius (m) — JEOD `earth.cc:37`.
pub const EARTH_R_EQ: f64 = 1000.0 * 6378.137;

/// Earth flattening coefficient — JEOD `earth.cc:36` (1 / 298.257223563).
pub const EARTH_FLAT_COEFF: f64 = 1.0 / 298.257_223_563;

/// Earth polar radius (m), derived as `EARTH_R_EQ * (1 - EARTH_FLAT_COEFF)`.
pub const EARTH_R_POL: f64 = EARTH_R_EQ * (1.0 - EARTH_FLAT_COEFF);

// ── Moon ───────────────────────────────────────────────────────────────

/// Moon gravitational parameter (m³/s²) — JEOD `moon_GRAIL150.cc:60`.
pub const MOON_MU: f64 = 4_902.799_806_931_69e9;

/// Moon equatorial radius (m) — JEOD `moon.cc:53`.
pub const MOON_R_EQ: f64 = 1000.0 * 1738.14;

/// Moon flattening coefficient — JEOD `moon.cc:52`.
pub const MOON_FLAT_COEFF: f64 = 0.00125;

/// Moon polar radius (m), derived as `MOON_R_EQ * (1 - MOON_FLAT_COEFF)`.
pub const MOON_R_POL: f64 = MOON_R_EQ * (1.0 - MOON_FLAT_COEFF);

// ── Sun ────────────────────────────────────────────────────────────────

/// Sun gravitational parameter (m³/s²) — JEOD `sun_spherical.cc:46`.
pub const SUN_MU: f64 = 1.327_124_40e20;

/// Sun equatorial radius (m) — JEOD `sun.cc:38`.
pub const SUN_R_EQ: f64 = 1000.0 * 696_000.0;

/// Sun flattening coefficient — JEOD `sun.cc:41`.
pub const SUN_FLAT_COEFF: f64 = 5.0e-5;

/// Sun polar radius (m), derived as `SUN_R_EQ * (1 - SUN_FLAT_COEFF)`.
pub const SUN_R_POL: f64 = SUN_R_EQ * (1.0 - SUN_FLAT_COEFF);

// ── Mars ───────────────────────────────────────────────────────────────

/// Mars gravitational parameter (m³/s²) — JEOD `mars_MRO110B2.cc:57`.
pub const MARS_MU: f64 = 4.282_837_452_7e13;

/// Mars equatorial radius (m) — JEOD `mars.cc:46`.
pub const MARS_R_EQ: f64 = 1000.0 * 3396.0;

/// Mars flattening coefficient — JEOD `mars.cc:45`.
pub const MARS_FLAT_COEFF: f64 = 0.005186;

/// Mars polar radius (m), derived as `MARS_R_EQ * (1 - MARS_FLAT_COEFF)`.
pub const MARS_R_POL: f64 = MARS_R_EQ * (1.0 - MARS_FLAT_COEFF);
