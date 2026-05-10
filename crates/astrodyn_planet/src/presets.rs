//! Canonical [`PlanetShape`] constants matching the
//! per-body data files under
//! [`models/environment/planet/data/src/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/planet/data/src/)
//! in JEOD v5.4.0.
//!
//! Each preset combines the shape parameters from `<body>.cc` with the
//! gravitational parameter from the corresponding `<body>_<model>.cc`
//! gravity-coefficient file, so a downstream caller can pull a single
//! `PlanetShape` constant rather than reconstructing the values
//! field-by-field. The numeric values themselves live in
//! [`astrodyn_quantities::body_constants`] (the universal-leaf crate),
//! so kernel-level test code can read the same constants without
//! depending on `astrodyn_planet`.

use astrodyn_quantities::body_constants::{
    EARTH_FLAT_COEFF, EARTH_MU, EARTH_R_EQ, EARTH_R_POL, MARS_FLAT_COEFF, MARS_MU, MARS_R_EQ,
    MARS_R_POL, MOON_FLAT_COEFF, MOON_MU, MOON_R_EQ, MOON_R_POL, SUN_FLAT_COEFF, SUN_MU, SUN_R_EQ,
    SUN_R_POL,
};

use crate::planet::PlanetShape;

/// Earth (WGS84 ellipsoid). Constants from JEOD `earth.cc` and
/// `earth_GGM05C.cc`; numeric values live in
/// [`astrodyn_quantities::body_constants`].
pub const EARTH: PlanetShape = PlanetShape {
    name: "Earth",
    mu: EARTH_MU,
    r_eq: EARTH_R_EQ,
    r_pol: EARTH_R_POL,
    flat_coeff: EARTH_FLAT_COEFF,
};

/// Moon. Constants from JEOD `moon.cc` and `moon_GRAIL150.cc`; numeric
/// values live in [`astrodyn_quantities::body_constants`].
pub const MOON: PlanetShape = PlanetShape {
    name: "Moon",
    mu: MOON_MU,
    r_eq: MOON_R_EQ,
    r_pol: MOON_R_POL,
    flat_coeff: MOON_FLAT_COEFF,
};

/// Sun. Constants from JEOD `sun.cc` and `sun_spherical.cc`; numeric
/// values live in [`astrodyn_quantities::body_constants`].
pub const SUN: PlanetShape = PlanetShape {
    name: "Sun",
    mu: SUN_MU,
    r_eq: SUN_R_EQ,
    r_pol: SUN_R_POL,
    flat_coeff: SUN_FLAT_COEFF,
};

/// Mars. Constants from JEOD `mars.cc` and `mars_MRO110B2.cc`; numeric
/// values live in [`astrodyn_quantities::body_constants`].
pub const MARS: PlanetShape = PlanetShape {
    name: "Mars",
    mu: MARS_MU,
    r_eq: MARS_R_EQ,
    r_pol: MARS_R_POL,
    flat_coeff: MARS_FLAT_COEFF,
};
