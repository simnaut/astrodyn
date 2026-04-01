use crate::planet::PlanetShape;

/// Earth (WGS84 ellipsoid).
///
/// Constants from JEOD `planet/data/src/earth.cc`:
/// - r_eq = 6378.137 km, flat_inv = 298.257223563
///
/// Gravitational parameter from JEOD `earth_GGM05C.cc`:
/// - mu = 398600.44150e9 m^3/s^2
///
/// Note: GGM05C mu (3.986004415e14) differs from IERS 2010 (3.986004418e14)
/// by 3e6 m^3/s^2. We use the GGM05C value to match JEOD source.
pub const EARTH: PlanetShape = PlanetShape {
    name: "Earth",
    mu: 398_600.441_50e9,    // JEOD earth_GGM05C.cc:40
    r_eq: 1000.0 * 6378.137, // JEOD earth.cc:37
    r_pol: 1000.0 * 6378.137 * (1.0 - 1.0 / 298.257_223_563), // JEOD: r_eq * (1 - flat_coeff)
    flat_coeff: 1.0 / 298.257_223_563, // JEOD earth.cc:36
};

/// Moon.
///
/// Constants from JEOD `planet/data/src/moon.cc`:
/// - r_eq = 1738.14 km, flat_coeff = 0.00125
///
/// Gravitational parameter from JEOD `moon_GRAIL150.cc`:
/// - mu = 4902.79980693169e9 m^3/s^2
pub const MOON: PlanetShape = PlanetShape {
    name: "Moon",
    mu: 4902.79980693169e9,                    // JEOD moon_GRAIL150.cc:60
    r_eq: 1000.0 * 1738.14,                    // JEOD moon.cc:53
    r_pol: 1000.0 * 1738.14 * (1.0 - 0.00125), // JEOD: r_eq * (1 - flat_coeff)
    flat_coeff: 0.00125,                       // JEOD moon.cc:52
};

/// Sun.
///
/// Constants from JEOD `planet/data/src/sun.cc`:
/// - r_eq = 696000 km, flat_coeff = 5e-5
///
/// Gravitational parameter from JEOD `sun_spherical.cc`:
/// - mu = 1.32712440E+20 m^3/s^2
pub const SUN: PlanetShape = PlanetShape {
    name: "Sun",
    mu: 1.327_124_40e20,                      // JEOD sun_spherical.cc:46
    r_eq: 1000.0 * 696_000.0,                 // JEOD sun.cc:38
    r_pol: 1000.0 * 696_000.0 * (1.0 - 5e-5), // JEOD: r_eq * (1 - flat_coeff)
    flat_coeff: 5.0e-5,                       // JEOD sun.cc:41
};

/// Mars.
///
/// Constants from JEOD `planet/data/src/mars.cc`:
/// - r_eq = 3396.0 km, flat_coeff = 0.005186
///
/// Gravitational parameter from JEOD `mars_MRO110B2.cc`:
/// - mu = 4.2828374527E+13 m^3/s^2
pub const MARS: PlanetShape = PlanetShape {
    name: "Mars",
    mu: 4.282_837_452_7e13,                    // JEOD mars_MRO110B2.cc:57
    r_eq: 1000.0 * 3396.0,                     // JEOD mars.cc:46
    r_pol: 1000.0 * 3396.0 * (1.0 - 0.005186), // JEOD: r_eq * (1 - flat_coeff)
    flat_coeff: 0.005186,                      // JEOD mars.cc:45
};
