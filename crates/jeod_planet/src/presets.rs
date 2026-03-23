use crate::planet::PlanetShape;

/// Earth (WGS84 ellipsoid).
///
/// Constants from JEOD `planet/data/src/earth.cc`:
/// - r_eq = 6378.137 km, flat_inv = 298.257223563
///
/// Gravitational parameter from JEOD `earth_GGM05C.cc`:
/// - mu = 398600.44150e9 m^3/s^2
pub const EARTH: PlanetShape = PlanetShape {
    name: "Earth",
    mu: 3.986_004_415e14,
    r_eq: 6_378_137.0,
    r_pol: 6_356_752.314_245_179, // r_eq * (1 - 1/298.257223563)
    flattening: 1.0 / 298.257_223_563,
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
    mu: 4.902_799_806_931_69e12,
    r_eq: 1_738_140.0,
    r_pol: 1_735_967.325, // 1738140.0 * (1 - 0.00125)
    flattening: 0.00125,
};

/// Sun.
///
/// Constants from JEOD `planet/data/src/sun.cc`:
/// - r_eq = 696000 km, flat_coeff = 5e-5
///
/// Gravitational parameter from JPL DE421 constants:
/// - mu = 1.32712440018e20 m^3/s^2 (GMS)
pub const SUN: PlanetShape = PlanetShape {
    name: "Sun",
    mu: 1.327_124_400_18e20,
    r_eq: 696_000_000.0,
    r_pol: 695_965_200.0, // 696000000.0 * (1 - 5e-5)
    flattening: 5.0e-5,
};

/// Mars.
///
/// Constants from JEOD `planet/data/src/mars.cc`:
/// - r_eq = 3396.0 km, flat_coeff = 0.005186
///
/// Gravitational parameter from JEOD `mars_MRO110B2.cc` or standard:
/// - mu = 4.2828372e13 m^3/s^2
pub const MARS: PlanetShape = PlanetShape {
    name: "Mars",
    mu: 4.282_837_2e13,
    r_eq: 3_396_000.0,
    r_pol: 3_378_388.584, // 3396000.0 * (1 - 0.005186)
    flattening: 0.005186,
};
