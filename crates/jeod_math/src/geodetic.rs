//! Geodetic coordinate conversions.
//!
//! Faithful port of JEOD's `planet_fixed_posn.cc` algorithms:
//! - Cartesian (PCPF) <-> ellipsoidal geodetic (latitude, longitude, altitude)
//! - Cartesian (PCPF) <-> spherical (latitude, longitude, altitude)
//!
//! All coordinates are in the planet-centered planet-fixed (PCPF) frame.

use glam::DVec3;

/// Geodetic coordinates on a reference ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GeodeticState {
    pub latitude: f64,  // rad, geodetic latitude
    pub longitude: f64, // rad, geodetic longitude
    pub altitude: f64,  // m, height above reference ellipsoid
}

/// Spherical coordinates relative to a spherical planet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalState {
    pub latitude: f64,  // rad, geocentric latitude
    pub longitude: f64, // rad, longitude
    pub altitude: f64,  // m, height above mean equatorial radius
}

/// Convert Cartesian PCPF coordinates to spherical coordinates.
///
/// Port of JEOD `PlanetFixedPosition::cart_to_spher()`.
///
/// # Arguments
/// * `cart` - Cartesian position in PCPF frame (m)
/// * `r_eq` - Equatorial radius of the planet (m)
pub fn cartesian_to_spherical(cart: DVec3, r_eq: f64) -> SphericalState {
    let r_local = cart.length();
    assert!(
        r_local > r_eq * 1e-10,
        "cartesian_to_spherical: position too close to planet center ({r_local} m)"
    );

    SphericalState {
        latitude: (cart.z / r_local).asin(),
        longitude: cart.y.atan2(cart.x),
        altitude: r_local - r_eq,
    }
}

/// Convert spherical coordinates to Cartesian PCPF coordinates.
///
/// Port of JEOD `PlanetFixedPosition::spher_to_cart()`.
pub fn spherical_to_cartesian(sph: &SphericalState, r_eq: f64) -> DVec3 {
    let radius = r_eq + sph.altitude;
    let cos_lat = sph.latitude.cos();
    let sin_lat = sph.latitude.sin();
    let cos_lon = sph.longitude.cos();
    let sin_lon = sph.longitude.sin();

    DVec3::new(
        radius * cos_lat * cos_lon,
        radius * cos_lat * sin_lon,
        radius * sin_lat,
    )
}

/// Convert Cartesian PCPF coordinates to geodetic (ellipsoidal) coordinates.
///
/// Port of JEOD `PlanetFixedPosition::cart_to_ellip()` and
/// `PlanetFixedPosition::get_elliptic_parameters()`.
///
/// Uses Borkowski's iterative method for the latitude/altitude computation.
///
/// # Arguments
/// * `cart` - Cartesian position in PCPF frame (m)
/// * `r_eq` - Equatorial radius (m)
/// * `r_pol` - Polar radius (m)
pub fn cartesian_to_geodetic(cart: DVec3, r_eq: f64, r_pol: f64) -> GeodeticState {
    // JEOD planet_fixed_posn.cc:155-162: check for NaN/Inf before proceeding.
    assert!(
        cart.x.is_finite() && cart.y.is_finite() && cart.z.is_finite(),
        "cartesian_to_geodetic: input contains NaN or Inf ({cart:?})"
    );

    let x_ellipse_sq = cart.x * cart.x + cart.y * cart.y;
    let x_ellipse = x_ellipse_sq.sqrt();
    let z_ellipse = cart.z;
    let r_ellipse = (x_ellipse_sq + z_ellipse * z_ellipse).sqrt();

    assert!(
        r_ellipse > r_eq * 1e-10,
        "cartesian_to_geodetic: position too close to planet center ({r_ellipse} m)"
    );

    let (lat, alt) = get_elliptic_parameters(x_ellipse, z_ellipse, r_eq, r_pol);

    let longitude = if x_ellipse != 0.0 {
        cart.y.atan2(cart.x)
    } else {
        // Directly over the pole — longitude is undefined, JEOD leaves it unchanged.
        // We return 0.0 as a convention.
        0.0
    };

    GeodeticState {
        latitude: lat,
        longitude,
        altitude: alt,
    }
}

/// Borkowski's iterative method for geodetic latitude and altitude.
///
/// Port of JEOD `PlanetFixedPosition::get_elliptic_parameters()`.
///
/// Reference: Borkowski, K.M., "Accurate Algorithms To Transform Geocentric
/// To Geodetic Coordinates", Bull. Géod., 63 (1989), pp. 50-56.
fn get_elliptic_parameters(r: f64, z: f64, r_eq: f64, r_pol: f64) -> (f64, f64) {
    let a = r_eq;
    let b = r_pol;

    let (lat, y);

    if r > 0.0 {
        let y0_init = (a * z / (b * r)).atan();
        let ar = a * r;
        let bz = b * z;
        let w = (bz / ar).atan();
        let c = (a * a - b * b) / (ar * ar + bz * bz).sqrt();

        let mut y0 = y0_init;
        let max_iters = 10; // JEOD: PlanetFixedPosition::Max_iteration_limit

        let mut y_val = y0;
        for _ in 0..max_iters {
            let d = 2.0 * ((y0 - w).cos() - c * (2.0 * y0).cos());
            y_val = y0 - (2.0 * (y0 - w).sin() - c * (2.0 * y0).sin()) / d;
            if (y_val - y0).abs() < 1.0e-12 {
                break;
            }
            y0 = y_val;
        }
        y = y_val;
        lat = (a * y.sin() / (b * y.cos())).atan();
    } else {
        // Directly over pole: lat = ±π/2
        y = 0.5 * z * std::f64::consts::PI / z.abs();
        lat = y;
    }

    let alt = (r - a * y.cos()) * lat.cos() + (z - b * y.sin()) * lat.sin();

    (lat, alt)
}

/// Convert geodetic (ellipsoidal) coordinates to Cartesian PCPF coordinates.
///
/// Port of JEOD `PlanetFixedPosition::ellip_to_cart()`.
///
/// # Arguments
/// * `geo` - Geodetic coordinates (latitude rad, longitude rad, altitude m)
/// * `r_eq` - Equatorial radius (m)
/// * `r_pol` - Polar radius (m)
pub fn geodetic_to_cartesian(geo: &GeodeticState, r_eq: f64, r_pol: f64) -> DVec3 {
    let sin_lat = geo.latitude.sin();
    let cos_lat = geo.latitude.cos();

    // Ellipsoid eccentricity squared
    let e_sq = 1.0 - (r_pol * r_pol) / (r_eq * r_eq);

    // Radius of curvature in the prime vertical
    let rc_ellipse = r_eq / (1.0 - e_sq * sin_lat * sin_lat).sqrt();

    // Position in the plane of the ellipse
    let x_ellipse = (rc_ellipse + geo.altitude) * cos_lat;

    DVec3::new(
        x_ellipse * geo.longitude.cos(),
        x_ellipse * geo.longitude.sin(),
        (rc_ellipse * (1.0 - e_sq) + geo.altitude) * sin_lat,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EARTH_R_EQ: f64 = 6_378_137.0; // WGS84 equatorial radius (m)
    const EARTH_R_POL: f64 = 6_356_752.314_245_179_3; // WGS84: r_eq * (1 - 1/298.257223563)

    #[test]
    fn spherical_equator_sea_level() {
        let cart = DVec3::new(EARTH_R_EQ, 0.0, 0.0);
        let sph = cartesian_to_spherical(cart, EARTH_R_EQ);
        assert!((sph.latitude).abs() < 1e-15);
        assert!((sph.longitude).abs() < 1e-15);
        assert!((sph.altitude).abs() < 1e-6);
    }

    #[test]
    fn spherical_round_trip() {
        let original = SphericalState {
            latitude: 0.7, // ~40 degrees
            longitude: -1.2,
            altitude: 400_000.0, // 400 km
        };
        let cart = spherical_to_cartesian(&original, EARTH_R_EQ);
        let recovered = cartesian_to_spherical(cart, EARTH_R_EQ);
        assert!((recovered.latitude - original.latitude).abs() < 1e-12);
        assert!((recovered.longitude - original.longitude).abs() < 1e-12);
        assert!((recovered.altitude - original.altitude).abs() < 1e-6);
    }

    #[test]
    fn geodetic_equator_sea_level() {
        let geo = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        };
        let cart = geodetic_to_cartesian(&geo, EARTH_R_EQ, EARTH_R_POL);
        assert!((cart.x - EARTH_R_EQ).abs() < 1e-6);
        assert!(cart.y.abs() < 1e-6);
        assert!(cart.z.abs() < 1e-6);
    }

    #[test]
    fn geodetic_north_pole() {
        let geo = GeodeticState {
            latitude: PI / 2.0,
            longitude: 0.0,
            altitude: 0.0,
        };
        let cart = geodetic_to_cartesian(&geo, EARTH_R_EQ, EARTH_R_POL);
        assert!(cart.x.abs() < 1e-6);
        assert!(cart.y.abs() < 1e-6);
        assert!((cart.z - EARTH_R_POL).abs() < 1e-6);
    }

    #[test]
    fn geodetic_south_pole() {
        let geo = GeodeticState {
            latitude: -PI / 2.0,
            longitude: 0.0,
            altitude: 0.0,
        };
        let cart = geodetic_to_cartesian(&geo, EARTH_R_EQ, EARTH_R_POL);
        assert!(cart.x.abs() < 1e-6);
        assert!(cart.y.abs() < 1e-6);
        assert!((cart.z + EARTH_R_POL).abs() < 1e-6);
    }

    #[test]
    fn geodetic_round_trip_equator() {
        let original = GeodeticState {
            latitude: 0.0,
            longitude: 1.5,
            altitude: 0.0,
        };
        let cart = geodetic_to_cartesian(&original, EARTH_R_EQ, EARTH_R_POL);
        let recovered = cartesian_to_geodetic(cart, EARTH_R_EQ, EARTH_R_POL);
        assert!((recovered.latitude - original.latitude).abs() < 1e-12);
        assert!((recovered.longitude - original.longitude).abs() < 1e-12);
        assert!((recovered.altitude - original.altitude).abs() < 1e-6);
    }

    #[test]
    fn geodetic_round_trip_mid_latitude() {
        let original = GeodeticState {
            latitude: 0.9, // ~51.6 degrees (ISS inclination)
            longitude: -0.5,
            altitude: 408_000.0, // ISS altitude
        };
        let cart = geodetic_to_cartesian(&original, EARTH_R_EQ, EARTH_R_POL);
        let recovered = cartesian_to_geodetic(cart, EARTH_R_EQ, EARTH_R_POL);
        assert!(
            (recovered.latitude - original.latitude).abs() < 1e-12,
            "latitude error: {}",
            (recovered.latitude - original.latitude).abs()
        );
        assert!(
            (recovered.longitude - original.longitude).abs() < 1e-12,
            "longitude error: {}",
            (recovered.longitude - original.longitude).abs()
        );
        assert!(
            (recovered.altitude - original.altitude).abs() < 1e-6,
            "altitude error: {} m",
            (recovered.altitude - original.altitude).abs()
        );
    }

    #[test]
    fn geodetic_round_trip_poles() {
        for &lat in &[PI / 2.0, -PI / 2.0] {
            let original = GeodeticState {
                latitude: lat,
                longitude: 0.0,
                altitude: 100_000.0,
            };
            let cart = geodetic_to_cartesian(&original, EARTH_R_EQ, EARTH_R_POL);
            let recovered = cartesian_to_geodetic(cart, EARTH_R_EQ, EARTH_R_POL);
            assert!(
                (recovered.latitude - original.latitude).abs() < 1e-10,
                "pole latitude error: {}",
                (recovered.latitude - original.latitude).abs()
            );
            assert!(
                (recovered.altitude - original.altitude).abs() < 1e-6,
                "pole altitude error: {} m",
                (recovered.altitude - original.altitude).abs()
            );
        }
    }

    #[test]
    fn geodetic_round_trip_high_altitude() {
        // Geostationary orbit altitude
        let original = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 35_786_000.0, // ~35,786 km
        };
        let cart = geodetic_to_cartesian(&original, EARTH_R_EQ, EARTH_R_POL);
        let recovered = cartesian_to_geodetic(cart, EARTH_R_EQ, EARTH_R_POL);
        assert!((recovered.latitude - original.latitude).abs() < 1e-12);
        assert!((recovered.altitude - original.altitude).abs() < 1e-6);
    }

    #[test]
    fn geodetic_round_trip_negative_altitude() {
        // Subsurface point (e.g., mine shaft)
        let original = GeodeticState {
            latitude: 0.5,
            longitude: 1.0,
            altitude: -1000.0, // 1 km below surface
        };
        let cart = geodetic_to_cartesian(&original, EARTH_R_EQ, EARTH_R_POL);
        let recovered = cartesian_to_geodetic(cart, EARTH_R_EQ, EARTH_R_POL);
        assert!((recovered.latitude - original.latitude).abs() < 1e-12);
        assert!((recovered.longitude - original.longitude).abs() < 1e-12);
        assert!((recovered.altitude - original.altitude).abs() < 1e-6);
    }

    /// Verify round-trip for 10+ diverse test points (Phase 3 exit criterion).
    #[test]
    fn geodetic_round_trip_ten_points() {
        let test_cases = [
            (0.0, 0.0, 0.0, "equator prime meridian"),
            (PI / 2.0, 0.0, 0.0, "north pole"),
            (-PI / 2.0, 0.0, 0.0, "south pole"),
            (0.4838, 1.5175, 8_848.0, "Mount Everest ~27.99N 86.93E"),
            (0.9, 0.5, 408_000.0, "ISS altitude"),
            (-0.6, 2.5, 200_000.0, "southern hemisphere LEO"),
            (0.0, PI, 35_786_000.0, "GEO at 180 longitude"),
            (1.0, -1.0, -500.0, "subsurface mid-lat"),
            (0.01, 3.0, 0.0, "near equator east"),
            (1.55, 0.0, 10_000.0, "near pole, 10 km up"),
            (-1.2, -2.8, 600_000.0, "deep south high alt"),
        ];

        for (lat, lon, alt, label) in test_cases {
            let original = GeodeticState {
                latitude: lat,
                longitude: lon,
                altitude: alt,
            };
            let cart = geodetic_to_cartesian(&original, EARTH_R_EQ, EARTH_R_POL);
            let recovered = cartesian_to_geodetic(cart, EARTH_R_EQ, EARTH_R_POL);
            let lat_err = (recovered.latitude - original.latitude).abs();
            let lon_err = if lat.abs() > 1.5 {
                0.0 // longitude undefined at poles
            } else {
                (recovered.longitude - original.longitude).abs()
            };
            let alt_err = (recovered.altitude - original.altitude).abs();

            assert!(lat_err < 1e-10, "{label}: latitude error = {lat_err}");
            assert!(lon_err < 1e-10, "{label}: longitude error = {lon_err}");
            assert!(alt_err < 1e-6, "{label}: altitude error = {alt_err} m");
        }
    }
}
