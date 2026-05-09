//! Geodetic coordinate conversions.
//!
//! Faithful port of JEOD's `planet_fixed_posn.cc` algorithms:
//! - Cartesian (PCPF) <-> ellipsoidal geodetic (latitude, longitude, altitude)
//! - Cartesian (PCPF) <-> spherical (latitude, longitude, altitude)
//!
//! All coordinates are in the planet-centered planet-fixed (PCPF) frame.
//!
//! Two API flavors are provided:
//! - The original `f64` variants (`cartesian_to_geodetic`, `geodetic_to_cartesian`)
//!   operate on raw `DVec3` / `f64` and are retained for backward compatibility.
//!   They are deprecated and slated for removal in Phase 10 of the type-system
//!   refactor (#101).
//! - The typed sibling variants (`cartesian_to_geodetic_typed`,
//!   `geodetic_to_cartesian_typed`) accept/return `astrodyn_quantities` typed
//!   quantities (`Position<PlanetFixed<P>>`, `Length`, `Angle`), giving
//!   frame-tagged, unit-safe signatures.

use astrodyn_quantities::aliases::Position;
use astrodyn_quantities::frame::{Planet, PlanetFixed};
use astrodyn_quantities::qty3::Qty3;
use glam::{DMat3, DVec3};
use uom::si::angle::radian;
use uom::si::f64::{Angle, Length};
use uom::si::length::meter;

/// Maximum iterations for Borkowski's geodetic latitude solver.
/// Matches JEOD `PlanetFixedPosition::Max_iteration_limit` (class-level constant).
const MAX_ITERATION_LIMIT: usize = 10;

/// Geodetic coordinates on a reference ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GeodeticState {
    /// Geodetic latitude in radians (positive north, range `±π/2`).
    pub latitude: f64,
    /// Geodetic longitude in radians (positive east).
    pub longitude: f64,
    /// Height above the reference ellipsoid, in meters.
    pub altitude: f64,
}

impl GeodeticState {
    /// Convert raw planet-fixed Cartesian coordinates to geodetic state.
    ///
    /// Planet-agnostic entry point that retains the f64 surface for
    /// callers (e.g., NED initializers) where the planet phantom is not
    /// available at the call site. Bit-identical numerics to
    /// [`cartesian_to_geodetic_typed`].
    pub fn from_planet_fixed(cart: DVec3, r_eq: f64, r_pol: f64) -> Self {
        cartesian_to_geodetic_impl(cart, r_eq, r_pol)
    }

    /// Convert this geodetic state back to raw planet-fixed Cartesian.
    ///
    /// Planet-agnostic entry point with the same usage pattern as
    /// [`Self::from_planet_fixed`].
    pub fn to_planet_fixed(&self, r_eq: f64, r_pol: f64) -> DVec3 {
        geodetic_to_cartesian_impl(self, r_eq, r_pol)
    }
}

/// Compute the geodetic state of a body in inertial coordinates.
///
/// Rotates the inertial position into the planet-fixed frame using the given
/// transformation matrix, then delegates to
/// [`GeodeticState::from_planet_fixed`] (sole owner of the JEOD Borkowski
/// iteration kernel). Bit-identical numerics to the typed sibling
/// [`compute_body_geodetic_typed`].
pub fn compute_body_geodetic(
    position: DVec3,
    t_inertial_pfix: &DMat3,
    r_eq: f64,
    r_pol: f64,
) -> GeodeticState {
    let pos_pfix = *t_inertial_pfix * position;
    GeodeticState::from_planet_fixed(pos_pfix, r_eq, r_pol)
}

/// Typed sibling of [`compute_body_geodetic`].
///
/// Accepts a typed inertial position and ellipsoid radii, applies the
/// inertial-to-planet-fixed rotation, then delegates to
/// [`GeodeticState::from_planet_fixed`] (sole owner of the JEOD Borkowski
/// iteration kernel). Returns the f64 [`GeodeticState`] used by Bevy
/// components; bit-identical to the f64 surface.
pub fn compute_body_geodetic_typed<P: Planet>(
    position: Position<astrodyn_quantities::frame::PlanetInertial<P>>,
    t_inertial_pfix: &DMat3,
    r_eq: Length,
    r_pol: Length,
) -> GeodeticState {
    let pos_pfix = *t_inertial_pfix * position.raw_si();
    GeodeticState::from_planet_fixed(pos_pfix, r_eq.get::<meter>(), r_pol.get::<meter>())
}

/// Typed geodetic coordinates on a reference ellipsoid.
///
/// Companion to [`GeodeticState`] carrying `uom` dimensioned scalars so
/// signatures expressed with this type are unit-safe.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GeodeticStateTyped {
    /// Geodetic latitude (positive north, range ±π/2).
    pub latitude: Angle,
    /// Geodetic longitude (positive east).
    pub longitude: Angle,
    /// Height above the reference ellipsoid.
    pub altitude: Length,
}

impl GeodeticStateTyped {
    /// Construct a typed state from an untyped [`GeodeticState`] assuming the
    /// stored components are in radians/meters (their documented base units).
    #[inline]
    pub fn from_raw(state: GeodeticState) -> Self {
        Self {
            latitude: Angle::new::<radian>(state.latitude),
            longitude: Angle::new::<radian>(state.longitude),
            altitude: Length::new::<meter>(state.altitude),
        }
    }

    /// Lower a typed state back to the raw [`GeodeticState`] (radians/meters).
    #[inline]
    pub fn into_raw(self) -> GeodeticState {
        GeodeticState {
            latitude: self.latitude.get::<radian>(),
            longitude: self.longitude.get::<radian>(),
            altitude: self.altitude.get::<meter>(),
        }
    }
}

/// Spherical coordinates relative to a spherical planet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalState {
    /// Geocentric latitude in radians.
    pub latitude: f64,
    /// Longitude in radians (positive east).
    pub longitude: f64,
    /// Height above the mean equatorial radius, in meters.
    pub altitude: f64,
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
    // JEOD_INV: PF.01 — position must be far from planet center (r_local > r_eq · 1e-10)
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
/// Internal numeric kernel shared by [`cartesian_to_geodetic_typed`]; new
/// callers should use the typed sibling. Kept module-private after the
/// Phase 10 purge of the bare-`f64` public surface.
///
/// # Arguments
/// * `cart` - Cartesian position in PCPF frame (m)
/// * `r_eq` - Equatorial radius (m)
/// * `r_pol` - Polar radius (m)
pub(crate) fn cartesian_to_geodetic_impl(cart: DVec3, r_eq: f64, r_pol: f64) -> GeodeticState {
    // JEOD_INV: PF.02 — input must be NaN/Inf free
    // JEOD planet_fixed_posn.cc:155-162: check for NaN/Inf before proceeding.
    assert!(
        cart.x.is_finite() && cart.y.is_finite() && cart.z.is_finite(),
        "cartesian_to_geodetic: input contains NaN or Inf ({cart:?})"
    );

    let x_ellipse_sq = cart.x * cart.x + cart.y * cart.y;
    let x_ellipse = x_ellipse_sq.sqrt();
    let z_ellipse = cart.z;
    let r_ellipse = (x_ellipse_sq + z_ellipse * z_ellipse).sqrt();

    // JEOD_INV: PF.01 — position must be far from planet center
    assert!(
        r_ellipse > r_eq * 1e-10,
        "cartesian_to_geodetic: position too close to planet center ({r_ellipse} m)"
    );

    let (lat, alt) = get_elliptic_parameters(x_ellipse, z_ellipse, r_eq, r_pol);

    // JEOD_INV: PF.03 — polar singularity: at x_ellipse==0, longitude is undefined
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

        let mut y_val = y0;
        let mut converged = false;
        for _ in 0..MAX_ITERATION_LIMIT {
            let d = 2.0 * ((y0 - w).cos() - c * (2.0 * y0).cos());
            // JEOD_INV: PF.05 — Borkowski denominator must be non-zero
            // Not in JEOD: guard against degenerate denominator. JEOD divides
            // unconditionally; we assert because d==0 implies zero flattening
            // which should never reach this code path with a real ellipsoid.
            assert!(
                d.abs() > f64::EPSILON,
                "geodetic iteration: denominator near zero (d={d})"
            );
            y_val = y0 - (2.0 * (y0 - w).sin() - c * (2.0 * y0).sin()) / d;
            if (y_val - y0).abs() < 1.0e-12 {
                converged = true;
                break;
            }
            y0 = y_val;
        }
        // JEOD_INV: PF.04 — Borkowski iteration must converge to 1e-12 within MAX_ITERATION_LIMIT
        // Not in JEOD: JEOD silently uses the last iterate on non-convergence.
        // We assert because the Borkowski iteration is guaranteed to converge for
        // physically valid inputs and failure indicates a bug in the caller.
        assert!(
            converged,
            "geodetic iteration did not converge after {MAX_ITERATION_LIMIT} iterations"
        );
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
/// Internal numeric kernel shared by [`geodetic_to_cartesian_typed`]; new
/// callers should use the typed sibling. Kept module-private after the
/// Phase 10 purge of the bare-`f64` public surface.
///
/// # Arguments
/// * `geo` - Geodetic coordinates (latitude rad, longitude rad, altitude m)
/// * `r_eq` - Equatorial radius (m)
/// * `r_pol` - Polar radius (m)
pub(crate) fn geodetic_to_cartesian_impl(geo: &GeodeticState, r_eq: f64, r_pol: f64) -> DVec3 {
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

/// Typed sibling for the file-private `cartesian_to_geodetic_impl` kernel.
/// Accepts a planet-fixed position and returns a dimensionally-typed
/// [`GeodeticStateTyped`].
///
/// Bit-identical to the kernel — this is a thin wrapper that
/// unwraps `Position<PlanetFixed<P>>` to its SI `DVec3` representation,
/// delegates to the f64 implementation, and re-wraps the scalar outputs.
///
/// The planet is carried only as a compile-time phantom tag on the input
/// frame; the ellipsoid dimensions are supplied numerically via `r_eq` /
/// `r_pol` (matching the existing f64 API). The geodetic representation is
/// defined with respect to the body-fixed frame of the named planet.
pub fn cartesian_to_geodetic_typed<P: Planet>(
    pos: Position<PlanetFixed<P>>,
    r_eq: Length,
    r_pol: Length,
) -> GeodeticStateTyped {
    let raw = cartesian_to_geodetic_impl(pos.raw_si(), r_eq.get::<meter>(), r_pol.get::<meter>());
    GeodeticStateTyped::from_raw(raw)
}

/// Typed sibling for the file-private `geodetic_to_cartesian_impl` kernel.
/// Returns a planet-fixed position parameterized by the planet tag `P`.
///
/// Bit-identical to the kernel.
pub fn geodetic_to_cartesian_typed<P: Planet>(
    state: GeodeticStateTyped,
    r_eq: Length,
    r_pol: Length,
) -> Position<PlanetFixed<P>> {
    let raw_state = state.into_raw();
    let cart = geodetic_to_cartesian_impl(&raw_state, r_eq.get::<meter>(), r_pol.get::<meter>());
    Qty3::from_raw_si(cart)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_quantities::ext::F64Ext;
    use astrodyn_quantities::frame::Earth;
    use std::f64::consts::PI;
    // Test aliases so the existing f64 test bodies keep their compact
    // call sites after the Phase 10 purge of the bare-`f64` public surface.
    use cartesian_to_geodetic_impl as cartesian_to_geodetic;
    use geodetic_to_cartesian_impl as geodetic_to_cartesian;

    const EARTH_R_EQ: f64 = 6_378_137.0; // WGS84 equatorial radius (m)
    const EARTH_R_POL: f64 = EARTH_R_EQ * (1.0 - 1.0 / 298.257_223_563); // JEOD: r_eq * (1 - flat_coeff)

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

    // =================================================================
    // Typed API tests
    // =================================================================

    /// ISS altitude landmark: typed round-trip (geodetic -> cartesian -> geodetic)
    /// must close to <1e-14 rad / <1e-9 m.
    #[test]
    fn typed_round_trip_iss_landmark() {
        // ISS-like: 51.6° N, 30° E, 408 km above WGS84 ellipsoid.
        let original = GeodeticStateTyped {
            latitude: 51.6.deg(),
            longitude: 30.0.deg(),
            altitude: 408_000.0.m(),
        };
        let r_eq = EARTH_R_EQ.m();
        let r_pol = EARTH_R_POL.m();

        let cart: Position<PlanetFixed<Earth>> = geodetic_to_cartesian_typed(original, r_eq, r_pol);
        let recovered = cartesian_to_geodetic_typed(cart, r_eq, r_pol);

        let lat_err =
            (recovered.latitude.get::<radian>() - original.latitude.get::<radian>()).abs();
        let lon_err =
            (recovered.longitude.get::<radian>() - original.longitude.get::<radian>()).abs();
        let alt_err = (recovered.altitude.get::<meter>() - original.altitude.get::<meter>()).abs();

        assert!(lat_err < 1e-14, "lat err = {lat_err}");
        assert!(lon_err < 1e-14, "lon err = {lon_err}");
        assert!(alt_err < 1e-9, "alt err = {alt_err} m");
    }

    /// The typed and f64 entry points must be bit-identical for the
    /// underlying scalar components.
    #[test]
    fn typed_matches_f64_cartesian_to_geodetic() {
        // Non-trivial position to exercise non-zero lat/lon/alt components.
        let raw_pos = DVec3::new(4_500_000.0, -2_800_000.0, 3_700_000.0);
        let pos: Position<PlanetFixed<Earth>> = Qty3::from_raw_si(raw_pos);

        let f64_result = cartesian_to_geodetic(raw_pos, EARTH_R_EQ, EARTH_R_POL);
        let typed_result = cartesian_to_geodetic_typed(pos, EARTH_R_EQ.m(), EARTH_R_POL.m());

        // Bit-identical: same numerical operation, just unit-wrapped.
        assert_eq!(typed_result.latitude.get::<radian>(), f64_result.latitude);
        assert_eq!(typed_result.longitude.get::<radian>(), f64_result.longitude);
        assert_eq!(typed_result.altitude.get::<meter>(), f64_result.altitude);
    }

    /// Dual of the previous test for `geodetic_to_cartesian`.
    #[test]
    fn typed_matches_f64_geodetic_to_cartesian() {
        let raw = GeodeticState {
            latitude: 0.9,
            longitude: -0.5,
            altitude: 408_000.0,
        };
        let typed = GeodeticStateTyped::from_raw(raw);

        let f64_cart = geodetic_to_cartesian(&raw, EARTH_R_EQ, EARTH_R_POL);
        let typed_cart: Position<PlanetFixed<Earth>> =
            geodetic_to_cartesian_typed(typed, EARTH_R_EQ.m(), EARTH_R_POL.m());

        let typed_raw = typed_cart.raw_si();
        assert_eq!(typed_raw.x, f64_cart.x);
        assert_eq!(typed_raw.y, f64_cart.y);
        assert_eq!(typed_raw.z, f64_cart.z);
    }

    /// `GeodeticStateTyped::from_raw` / `into_raw` must be inverses.
    #[test]
    fn typed_raw_conversion_round_trip() {
        let raw = GeodeticState {
            latitude: 0.123_456_789,
            longitude: -1.987_654_321,
            altitude: 12_345.678,
        };
        let typed = GeodeticStateTyped::from_raw(raw);
        let back = typed.into_raw();
        assert_eq!(raw.latitude, back.latitude);
        assert_eq!(raw.longitude, back.longitude);
        assert_eq!(raw.altitude, back.altitude);
    }

    // ---- proptest round-trips (#398) ----------------------------------
    //
    // Defends against the regression class fixed in #393: a typed sibling
    // whose `from_raw → into_raw` round-trip silently drops a field.

    use proptest::prelude::*;

    fn arb_finite_f64() -> impl Strategy<Value = f64> {
        proptest::num::f64::ANY.prop_filter("finite", |x| x.is_finite())
    }

    fn arb_geodetic_state() -> impl Strategy<Value = GeodeticState> {
        (arb_finite_f64(), arb_finite_f64(), arb_finite_f64()).prop_map(
            |(latitude, longitude, altitude)| GeodeticState {
                latitude,
                longitude,
                altitude,
            },
        )
    }

    proptest! {
        #[test]
        fn round_trip_geodetic_untyped_typed_untyped(orig in arb_geodetic_state()) {
            let typed = GeodeticStateTyped::from_raw(orig);
            prop_assert_eq!(typed.into_raw(), orig);
        }

        #[test]
        fn round_trip_geodetic_typed_untyped_typed(orig in arb_geodetic_state()) {
            let typed = GeodeticStateTyped::from_raw(orig);
            let lifted = GeodeticStateTyped::from_raw(typed.into_raw());
            prop_assert_eq!(lifted, typed);
        }
    }
}
