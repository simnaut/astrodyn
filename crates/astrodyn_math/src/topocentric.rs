//! Site-anchored topocentric (East-North-Up) frame construction.
//!
//! Builds the rotation from a planet's body-fixed (PCPF) frame to a local
//! East-North-Up frame anchored at a fixed geodetic site — a landing site, a
//! ground station, a DEM tile origin. The result is a typed
//! [`FrameTransform<PlanetFixed<P>, Topocentric<P>>`] that composes directly
//! with the inertial→body-fixed rotation from the ephemeris.
//!
//! Convention follows JEOD's `NorthEastDown::build_ned_orientation()`
//! (`models/utils/planet_fixed/north_east_down/src/north_east_down.cc`), with
//! the rows permuted/sign-flipped from North-East-Down to East-North-Up:
//!
//! - East  = `(-sinλ,        cosλ,        0   )`
//! - North = `(-sinφ·cosλ,  -sinφ·sinλ,   cosφ)`
//! - Up    = `( cosφ·cosλ,   cosφ·sinλ,   sinφ)`   (= −Down)
//!
//! where φ is geodetic latitude and λ is geodetic longitude. These rows are the
//! ENU basis axes expressed in PCPF, i.e. the `parent → this` rotation matrix
//! (PCPF → ENU). `Up` is the outward geodetic ellipsoid normal, so passing a
//! *geodetic* (not geocentric) latitude makes the local horizon agree with the
//! reference ellipsoid the DEM is tied to.
//!
//! Only the site's latitude/longitude affect the ENU *orientation*; altitude
//! and the ellipsoid shape shift the frame *origin*, which a rotation-only
//! [`FrameTransform`] does not carry. The builder accepts a full
//! [`GeodeticStateTyped`] for ergonomics (callers already have one) but reads
//! only its latitude and longitude — the `altitude` field is ignored.

use astrodyn_quantities::frame::{Planet, PlanetFixed, Topocentric};
use astrodyn_quantities::frame_transform::FrameTransform;
use glam::{DMat3, DVec3};
use uom::si::angle::radian;

use crate::geodetic::GeodeticStateTyped;

/// Rotation from planet `P`'s body-fixed (PCPF) frame to the East-North-Up
/// frame anchored at geodetic site `site`.
///
/// Apply it to a `Position`/`Velocity`/`Acceleration` in `PlanetFixed<P>` to
/// get the same quantity in the site-local ENU frame; compose with the
/// ephemeris `FrameTransform<RootInertial, PlanetFixed<P>>` for an inertial→ENU
/// chain. The returned transform depends only on the site's geodetic
/// latitude/longitude (see the module docs).
pub fn topocentric_enu_transform<P: Planet>(
    site: &GeodeticStateTyped,
) -> FrameTransform<PlanetFixed<P>, Topocentric<P>> {
    let (sinlat, coslat) = site.latitude.get::<radian>().sin_cos();
    let (sinlon, coslon) = site.longitude.get::<radian>().sin_cos();

    // ENU basis axes expressed in PCPF coordinates (the rows of the PCPF→ENU
    // rotation). `from_cols(...).transpose()` lays these vectors in as rows.
    let east = DVec3::new(-sinlon, coslon, 0.0);
    let north = DVec3::new(-sinlat * coslon, -sinlat * sinlon, coslat);
    let up = DVec3::new(coslat * coslon, coslat * sinlon, sinlat);
    let pcpf_to_enu = DMat3::from_cols(east, north, up).transpose();

    FrameTransform::from_matrix(pcpf_to_enu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_quantities::frame::Earth;
    use astrodyn_quantities::prelude::{PlanetFixed, Vec3Ext};
    use uom::si::angle::degree;
    use uom::si::f64::{Angle, Length};
    use uom::si::length::meter;

    fn site(lat_deg: f64, lon_deg: f64) -> GeodeticStateTyped {
        GeodeticStateTyped {
            latitude: Angle::new::<degree>(lat_deg),
            longitude: Angle::new::<degree>(lon_deg),
            altitude: Length::new::<meter>(0.0),
        }
    }

    // At the sub-(0°,0°) site on the prime meridian/equator, PCPF axes map to
    // ENU as: PCPF +X (out through 0°,0°) → Up, PCPF +Y (90°E) → East,
    // PCPF +Z (north pole) → North.
    #[test]
    fn enu_at_lat0_lon0_matches_known_axes() {
        let t = topocentric_enu_transform::<Earth>(&site(0.0, 0.0));
        let x = t.apply(DVec3::X.m_at::<PlanetFixed<Earth>>()).raw_si();
        let y = t.apply(DVec3::Y.m_at::<PlanetFixed<Earth>>()).raw_si();
        let z = t.apply(DVec3::Z.m_at::<PlanetFixed<Earth>>()).raw_si();
        // x (PCPF) → Up = +Z_enu; y → East = +X_enu; z → North = +Y_enu.
        assert!(
            (x - DVec3::Z).length() < 1e-12,
            "PCPF X should map to ENU Up, got {x:?}"
        );
        assert!(
            (y - DVec3::X).length() < 1e-12,
            "PCPF Y should map to ENU East, got {y:?}"
        );
        assert!(
            (z - DVec3::Y).length() < 1e-12,
            "PCPF Z should map to ENU North, got {z:?}"
        );
    }

    // Round-trip a non-trivial PCPF vector through the transform and its
    // inverse at an off-axis site (not equator, not pole).
    #[test]
    fn enu_round_trips_through_inverse() {
        let t = topocentric_enu_transform::<Earth>(&site(34.5, -118.25));
        let v = DVec3::new(1_234.5, -6_789.0, 4_242.0);
        let there = t.apply(v.m_at::<PlanetFixed<Earth>>());
        let back = t.inverse().apply(there).raw_si();
        assert!(
            (back - v).length() < 1e-9,
            "round trip drifted: {back:?} vs {v:?}"
        );
    }

    // The North pole's geodetic Up must be PCPF +Z, and at a non-zero longitude
    // there the East/North still form a right-handed triad with Up (det = +1 is
    // already enforced by FrameTransform::from_matrix; this pins the Up axis).
    #[test]
    fn enu_up_axis_is_geodetic_normal() {
        let t = topocentric_enu_transform::<Earth>(&site(90.0, 45.0));
        // Up is the third row of the PCPF→ENU matrix; apply to the pole normal.
        let up_image = t.apply(DVec3::Z.m_at::<PlanetFixed<Earth>>()).raw_si();
        // At the pole, PCPF +Z is the geodetic normal → maps to ENU Up (+Z_enu).
        assert!(
            (up_image - DVec3::Z).length() < 1e-12,
            "pole normal should map to ENU Up, got {up_image:?}"
        );
    }
}
