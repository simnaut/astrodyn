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

/// Local-level (NED) basis axes at geodetic `(lat, lon)` (radians), expressed
/// in PCPF, as the three row vectors `(north, east, down)` of the PCPF→NED
/// rotation.
///
/// Single source of truth for both the NED rotation
/// (`astrodyn_dynamics::compute_ned_rotation`, which lays these rows into a
/// matrix via `mat3_from_rows`) and the ENU rotation
/// ([`topocentric_enu_transform`], which permutes to `(east, north, −down)`).
/// The arithmetic here is bit-identical to the historical `compute_ned_rotation`
/// body — four separate `.sin()`/`.cos()` calls and the same row expressions —
/// so the Tier-3 NED-init freeze is preserved when that function delegates here.
///
/// JEOD source: `NorthEastDown::build_ned_orientation`
/// (`models/utils/planet_fixed/north_east_down/src/north_east_down.cc`).
pub fn local_level_ned_axes(lat: f64, lon: f64) -> (DVec3, DVec3, DVec3) {
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let down = DVec3::new(-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat);
    (north, east, down)
}

/// Rotation from planet `P`'s body-fixed (PCPF) frame to the East-North-Up
/// axes of the geodetic site `site`.
///
/// This is **orientation only** — a proper rotation, no origin shift.
///
/// - `Velocity`/`Acceleration` (free vectors) and direction vectors: apply
///   directly to re-express them in ENU axes. This is the intended use.
/// - `Position`: `apply` yields the PCPF position re-expressed in ENU *axes*,
///   still rooted at the planet **centre** — NOT site-relative. Site-relative
///   is `R·(p − s)`; `apply` computes only `R·p`, off by the site radius
///   (~6378 km on Earth's surface). To transform a `Position` into a site-
///   anchored frame, use a construct that carries the origin: the typed
///   `topocentric_enu_state` /
///   `RefFrameStateTyped<PlanetFixed<P>, Topocentric<P>>` (rotation **and**
///   origin) in `astrodyn_frames`, or
///   `astrodyn_dynamics::body_init::ned_reference_frame_state`. A rotation-only
///   `FrameTransform` structurally cannot carry the offset (see the module
///   docs and issue #689).
///
/// Compose with the ephemeris `FrameTransform<RootInertial, PlanetFixed<P>>`
/// for an inertial→ENU axis chain. The transform depends only on the site's
/// geodetic latitude/longitude; altitude is ignored.
pub fn topocentric_enu_transform<P: Planet>(
    site: &GeodeticStateTyped,
) -> FrameTransform<PlanetFixed<P>, Topocentric<P>> {
    // Shared site-orientation kernel: `(north, east, down)` rows of the
    // PCPF→NED rotation. ENU permutes to `(east, north, up)` with `up = −down`
    // (exact f64 negation), so this is the same rotation as the NED builder.
    let (north, east, down) = local_level_ned_axes(
        site.latitude.get::<radian>(),
        site.longitude.get::<radian>(),
    );

    // ENU basis axes as the rows of the PCPF→ENU rotation.
    // `from_cols(...).transpose()` lays these vectors in as rows.
    let pcpf_to_enu = DMat3::from_cols(east, north, -down).transpose();

    FrameTransform::from_matrix(pcpf_to_enu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_quantities::frame::Earth;
    use astrodyn_quantities::prelude::{PlanetFixed, Vec3Ext};
    use uom::si::angle::{degree, radian};
    use uom::si::f64::{Angle, Length};
    use uom::si::length::meter;

    fn site(lat_deg: f64, lon_deg: f64) -> GeodeticStateTyped {
        GeodeticStateTyped {
            latitude: Angle::new::<degree>(lat_deg),
            longitude: Angle::new::<degree>(lon_deg),
            altitude: Length::new::<meter>(0.0),
        }
    }

    // ENU is the same rotation as NED up to a fixed row permutation + sign:
    // row 0 = East, row 1 = North, row 2 = Up = −Down. Pin this bit-exactly
    // (feeding the kernel the same radian values the transform used, so any
    // mismatch is the permutation, not a lat/lon ULP) so a future change that
    // diverges the ENU builder from the shared NED kernel is caught here, not
    // in a slow Tier-3 run. `from_matrix` preserves the input matrix
    // bit-exactly, so `.matrix()` rows equal the constructed `(east, north,
    // −down)`.
    #[test]
    fn enu_rows_are_permuted_ned() {
        for &(lat_deg, lon_deg) in &[(34.5, -118.25), (-12.0, 77.0), (89.5, 145.0)] {
            let s = site(lat_deg, lon_deg);
            let lat = s.latitude.get::<radian>();
            let lon = s.longitude.get::<radian>();
            let enu = topocentric_enu_transform::<Earth>(&s).matrix();
            let (north, east, down) = local_level_ned_axes(lat, lon);
            // glam stores column-major; reconstruct rows from the column array.
            let c = enu.to_cols_array_2d();
            let row = |i: usize| DVec3::new(c[0][i], c[1][i], c[2][i]);
            assert_eq!(
                row(0),
                east,
                "ENU row 0 must be NED East at {lat_deg},{lon_deg}"
            );
            assert_eq!(
                row(1),
                north,
                "ENU row 1 must be NED North at {lat_deg},{lon_deg}"
            );
            assert_eq!(
                row(2),
                -down,
                "ENU row 2 (Up) must be −(NED Down) at {lat_deg},{lon_deg}"
            );
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
