//! Typed origin-anchored topocentric (East-North-Up) pose.
//!
//! [`topocentric_enu_state`] builds the full pfix→site pose — origin **and**
//! orientation — as a typed [`RefFrameStateTyped<PlanetFixed<P>,
//! Topocentric<P>>`]. This is the typed analog of JEOD's `NorthEastDown`
//! `RefFrame` (orientation + translation), and the origin-carrying companion to
//! the rotation-only [`astrodyn_math::topocentric_enu_transform`]: where that
//! function returns just the `.rot` half (a `FrameTransform`, which cannot move
//! a `Position` off the planet centre), this builder carries the site origin so
//! a `Position` can be transformed site-relative via
//! [`RefFrameStateTyped::position_parent_to_child`].
//!
//! The pose is **static** wrt the planet-fixed frame — zero velocity, zero
//! angular rate. Composing it up to an inertial frame (which adds the planet's
//! `ω×r` terms) is the caller's job via [`RefFrameStateTyped::incr_right`] with
//! the inertial→pfix state, exactly as
//! `astrodyn_dynamics::body_init::ned_reference_frame_state` does for the
//! untyped NED frame.

use astrodyn_math::{local_level_ned_axes, GeodeticStateTyped, JeodQuat};
use astrodyn_quantities::aliases::{AngularVelocity, Position, Velocity};
use astrodyn_quantities::frame::{Planet, PlanetFixed, Topocentric};
use astrodyn_quantities::quat::NormalizedQuat;
use glam::DMat3;
use uom::si::f64::Length;
use uom::si::length::meter;

use crate::ref_frame_state::{RefFrameRotTyped, RefFrameStateTyped, RefFrameTransTyped};

/// Build the typed pfix→ENU pose for the geodetic site `site` on planet `P`.
///
/// The returned [`RefFrameStateTyped<PlanetFixed<P>, Topocentric<P>>`] carries:
/// - **origin** = the site's geodetic→PCPF Cartesian position, including
///   altitude (via [`GeodeticState::to_planet_fixed`](astrodyn_math::GeodeticState::to_planet_fixed)),
/// - **orientation** = the PCPF→ENU rotation (the same axes as
///   [`astrodyn_math::topocentric_enu_transform`], from the shared
///   [`astrodyn_math::local_level_ned_axes`] kernel),
/// - **zero** velocity and angular rate (the site is fixed wrt the rotating
///   planet).
///
/// `r_eq` / `r_pol` are the planet's equatorial / polar radii.
///
/// # Origin-shift, the whole point
///
/// Apply [`RefFrameStateTyped::position_parent_to_child`] to a
/// `Position<PlanetFixed<P>>` to get the **site-relative** position
/// `R·(p − s)` — the operation the rotation-only `FrameTransform` cannot
/// express. Free vectors (`Velocity`/`Acceleration`) go through
/// [`RefFrameStateTyped::rotate_vector_parent_to_child`].
///
/// # Note on the cached rotation matrix
///
/// The rotation is stored quaternion-canonically (JEOD_INV RF.04), so
/// `state.rot.t_parent_this()` agrees with
/// `topocentric_enu_transform::<P>(site).matrix()` only to within a few ULPs
/// (the typed path round-trips through the quaternion; `from_matrix` is
/// matrix-preserving). Both describe the same rotation.
pub fn topocentric_enu_state<P: Planet>(
    site: &GeodeticStateTyped,
    r_eq: Length,
    r_pol: Length,
) -> RefFrameStateTyped<PlanetFixed<P>, Topocentric<P>> {
    let geo = (*site).into_raw();

    // Origin in PCPF, full ellipsoid conversion (altitude included).
    let origin = geo.to_planet_fixed(r_eq.get::<meter>(), r_pol.get::<meter>());

    // ENU axes from the shared kernel, using the same lat/lon as the origin.
    let (north, east, down) = local_level_ned_axes(geo.latitude, geo.longitude);
    let t_pfix_enu = DMat3::from_cols(east, north, -down).transpose();

    // RF.04 canonical path: derive the quaternion from the matrix; the typed
    // rotation re-derives its cached matrix from the quaternion.
    let q = NormalizedQuat::renormalize(JeodQuat::left_quat_from_transformation(&t_pfix_enu))
        .expect("ENU rotation matrix is orthonormal, so its quaternion is non-zero");

    let trans = RefFrameTransTyped {
        position: Position::<PlanetFixed<P>>::from_raw_si(origin),
        velocity: Velocity::<PlanetFixed<P>>::zero(),
    };
    let rot = RefFrameRotTyped::<PlanetFixed<P>, Topocentric<P>>::new(
        q,
        AngularVelocity::<Topocentric<P>>::zero(),
    );
    RefFrameStateTyped::new(trans, rot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_math::test_utils::approx_eq_vec3;
    use astrodyn_math::topocentric_enu_transform;
    use astrodyn_quantities::frame::Earth;
    use glam::DVec3;
    use uom::si::angle::degree;
    use uom::si::f64::Angle;

    // WGS84 (the radii the NED/orbinit fixtures use).
    const R_EQ: f64 = 6_378_137.0;
    const R_POL: f64 = 6_356_752.314_245;

    fn site(lat_deg: f64, lon_deg: f64, alt_m: f64) -> GeodeticStateTyped {
        GeodeticStateTyped {
            latitude: Angle::new::<degree>(lat_deg),
            longitude: Angle::new::<degree>(lon_deg),
            altitude: Length::new::<meter>(alt_m),
        }
    }

    fn earth_pose(
        lat: f64,
        lon: f64,
        alt: f64,
    ) -> RefFrameStateTyped<PlanetFixed<Earth>, Topocentric<Earth>> {
        topocentric_enu_state::<Earth>(
            &site(lat, lon, alt),
            Length::new::<meter>(R_EQ),
            Length::new::<meter>(R_POL),
        )
    }

    // The pose origin is the full geodetic→PCPF cartesian (altitude included),
    // bit-exact to a direct conversion; velocity and rate are zero.
    #[test]
    fn origin_is_geodetic_cartesian() {
        let s = site(34.5, -118.25, 700.0);
        let pose = earth_pose(34.5, -118.25, 700.0);
        let expected = s.into_raw().to_planet_fixed(R_EQ, R_POL);
        assert_eq!(
            pose.trans.position.raw_si(),
            expected,
            "origin must include altitude"
        );
        assert_eq!(pose.trans.velocity.raw_si(), DVec3::ZERO);
        assert_eq!(pose.rot.ang_vel_this().raw_si(), DVec3::ZERO);
    }

    // The pose rotation describes the same ENU axes as the rotation-only
    // transform — to ULP level, since the typed path round-trips through the
    // canonical quaternion (RF.04) while `from_matrix` is matrix-preserving.
    #[test]
    fn rotation_matches_topo_transform() {
        let s = site(-12.0, 77.0, 0.0);
        let pose = earth_pose(-12.0, 77.0, 0.0);
        let transform = topocentric_enu_transform::<Earth>(&s).matrix();
        let r = pose.rot.t_parent_this();
        assert!(
            approx_eq_vec3(r.x_axis, transform.x_axis, 1e-12)
                && approx_eq_vec3(r.y_axis, transform.y_axis, 1e-12)
                && approx_eq_vec3(r.z_axis, transform.z_axis, 1e-12),
            "pose rotation diverges from topocentric_enu_transform: {r:?} vs {transform:?}"
        );
    }

    // The load-bearing distinction from a rotation-only `FrameTransform`: a
    // `Position` at the site origin maps to the child-frame origin (zero),
    // because the pose subtracts the origin offset `R·(s − s) = 0`. A
    // rotation-only apply would leave it ~6400 km from zero.
    #[test]
    fn site_origin_maps_to_child_zero() {
        let pose = earth_pose(51.6, -0.13, 25.0);
        let origin_pcpf = pose.trans.position; // Position<PlanetFixed<Earth>>
        let in_enu = pose.position_parent_to_child(origin_pcpf);
        assert!(
            in_enu.raw_si().length() < 1e-6,
            "site origin must map to ENU zero, got {:?}",
            in_enu.raw_si()
        );
    }

    // Position pose-apply round-trips through its inverse.
    #[test]
    fn position_round_trips() {
        let pose = earth_pose(34.5, -118.25, 700.0);
        let p = Position::<PlanetFixed<Earth>>::from_raw_si(DVec3::new(
            6_300_000.0,
            100_000.0,
            3_600_000.0,
        ));
        let there = pose.position_parent_to_child(p);
        let back = pose.position_child_to_parent(there);
        assert!(
            (back.raw_si() - p.raw_si()).length() < 1e-6,
            "round trip drifted: {:?} vs {:?}",
            back.raw_si(),
            p.raw_si()
        );
    }

    // Free-vector rotate matches the rotation-only transform (both are pure
    // rotation; no origin shift on a Velocity).
    #[test]
    fn rotate_vector_matches_transform() {
        let s = site(34.5, -118.25, 0.0);
        let pose = earth_pose(34.5, -118.25, 0.0);
        let v = Velocity::<PlanetFixed<Earth>>::from_raw_si(DVec3::new(12.0, -7.0, 3.0));
        let via_pose = pose.rotate_vector_parent_to_child(v).raw_si();
        let via_transform = topocentric_enu_transform::<Earth>(&s).apply(v).raw_si();
        assert!(
            approx_eq_vec3(via_pose, via_transform, 1e-12),
            "rotate_vector diverged: {via_pose:?} vs {via_transform:?}"
        );
    }
}
