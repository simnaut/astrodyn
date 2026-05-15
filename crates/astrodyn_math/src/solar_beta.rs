//! Solar beta angle computation.
//!
//! The solar beta angle is the angle between the orbital plane and the
//! Sun direction vector. It determines the eclipse geometry and thermal
//! environment of a satellite.
//!
//! β = π/2 - acos(ĥ · ŝ)
//!
//! where ĥ is the orbit normal unit vector and ŝ is the Sun direction unit vector.
//! When β = 0, the Sun is in the orbital plane.
//! When β = ±90°, the Sun is perpendicular to the orbital plane.

use astrodyn_quantities::dims::SpecificAngMomDim;
use astrodyn_quantities::prelude::{Position, Qty3, RootInertial};
use glam::DVec3;
use uom::si::angle::radian;
use uom::si::f64::Angle;

/// Compute the solar beta angle.
///
/// Internal numeric kernel shared by [`solar_beta_angle_typed`]. New
/// callers should use the typed sibling; this `_impl` is kept
/// module-private after the Phase 10 purge of the bare-`f64` public
/// surface.
///
/// # Arguments
/// * `orbit_ang_momentum` - Orbital angular momentum vector (r × v), does not need to be unit
/// * `sun_direction` - Direction vector toward the Sun, does not need to be unit
///
/// # Returns
/// Solar beta angle in radians, in range [-π/2, π/2].
/// Positive when the Sun is on the same side as the angular momentum vector.
pub(crate) fn solar_beta_angle_impl(orbit_ang_momentum: DVec3, sun_direction: DVec3) -> f64 {
    assert!(
        orbit_ang_momentum.length_squared() > 0.0,
        "orbit_ang_momentum must be non-zero"
    );
    assert!(
        sun_direction.length_squared() > 0.0,
        "sun_direction must be non-zero"
    );

    let h_hat = orbit_ang_momentum.normalize();
    let s_hat = sun_direction.normalize();

    // β = π/2 - acos(h_hat · s_hat)
    // Equivalently: β = asin(h_hat · s_hat)
    // Clamp to [-1, 1] to guard against floating-point noise outside asin domain.
    let dot = h_hat.dot(s_hat).clamp(-1.0, 1.0);
    dot.asin()
}

/// Typed counterpart of `solar_beta_angle_impl` (the file-private kernel above).
///
/// Computes the solar beta angle from a frame-tagged specific angular
/// momentum vector (r × v, m²/s) and an inertial-frame sun direction
/// vector (interpreted as meters; only direction matters, so a unit
/// vector or a full position difference are both acceptable).
///
/// # Arguments
/// * `orbit_ang_momentum` - Specific orbital angular momentum `h = r × v`
///   in the inertial frame (does not need to be unit)
/// * `sun_direction` - Direction toward the Sun in the inertial frame
///   (does not need to be unit)
///
/// # Returns
/// Solar beta angle as a typed `Angle`, in range [-π/2, π/2].
/// Positive when the Sun is on the same side as the angular momentum
/// vector.
pub fn solar_beta_angle_typed(
    orbit_ang_momentum: Qty3<SpecificAngMomDim, RootInertial>,
    sun_direction: Position<RootInertial>,
) -> Angle {
    let beta = solar_beta_angle_impl(orbit_ang_momentum.raw_si(), sun_direction.raw_si());
    Angle::new::<radian>(beta)
}

/// Compute the solar beta angle from raw position / velocity / Sun position.
///
/// Computes the orbital angular momentum vector `h = r × v`, then delegates
/// to [`solar_beta_angle_typed`].
///
/// # Panics
///
/// Panics if the orbital angular momentum `h = r × v` is zero (degenerate
/// orbit) or if the Sun position coincides with the body position.
pub fn compute_body_solar_beta(position: DVec3, velocity: DVec3, sun_position: DVec3) -> f64 {
    let h = position.cross(velocity);
    let rel_sun = sun_position - position;

    assert!(
        h.length_squared() > 0.0,
        "compute_body_solar_beta: orbital angular momentum is zero; \
         solar beta angle is undefined"
    );
    assert!(
        rel_sun.length_squared() > 0.0,
        "compute_body_solar_beta: sun_position coincides with position; \
         solar beta angle is undefined"
    );

    // Delegate to the typed kernel (which clamps & normalizes internally) and
    // unwrap the typed `Angle` to radians for f64 storage. Bit-identical to
    // the historical f64 path.
    let h_typed = Qty3::<SpecificAngMomDim, RootInertial>::from_raw_si(h);
    let sun_typed = Position::<RootInertial>::from_raw_si(rel_sun);
    solar_beta_angle_typed(h_typed, sun_typed).get::<radian>()
}

/// Typed sibling of [`compute_body_solar_beta`].
///
/// Accepts inertial-frame typed position/velocity/sun position and returns
/// the solar beta angle as a typed [`Angle`]. Bit-identical to the f64
/// surface — this wrapper computes `h = r × v` and delegates to the typed
/// [`solar_beta_angle_typed`] kernel.
///
/// # Panics
///
/// Panics if the orbital angular momentum `h = r × v` is zero (degenerate
/// orbit) or if the Sun position coincides with the body position.
pub fn compute_body_solar_beta_typed(
    position: Position<RootInertial>,
    velocity: astrodyn_quantities::aliases::Velocity<RootInertial>,
    sun_position: Position<RootInertial>,
) -> Angle {
    let pos = position.raw_si();
    let vel = velocity.raw_si();
    let sun = sun_position.raw_si();
    let h = pos.cross(vel);
    let rel_sun = sun - pos;

    assert!(
        h.length_squared() > 0.0,
        "compute_body_solar_beta_typed: orbital angular momentum is zero; \
         solar beta angle is undefined"
    );
    assert!(
        rel_sun.length_squared() > 0.0,
        "compute_body_solar_beta_typed: sun_position coincides with position; \
         solar beta angle is undefined"
    );

    // Construct typed angular-momentum and unit-direction quantities for the
    // typed kernel. The kernel normalizes internally — direction-only inputs
    // are accepted, so the unnormalized `rel_sun` works as a Position<RootInertial>.
    let h_typed = Qty3::<SpecificAngMomDim, RootInertial>::from_raw_si(h);
    let sun_typed = Position::<RootInertial>::from_raw_si(rel_sun);
    solar_beta_angle_typed(h_typed, sun_typed)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary"
)]
mod tests {
    use super::*;
    use astrodyn_quantities::prelude::Vec3Ext;
    use std::f64::consts::PI;
    // Test alias so the existing f64 test bodies keep their compact
    // `solar_beta_angle` call sites after the Phase 10 purge of the
    // bare-`f64` public surface.
    use solar_beta_angle_impl as solar_beta_angle;

    #[test]
    fn sun_in_orbit_plane() {
        // Sun direction lies in the orbital plane (perpendicular to h)
        let h = DVec3::new(0.0, 0.0, 1.0); // orbit normal = +Z
        let sun = DVec3::new(1.0, 0.0, 0.0); // Sun in +X (in plane)
        let beta = solar_beta_angle(h, sun);
        assert!(beta.abs() < 1e-15);
    }

    #[test]
    fn sun_perpendicular_to_orbit_plane() {
        // Sun direction is along the orbit normal
        let h = DVec3::new(0.0, 0.0, 1.0);
        let sun = DVec3::new(0.0, 0.0, 1.0);
        let beta = solar_beta_angle(h, sun);
        assert!((beta - PI / 2.0).abs() < 1e-15);
    }

    #[test]
    fn sun_opposite_to_orbit_normal() {
        let h = DVec3::new(0.0, 0.0, 1.0);
        let sun = DVec3::new(0.0, 0.0, -1.0);
        let beta = solar_beta_angle(h, sun);
        assert!((beta + PI / 2.0).abs() < 1e-15);
    }

    #[test]
    fn forty_five_degree_beta() {
        let h = DVec3::new(0.0, 0.0, 1.0);
        let sun = DVec3::new(1.0, 0.0, 1.0); // 45 degrees from plane
        let beta = solar_beta_angle(h, sun);
        assert!((beta - PI / 4.0).abs() < 1e-14);
    }

    #[test]
    fn unnormalized_inputs() {
        // Inputs don't need to be unit vectors
        let h = DVec3::new(0.0, 0.0, 1e10);
        let sun = DVec3::new(1e8, 0.0, 0.0);
        let beta = solar_beta_angle(h, sun);
        assert!(beta.abs() < 1e-14);
    }

    #[test]
    fn iss_like_orbit() {
        // ISS orbit normal (inclined ~51.6 degrees from ecliptic)
        let inc = 51.6_f64.to_radians();
        let h = DVec3::new(0.0, -inc.sin(), inc.cos());
        // Sun approximately in ecliptic plane along +X
        let sun = DVec3::new(1.0, 0.0, 0.0);
        let beta = solar_beta_angle(h, sun);
        // Beta should be small (sun nearly in orbit plane at equinox)
        assert!(beta.abs() < 1e-14);
    }

    // --- Typed-variant tests (Phase 2 #104) --------------------------------

    /// Wrap a raw `DVec3` as a frame-tagged, dimensioned specific
    /// angular-momentum vector in the inertial frame.
    fn h_in<F: astrodyn_quantities::frame::Frame>(v: DVec3) -> Qty3<SpecificAngMomDim, F> {
        Qty3::from_raw_si(v)
    }

    #[test]
    fn typed_bounded_in_pi_over_two() {
        // Beta is always in [-π/2, π/2]; sweep a handful of inertial sun
        // directions around an ISS-like orbit normal and verify the
        // returned typed angle stays within that bound (plus a small
        // floating-point slop).
        let inc = 51.6_f64.to_radians();
        let h = h_in::<RootInertial>(DVec3::new(0.0, -inc.sin(), inc.cos()));
        for (sx, sy, sz) in [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 2.0, -0.5),
            (0.0, 0.0, -1.0),
        ] {
            let sun = DVec3::new(sx, sy, sz).m_at::<RootInertial>();
            let beta = solar_beta_angle_typed(h, sun);
            let rad = beta.get::<radian>();
            assert!(
                rad.abs() <= PI / 2.0 + 1e-15,
                "beta out of range: {rad} for sun=({sx},{sy},{sz})"
            );
        }
    }

    #[test]
    fn typed_iss_like_orbit() {
        // Mirror of `iss_like_orbit` but through the typed API. Tolerance
        // matches the f64 test exactly.
        let inc = 51.6_f64.to_radians();
        let h_raw = DVec3::new(0.0, -inc.sin(), inc.cos());
        let sun_raw = DVec3::new(1.0, 0.0, 0.0);

        let beta =
            solar_beta_angle_typed(h_in::<RootInertial>(h_raw), sun_raw.m_at::<RootInertial>());
        assert!(beta.get::<radian>().abs() < 1e-14);
    }

    #[test]
    fn typed_bit_identical_to_f64() {
        // The typed variant must delegate verbatim to the f64 variant.
        // Check a range of non-trivial inputs (orthogonal, oblique, nearly
        // aligned, negative, unnormalized) and assert bit-exact equality
        // between `solar_beta_angle_typed(...).get::<radian>()` and
        // `solar_beta_angle(...)`.
        let cases: &[(DVec3, DVec3)] = &[
            (DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.0, 0.0, 0.0)),
            (DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.0, 0.0, 1.0)),
            (DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.0, 0.0, -1.0)),
            (DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.0, 0.0, 1.0)),
            (DVec3::new(0.0, 0.0, 1e10), DVec3::new(1e8, 0.0, 0.0)),
            (DVec3::new(1.0, 2.0, 3.0), DVec3::new(-0.5, 0.25, 0.75)),
            (DVec3::new(-1.0, 4.0, -2.0), DVec3::new(7.0, -3.0, 1.0)),
        ];

        for &(h, sun) in cases {
            let f64_result = solar_beta_angle(h, sun);
            let typed_result =
                solar_beta_angle_typed(h_in::<RootInertial>(h), sun.m_at::<RootInertial>());
            assert_eq!(
                typed_result.get::<radian>(),
                f64_result,
                "typed != f64 for h={h:?} sun={sun:?}"
            );
        }
    }
}
