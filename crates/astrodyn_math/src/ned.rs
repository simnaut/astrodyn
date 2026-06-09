//! Runtime North-East-Down (NED) frame for a moving body — the per-step
//! analog of JEOD's `NedDerivedState`
//! (`models/dynamics/derived_state/src/ned_derived_state.cc`).
//!
//! Where [`crate::topocentric`] builds a *fixed-site* ENU rotation and the
//! `astrodyn_frames` typed pose anchors a fixed site, this module tracks the
//! NED frame at a **subject body's instantaneous sub-point**: each step the
//! frame origin is the body's planet-fixed (pfix) position, and the axes are
//! the NED triad at that point's geodetic latitude/longitude.
//!
//! JEOD computes this as `compute_relative_state(pfix)` (body state relative to
//! the rotating pfix frame) followed by `build_ned_orientation()`. The NED
//! frame is **stationary wrt pfix** (zero rate); its parent is the pfix frame,
//! not the inertial frame (contrast [`crate::lvlh::LvlhFrame`], whose parent is
//! the planet-inertial frame and which rotates at orbital rate).

use glam::{DMat3, DVec3};

use astrodyn_quantities::aliases::Position;
use astrodyn_quantities::frame::{Planet, PlanetInertial};
use uom::si::f64::Length;
use uom::si::length::meter;

use crate::geodetic::GeodeticState;
use crate::topocentric::local_level_ned_axes;
use crate::types::mat3_from_rows;

/// Runtime NED frame pose: the NED frame of a body's current sub-point,
/// expressed relative to the planet-fixed (pfix) parent frame.
///
/// Carries the **origin** (the body's pfix position — JEOD's
/// `ned_state.cart_coords`) and the **orientation** (NED axes at the
/// sub-point), plus the structurally-zero rate. Mirrors
/// [`crate::lvlh::LvlhFrame`], but the parent frame is **pfix** (planet-fixed),
/// not inertial, and the frame is stationary wrt that parent.
///
/// # Deferred: origin velocity
///
/// JEOD's NED frame also carries an origin *velocity* — the body's
/// pfix-relative velocity `R·v_inertial − ω_pfix × r_pfix`
/// (`set_ned_trans_states`). It is **not** modelled here: JEOD's SIM_NED never
/// logs it (so there is no cross-validation reference), and computing it
/// requires the planet's rotation *rate* `ω_pfix`, which the Bevy adapter does
/// not currently surface on the planet entity (only the rotation matrix, via
/// `PlanetFixedRotationC`). Adding it would need planet-rate plumbing across
/// both adapters; deferred to keep the runner and Bevy outputs in parity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NedFrame {
    /// Transformation matrix from the pfix parent frame to the NED frame.
    /// Rows are the North/East/Down axes expressed in pfix
    /// (JEOD `ned_frame.state.rot.T_parent_this`).
    pub t_parent_this: DMat3,
    /// Angular velocity of the NED frame wrt pfix, in NED coordinates (rad/s).
    /// **Always zero** — the NED frame is fixed to the rotating planet, so it
    /// has no rate relative to pfix (JEOD zeroes `ang_vel_this` in
    /// `build_ned_orientation`). Kept explicit to document the invariant.
    pub ang_vel_this: DVec3,
    /// NED frame origin (the body's sub-point) in pfix coordinates (m).
    /// JEOD `ned_state.cart_coords` / `ned_frame.state.trans.position`.
    pub position: DVec3,
}

impl Default for NedFrame {
    fn default() -> Self {
        Self {
            t_parent_this: DMat3::IDENTITY,
            ang_vel_this: DVec3::ZERO,
            position: DVec3::ZERO,
        }
    }
}

/// Build the runtime [`NedFrame`] pose for a body from its inertial position
/// and the planet-fixed frame's rotation.
///
/// Port of JEOD `NedDerivedState::update` → `compute_ned_frame` (the
/// origin/orientation part): the body's pfix position is taken (the
/// `compute_relative_state(pfix)` position), then the NED axes are built from
/// the geodetic latitude/longitude of that sub-point (`build_ned_orientation`).
/// The pfix frame shares the planet-inertial origin (planet centre), so the
/// body's pfix position is simply `R·r_inertial`. See the [`NedFrame`] docs for
/// the deferred origin velocity.
///
/// # Arguments
/// * `position` — body position in the planet-inertial frame (m).
/// * `t_inertial_pfix` — inertial→pfix rotation (the pfix node's
///   `t_parent_this`).
/// * `r_eq` / `r_pol` — planet equatorial / polar radii (m).
///
/// Geodetic preconditions (`PF.01`–`PF.05`: position far from centre, finite
/// input, Borkowski convergence) are inherited from
/// [`GeodeticState::from_planet_fixed`].
pub fn compute_body_ned_frame(
    position: DVec3,
    t_inertial_pfix: &DMat3,
    r_eq: f64,
    r_pol: f64,
) -> NedFrame {
    // Body position relative to the rotating pfix frame (origins coincide at
    // the planet centre).
    let pos_pfix = *t_inertial_pfix * position;

    // NED axes at the sub-point's geodetic latitude/longitude. Same shared
    // kernel as `astrodyn_dynamics::compute_ned_rotation` and the ENU builder.
    let geo = GeodeticState::from_planet_fixed(pos_pfix, r_eq, r_pol);
    let (north, east, down) = local_level_ned_axes(geo.latitude, geo.longitude);
    let t_pfix_ned = mat3_from_rows(north, east, down);

    NedFrame {
        t_parent_this: t_pfix_ned,
        ang_vel_this: DVec3::ZERO,
        position: pos_pfix,
    }
}

/// Typed sibling of [`compute_body_ned_frame`]. Accepts a typed planet-inertial
/// position and `uom` radii; bit-identical numerics to the f64 surface (both
/// drop to `raw_si()` at the kernel boundary).
pub fn compute_body_ned_frame_typed<P: Planet>(
    position: Position<PlanetInertial<P>>,
    t_inertial_pfix: &DMat3,
    r_eq: Length,
    r_pol: Length,
) -> NedFrame {
    compute_body_ned_frame(
        position.raw_si(),
        t_inertial_pfix,
        r_eq.get::<meter>(),
        r_pol.get::<meter>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const R_EQ: f64 = 6_378_137.0;
    const R_POL: f64 = 6_356_752.314_245;

    // With an identity inertial→pfix rotation the NED origin is just the
    // inertial position, and the orientation matches the shared NED kernel at
    // the sub-point's geodetic lat/lon. The frame is stationary wrt pfix.
    #[test]
    fn identity_pfix_origin_and_axes() {
        let r = DVec3::new(6_500_000.0, 200_000.0, 3_000_000.0);
        let ned = compute_body_ned_frame(r, &DMat3::IDENTITY, R_EQ, R_POL);
        assert_eq!(
            ned.position, r,
            "origin must be R·r_inertial (= r at identity)"
        );
        assert_eq!(ned.ang_vel_this, DVec3::ZERO, "NED rate wrt pfix is zero");

        let geo = GeodeticState::from_planet_fixed(r, R_EQ, R_POL);
        let (n, e, d) = local_level_ned_axes(geo.latitude, geo.longitude);
        assert_eq!(ned.t_parent_this, mat3_from_rows(n, e, d));
    }

    // A non-identity pfix rotation rotates both the origin and the axes: the
    // origin is `R·r`, and the NED axes are built at the geodetic lat/lon of
    // that rotated sub-point (so the orientation matches the shared kernel
    // evaluated on `R·r`, not on the inertial `r`).
    #[test]
    fn rotated_pfix_uses_subpoint_axes() {
        let r = DVec3::new(5_000_000.0, -3_000_000.0, 2_500_000.0);
        // 30° about z as the inertial→pfix rotation.
        let (s, c) = (30.0_f64).to_radians().sin_cos();
        let t = mat3_from_rows(
            DVec3::new(c, s, 0.0),
            DVec3::new(-s, c, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let ned = compute_body_ned_frame(r, &t, R_EQ, R_POL);
        assert_eq!(ned.position, t * r, "origin must be R·r");
        let geo = GeodeticState::from_planet_fixed(t * r, R_EQ, R_POL);
        let (n, e, d) = local_level_ned_axes(geo.latitude, geo.longitude);
        assert_eq!(ned.t_parent_this, mat3_from_rows(n, e, d));
    }
}
