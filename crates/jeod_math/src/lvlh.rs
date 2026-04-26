//! LVLH (Local Vertical Local Horizontal) frame computation.
//!
//! Faithful port of JEOD's `lvlh_frame.cc:247-285`.
//!
//! The LVLH frame is defined relative to a planet-centered inertial frame:
//! - Z-hat = -r̂ (nadir, toward planet center)
//! - Y-hat = -ĥ (negative orbit normal)
//! - X-hat = Y-hat × Z-hat (approximately along velocity for circular orbits)
//!
//! The frame origin is co-located and co-moving with the vehicle.

use glam::{DMat3, DVec3};

use jeod_quantities::aliases::{Position, Velocity};
use jeod_quantities::frame::{Inertial, Lvlh, Vehicle};
use jeod_quantities::frame_transform::FrameTransform;
use jeod_quantities::quat::{JeodQuat, NormalizedQuat};

/// LVLH frame state: rotation matrix, angular velocity, and origin position/velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LvlhFrame {
    /// Transformation matrix from parent (inertial) frame to LVLH frame.
    /// Rows are the LVLH frame axes expressed in the inertial frame.
    pub t_parent_this: DMat3,
    /// Angular velocity of the LVLH frame in the LVLH frame (rad/s).
    pub ang_vel_this: DVec3,
    /// Position of the LVLH frame origin (same as vehicle) in the inertial frame (m).
    pub position: DVec3,
    /// Velocity of the LVLH frame origin (same as vehicle) in the inertial frame (m/s).
    pub velocity: DVec3,
}

impl Default for LvlhFrame {
    fn default() -> Self {
        Self {
            t_parent_this: DMat3::IDENTITY,
            ang_vel_this: DVec3::ZERO,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }
    }
}

impl LvlhFrame {
    /// Build a complete `LvlhFrame` (rotation, angular velocity, origin
    /// position/velocity) from typed inertial position and velocity.
    ///
    /// This is the typed entry point for callers that need the full
    /// struct, including the auxiliary fields that
    /// [`compute_lvlh_frame_typed`] discards.
    ///
    /// Bit-identical numerics to [`compute_lvlh_frame_typed`] — both share
    /// the [`compute_lvlh_frame_impl`] kernel.
    ///
    /// # Panics
    /// Panics if position or angular momentum magnitude is zero.
    pub fn compute(position: Position<Inertial>, velocity: Velocity<Inertial>) -> Self {
        compute_lvlh_frame_impl(position.raw_si(), velocity.raw_si())
    }
}

/// Compute the LVLH frame from position and velocity in a planet-centered inertial frame.
///
/// Port of JEOD `LvlhFrame::compute_lvlh_frame()` (lvlh_frame.cc:247-285).
///
/// Numeric kernel shared by the typed [`compute_lvlh_frame_typed`] entry
/// point and the [`LvlhFrame::compute`] full-state constructor (used when
/// the auxiliary fields — angular velocity, origin position/velocity —
/// are needed in addition to the orientation that the typed sibling
/// returns).
///
/// New callers that only need the inertial→LVLH orientation should use
/// [`compute_lvlh_frame_typed`]; callers that need the full struct should
/// use [`LvlhFrame::compute`].
///
/// # Arguments
/// * `position` - Vehicle position in planet-centered inertial frame (m)
/// * `velocity` - Vehicle velocity in planet-centered inertial frame (m/s)
///
/// # Panics
/// Panics if position or angular momentum magnitude is zero.
fn compute_lvlh_frame_impl(position: DVec3, velocity: DVec3) -> LvlhFrame {
    // Compute angular momentum vector: h = r × v
    let angmom = position.cross(velocity);
    let hmag = angmom.length();
    let rmagsq = position.length_squared();
    let rmag = rmagsq.sqrt();

    assert!(rmag > 0.0, "compute_lvlh_frame: position magnitude is zero");
    assert!(
        hmag > 0.0,
        "compute_lvlh_frame: angular momentum is zero (radial trajectory)"
    );

    // Orbital angular velocity magnitude
    let wmag = hmag / rmagsq;

    // LVLH frame axes (rows of T_parent_this):
    //   z_hat = -r/rmag (nadir, toward planet center)
    //   y_hat = -h/hmag (negative orbit normal)
    //   x_hat = y_hat × z_hat (forward, approximately along velocity)
    let z_hat = -position / rmag;
    let y_hat = -angmom / hmag;
    let x_hat = y_hat.cross(z_hat).normalize();

    // T_parent_this is a row-major transformation matrix where rows are frame axes
    // In glam DMat3 (column-major), we construct from columns then transpose,
    // or equivalently construct from rows using from_cols on the transposed data.
    //
    // JEOD stores T_parent_this[0] = x_hat, T_parent_this[1] = y_hat, T_parent_this[2] = z_hat
    // These are rows of the transformation matrix. In glam's column-major DMat3:
    let t_parent_this = DMat3::from_cols(
        DVec3::new(x_hat.x, y_hat.x, z_hat.x),
        DVec3::new(x_hat.y, y_hat.y, z_hat.y),
        DVec3::new(x_hat.z, y_hat.z, z_hat.z),
    );

    // Angular velocity of the LVLH frame in the LVLH frame:
    // ω = [0, -wmag, 0] (rotation about the -Y axis at orbital rate)
    let ang_vel_this = DVec3::new(0.0, -wmag, 0.0);

    LvlhFrame {
        t_parent_this,
        ang_vel_this,
        position,
        velocity,
    }
}

/// Typed sibling of [`compute_lvlh_frame`]: returns a `FrameTransform` from
/// the planet-centered inertial frame to the chief vehicle's LVLH frame.
///
/// Delegates to the f64-based computation, then wraps the resulting rotation
/// matrix (`T_parent_this` — inertial → LVLH, rows are LVLH axes in the
/// inertial frame) as a canonical JEOD scalar-first, left-transformation
/// quaternion inside the `FrameTransform`.
///
/// Angular velocity, origin position, and origin velocity produced by the
/// underlying LVLH computation are discarded here — this function returns
/// only the frame orientation. Callers that need those auxiliary quantities
/// should continue to use the (deprecated) f64 API until Phase 3+ migrates
/// the full `LvlhFrame` struct to typed quantities.
///
/// # Arguments
/// * `position` - Chief vehicle position in the planet-centered inertial frame.
/// * `velocity` - Chief vehicle velocity in the planet-centered inertial frame.
///
/// # Panics
/// Panics if position or angular momentum magnitude is zero (radial trajectory).
pub fn compute_lvlh_frame_typed<Chief>(
    position: Position<Inertial>,
    velocity: Velocity<Inertial>,
) -> FrameTransform<Inertial, Lvlh<Chief>>
where
    Chief: Vehicle,
{
    // Delegate to the shared kernel so physics stays in one place.
    let lvlh = compute_lvlh_frame_impl(position.raw_si(), velocity.raw_si());

    // Convert the transformation matrix T_parent_this (inertial → LVLH) into
    // a canonical JEOD left quaternion. JEOD's `left_quat_from_transformation`
    // extractor already re-normalizes internally, so we wrap the result with
    // `NormalizedQuat::new` — that witnesses the existing unit norm without
    // re-dividing by `‖q‖`, preserving bit-identity with the f64 surface.
    let q: JeodQuat = JeodQuat::left_quat_from_transformation(&lvlh.t_parent_this);
    let q_norm = NormalizedQuat::new(q)
        .unwrap_or_else(|err| panic!("left_quat_from_transformation guarantees unit norm: {err}"));
    FrameTransform::<Inertial, Lvlh<Chief>>::from_quat(q_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test alias so the existing test bodies keep their compact `compute_lvlh_frame`
    // call sites after the Phase 10 purge of the bare-`f64` public surface.
    use compute_lvlh_frame_impl as compute_lvlh_frame;

    const EARTH_MU: f64 = 3.986_004_415e14; // m^3/s^2

    #[test]
    fn circular_equatorial_orbit() {
        // Circular orbit at 400 km altitude, equatorial
        let r = 6_778_137.0; // r_eq + 400 km
        let v = (EARTH_MU / r).sqrt(); // circular velocity

        let position = DVec3::new(r, 0.0, 0.0);
        let velocity = DVec3::new(0.0, v, 0.0);

        let lvlh = compute_lvlh_frame(position, velocity);

        // Z-hat should point toward planet center (nadir): -r/|r| = [-1, 0, 0]
        let z_hat = DVec3::new(
            lvlh.t_parent_this.col(0).z,
            lvlh.t_parent_this.col(1).z,
            lvlh.t_parent_this.col(2).z,
        );
        assert!((z_hat.x + 1.0).abs() < 1e-14, "Z-hat x: {}", z_hat.x);
        assert!(z_hat.y.abs() < 1e-14, "Z-hat y: {}", z_hat.y);
        assert!(z_hat.z.abs() < 1e-14, "Z-hat z: {}", z_hat.z);

        // Y-hat should be negative orbit normal: -(r×v)/|r×v| = [0, 0, -1]
        let y_hat = DVec3::new(
            lvlh.t_parent_this.col(0).y,
            lvlh.t_parent_this.col(1).y,
            lvlh.t_parent_this.col(2).y,
        );
        assert!(y_hat.x.abs() < 1e-14, "Y-hat x: {}", y_hat.x);
        assert!(y_hat.y.abs() < 1e-14, "Y-hat y: {}", y_hat.y);
        assert!((y_hat.z + 1.0).abs() < 1e-14, "Y-hat z: {}", y_hat.z);

        // X-hat should be along velocity direction: [0, 1, 0]
        let x_hat = DVec3::new(
            lvlh.t_parent_this.col(0).x,
            lvlh.t_parent_this.col(1).x,
            lvlh.t_parent_this.col(2).x,
        );
        assert!(x_hat.x.abs() < 1e-14, "X-hat x: {}", x_hat.x);
        assert!((x_hat.y - 1.0).abs() < 1e-14, "X-hat y: {}", x_hat.y);
        assert!(x_hat.z.abs() < 1e-14, "X-hat z: {}", x_hat.z);

        // Angular velocity should be [0, -n, 0] where n = orbital rate
        let n = (EARTH_MU / (r * r * r)).sqrt();
        assert!(lvlh.ang_vel_this.x.abs() < 1e-14);
        assert!((lvlh.ang_vel_this.y + n).abs() / n < 1e-10);
        assert!(lvlh.ang_vel_this.z.abs() < 1e-14);
    }

    #[test]
    fn transformation_is_orthonormal() {
        let position = DVec3::new(5e6, 3e6, 4e6);
        let velocity = DVec3::new(-2000.0, 5000.0, 3000.0);

        let lvlh = compute_lvlh_frame(position, velocity);
        let t = lvlh.t_parent_this;

        // T * T^T should be identity (orthogonal matrix)
        let product = t * t.transpose();
        let diff = product - DMat3::IDENTITY;
        assert!(diff.x_axis.length() < 1e-14);
        assert!(diff.y_axis.length() < 1e-14);
        assert!(diff.z_axis.length() < 1e-14);

        // Determinant should be +1 (proper rotation)
        assert!((t.determinant() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn inclined_orbit() {
        // ISS-like inclined orbit
        let r = 6_778_137.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();

        let position = DVec3::new(r, 0.0, 0.0);
        let velocity = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        let lvlh = compute_lvlh_frame(position, velocity);

        // Z-hat should still point toward center: [-1, 0, 0]
        let z_hat = DVec3::new(
            lvlh.t_parent_this.col(0).z,
            lvlh.t_parent_this.col(1).z,
            lvlh.t_parent_this.col(2).z,
        );
        assert!((z_hat.x + 1.0).abs() < 1e-14);
        assert!(z_hat.y.abs() < 1e-14);
        assert!(z_hat.z.abs() < 1e-14);

        // Angular velocity magnitude should match orbital rate
        let n = (EARTH_MU / (r * r * r)).sqrt();
        assert!((lvlh.ang_vel_this.length() - n).abs() / n < 1e-10);

        // Origin matches vehicle
        assert_eq!(lvlh.position, position);
        assert_eq!(lvlh.velocity, velocity);
    }

    #[test]
    fn half_orbit_frame_rotates() {
        // At position [r, 0, 0], LVLH Z = [-1, 0, 0]
        // At position [0, r, 0], LVLH Z = [0, -1, 0]
        let r = 6_778_137.0;
        let v = (EARTH_MU / r).sqrt();

        let pos2 = DVec3::new(0.0, r, 0.0);
        let vel2 = DVec3::new(-v, 0.0, 0.0);
        let lvlh2 = compute_lvlh_frame(pos2, vel2);

        let z_hat = DVec3::new(
            lvlh2.t_parent_this.col(0).z,
            lvlh2.t_parent_this.col(1).z,
            lvlh2.t_parent_this.col(2).z,
        );
        assert!(z_hat.x.abs() < 1e-14);
        assert!((z_hat.y + 1.0).abs() < 1e-14);
        assert!(z_hat.z.abs() < 1e-14);
    }

    // Re-alias the sealed-but-test-only chief marker from `jeod_quantities`.
    // `Vehicle` is sealed, so we can't implement it here — the `test-utils`
    // feature on `jeod_quantities` (dev-dep in our Cargo.toml) opens up
    // `TestVehicle` specifically for tests like this.
    use jeod_quantities::frame::TestVehicle as IssChief;

    #[test]
    fn typed_lvlh_matches_f64_port() {
        // ISS-like inclined state in planet-centered inertial coordinates.
        let r = 6_778_137.0; // ~ r_eq + 400 km
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();

        let position_raw = DVec3::new(r, 0.0, 0.0);
        let velocity_raw = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        // f64 baseline: derive the JEOD canonical quaternion from the old
        // rotation-matrix output via the same extractor the typed variant uses.
        let lvlh_f64 = compute_lvlh_frame(position_raw, velocity_raw);
        let q_f64 = JeodQuat::left_quat_from_transformation(&lvlh_f64.t_parent_this);

        // Typed sibling: build Position/Velocity and call compute_lvlh_frame_typed.
        let position = Position::<Inertial>::from_raw_si(position_raw);
        let velocity = Velocity::<Inertial>::from_raw_si(velocity_raw);
        let xform = compute_lvlh_frame_typed::<IssChief>(position, velocity);

        // Quaternion components must match bit-exactly — same extractor, same
        // input matrix, same renormalization code path.
        let q_typed = xform.quat().inner();
        assert_eq!(
            q_typed.data, q_f64.data,
            "typed and f64 quaternion components must be bit-identical"
        );

        // Matrix cached inside the FrameTransform must round-trip through the
        // quaternion → matrix derivation without drifting from orthonormality.
        let m = xform.matrix();
        let should_be_identity = m * m.transpose();
        let diff = should_be_identity - DMat3::IDENTITY;
        assert!(diff.x_axis.length() < 1e-14);
        assert!(diff.y_axis.length() < 1e-14);
        assert!(diff.z_axis.length() < 1e-14);
        assert!((m.determinant() - 1.0).abs() < 1e-14);
    }
}
