//! Declaratively driven kinematic joints.
//!
//! Pure-math kernel for advancing a single-axis kinematic joint under a
//! prescribed angular rate. The joint frame's rotation about its parent
//! is `θ(t) = θ₀ + ω · t` about a fixed axis expressed in parent-frame
//! coordinates; the angular velocity in this-frame coordinates is the
//! constant `ω · axis_in_this`. Because the joint is a pure rotation
//! about a fixed axis (the axis is parallel to itself in both parent
//! and this frames — only its direction rotates with the joint), the
//! "axis in this-frame coordinates" is the same vector as
//! "axis in parent-frame coordinates" for the kinematic-velocity
//! component.
//!
//! "Kinematic" here means: the angle (and thus the rotation and
//! angular velocity of the joint frame) is an *input*, not an
//! integrated state. There is no torque, no inertia, no momentum
//! exchange — the consumer simply specifies the angular trajectory
//! and this kernel computes the corresponding `(rotation, angular
//! velocity)` snapshot. Joint dynamics (free-swinging joints under
//! torque, inverse dynamics, constraint-derived joint forces) are
//! explicitly out of scope here and tracked separately under the
//! deferred-dynamics meta.
//!
//! The kernel is the analog of `planet_fixed_rotation_system`'s spin
//! about the planet pole, generalized to an arbitrary axis declared
//! per joint. JEOD has no native joint primitive — articulated
//! sub-trees in JEOD are wired through `attach_to_frame` (kinematic
//! attachment to a non-body reference frame) plus mission-code
//! handlers that update the attached frame's rotation each step. This
//! kernel formalises that mission-code pattern as a first-class API
//! so a Bevy mission crate can declare "this joint rotates about Z at
//! 10 deg/s" without writing a per-mission kinematics driver.

use glam::DVec3;
use jeod_math::JeodQuat;

/// Declarative specification for a kinematically driven single-axis
/// joint.
///
/// The joint frame's rotation about its parent is
/// `θ(t) = initial_angle_rad + rate_rad_per_s · t`, applied about
/// `axis_in_parent` (a unit vector in the parent frame).
///
/// Sign convention follows JEOD's left-transformation quaternion:
/// `q.scalar = cos(θ/2)`, `q.vector = -sin(θ/2) · axis`. A positive
/// `rate_rad_per_s` therefore rotates a vector expressed in the parent
/// frame *negatively* about `axis_in_parent` when re-expressed in the
/// child (this-) frame, matching the rest of the codebase's
/// quaternion convention.
///
/// The angular velocity expressed in this-frame coordinates is
/// `rate_rad_per_s · axis_in_parent` (the rotation axis of a single-
/// axis joint is invariant under that same rotation, so the parent
/// and this representations of the *axis direction* coincide).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointKinematicsSpec {
    /// Unit vector in the parent frame about which the joint rotates.
    /// Caller must supply a unit-norm vector — the kernel asserts this
    /// in [`evaluate`] so a mis-spec fails loudly rather than producing
    /// silently scaled rotation.
    pub axis_in_parent: DVec3,
    /// Prescribed angular rate (rad/s). May be negative.
    pub rate_rad_per_s: f64,
    /// Joint angle at `t = 0` in radians.
    pub initial_angle_rad: f64,
}

/// Tolerance on `axis_in_parent.length_squared() - 1` accepted by
/// [`evaluate`].
///
/// The kernel thresholds the squared distance from unit norm
/// (`|‖a‖² − 1| < AXIS_NORM_TOL`) rather than the linear distance
/// `|‖a‖ − 1|` because the squared form is cheaper (no `sqrt`) and is
/// the standard pattern in `glam`/`nalgebra`. Near `‖a‖ = 1` the two
/// forms differ by a factor of two to first order
/// (`‖a‖² − 1 = (‖a‖ − 1)(‖a‖ + 1) ≈ 2(‖a‖ − 1)`), so a caller that
/// pre-validates against a linear `‖a‖ − 1` bound must square the
/// comparison or scale the threshold to match.
///
/// Loose enough that float-arithmetic round-off in user-constructed
/// axes (e.g. normalising a 3-vector once at vehicle setup) does not
/// fire; tight enough that an obviously non-unit input (e.g. a raw
/// `[0, 0, 2]`) panics. Same order of magnitude as
/// `jeod_math::quaternion::NORM_LIMIT` (the JEOD fast-path bound on
/// quaternion magnitude).
///
/// Exposed as part of the public surface so callers that pre-validate
/// axes upstream of [`evaluate`] can match the kernel's exact bound
/// rather than hard-coding a parallel literal that might drift.
pub const AXIS_NORM_TOL: f64 = 1.0e-6;

/// Evaluate the joint kinematics at elapsed time `t` (seconds since
/// the joint's `initial_angle_rad` reference).
///
/// Returns:
/// - the left-transformation quaternion `q_parent_this` that maps a
///   vector expressed in the parent frame into this-frame
///   coordinates;
/// - the angular velocity of this frame relative to its parent,
///   expressed in this-frame coordinates (rad/s).
///
/// # Panics
///
/// - `axis_in_parent` is not a unit vector (within
///   [`AXIS_NORM_TOL`]). A non-unit axis would silently scale the
///   resulting rotation angle and produce an angular velocity whose
///   magnitude no longer matches `rate_rad_per_s`.
/// - `t` is non-finite. Joint angle would propagate `NaN`/`±∞` into
///   downstream consumers.
/// - `rate_rad_per_s` or `initial_angle_rad` are non-finite, for the
///   same reason.
pub fn evaluate(spec: &JointKinematicsSpec, elapsed_seconds: f64) -> (JeodQuat, DVec3) {
    let axis_norm_sq = spec.axis_in_parent.length_squared();
    assert!(
        (axis_norm_sq - 1.0).abs() < AXIS_NORM_TOL,
        "JointKinematicsSpec.axis_in_parent must be a unit vector \
         (found length² = {axis_norm_sq}, axis = {:?}). Normalize the \
         axis once at vehicle-setup time, e.g. \
         `axis_in_parent: DVec3::Z` or `axis_in_parent: raw.normalize()`.",
        spec.axis_in_parent,
    );
    assert!(
        elapsed_seconds.is_finite(),
        "joint kinematics elapsed_seconds must be finite, got {elapsed_seconds}. \
         The simulation time has gone non-finite — fix the time-update path."
    );
    assert!(
        spec.rate_rad_per_s.is_finite(),
        "JointKinematicsSpec.rate_rad_per_s must be finite, got {}. \
         Replace the spec with a finite rate or remove the joint.",
        spec.rate_rad_per_s,
    );
    assert!(
        spec.initial_angle_rad.is_finite(),
        "JointKinematicsSpec.initial_angle_rad must be finite, got {}. \
         Replace the spec with a finite angle.",
        spec.initial_angle_rad,
    );

    let angle = spec.initial_angle_rad + spec.rate_rad_per_s * elapsed_seconds;
    let q_parent_this = JeodQuat::left_quat_from_eigen_rotation(angle, spec.axis_in_parent);
    // The rotation axis of a single-axis joint is invariant under that
    // same rotation, so the axis vector is identical in parent and
    // this-frame coordinates. The angular velocity in this-frame
    // coordinates is therefore `rate · axis`.
    let ang_vel_this = spec.axis_in_parent * spec.rate_rad_per_s;
    (q_parent_this, ang_vel_this)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Identity case: zero rate, zero initial angle ⇒ identity rotation
    /// and zero angular velocity at any time.
    #[test]
    fn evaluate_zero_rate_zero_initial_angle_is_identity() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.0,
            initial_angle_rad: 0.0,
        };
        let (q, omega) = evaluate(&spec, 12_345.678_9);
        assert_eq!(q, JeodQuat::identity());
        assert_eq!(omega, DVec3::ZERO);
    }

    /// Closed-form: angle(t) = initial + rate * t. A 10 deg/s joint
    /// starting at 0 ° hits π/2 after π/(2·rate) seconds.
    #[test]
    fn evaluate_constant_rate_quarter_turn_about_z() {
        let rate = 10.0_f64.to_radians();
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: rate,
            initial_angle_rad: 0.0,
        };
        let t_quarter = (PI / 2.0) / rate;
        let (q, _omega) = evaluate(&spec, t_quarter);
        let expected = JeodQuat::left_quat_from_eigen_rotation(PI / 2.0, DVec3::Z);
        // Bit-identity expected (same closed-form expression, same axis).
        assert_eq!(q, expected);
    }

    /// Angular velocity is `rate · axis` (in this-frame coordinates).
    /// Sign of `rate` flips the angular-velocity vector.
    #[test]
    fn evaluate_angular_velocity_tracks_rate_and_axis() {
        let axis = DVec3::Y;
        for rate in &[-3.5, 0.0, 1.25] {
            let spec = JointKinematicsSpec {
                axis_in_parent: axis,
                rate_rad_per_s: *rate,
                initial_angle_rad: 0.0,
            };
            let (_q, omega) = evaluate(&spec, 0.0);
            assert_eq!(omega, axis * *rate);
            // And again at a non-zero time — angular velocity is constant
            // for a constant-rate joint.
            let (_q2, omega2) = evaluate(&spec, 7.5);
            assert_eq!(omega2, axis * *rate);
        }
    }

    /// `initial_angle_rad` shifts the angle at `t = 0`.
    #[test]
    fn evaluate_respects_initial_angle() {
        let theta0 = 0.4;
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::X,
            rate_rad_per_s: 0.0,
            initial_angle_rad: theta0,
        };
        let (q, _omega) = evaluate(&spec, 0.0);
        let expected = JeodQuat::left_quat_from_eigen_rotation(theta0, DVec3::X);
        assert_eq!(q, expected);
    }

    /// JEOD left-quat sign convention: the vector part is
    /// `-sin(θ/2)·axis`, so a positive rotation about `+Z` produces a
    /// quaternion with negative `z` component.
    #[test]
    fn evaluate_left_quat_sign_convention() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 1.0,
            initial_angle_rad: 0.0,
        };
        let (q, _omega) = evaluate(&spec, 1.0); // angle = 1 rad
        let half = 0.5_f64;
        let scalar = half.cos();
        let zv = -half.sin();
        let dq = q.to_glam();
        assert!((dq.w - scalar).abs() < 1.0e-15);
        assert!((dq.z - zv).abs() < 1.0e-15);
    }

    /// Non-unit axis must panic — silently rescaling the rotation
    /// angle would produce a wrong physics answer with no diagnostic.
    #[test]
    #[should_panic(expected = "must be a unit vector")]
    fn evaluate_panics_on_non_unit_axis() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
            rate_rad_per_s: 1.0,
            initial_angle_rad: 0.0,
        };
        let _ = evaluate(&spec, 0.0);
    }

    #[test]
    #[should_panic(expected = "elapsed_seconds must be finite")]
    fn evaluate_panics_on_nan_elapsed() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 1.0,
            initial_angle_rad: 0.0,
        };
        let _ = evaluate(&spec, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "rate_rad_per_s must be finite")]
    fn evaluate_panics_on_nan_rate() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: f64::NAN,
            initial_angle_rad: 0.0,
        };
        let _ = evaluate(&spec, 0.0);
    }

    #[test]
    #[should_panic(expected = "initial_angle_rad must be finite")]
    fn evaluate_panics_on_nan_initial_angle() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.0,
            initial_angle_rad: f64::INFINITY,
        };
        let _ = evaluate(&spec, 0.0);
    }

    /// Round-trip: the returned quaternion's transformation matrix
    /// applied to `axis_in_parent` returns the same axis (rotation
    /// about the axis fixes the axis).
    #[test]
    fn evaluate_axis_is_rotation_eigenvector() {
        let axis = DVec3::new(1.0, 1.0, 1.0).normalize();
        let spec = JointKinematicsSpec {
            axis_in_parent: axis,
            rate_rad_per_s: 0.7,
            initial_angle_rad: 0.2,
        };
        let (q, _omega) = evaluate(&spec, 3.0);
        let mat = q.left_quat_to_transformation();
        let rotated = mat * axis;
        for i in 0..3 {
            assert!(
                (rotated[i] - axis[i]).abs() < 1.0e-12,
                "axis component {i}: rotated={} axis={}",
                rotated[i],
                axis[i],
            );
        }
    }
}
