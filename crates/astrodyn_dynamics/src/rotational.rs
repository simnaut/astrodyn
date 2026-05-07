//! Rotational state: attitude quaternion, angular velocity, and the
//! integration kernels that propagate them under applied torque.
//!
//! Mirrors the rotational side of JEOD's
//! [`models/dynamics/dyn_body/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/)
//! integrator (v5.4.0). Quaternions follow the JEOD scalar-first
//! left-transformation convention; the typed siblings live in
//! [`astrodyn_quantities::body_attitude`].

use core::marker::PhantomData;

use crate::state::TranslationalState;
use astrodyn_math::quaternion::NORM_LIMIT;
use astrodyn_math::JeodQuat;
use astrodyn_quantities::aliases::AngularVelocity;
use astrodyn_quantities::body_attitude::BodyAttitude;
use astrodyn_quantities::frame::{BodyFrame, Vehicle};
use astrodyn_quantities::quat::NormalizedQuat;
use glam::{DMat3, DVec3};

/// Rotational state of a rigid body.
///
/// The quaternion is a scalar-first, left-transformation quaternion
/// (JEOD convention) describing the parent-to-body rotation.
/// Angular velocity is in rad/s, expressed in the body frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationalState {
    /// Scalar-first, left-transformation quaternion: parent-to-body.
    pub quaternion: JeodQuat,
    /// Angular velocity in rad/s, expressed in body frame.
    pub ang_vel_body: DVec3,
}

impl Default for RotationalState {
    fn default() -> Self {
        Self {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }
    }
}

/// Combined translational + rotational state for 6-DOF integration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SixDofState {
    /// Translational position + velocity.
    pub trans: TranslationalState,
    /// Attitude quaternion + angular velocity.
    pub rot: RotationalState,
}

/// Typed sibling of [`RotationalState`] parameterized by a vehicle marker
/// `V`. The attitude is a [`BodyAttitude<V>`] — the wrapper owns the
/// JEOD left-multiply convention so callers cannot integrate via raw
/// `multiply` and accidentally swap operand order (issue #252).
/// Angular velocity carries the `BodyFrame<V>` phantom tag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationalStateTyped<V: Vehicle> {
    /// RootInertial → body attitude (typed wrapper around a witnessed
    /// scalar-first / left-transformation quaternion).
    pub q_inertial_body: BodyAttitude<V>,
    /// Angular velocity in `BodyFrame<V>`.
    pub ang_vel_body: AngularVelocity<BodyFrame<V>>,
    _v: PhantomData<V>,
}

impl<V: Vehicle> Default for RotationalStateTyped<V> {
    #[inline]
    fn default() -> Self {
        Self {
            q_inertial_body: BodyAttitude::identity(),
            ang_vel_body: AngularVelocity::<BodyFrame<V>>::zero(),
            _v: PhantomData,
        }
    }
}

impl<V: Vehicle> RotationalStateTyped<V> {
    /// Construct from a typed [`BodyAttitude`] plus typed angular
    /// velocity.
    #[inline]
    pub fn new(
        q_inertial_body: BodyAttitude<V>,
        ang_vel_body: AngularVelocity<BodyFrame<V>>,
    ) -> Self {
        Self {
            q_inertial_body,
            ang_vel_body,
            _v: PhantomData,
        }
    }

    /// Drop the phantom and emit the untyped storage form. Numeric
    /// values (unitless quaternion components, rad/s for the angular
    /// velocity) are preserved exactly.
    #[inline]
    pub fn to_untyped(&self) -> RotationalState {
        RotationalState {
            quaternion: self.q_inertial_body.to_jeod_quat(),
            ang_vel_body: self.ang_vel_body.raw_si(),
        }
    }

    /// Wrap an untyped [`RotationalState`] as typed for vehicle `V`.
    /// **The caller asserts** the angular velocity is expressed in
    /// `BodyFrame<V>`. The inner quaternion is checked against
    /// [`NormalizedQuat::DEFAULT_TOLERANCE`]: panics if it has drifted
    /// more than 1e-12 from unit norm, indicating an upstream
    /// re-normalization was missed.
    pub fn from_untyped_unchecked(s: &RotationalState) -> Self {
        let q = NormalizedQuat::new(s.quaternion)
            .unwrap_or_else(|err| panic!("RotationalState quaternion is not unit-norm: {err}"));
        Self {
            q_inertial_body: BodyAttitude::from_witness(q),
            ang_vel_body: AngularVelocity::<BodyFrame<V>>::from_raw_si(s.ang_vel_body),
            _v: PhantomData,
        }
    }
}

/// Compute rotational acceleration from Euler's rigid-body equation.
///
/// Faithful port of JEOD `dyn_body_collect.cc` lines 238-267:
/// ```text
/// ang_mom = inertia * ang_vel
/// inertial_torq = ang_vel x ang_mom
/// torque_body = extern_torq_body - inertial_torq
/// rot_accel = inverse_inertia * torque_body
/// ```
/// Components with magnitude below 1e-20 are zeroed (JEOD `zero_small`).
// JEOD_INV: FD.02 — rot_accel = I^-1 * (tau - omega x I*omega)
// JEOD_INV: DB.19 — inverse_inertia used for Euler equation
pub fn compute_rotational_acceleration(
    inertia: &DMat3,
    inverse_inertia: &DMat3,
    ang_vel: DVec3,
    extern_torq_body: DVec3,
) -> DVec3 {
    // Angular momentum: L = I * omega
    let ang_mom = *inertia * ang_vel;

    // RootInertial (gyroscopic) torque: omega x L
    let inertial_torq = ang_vel.cross(ang_mom);

    // Net body-frame torque: external minus gyroscopic
    let torque_body = extern_torq_body - inertial_torq;

    // Rotational acceleration: alpha = I^-1 * torque
    let rot_accel = *inverse_inertia * torque_body;

    // JEOD_INV: DB.20 — small rot_accel truncated to zero (< 1e-20 per component)
    zero_small(rot_accel)
}

/// Zero components with magnitude below 1e-20, matching JEOD `Vector3::zero_small`.
fn zero_small(v: DVec3) -> DVec3 {
    const THRESHOLD: f64 = 1e-20;
    DVec3::new(
        if v.x.abs() < THRESHOLD { 0.0 } else { v.x },
        if v.y.abs() < THRESHOLD { 0.0 } else { v.y },
        if v.z.abs() < THRESHOLD { 0.0 } else { v.z },
    )
}

/// Compute the time derivative of a left quaternion.
///
/// Faithful port of JEOD `quat_inline.hh` lines 495-502:
/// ```text
/// mhang_vel = -0.5 * ang_vel
/// qdot[0] = -(q.vector . mhang_vel)   // scalar derivative
/// qdot[1..3] = q.scalar * mhang_vel + mhang_vel x q.vector
/// ```
///
/// Returns `[qdot_scalar, qdot_vx, qdot_vy, qdot_vz]` in JEOD scalar-first order.
pub fn compute_left_quat_deriv(q: &JeodQuat, ang_vel: DVec3) -> [f64; 4] {
    let mhang_vel = -0.5 * ang_vel;
    let qv = q.vector();
    let qs = q.scalar();

    // Scalar derivative: qdot[0] = -(qv . mhang_vel)
    let qdot_s = -qv.dot(mhang_vel);

    // Vector derivative: qdot[1..3] = qs * mhang_vel + mhang_vel x qv
    let qdot_v = qs * mhang_vel + mhang_vel.cross(qv);

    [qdot_s, qdot_v.x, qdot_v.y, qdot_v.z]
}

/// Normalize a quaternion without forcing scalar non-negative.
///
/// Faithful port of JEOD `quat_norm.cc` lines 83-101 (`normalize_integ`).
/// Uses a Pade approximant when the quaternion is near unit length,
/// and standard sqrt normalization otherwise.
///
/// Unlike `JeodQuat::normalize()`, this does NOT flip the sign to force
/// scalar >= 0. This is intentional: during integration the quaternion
/// may pass through the scalar-negative hemisphere, and forcing it back
/// would introduce discontinuities.
// JEOD_INV: DB.09 — quaternion normalized after every integration step
// JEOD_INV: RF.09 — quaternion assumed normalized for left_quat_to_transformation
pub fn normalize_integ(q: &mut JeodQuat) {
    let qmagsq = q.norm_sq();
    assert!(
        qmagsq > 0.0,
        "normalize_integ called with zero-magnitude quaternion (norm_sq == 0.0)"
    );
    let diff1 = 1.0 - qmagsq;

    let fact = if diff1 > -NORM_LIMIT && diff1 < NORM_LIMIT {
        // Near-unit: Pade approximant
        2.0 / (1.0 + qmagsq)
    } else {
        // Standard normalization
        1.0 / qmagsq.sqrt()
    };

    // Scale all 4 components
    q.data[0] *= fact;
    q.data[1] *= fact;
    q.data[2] *= fact;
    q.data[3] *= fact;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-14;

    // ---------------------------------------------------------------
    // compute_rotational_acceleration tests
    // ---------------------------------------------------------------

    /// Zero torque and zero angular velocity: rotational acceleration is zero.
    #[test]
    fn euler_zero_torque_zero_omega() {
        let inertia = DMat3::from_diagonal(DVec3::new(10.0, 20.0, 30.0));
        let inv_inertia = DMat3::from_diagonal(DVec3::new(0.1, 0.05, 1.0 / 30.0));

        let alpha =
            compute_rotational_acceleration(&inertia, &inv_inertia, DVec3::ZERO, DVec3::ZERO);
        assert!(
            alpha.length() < TOL,
            "Expected zero rot_accel, got {:?}",
            alpha,
        );
    }

    /// Zero torque, angular velocity on a principal axis:
    /// omega x (I * omega) = omega x (I_i * omega) = 0 since parallel.
    /// So rot_accel = I^-1 * (0 - 0) = 0.
    #[test]
    fn euler_zero_torque_principal_axis() {
        let inertia = DMat3::from_diagonal(DVec3::new(10.0, 20.0, 30.0));
        let inv_inertia = DMat3::from_diagonal(DVec3::new(0.1, 0.05, 1.0 / 30.0));
        let omega = DVec3::new(0.0, 0.0, 5.0); // spin about z principal axis

        let alpha = compute_rotational_acceleration(&inertia, &inv_inertia, omega, DVec3::ZERO);
        assert!(
            alpha.length() < TOL,
            "Expected zero rot_accel for principal-axis spin, got {:?}",
            alpha,
        );
    }

    /// Zero torque, off-axis angular velocity: gyroscopic coupling should
    /// produce nonzero rotational acceleration when moments differ.
    #[test]
    fn euler_zero_torque_off_axis() {
        let inertia = DMat3::from_diagonal(DVec3::new(10.0, 20.0, 30.0));
        let inv_inertia = DMat3::from_diagonal(DVec3::new(0.1, 0.05, 1.0 / 30.0));
        let omega = DVec3::new(1.0, 2.0, 3.0);

        let alpha = compute_rotational_acceleration(&inertia, &inv_inertia, omega, DVec3::ZERO);

        // Manual calculation:
        // ang_mom = [10, 40, 90]
        // inertial_torq = omega x ang_mom = [1,2,3] x [10,40,90]
        //   = [2*90 - 3*40, 3*10 - 1*90, 1*40 - 2*10]
        //   = [60, -60, 20]
        // torque_body = [0,0,0] - [60,-60,20] = [-60, 60, -20]
        // rot_accel = inv_I * torque = [-6, 3, -2/3]
        let expected = DVec3::new(-6.0, 3.0, -20.0 / 30.0);
        let diff = (alpha - expected).length();
        assert!(
            diff < 1e-12,
            "Off-axis gyroscopic coupling: expected {:?}, got {:?}, diff={}",
            expected,
            alpha,
            diff,
        );
    }

    // ---------------------------------------------------------------
    // compute_left_quat_deriv tests
    // ---------------------------------------------------------------

    /// Zero angular velocity: quaternion derivative should be zero.
    #[test]
    fn quat_deriv_zero_omega() {
        let q = JeodQuat::left_quat_from_eigen_rotation(0.5, DVec3::Z);
        let qdot = compute_left_quat_deriv(&q, DVec3::ZERO);

        for (i, &val) in qdot.iter().enumerate() {
            assert!(
                val.abs() < TOL,
                "qdot[{}] = {} should be zero for zero angular velocity",
                i,
                val,
            );
        }
    }

    /// Pure z-rotation: omega = [0, 0, omega_z].
    /// For identity quaternion q = [1, 0, 0, 0]:
    ///   mhang_vel = [0, 0, -omega_z/2]
    ///   qdot[0] = -(q.vector . mhang_vel) = -([0,0,0] . [0,0,-omega_z/2]) = 0
    ///   qdot[1..3] = 1.0 * [0,0,-omega_z/2] + [0,0,-omega_z/2] x [0,0,0]
    ///              = [0, 0, -omega_z/2]
    #[test]
    fn quat_deriv_z_rotation_identity() {
        let q = JeodQuat::identity();
        let omega_z = 0.1;
        let qdot = compute_left_quat_deriv(&q, DVec3::new(0.0, 0.0, omega_z));

        assert!(
            qdot[0].abs() < TOL,
            "qdot scalar should be ~0, got {}",
            qdot[0],
        );
        assert!(qdot[1].abs() < TOL, "qdot vx should be ~0, got {}", qdot[1],);
        assert!(qdot[2].abs() < TOL, "qdot vy should be ~0, got {}", qdot[2],);
        let expected_vz = -omega_z * 0.5;
        assert!(
            (qdot[3] - expected_vz).abs() < TOL,
            "qdot vz expected {}, got {}",
            expected_vz,
            qdot[3],
        );
    }

    /// Non-trivial quaternion with z-rotation: verify consistency with
    /// the formula applied to a 30-degree rotation about z.
    #[test]
    fn quat_deriv_z_rotation_nontrivial() {
        let angle = PI / 6.0; // 30 degrees
        let q = JeodQuat::left_quat_from_eigen_rotation(angle, DVec3::Z);
        let omega_z = 2.0;
        let qdot = compute_left_quat_deriv(&q, DVec3::new(0.0, 0.0, omega_z));

        // Manual: mhang_vel = [0, 0, -1.0]
        // qs = cos(15 deg), qv = [0, 0, -sin(15 deg)]
        // qdot[0] = -(qv . mhang_vel) = -((-sin15)*(-1)) = -sin(15 deg)
        // qdot_v = qs * [0,0,-1] + [0,0,-1] x [0,0,-sin15]
        //        = [0, 0, -cos15] + [0,0,0]   (parallel vectors -> cross = 0)
        //        = [0, 0, -cos15]
        let half = angle * 0.5;
        let expected_s = -half.sin();
        let expected_vz = -half.cos();

        assert!(
            (qdot[0] - expected_s).abs() < 1e-12,
            "qdot scalar expected {}, got {}",
            expected_s,
            qdot[0],
        );
        assert!(
            qdot[1].abs() < 1e-12,
            "qdot vx should be ~0, got {}",
            qdot[1],
        );
        assert!(
            qdot[2].abs() < 1e-12,
            "qdot vy should be ~0, got {}",
            qdot[2],
        );
        assert!(
            (qdot[3] - expected_vz).abs() < 1e-12,
            "qdot vz expected {}, got {}",
            expected_vz,
            qdot[3],
        );
    }

    // ---------------------------------------------------------------
    // normalize_integ tests
    // ---------------------------------------------------------------

    /// Near-unity quaternion: should use Pade approximant path.
    #[test]
    fn normalize_integ_near_unity() {
        let mut q = JeodQuat::new(1.0 + 1e-10, 0.0, 0.0, 0.0);
        normalize_integ(&mut q);

        let norm_err = (q.norm_sq() - 1.0).abs();
        assert!(
            norm_err < 1e-14,
            "After normalize_integ, |q|^2 - 1 = {}",
            norm_err,
        );
        // Should be [1, 0, 0, 0]
        assert!(
            (q.scalar() - 1.0).abs() < 1e-9,
            "scalar should be ~1.0, got {}",
            q.scalar(),
        );
    }

    /// Far from unity: uses sqrt normalization path.
    #[test]
    fn normalize_integ_far_from_unity() {
        let mut q = JeodQuat::new(2.0, 0.0, 0.0, 0.0);
        normalize_integ(&mut q);

        let norm_err = (q.norm_sq() - 1.0).abs();
        assert!(
            norm_err < 1e-14,
            "After normalize_integ, |q|^2 - 1 = {}",
            norm_err,
        );
        assert!(
            (q.scalar() - 1.0).abs() < 1e-14,
            "scalar should be 1.0, got {}",
            q.scalar(),
        );
    }

    /// normalize_integ does NOT flip sign (unlike JeodQuat::normalize).
    #[test]
    fn normalize_integ_preserves_sign() {
        let mut q = JeodQuat::new(-1.0, 0.0, 0.0, 0.0);
        normalize_integ(&mut q);

        // scalar should remain negative
        assert!(
            q.scalar() < 0.0,
            "normalize_integ should NOT flip sign, scalar = {}",
            q.scalar(),
        );
        let norm_err = (q.norm_sq() - 1.0).abs();
        assert!(
            norm_err < 1e-14,
            "After normalize_integ, |q|^2 - 1 = {}",
            norm_err,
        );
    }

    /// Non-trivial quaternion normalization.
    #[test]
    fn normalize_integ_nontrivial() {
        let mut q = JeodQuat::new(3.0, 4.0, 0.0, 0.0);
        normalize_integ(&mut q);

        let norm_err = (q.norm_sq() - 1.0).abs();
        assert!(
            norm_err < 1e-14,
            "After normalize_integ, |q|^2 - 1 = {}",
            norm_err,
        );
        // 3/5, 4/5
        assert!(
            (q.scalar() - 0.6).abs() < 1e-14,
            "scalar should be 0.6, got {}",
            q.scalar(),
        );
        assert!(
            (q.data[1] - 0.8).abs() < 1e-14,
            "vx should be 0.8, got {}",
            q.data[1],
        );
    }

    #[test]
    fn typed_rotational_state_round_trips() {
        use astrodyn_quantities::frame::TestVehicle;

        let untyped = RotationalState {
            quaternion: JeodQuat::left_quat_from_eigen_rotation(0.7, DVec3::Z),
            ang_vel_body: DVec3::new(0.01, 0.02, 0.03),
        };

        let typed = RotationalStateTyped::<TestVehicle>::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();

        assert_eq!(back.quaternion, untyped.quaternion);
        assert_eq!(back.ang_vel_body, untyped.ang_vel_body);
    }

    #[test]
    fn typed_rotational_default_is_identity() {
        use astrodyn_quantities::frame::TestVehicle;

        let s = RotationalStateTyped::<TestVehicle>::default();
        let untyped = s.to_untyped();
        assert_eq!(untyped.quaternion, JeodQuat::identity());
        assert_eq!(untyped.ang_vel_body, DVec3::ZERO);
    }

    // The constant-ω attitude advance helper that used to live here
    // (`advance_left_quat_body_rate`) was promoted into a typed
    // sibling, [`astrodyn_quantities::body_attitude::BodyAttitude::advance_under_body_rate`],
    // as part of issue #252. The runtime regression test
    // `advance_left_quat_uses_left_multiply_not_right` was likewise
    // moved there; the type-level guarantee that no caller can write
    // the wrong multiply at all is exercised by the compile-fail
    // doctests on `BodyAttitude` itself.
}
