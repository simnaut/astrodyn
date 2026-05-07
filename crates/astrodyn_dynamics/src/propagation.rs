//! State propagation: compute derived frame states from an integrated frame
//! state and mass offsets.
//!
//! Port of JEOD `dyn_body_propagate_state.cc`. In JEOD, a DynBody has three
//! reference frames: structure (geometric origin), composite_body (composite
//! CoM), and core_body (this body's CoM only). The state is integrated in one
//! of these frames, then propagated to the others using the mass offsets.
//!
//! These are pure functions (no ECS dependency).

use crate::mass_body::MassPointState;
use astrodyn_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use astrodyn_math::JeodQuat;

/// Forward propagation: compute a derived frame state from a source frame state
/// and a relative mass point offset.
///
/// Port of JEOD `compute_derived_state_forward()` from
/// `dyn_body_propagate_state.cc:159-198`.
///
/// Given a source frame state (e.g., structure) and a relative mass offset
/// (e.g., structure-to-composite), compute the derived frame state
/// (e.g., composite_body).
///
/// The `rel` offset describes how the derived frame relates to the source frame:
/// - `rel.position`: offset from source origin to derived origin, in source frame coords.
/// - `rel.t_parent_this`: rotation from source frame to derived frame.
pub fn propagate_forward(source: &RefFrameState, rel: &MassPointState) -> RefFrameState {
    // Rotation: T_derived = T_rel * T_source
    let t_derived = rel.t_parent_this * source.rot.t_parent_this;

    // Quaternion from the derived transformation matrix
    let q_derived = JeodQuat::left_quat_from_transformation(&t_derived);

    // Angular velocity: omega_derived = T_rel * omega_source  (rigid body assumption)
    let ang_vel_derived = rel.t_parent_this * source.rot.ang_vel_this;

    // Position: r_derived = r_source + T_source^T * rel.position
    // T_source transforms from parent to source, so T^T goes source to parent.
    let pos_derived = source.trans.position + source.rot.t_parent_this.transpose() * rel.position;

    // Velocity: v_derived = v_source + T_source^T * (omega_source x rel.position)
    let omega_cross_r = source.rot.ang_vel_this.cross(rel.position);
    let vel_derived = source.trans.velocity + source.rot.t_parent_this.transpose() * omega_cross_r;

    RefFrameState {
        trans: RefFrameTrans {
            position: pos_derived,
            velocity: vel_derived,
        },
        rot: RefFrameRot {
            q_parent_this: q_derived,
            t_parent_this: t_derived,
            ang_vel_this: ang_vel_derived,
        },
    }
}

/// Reverse propagation: recover the source frame state from a derived frame
/// state and the forward offset.
///
/// The inverse of [`propagate_forward`]. Given the integrated frame
/// (e.g., composite_body) and the offset from structure-to-composite,
/// recover the structure frame state.
///
/// `source` here is the derived/integrated state, and `rel` is the same
/// forward offset used in `propagate_forward`.
pub fn propagate_reverse(derived: &RefFrameState, rel: &MassPointState) -> RefFrameState {
    // T_source = T_rel^T * T_derived
    let t_source = rel.t_parent_this.transpose() * derived.rot.t_parent_this;

    // Quaternion from the source transformation matrix
    let q_source = JeodQuat::left_quat_from_transformation(&t_source);

    // omega_source = T_rel^T * omega_derived
    let ang_vel_source = rel.t_parent_this.transpose() * derived.rot.ang_vel_this;

    // r_source = r_derived - T_source^T * rel.position
    let pos_source = derived.trans.position - t_source.transpose() * rel.position;

    // v_source = v_derived - T_source^T * (omega_source x rel.position)
    let omega_cross_r = ang_vel_source.cross(rel.position);
    let vel_source = derived.trans.velocity - t_source.transpose() * omega_cross_r;

    RefFrameState {
        trans: RefFrameTrans {
            position: pos_source,
            velocity: vel_source,
        },
        rot: RefFrameRot {
            q_parent_this: q_source,
            t_parent_this: t_source,
            ang_vel_this: ang_vel_source,
        },
    }
}

/// Convenience: given structure state and mass offsets, compute both
/// composite_body and core_body frame states.
///
/// Returns `(composite_state, core_state)`.
///
/// - `composite_offset`: structure-to-composite CoM offset (from mass tree).
/// - `core_offset`: structure-to-core CoM offset (from mass tree).
pub fn propagate_body_frames(
    structure: &RefFrameState,
    composite_offset: &MassPointState,
    core_offset: &MassPointState,
) -> (RefFrameState, RefFrameState) {
    let composite = propagate_forward(structure, composite_offset);
    let core = propagate_forward(structure, core_offset);
    (composite, core)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_frames::RefFrameRot;
    use astrodyn_math::test_utils::{approx_eq_f64, approx_eq_mat3, approx_eq_vec3};
    use glam::{DMat3, DVec3};
    use std::f64::consts::FRAC_PI_4;

    const TOL: f64 = 1e-14;

    /// Helper: create a RefFrameState with a rotation about an axis.
    fn make_state(
        angle: f64,
        axis: DVec3,
        pos: DVec3,
        vel: DVec3,
        ang_vel: DVec3,
    ) -> RefFrameState {
        let q = JeodQuat::left_quat_from_eigen_rotation(angle, axis);
        let t = q.left_quat_to_transformation();
        RefFrameState {
            trans: RefFrameTrans {
                position: pos,
                velocity: vel,
            },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: t,
                ang_vel_this: ang_vel,
            },
        }
    }

    // -----------------------------------------------------------------------
    // 1. Forward propagation with identity offset
    // -----------------------------------------------------------------------
    #[test]
    fn forward_identity_offset() {
        let source = make_state(
            0.5,
            DVec3::Z,
            DVec3::new(1e7, 0.0, 0.0),
            DVec3::new(7000.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.001),
        );
        let rel = MassPointState::default(); // zero offset, identity rotation

        let derived = propagate_forward(&source, &rel);

        assert!(
            approx_eq_vec3(derived.trans.position, source.trans.position, TOL),
            "identity offset: position"
        );
        assert!(
            approx_eq_vec3(derived.trans.velocity, source.trans.velocity, TOL),
            "identity offset: velocity"
        );
        assert!(
            approx_eq_mat3(&derived.rot.t_parent_this, &source.rot.t_parent_this, TOL),
            "identity offset: T"
        );
        assert!(
            approx_eq_vec3(derived.rot.ang_vel_this, source.rot.ang_vel_this, TOL),
            "identity offset: ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Forward propagation with translation offset
    // -----------------------------------------------------------------------
    #[test]
    fn forward_translation_offset() {
        // Source at origin, no rotation, no angular velocity
        let source = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(1e7, 0.0, 0.0),
                velocity: DVec3::new(7000.0, 0.0, 0.0),
            },
            rot: RefFrameRot::default(),
        };

        // Offset of 2m along X in source frame, identity rotation
        let rel = MassPointState {
            position: DVec3::new(2.0, 0.0, 0.0),
            t_parent_this: DMat3::IDENTITY,
        };

        let derived = propagate_forward(&source, &rel);

        // With identity T_source, position shifts by the offset directly
        let expected_pos = DVec3::new(1e7 + 2.0, 0.0, 0.0);
        assert!(
            approx_eq_vec3(derived.trans.position, expected_pos, TOL),
            "translation offset: position, expected {:?} got {:?}",
            expected_pos,
            derived.trans.position
        );

        // No angular velocity, so velocity is unchanged
        assert!(
            approx_eq_vec3(derived.trans.velocity, source.trans.velocity, TOL),
            "translation offset: velocity"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Forward-reverse round-trip
    // -----------------------------------------------------------------------
    #[test]
    fn forward_reverse_round_trip() {
        let source = make_state(
            0.7,
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(6.778e6, 0.0, 0.0),
            DVec3::new(0.0, 7672.0, 0.0),
            DVec3::new(0.001, -0.002, 0.0011),
        );

        let rel = MassPointState {
            position: DVec3::new(0.5, -0.3, 0.1),
            t_parent_this: {
                let q = JeodQuat::left_quat_from_eigen_rotation(0.05, DVec3::Y);
                q.left_quat_to_transformation()
            },
        };

        let derived = propagate_forward(&source, &rel);
        let recovered = propagate_reverse(&derived, &rel);

        assert!(
            approx_eq_vec3(recovered.trans.position, source.trans.position, 1e-8),
            "round-trip position: expected {:?}, got {:?}",
            source.trans.position,
            recovered.trans.position
        );
        assert!(
            approx_eq_vec3(recovered.trans.velocity, source.trans.velocity, 1e-8),
            "round-trip velocity: expected {:?}, got {:?}",
            source.trans.velocity,
            recovered.trans.velocity
        );
        assert!(
            approx_eq_mat3(
                &recovered.rot.t_parent_this,
                &source.rot.t_parent_this,
                1e-12
            ),
            "round-trip T"
        );
        assert!(
            approx_eq_vec3(recovered.rot.ang_vel_this, source.rot.ang_vel_this, 1e-12),
            "round-trip ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Propagate body frames: ISS-like CoM offset
    // -----------------------------------------------------------------------
    #[test]
    fn propagate_body_frames_iss() {
        // Structure frame at some orbital position
        let structure = make_state(
            FRAC_PI_4,
            DVec3::Z,
            DVec3::new(6.778e6, 0.0, 0.0),
            DVec3::new(0.0, 7672.0, 0.0),
            DVec3::new(0.0, 0.0, 0.0011), // ~ISS rotation rate
        );

        // Composite CoM offset from structure: 1m along X, 0.5m along Y
        let composite_offset = MassPointState {
            position: DVec3::new(1.0, 0.5, 0.0),
            t_parent_this: DMat3::IDENTITY,
        };

        // Core CoM offset: slightly different
        let core_offset = MassPointState {
            position: DVec3::new(0.8, 0.4, 0.1),
            t_parent_this: DMat3::IDENTITY,
        };

        let (composite, core) = propagate_body_frames(&structure, &composite_offset, &core_offset);

        // Composite position should differ from structure by T^T * offset
        let t_t = structure.rot.t_parent_this.transpose();
        let expected_comp_pos = structure.trans.position + t_t * composite_offset.position;
        assert!(
            approx_eq_vec3(composite.trans.position, expected_comp_pos, 1e-8),
            "body frames: composite position, expected {:?} got {:?}",
            expected_comp_pos,
            composite.trans.position
        );

        let expected_core_pos = structure.trans.position + t_t * core_offset.position;
        assert!(
            approx_eq_vec3(core.trans.position, expected_core_pos, 1e-8),
            "body frames: core position, expected {:?} got {:?}",
            expected_core_pos,
            core.trans.position
        );

        // Composite and core should have different positions
        let diff = (composite.trans.position - core.trans.position).length();
        assert!(
            diff > 0.01,
            "composite and core positions should differ, diff = {}",
            diff
        );
    }

    // -----------------------------------------------------------------------
    // 5. Rotation consistency: derived T is orthogonal
    // -----------------------------------------------------------------------
    #[test]
    fn rotation_consistency() {
        let source = make_state(
            1.2,
            DVec3::new(1.0, 2.0, 3.0).normalize(),
            DVec3::new(5e6, -3e6, 1e6),
            DVec3::new(500.0, -300.0, 100.0),
            DVec3::new(0.05, -0.02, 0.01),
        );

        let rel = MassPointState {
            position: DVec3::new(1.5, -0.8, 0.3),
            t_parent_this: {
                let q = JeodQuat::left_quat_from_eigen_rotation(0.3, DVec3::X);
                q.left_quat_to_transformation()
            },
        };

        let derived = propagate_forward(&source, &rel);

        // Check orthogonality: T * T^T = I
        let product = derived.rot.t_parent_this * derived.rot.t_parent_this.transpose();
        assert!(
            approx_eq_mat3(&product, &DMat3::IDENTITY, 1e-12),
            "T * T^T should be identity, got {:?}",
            product
        );

        // Check determinant = 1 (proper rotation)
        let det = derived.rot.t_parent_this.determinant();
        assert!(
            approx_eq_f64(det, 1.0, 1e-12),
            "det(T) should be 1, got {}",
            det
        );
    }
}
