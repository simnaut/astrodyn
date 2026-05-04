//! Kinematic state derivation for a passive rigid attachment.
//!
//! Computes a child body's instantaneous **composite-body** translational
//! and rotational state from its parent's composite-body state plus the
//! per-link attach geometry, under the composite-rigid-body assumption
//! that JEOD enforces in
//! [`models/dynamics/dyn_body/src/dyn_body_propagate_state.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_propagate_state.cc)
//! (`DynBody::propagate_state_from_structure` and friends): only the
//! root integrates; every kinematic child rides the root rigidly,
//! routed through structural frames via each link's `MassPointState`
//! offset and rotation.
//!
//! # JEOD precedent
//!
//! JEOD propagates state from one body's structure frame to a child's
//! structure frame via [`crate::propagation::propagate_forward`]
//! (port of `compute_derived_state_forward`). The same kernel is then
//! re-applied in two more shifts to land the body's *composite_body*
//! state from its structure state via the body's
//! `mass.composite_properties` mass point. This module composes those
//! three shifts (parent composite → parent struct → child struct →
//! child composite) into a single direct kernel that consumes only
//! the per-edge geometry (`MassChildOf` analogue) and the per-body
//! struct→body rotations. Bit-equivalent to running
//! `propagate_forward` three times when no `JointKinematicsC`-class
//! prescribed motion sits on the link — the "passive rigid attachment"
//! scope. Equivalent to JEOD's `compute_derived_state_forward` chain
//! in `models/dynamics/dyn_body/src/dyn_body.cc`.
//!
//! # Frame discipline
//!
//! The kernel operates on raw `glam::DVec3`/`DQuat`/`DMat3` per the
//! project's "internal physics-crate kernels" rule (CLAUDE.md
//! "Precision"). Typed siblings live at
//! [`compute_kinematic_child_state_typed`] for callers that want to
//! preserve `Position<RootInertial>` / `Velocity<RootInertial>` /
//! `BodyAttitude<V>` / `AngularVelocity<BodyFrame<V>>` phantoms across
//! the boundary; the orchestration layer in `jeod_sim` and the Bevy
//! adapter use the typed surface.
//!
//! # Out of scope
//!
//! - **Joint-kinematics drivers** (`JointKinematicsC`-class prescribed
//!   relative motion at a `MassChildOf` link). If a chain member's
//!   link carries prescribed rotation rates from a joint driver, that
//!   system writes its own rotation onto the *frame* entity — a
//!   parallel surface. This kernel operates on bodies in the
//!   `MassChildOf` mass tree only, and treats every link as passive
//!   rigid (zero relative angular velocity at the joint). Composing
//!   prescribed joint motion is design-doc § 15.4 work.
//! - **Partial-state propagation** (JEOD's `compute_state_elements_*`
//!   path that drops position/velocity when attitude is missing). We
//!   panic on a missing parent attitude rather than silently producing
//!   degraded output, per "Fail Loudly".

use glam::{DMat3, DQuat, DVec3};
use jeod_math::JeodQuat;
use jeod_quantities::aliases::{AngularVelocity, Position, Velocity};
use jeod_quantities::body_attitude::BodyAttitude;
use jeod_quantities::frame::{BodyFrame, RootInertial, Vehicle};
use jeod_quantities::quat::NormalizedQuat;

use crate::rotational::RotationalState;
use crate::state::TranslationalState;

/// Inputs to [`compute_kinematic_child_state`].
///
/// Every field is in **raw SI units** (m, m/s, rad/s, unitless rotation
/// matrices / quaternions) and uses the JEOD scalar-first
/// left-transformation quaternion convention. The orchestration layer
/// (`jeod_sim::propagate_state_via_storage`) and the Bevy adapter
/// (`propagate_state_from_root_system`) marshal typed components into
/// this raw shape at the kernel boundary.
#[derive(Debug, Clone, Copy)]
pub struct KinematicChildInputs {
    /// Parent's inertial→body rotation as a transformation matrix
    /// (`T_inertial_body`: `v_body = T · v_inertial`).
    pub parent_t_inertial_body: DMat3,
    /// Parent's angular velocity expressed in the parent's body frame
    /// (rad/s).
    pub parent_ang_vel_body: DVec3,
    /// Parent's composite-body position in the inertial frame (m).
    pub parent_position_inertial: DVec3,
    /// Parent's composite-body velocity in the inertial frame (m/s).
    pub parent_velocity_inertial: DVec3,
    /// Parent's structural→body rotation (`T_struct_body`: rotates
    /// vector components from parent's structural frame into parent's
    /// body frame). Identity for vehicles whose structural frame
    /// coincides with the body frame (the default).
    pub parent_t_struct_body: DMat3,
    /// Parent's composite center of mass in the parent's structural
    /// frame (m). Mirrors JEOD `mass.composite_properties.position`
    /// (CoM expressed in this body's structural frame).
    pub parent_composite_in_pstr: DVec3,
    /// Per-link rotation `T_parent_child` (`v_child_struct =
    /// T_parent_child · v_parent_struct`). This is JEOD's
    /// `MassPointState::T_parent_this` for the parent → child link
    /// and matches `MassChildOf.t_parent_child`.
    pub t_parent_child: DMat3,
    /// Child's structural origin in the **parent's** structural
    /// frame (m). Matches `MassChildOf.offset` and JEOD
    /// `MassPointState::position`.
    pub link_offset_in_pstr: DVec3,
    /// Child's structural→body rotation (`T_struct_body_child`).
    /// Identity when the child has no `StructuralTransformC`.
    pub child_t_struct_body: DMat3,
    /// Child's composite center of mass in the child's structural
    /// frame (m). Mirrors JEOD
    /// `mass.composite_properties.position` for the child.
    pub child_composite_in_cstr: DVec3,
}

/// Outputs of [`compute_kinematic_child_state`].
///
/// Every value is the child's instantaneous composite-body state under
/// passive rigid attachment, in the same frame conventions the parent's
/// inputs used.
#[derive(Debug, Clone, Copy)]
pub struct KinematicChildOutputs {
    /// Child's inertial→body rotation as a transformation matrix
    /// (`T_inertial_body_child`).
    pub child_t_inertial_body: DMat3,
    /// Child's inertial→body rotation as a JEOD scalar-first
    /// left-transformation quaternion. Bit-equivalent to
    /// `JeodQuat::left_quat_from_transformation(&child_t_inertial_body)`
    /// — exposed alongside the matrix so callers can land directly
    /// into [`RotationalState::quaternion`] without re-deriving it.
    pub child_q_inertial_body: JeodQuat,
    /// Child's angular velocity expressed in the **child's** body
    /// frame (rad/s). Equal to the parent's angular velocity vector
    /// re-expressed in the child's body coordinates — passive rigid
    /// attachment carries no joint-relative rotation.
    pub child_ang_vel_body: DVec3,
    /// Child's composite-body position in the inertial frame (m).
    pub child_position_inertial: DVec3,
    /// Child's composite-body velocity in the inertial frame (m/s).
    pub child_velocity_inertial: DVec3,
}

/// Derive a kinematic child's instantaneous composite-body state from
/// the parent's composite-body state and the per-link attach geometry.
///
/// The derivation routes through structural frames the way JEOD does:
///
/// 1. **Rotation chain**:
///    `T_inertial_body_child =
///         T_struct_body_child · T_parent_child · T_struct_body_parent^T
///       · T_inertial_body_parent`.
///    Composes parent inertial→body, body→struct, parent struct→child
///    struct, and child struct→body in the JEOD-faithful order.
///
/// 2. **Inertial-frame angular velocity**:
///    `omega_inertial = T_inertial_body_parent^T · parent_ang_vel_body`.
///    Passive rigid attachment ⇒ child shares this same physical
///    angular-velocity vector. The components in the child body
///    frame are
///    `child_ang_vel_body = T_inertial_body_child · omega_inertial`.
///
/// 3. **Composite-CoM offset in the parent's structural frame**:
///    `pcm_to_ccm =
///         link_offset_in_pstr
///         + T_parent_child^T · child_composite_in_cstr
///         − parent_composite_in_pstr`.
///    Same arithmetic the wrench walk's
///    [`crate::wrench::shift_wrench_to_parent`] orchestrator uses
///    (mirrors JEOD `dyn_body_collect.cc:181`).
///
/// 4. **Inertial-frame offset**:
///    `r_inertial = T_inertial_struct_parent^T · pcm_to_ccm`,
///    where
///    `T_inertial_struct_parent = T_struct_body_parent^T
///                              · T_inertial_body_parent`.
///    `T_inertial_struct^T` rotates parent-structural components into
///    inertial components — the same composition
///    `force_collection_system` and `wrench_aggregation_system` use
///    at their root-exit boundaries.
///
/// 5. **Rigid-body velocity**:
///    `v_child = v_parent + omega_inertial × r_inertial`.
///    `r_inertial` is constant in the parent body frame for a passive
///    rigid attachment, so the only velocity contribution is the
///    instantaneous rotation of the link.
///
/// Identity-attitude inputs (`T_inertial_body_parent = I`,
/// `T_struct_body_* = I`, `T_parent_child = I`,
/// `parent_ang_vel_body = 0`) collapse to:
/// - `child_t_inertial_body = I`,
/// - `child_ang_vel_body = 0`,
/// - `child_position_inertial = parent_position_inertial +
///        link_offset_in_pstr + child_composite_in_cstr − parent_composite_in_pstr`,
/// - `child_velocity_inertial = parent_velocity_inertial`.
///
/// # Panics
///
/// Panics with a descriptive diagnostic when the composed
/// `child_t_inertial_body` produces a quaternion that drifts past
/// `NormalizedQuat::DEFAULT_TOLERANCE` (1e-12) from unit norm. Per
/// CLAUDE.md "Fail Loudly", a quaternion that quietly fails this
/// invariant indicates a numerically degenerate input rotation matrix
/// (a non-orthonormal `t_parent_child` from mission-code construction,
/// say) and writing it back into a typed
/// [`crate::rotational::RotationalStateTyped`] would silently corrupt
/// every downstream consumer's attitude reads.
// JEOD_INV: DB.13 — kinematic state propagation routed through structural frames (parent → struct → link → child struct → child body), mirroring `DynBody::propagate_state_from_structure`
// JEOD_INV: DB.17 — kinematic children's state is derived from the root each step (only the root integrates)
#[inline]
pub fn compute_kinematic_child_state(inputs: KinematicChildInputs) -> KinematicChildOutputs {
    let KinematicChildInputs {
        parent_t_inertial_body,
        parent_ang_vel_body,
        parent_position_inertial,
        parent_velocity_inertial,
        parent_t_struct_body,
        parent_composite_in_pstr,
        t_parent_child,
        link_offset_in_pstr,
        child_t_struct_body,
        child_composite_in_cstr,
    } = inputs;

    // 1. Rotation chain: parent inertial→body, body→struct,
    //    struct→struct (link), child struct→body.
    let parent_t_inertial_struct = parent_t_struct_body.transpose() * parent_t_inertial_body;
    // T_inertial_struct_child = T_parent_child · T_inertial_struct_parent
    //   (apply the parent→child structural rotation after landing in
    //    parent struct).
    let child_t_inertial_struct = t_parent_child * parent_t_inertial_struct;
    // T_inertial_body_child = T_struct_body_child · T_inertial_struct_child
    let child_t_inertial_body = child_t_struct_body * child_t_inertial_struct;

    // Quaternion form of the composed rotation. Witness-checked unit-
    // norm (Fail Loudly): a degenerate `t_parent_child` would silently
    // push the matrix off SO(3); the normalized-quat constructor
    // catches it before it reaches the typed surface.
    let child_q_inertial_body = JeodQuat::left_quat_from_transformation(&child_t_inertial_body);
    let _normalized_witness = NormalizedQuat::new(child_q_inertial_body).unwrap_or_else(|err| {
        panic!(
            "compute_kinematic_child_state: composed inertial→body rotation produced a \
                 non-unit quaternion ({err:?}). This indicates a non-orthonormal input — most \
                 likely a `MassChildOf.t_parent_child` rotation matrix that has drifted off SO(3). \
                 Fix the upstream rotation: set `MassChildOf::with_rotation(parent, offset, T)` \
                 with an orthonormal `T`, or rebuild it from a unit quaternion via \
                 `JeodQuat::left_quat_to_transformation`."
        )
    });

    // 2. Angular velocity: same physical vector, expressed in the
    //    child's body frame.
    //    omega_inertial = T_inertial_body_parent^T · parent_ang_vel_body
    //    omega_child_body = T_inertial_body_child · omega_inertial
    let omega_inertial = parent_t_inertial_body.transpose() * parent_ang_vel_body;
    let child_ang_vel_body = child_t_inertial_body * omega_inertial;

    // 3. Parent-structural offset from parent CoM to child CoM.
    //    Same arithmetic as the wrench walk's `pcm_to_ccm`.
    //    `T_parent_child^T` rotates child-struct components back into
    //    parent-struct components.
    let pcm_to_ccm = link_offset_in_pstr + t_parent_child.transpose() * child_composite_in_cstr
        - parent_composite_in_pstr;

    // 4. Inertial-frame offset.
    //    `T_inertial_struct.transpose()` rotates parent-struct
    //    components into inertial components — same identity
    //    `force_collection_system` and `wrench_aggregation_system` use
    //    at their root-exit boundaries.
    let r_offset_inertial = parent_t_inertial_struct.transpose() * pcm_to_ccm;
    let child_position_inertial = parent_position_inertial + r_offset_inertial;

    // 5. Rigid-body velocity: parent CoM velocity plus instantaneous
    //    rotation of the offset vector. `omega × r` only — `dr/dt|_body`
    //    is zero for a passive rigid link.
    let child_velocity_inertial =
        parent_velocity_inertial + omega_inertial.cross(r_offset_inertial);

    KinematicChildOutputs {
        child_t_inertial_body,
        child_q_inertial_body,
        child_ang_vel_body,
        child_position_inertial,
        child_velocity_inertial,
    }
}

/// Convenience: extract the inputs and call
/// [`compute_kinematic_child_state`] from `RotationalState` /
/// `TranslationalState` shapes.
///
/// The `parent_*` arguments are the parent's **composite-body**
/// rotational and translational state in the inertial frame; the
/// `link_*` and `child_*` arguments mirror the storage layout in
/// `MassChildOf` plus the per-body structural transform. The kernel
/// inputs are JEOD-faithful at this granularity (parent body state +
/// parent struct→body + parent composite + link rotation + link
/// offset + child struct→body + child composite), so we accept the
/// long argument list rather than collapse fields into a struct that
/// would just be the kernel's `KinematicChildInputs` again.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn derive_kinematic_child_from_states(
    parent_rot: &RotationalState,
    parent_trans: &TranslationalState,
    parent_t_struct_body: DMat3,
    parent_composite_in_pstr: DVec3,
    t_parent_child: DMat3,
    link_offset_in_pstr: DVec3,
    child_t_struct_body: DMat3,
    child_composite_in_cstr: DVec3,
) -> (RotationalState, TranslationalState) {
    let parent_t_inertial_body = parent_rot.quaternion.left_quat_to_transformation();
    let inputs = KinematicChildInputs {
        parent_t_inertial_body,
        parent_ang_vel_body: parent_rot.ang_vel_body,
        parent_position_inertial: parent_trans.position,
        parent_velocity_inertial: parent_trans.velocity,
        parent_t_struct_body,
        parent_composite_in_pstr,
        t_parent_child,
        link_offset_in_pstr,
        child_t_struct_body,
        child_composite_in_cstr,
    };
    let out = compute_kinematic_child_state(inputs);
    let rot = RotationalState {
        quaternion: out.child_q_inertial_body,
        ang_vel_body: out.child_ang_vel_body,
    };
    let trans = TranslationalState {
        position: out.child_position_inertial,
        velocity: out.child_velocity_inertial,
    };
    (rot, trans)
}

/// Typed sibling of [`compute_kinematic_child_state`] for callers that
/// want to preserve frame phantoms across the boundary.
///
/// All angles enter / exit the kernel as [`BodyAttitude<V>`],
/// [`AngularVelocity<BodyFrame<V>>`], [`Position<RootInertial>`],
/// [`Velocity<RootInertial>`]. The struct→body rotation matrices, the
/// per-link `t_parent_child`, and the per-body composite CoM positions
/// are kept as raw `DMat3` / `DVec3` because they live in structural
/// frames the typed-quantity facade does not yet have phantoms for —
/// the same boundary every wrench-aggregation call site uses.
///
/// `VParent` and `VChild` may be the same vehicle marker (the typical
/// SelfRef pattern in the Bevy adapter, where both bodies belong to
/// "this entity's vehicle"). They are kept as separate type
/// parameters so a future articulated-vehicle topology that splits
/// vehicles can name the child's vehicle independently without
/// touching the kernel.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn compute_kinematic_child_state_typed<VParent: Vehicle, VChild: Vehicle>(
    parent_q_inertial_body: BodyAttitude<VParent>,
    parent_ang_vel_body: AngularVelocity<BodyFrame<VParent>>,
    parent_position_inertial: Position<RootInertial>,
    parent_velocity_inertial: Velocity<RootInertial>,
    parent_t_struct_body: DMat3,
    parent_composite_in_pstr: DVec3,
    t_parent_child: DMat3,
    link_offset_in_pstr: DVec3,
    child_t_struct_body: DMat3,
    child_composite_in_cstr: DVec3,
) -> (
    BodyAttitude<VChild>,
    AngularVelocity<BodyFrame<VChild>>,
    Position<RootInertial>,
    Velocity<RootInertial>,
) {
    let parent_t_inertial_body = parent_q_inertial_body
        .as_witness()
        .left_quat_to_transformation();
    let inputs = KinematicChildInputs {
        parent_t_inertial_body,
        parent_ang_vel_body: parent_ang_vel_body.raw_si(),
        parent_position_inertial: parent_position_inertial.raw_si(),
        parent_velocity_inertial: parent_velocity_inertial.raw_si(),
        parent_t_struct_body,
        parent_composite_in_pstr,
        t_parent_child,
        link_offset_in_pstr,
        child_t_struct_body,
        child_composite_in_cstr,
    };
    let out = compute_kinematic_child_state(inputs);
    // Re-witness the composed quaternion. The kernel already verified
    // unit-norm via NormalizedQuat::new — re-calling here is the
    // canonical typed boundary.
    let q = NormalizedQuat::new(out.child_q_inertial_body)
        .expect("child_q_inertial_body unit-norm: enforced inside compute_kinematic_child_state");
    let attitude = BodyAttitude::<VChild>::from_witness(q);
    let ang_vel = AngularVelocity::<BodyFrame<VChild>>::from_raw_si(out.child_ang_vel_body);
    let position = Position::<RootInertial>::from_raw_si(out.child_position_inertial);
    let velocity = Velocity::<RootInertial>::from_raw_si(out.child_velocity_inertial);
    (attitude, ang_vel, position, velocity)
}

/// glam-quaternion convenience wrapper used by orchestration callers
/// that already hold a [`DQuat`] (Bevy adapter's `RotationalStateC`
/// path lifts `JeodQuat` → `DQuat` at boundary; arena callers stay
/// in `JeodQuat`). Kept thin — defers to
/// [`compute_kinematic_child_state`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn compute_kinematic_child_state_dquat(
    parent_q_inertial_body: DQuat,
    parent_ang_vel_body: DVec3,
    parent_position_inertial: DVec3,
    parent_velocity_inertial: DVec3,
    parent_t_struct_body: DMat3,
    parent_composite_in_pstr: DVec3,
    t_parent_child: DMat3,
    link_offset_in_pstr: DVec3,
    child_t_struct_body: DMat3,
    child_composite_in_cstr: DVec3,
) -> KinematicChildOutputs {
    // DQuat to rotation matrix (right-mul convention) is identical
    // numerics to JeodQuat::left_quat_to_transformation when the
    // glam quaternion was constructed from JEOD components — the
    // boundary lift in `RotationalStateC::from_untyped_unchecked`
    // performs that swap once at insertion time. Here we go through
    // the rotation matrix derived directly from DQuat to avoid a
    // second swap.
    let parent_t_inertial_body = DMat3::from_quat(parent_q_inertial_body);
    let inputs = KinematicChildInputs {
        parent_t_inertial_body,
        parent_ang_vel_body,
        parent_position_inertial,
        parent_velocity_inertial,
        parent_t_struct_body,
        parent_composite_in_pstr,
        t_parent_child,
        link_offset_in_pstr,
        child_t_struct_body,
        child_composite_in_cstr,
    };
    compute_kinematic_child_state(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::propagate_forward;
    use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
    use jeod_math::test_utils::{approx_eq_mat3, approx_eq_vec3};
    use std::f64::consts::FRAC_PI_4;

    const TOL: f64 = 1e-12;

    fn jeod_rot_z(angle: f64) -> DMat3 {
        JeodQuat::left_quat_from_eigen_rotation(angle, DVec3::Z).left_quat_to_transformation()
    }

    fn jeod_rot_axis(angle: f64, axis: DVec3) -> DMat3 {
        JeodQuat::left_quat_from_eigen_rotation(angle, axis).left_quat_to_transformation()
    }

    /// Identity attach, identity parent attitude: child must equal
    /// parent verbatim — no rotation, no offset, same physics.
    #[test]
    fn identity_attach_returns_parent_state() {
        let parent_pos = DVec3::new(7e6, 0.0, 0.0);
        let parent_vel = DVec3::new(0.0, 7500.0, 0.0);
        let inputs = KinematicChildInputs {
            parent_t_inertial_body: DMat3::IDENTITY,
            parent_ang_vel_body: DVec3::ZERO,
            parent_position_inertial: parent_pos,
            parent_velocity_inertial: parent_vel,
            parent_t_struct_body: DMat3::IDENTITY,
            parent_composite_in_pstr: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
            link_offset_in_pstr: DVec3::ZERO,
            child_t_struct_body: DMat3::IDENTITY,
            child_composite_in_cstr: DVec3::ZERO,
        };
        let out = compute_kinematic_child_state(inputs);

        assert!(approx_eq_mat3(
            &out.child_t_inertial_body,
            &DMat3::IDENTITY,
            TOL
        ));
        assert!(approx_eq_vec3(out.child_ang_vel_body, DVec3::ZERO, TOL));
        assert!(approx_eq_vec3(out.child_position_inertial, parent_pos, TOL));
        assert!(approx_eq_vec3(out.child_velocity_inertial, parent_vel, TOL));
    }

    /// Passive 30°-Z attach with a parent at identity attitude.
    /// The child's body frame is rotated 30° about Z relative to the
    /// parent — `T_inertial_body_child = T_parent_child` (since
    /// parent is identity and there's no struct→body rotation).
    #[test]
    fn passive_z_rotation_attach_composes_correctly() {
        let angle = std::f64::consts::PI / 6.0;
        let t_pc = jeod_rot_z(angle);
        let parent_pos = DVec3::new(7e6, 0.0, 0.0);
        let parent_vel = DVec3::new(0.0, 7500.0, 0.0);
        let inputs = KinematicChildInputs {
            parent_t_inertial_body: DMat3::IDENTITY,
            parent_ang_vel_body: DVec3::ZERO,
            parent_position_inertial: parent_pos,
            parent_velocity_inertial: parent_vel,
            parent_t_struct_body: DMat3::IDENTITY,
            parent_composite_in_pstr: DVec3::ZERO,
            t_parent_child: t_pc,
            link_offset_in_pstr: DVec3::ZERO,
            child_t_struct_body: DMat3::IDENTITY,
            child_composite_in_cstr: DVec3::ZERO,
        };
        let out = compute_kinematic_child_state(inputs);

        assert!(approx_eq_mat3(&out.child_t_inertial_body, &t_pc, TOL));
        // Identity parent attitude + zero ang_vel ⇒ child ang_vel
        // must be zero in either frame.
        assert!(approx_eq_vec3(out.child_ang_vel_body, DVec3::ZERO, TOL));
        // Zero offset + zero composite CoMs ⇒ child sits on the parent.
        assert!(approx_eq_vec3(out.child_position_inertial, parent_pos, TOL));
        assert!(approx_eq_vec3(out.child_velocity_inertial, parent_vel, TOL));

        // Quaternion form must round-trip back to the same matrix.
        let q_to_t = out.child_q_inertial_body.left_quat_to_transformation();
        assert!(approx_eq_mat3(&q_to_t, &t_pc, TOL));
    }

    /// Non-zero offset under non-zero parent angular velocity:
    /// child velocity = parent velocity + ω × r.
    #[test]
    fn non_zero_offset_under_parent_omega_yields_omega_cross_r() {
        // Parent rotating about +Z at 0.001 rad/s, child offset (1, 0, 0)
        // along parent's body x-axis. Identity parent attitude ⇒
        // r_inertial = (1, 0, 0), omega_inertial = (0, 0, 0.001).
        // ω × r = (0, 1e-3, 0).
        let omega = DVec3::new(0.0, 0.0, 1e-3);
        let offset = DVec3::new(1.0, 0.0, 0.0);
        let parent_pos = DVec3::new(7e6, 0.0, 0.0);
        let parent_vel = DVec3::new(0.0, 7500.0, 0.0);

        let inputs = KinematicChildInputs {
            parent_t_inertial_body: DMat3::IDENTITY,
            parent_ang_vel_body: omega,
            parent_position_inertial: parent_pos,
            parent_velocity_inertial: parent_vel,
            parent_t_struct_body: DMat3::IDENTITY,
            parent_composite_in_pstr: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
            link_offset_in_pstr: offset,
            child_t_struct_body: DMat3::IDENTITY,
            child_composite_in_cstr: DVec3::ZERO,
        };
        let out = compute_kinematic_child_state(inputs);

        assert!(approx_eq_vec3(
            out.child_position_inertial,
            parent_pos + offset,
            TOL
        ));
        let expected_vel = parent_vel + DVec3::new(0.0, 1e-3, 0.0);
        assert!(
            approx_eq_vec3(out.child_velocity_inertial, expected_vel, TOL),
            "expected {expected_vel:?}, got {:?}",
            out.child_velocity_inertial
        );
        // Same omega, different frame components (identity → unchanged).
        assert!(approx_eq_vec3(out.child_ang_vel_body, omega, TOL));
    }

    /// High-frequency attach rotation (45° about a non-axial vector)
    /// under a non-trivial parent attitude (90° about Z) — verify
    /// the rotation chain composes correctly by comparing against
    /// three sequential `propagate_forward` calls (the JEOD
    /// composition this kernel collapses).
    #[test]
    fn matches_three_propagate_forward_calls_under_complex_attitudes() {
        // Parent attitude: 90° about Z (so parent body x-axis points
        // along inertial +y).
        let parent_q =
            JeodQuat::left_quat_from_eigen_rotation(std::f64::consts::FRAC_PI_2, DVec3::Z);
        let parent_t_ib = parent_q.left_quat_to_transformation();
        let parent_omega = DVec3::new(0.001, -0.002, 0.0011);
        let parent_pos = DVec3::new(6.778e6, 0.0, 0.0);
        let parent_vel = DVec3::new(0.0, 7672.0, 0.0);
        // Parent struct→body 30° about Y.
        let parent_t_sb = jeod_rot_axis(std::f64::consts::FRAC_PI_6, DVec3::Y);
        // Parent composite CoM in parent struct.
        let parent_cm_in_pstr = DVec3::new(0.1, 0.0, -0.05);
        // Link: 45° about (1,1,1).normalized, offset (2, 0.5, -0.3).
        let link_axis = DVec3::new(1.0, 1.0, 1.0).normalize();
        let t_pc = jeod_rot_axis(FRAC_PI_4, link_axis);
        let link_offset = DVec3::new(2.0, 0.5, -0.3);
        // Child struct→body 15° about X.
        let child_t_sb = jeod_rot_axis(0.15, DVec3::X);
        // Child composite CoM in child struct.
        let child_cm_in_cstr = DVec3::new(-0.05, 0.0, 0.02);

        let inputs = KinematicChildInputs {
            parent_t_inertial_body: parent_t_ib,
            parent_ang_vel_body: parent_omega,
            parent_position_inertial: parent_pos,
            parent_velocity_inertial: parent_vel,
            parent_t_struct_body: parent_t_sb,
            parent_composite_in_pstr: parent_cm_in_pstr,
            t_parent_child: t_pc,
            link_offset_in_pstr: link_offset,
            child_t_struct_body: child_t_sb,
            child_composite_in_cstr: child_cm_in_cstr,
        };
        let out = compute_kinematic_child_state(inputs);

        // Reference: three sequential propagate_forward calls from
        // parent composite-body → parent struct → child struct →
        // child composite-body. Each leg is a `MassPointState` with
        // (offset, t_parent_this) per the JEOD pipeline.
        // Leg 1 (parent body → parent struct): offset = -parent CoM
        // in parent struct, rotation = T_struct_body^T (body→struct),
        // i.e. propagate_forward source frame is parent body, derived
        // is parent struct.
        let parent_body_state = RefFrameState {
            trans: RefFrameTrans {
                position: parent_pos,
                velocity: parent_vel,
            },
            rot: RefFrameRot {
                q_parent_this: parent_q,
                t_parent_this: parent_t_ib,
                ang_vel_this: parent_omega,
            },
        };
        // body→struct mass point: position offset (in body frame) is
        // T_struct_body · (-CoM_in_pstr), rotation t_parent_this is
        // T_body_struct = T_struct_body^T. propagate_forward then
        // computes a struct-frame state expressed in inertial
        // coordinates.
        let body_to_struct = crate::mass_body::MassPointState {
            position: parent_t_sb * (-parent_cm_in_pstr),
            t_parent_this: parent_t_sb.transpose(),
        };
        let parent_struct_state = propagate_forward(&parent_body_state, &body_to_struct);

        // Leg 2 (parent struct → child struct): offset = link_offset
        // (in parent struct, by definition), rotation = t_parent_child.
        let pstr_to_cstr = crate::mass_body::MassPointState {
            position: link_offset,
            t_parent_this: t_pc,
        };
        let child_struct_state = propagate_forward(&parent_struct_state, &pstr_to_cstr);

        // Leg 3 (child struct → child composite-body): offset =
        // child_cm_in_cstr (in child struct), rotation =
        // child_t_struct_body.
        let cstr_to_cbody = crate::mass_body::MassPointState {
            position: child_cm_in_cstr,
            t_parent_this: child_t_sb,
        };
        let child_body_state = propagate_forward(&child_struct_state, &cstr_to_cbody);

        // Compare. propagate_forward expresses the derived state in
        // the SAME inertial frame as the source — so positions and
        // velocities are inertial, t_parent_this is inertial→derived
        // (= inertial→child body for leg 3), and ang_vel_this is in
        // the derived frame's coordinates (= child body for leg 3).
        assert!(
            approx_eq_mat3(
                &out.child_t_inertial_body,
                &child_body_state.rot.t_parent_this,
                1e-12
            ),
            "T_inertial_body mismatch:\n kernel: {:?}\n propagate: {:?}",
            out.child_t_inertial_body,
            child_body_state.rot.t_parent_this
        );
        assert!(
            approx_eq_vec3(
                out.child_ang_vel_body,
                child_body_state.rot.ang_vel_this,
                1e-12
            ),
            "ang_vel_body mismatch:\n kernel: {:?}\n propagate: {:?}",
            out.child_ang_vel_body,
            child_body_state.rot.ang_vel_this
        );
        assert!(
            approx_eq_vec3(
                out.child_position_inertial,
                child_body_state.trans.position,
                1e-9
            ),
            "position mismatch:\n kernel: {:?}\n propagate: {:?}",
            out.child_position_inertial,
            child_body_state.trans.position
        );
        assert!(
            approx_eq_vec3(
                out.child_velocity_inertial,
                child_body_state.trans.velocity,
                1e-9
            ),
            "velocity mismatch:\n kernel: {:?}\n propagate: {:?}",
            out.child_velocity_inertial,
            child_body_state.trans.velocity
        );
    }

    /// Typed-sibling round trip: same inputs as the
    /// `non_zero_offset_under_parent_omega_yields_omega_cross_r` case,
    /// but routed through `compute_kinematic_child_state_typed` so the
    /// `Position<RootInertial>` / `Velocity<RootInertial>` /
    /// `BodyAttitude<V>` / `AngularVelocity<BodyFrame<V>>` boundary is
    /// exercised.
    #[test]
    fn typed_sibling_matches_raw_kernel() {
        use jeod_quantities::frame::SelfRef;

        let omega = DVec3::new(0.0, 0.0, 1e-3);
        let offset = DVec3::new(1.0, 0.0, 0.0);
        let parent_pos = DVec3::new(7e6, 0.0, 0.0);
        let parent_vel = DVec3::new(0.0, 7500.0, 0.0);

        let parent_q = BodyAttitude::<SelfRef>::identity();
        let parent_omega_typed = AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(omega);
        let parent_pos_typed = Position::<RootInertial>::from_raw_si(parent_pos);
        let parent_vel_typed = Velocity::<RootInertial>::from_raw_si(parent_vel);

        let (child_q, child_omega, child_pos, child_vel) =
            compute_kinematic_child_state_typed::<SelfRef, SelfRef>(
                parent_q,
                parent_omega_typed,
                parent_pos_typed,
                parent_vel_typed,
                DMat3::IDENTITY,
                DVec3::ZERO,
                DMat3::IDENTITY,
                offset,
                DMat3::IDENTITY,
                DVec3::ZERO,
            );

        // Same expectations as the raw-DVec3 case.
        assert!(approx_eq_vec3(child_pos.raw_si(), parent_pos + offset, TOL));
        let expected_vel = parent_vel + DVec3::new(0.0, 1e-3, 0.0);
        assert!(approx_eq_vec3(child_vel.raw_si(), expected_vel, TOL));
        assert!(approx_eq_vec3(child_omega.raw_si(), omega, TOL));
        // Identity rotation in, identity rotation out.
        assert!(approx_eq_mat3(
            &child_q.as_witness().left_quat_to_transformation(),
            &DMat3::IDENTITY,
            TOL,
        ));
    }
}
