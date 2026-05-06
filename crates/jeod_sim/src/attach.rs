//! Orchestration: stage and apply a JEOD-faithful mass-tree attach /
//! detach in an ECS-agnostic context.
//!
//! The pure momentum-conservation kernel
//! ([`jeod_dynamics::combine_states_at_attach`]) lives one layer down
//! in [`jeod_dynamics`]; the integrator-history reset hook
//! ([`crate::reset_integrators`]) lives in this crate. This module
//! pulls them together into helpers that an ECS adapter (or the
//! `jeod_runner` arena consumer) can call without re-deriving the
//! input plumbing.
//!
//! ## What "orchestration" means here
//!
//! The kernel takes pre-attach states + masses for parent and child
//! plus the *post-attach* combined mass and returns the merged
//! composite-body inertial state. The orchestration concern is how
//! the caller sequences:
//!
//! 1. snapshot pre-attach state from storage,
//! 2. mutate the topology (this also recomputes composite mass props),
//! 3. read the post-attach combined mass,
//! 4. run the kernel,
//! 5. write the merged state back into the parent's storage,
//! 6. reset multi-step integrator history (IG.37).
//!
//! Steps 2 and 6 are storage-coupled, so this module exposes the
//! between-snapshot-and-write pure-data helpers
//! ([`stage_attach_combine`]) and leaves the topology mutation +
//! write-back to the calling adapter.
//!
//! For detach, the symmetric helper [`stage_detach_capture`] returns
//! the [`DetachedSubtreeState`] derived from the subtree's pre-detach
//! composite-body inertial state — what the caller then attaches to
//! the ECS (or stores in a runner-side map) so the detached subtree
//! can drift ballistically.
//!
//! ## Cross-integration-frame attach
//!
//! When the parent and child of an attach resolve to different
//! integration frames, the merged body switches integration frame —
//! the child (and every kinematic descendant of the child in the mass
//! tree) must be reinterpreted under the parent's integ frame post-
//! attach. JEOD's `dyn_body_integration.cc::set_integ_frame` performs
//! the topology reparent only ("does not update state"); the next
//! `propagate_state()` pass refills descendants' parent-relative storage
//! from the merged root.
//!
//! Adapters that lack JEOD's same-call `propagate_state` recursion
//! (e.g. the Bevy adapter, where the kinematic propagation walk runs
//! many systems later) must additionally **rewrite each descendant's
//! stored translational state** from the old integ frame's coordinates
//! into the new frame's coordinates inside the staging window. The pure
//! arithmetic of that rewrite is captured by [`CrossIntegFrameStateShift`]:
//! a typed translation between two non-rotating integration frames whose
//! root-inertial origins are known. Both legitimate integ-frame entities
//! in this codebase are non-rotating (`RootInertial` or
//! `PlanetInertial<P>`), so the rewrite is a pure translation — no
//! attitude or angular-velocity correction is required.

use jeod_dynamics::{
    combine_states_at_attach, AttachCombineInputs, AttachCombineOutputs, DetachedSubtreeState,
    MassProperties,
};
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_math::JeodQuat;
use jeod_quantities::quat::NormalizedQuat;

/// Internal type alias used solely so the unit-norm check in
/// [`stage_detach_capture`] can name a concrete `NormalizedQuat`
/// without needing the caller to thread `Layout` / `Transform` type
/// parameters through the orchestration boundary. The witness value
/// is discarded — only the panic-on-non-unit side-effect matters.
type ScalarFirstLeftNormalizedQuat =
    NormalizedQuat<jeod_quantities::quat::ScalarFirst, jeod_quantities::quat::LeftTransform>;

/// Inputs to [`stage_attach_combine`].
///
/// All composite states are in the simulation's integration frame
/// (typically the central body's inertial frame). `parent_*` and
/// `child_*` are the pre-attach snapshots; `combined_mass` is the
/// recomputed composite mass after the topology mutation has been
/// applied to the storage (e.g. `MassTree::attach` has already run);
/// `parent_t_inertial_struct` is the pre-attach
/// `T_inertial_to_struct` for the parent.
#[derive(Debug, Clone, Copy)]
pub struct StageAttachInputs {
    /// Parent's pre-attach composite-body inertial position.
    pub parent_position: glam::DVec3,
    /// Parent's pre-attach composite-body inertial velocity.
    pub parent_velocity: glam::DVec3,
    /// Parent's pre-attach inertial-to-body attitude quaternion
    /// (scalar-first, left-transformation).
    pub parent_quaternion: JeodQuat,
    /// Parent's pre-attach body-frame angular velocity.
    pub parent_ang_vel_body: glam::DVec3,
    /// Parent's pre-attach composite mass properties.
    pub parent_mass: MassProperties,
    /// Pre-attach parent CoM offset in the parent's structural frame
    /// (i.e. `parent_mass.position` *before* the tree mutation).
    pub orig_parent_cm_struct: glam::DVec3,
    /// Pre-attach `T_inertial_to_struct` for the parent body, used to
    /// rotate the structure-frame CoM-shift vector into the inertial
    /// frame for the position update.
    pub parent_t_inertial_struct: glam::DMat3,

    /// Child's pre-attach composite-body inertial position.
    pub child_position: glam::DVec3,
    /// Child's pre-attach composite-body inertial velocity.
    pub child_velocity: glam::DVec3,
    /// Child's pre-attach inertial-to-body attitude quaternion.
    pub child_quaternion: JeodQuat,
    /// Child's pre-attach body-frame angular velocity.
    pub child_ang_vel_body: glam::DVec3,
    /// Child's pre-attach composite mass properties.
    pub child_mass: MassProperties,

    /// Post-attach composite mass properties of the *combined* body
    /// (caller computes this by running the topology-only mass-tree
    /// recomposition first).
    pub combined_mass: MassProperties,
}

/// Outputs of [`stage_attach_combine`] — the merged composite-body
/// inertial state ready for the caller to write back into the parent's
/// storage.
#[derive(Debug, Clone, Copy)]
pub struct StageAttachOutputs {
    /// Post-attach composite-body inertial position of the merged
    /// root.
    pub position: glam::DVec3,
    /// Post-attach composite-body inertial velocity of the merged
    /// root (mass-weighted average of the pre-attach pair).
    pub velocity: glam::DVec3,
    /// Post-attach inertial-to-body attitude quaternion. Per JEOD's
    /// `dyn_body_attach.cc` "the new parent body frame is colinear
    /// with the original body frame, just offset and moving" — i.e.
    /// the merged body inherits the parent's attitude.
    pub quaternion: JeodQuat,
    /// Post-attach body-frame angular velocity, solving the angular-
    /// momentum conservation about the new combined CoM.
    pub ang_vel_body: glam::DVec3,
}

/// Run the JEOD momentum-conservation combine algorithm on a pair of
/// pre-attach body states.
///
/// Pure function — no I/O, no storage mutation. The caller is
/// responsible for having already mutated the mass-tree topology
/// (which yields `combined_mass`) and for writing the returned
/// composite-body state back into the merged body's storage.
///
/// Wraps [`jeod_dynamics::combine_states_at_attach`] with a flatter
/// argument shape that an ECS adapter can populate directly from
/// component reads, without first assembling
/// [`jeod_frames::RefFrameState`] structs.
pub fn stage_attach_combine(input: StageAttachInputs) -> StageAttachOutputs {
    let parent_t_parent_this = input.parent_quaternion.left_quat_to_transformation();
    let child_t_parent_this = input.child_quaternion.left_quat_to_transformation();

    let parent_composite = RefFrameState {
        trans: RefFrameTrans {
            position: input.parent_position,
            velocity: input.parent_velocity,
        },
        rot: RefFrameRot {
            q_parent_this: input.parent_quaternion,
            t_parent_this: parent_t_parent_this,
            ang_vel_this: input.parent_ang_vel_body,
        },
    };
    let child_composite = RefFrameState {
        trans: RefFrameTrans {
            position: input.child_position,
            velocity: input.child_velocity,
        },
        rot: RefFrameRot {
            q_parent_this: input.child_quaternion,
            t_parent_this: child_t_parent_this,
            ang_vel_this: input.child_ang_vel_body,
        },
    };

    let AttachCombineOutputs { composite_state } = combine_states_at_attach(AttachCombineInputs {
        parent_composite,
        parent_mass: input.parent_mass,
        parent_t_inertial_struct: input.parent_t_inertial_struct,
        child_composite,
        child_mass: input.child_mass,
        combined_mass: input.combined_mass,
        orig_parent_cm_struct: input.orig_parent_cm_struct,
    });

    StageAttachOutputs {
        position: composite_state.trans.position,
        velocity: composite_state.trans.velocity,
        quaternion: composite_state.rot.q_parent_this,
        ang_vel_body: composite_state.rot.ang_vel_this,
    }
}

/// Capture a subtree's pre-detach composite-body inertial state into
/// a [`DetachedSubtreeState`] suitable for ballistic propagation.
///
/// The detached subtree's instantaneous inertial state at the detach
/// instant *is* its composite-body state immediately before the
/// topology change — the rigid-body composite hasn't moved yet, only
/// the parent's mass distribution has. This helper exists so the
/// caller doesn't have to assemble a [`RefFrameState`] just to call
/// [`DetachedSubtreeState::from_ref_frame_state`].
///
/// `quaternion` must be unit-norm to within
/// [`NormalizedQuat::DEFAULT_TOLERANCE`] (the integrators' end-of-step
/// `normalize_integ` and the frame-tree assembly code both satisfy
/// this; mission code that constructs a quaternion by hand should
/// call `JeodQuat::normalize` first). Panics with a fail-loud
/// diagnostic if the input is not unit-norm.
pub fn stage_detach_capture(
    position: glam::DVec3,
    velocity: glam::DVec3,
    quaternion: JeodQuat,
    ang_vel_body: glam::DVec3,
) -> DetachedSubtreeState {
    let _: ScalarFirstLeftNormalizedQuat = NormalizedQuat::new(quaternion).unwrap_or_else(|err| {
        panic!(
            "stage_detach_capture: pre-detach quaternion is not unit-norm: {err}. \
             Call `JeodQuat::normalize` on the input before passing it here \
             (the integrators' end-of-step normalize_integ guarantees this for \
             integrated bodies — mission code constructing detach state by hand \
             must do the same)."
        )
    });
    DetachedSubtreeState::from_ref_frame_state(&RefFrameState {
        trans: RefFrameTrans { position, velocity },
        rot: RefFrameRot {
            q_parent_this: quaternion,
            t_parent_this: quaternion.left_quat_to_transformation(),
            ang_vel_this: ang_vel_body,
        },
    })
}

/// Pure translation between two non-rotating integration frames whose
/// root-inertial origins are known.
///
/// Built from a body's *pre-attach* integ-frame origin and the
/// post-attach (parent's) integ-frame origin, both expressed in
/// root-inertial coordinates. The shift carried by this struct is
/// `old - new`: applying it to a `(position, velocity)` pair stored in
/// the *old* integ frame's coordinates yields the same physical pose
/// re-expressed in the *new* integ frame's coordinates.
///
/// Used by the cross-integration-frame attach/detach path on every
/// adapter that needs to translate a body's stored translational state
/// across an integ-frame switch:
///
/// - The Bevy adapter calls this **per descendant** in the child's
///   pre-attach mass-tree subtree, because each descendant may have
///   had a different pre-attach integ frame than the new parent's and
///   each one's stored coordinates must be re-expressed in the new
///   parent's integ frame so consumers running between staging and the
///   next kinematic propagation pass don't read pre-attach numerics
///   through post-attach frame-tree topology.
///
/// - The runner's `Simulation::attach` / `Simulation::detach` call this
///   **per side** at the kernel boundary: the seed-time lift uses
///   `between_integ_origins(body_integ_origin, ZERO)` to re-express
///   stored integ-frame state in root-inertial coordinates before
///   `combine_states_at_attach` runs, and the writeback uses the
///   symmetric `between_integ_origins(ZERO, body_integ_origin)` to
///   lower the merged root-inertial result back into integ-frame
///   storage.
///
/// JEOD's `dyn_body_integration.cc::set_integ_frame` only reparents the
/// frame nodes ("does not update state") and relies on a same-call
/// `propagate_state()` recursion; adapters without that recursion do
/// the translation here.
///
/// The translation is valid only because every legitimate integ-frame
/// entity in this codebase is non-rotating (`RootInertial` or a planet
/// inertial frame, all co-aligned with root inertial axes by the frame
/// tree's construction). A rotating integ frame would additionally
/// require an attitude / angular-velocity rewrite — that case is not
/// supported and is structurally rejected upstream by the Bevy
/// adapter's cross-integ-frame fence.
///
/// All inputs and outputs are raw `glam::DVec3` (root-inertial for the
/// origin pairs, integration-frame for the body state). Typed quantity
/// shifts are available in [`jeod_quantities::IntegOrigin`]; this
/// helper sits one layer below to keep the per-descendant arithmetic
/// fast and to match the mass-tree kernel's untyped surface.
#[derive(Debug, Clone, Copy)]
pub struct CrossIntegFrameStateShift {
    shift_pos: glam::DVec3,
    shift_vel: glam::DVec3,
}

impl CrossIntegFrameStateShift {
    /// Build a state-shift from the body's pre-attach integ-frame
    /// origin and the post-attach (parent's) integ-frame origin, both
    /// expressed in root-inertial coordinates.
    ///
    /// Returns the no-op shift (zero pos, zero vel) when both origins
    /// agree — useful for adapters that uniformly funnel every
    /// descendant through this kernel without first checking whether
    /// the body actually changed integ frame.
    #[inline]
    pub fn between_integ_origins(
        old_origin_pos: glam::DVec3,
        old_origin_vel: glam::DVec3,
        new_origin_pos: glam::DVec3,
        new_origin_vel: glam::DVec3,
    ) -> Self {
        Self {
            shift_pos: old_origin_pos - new_origin_pos,
            shift_vel: old_origin_vel - new_origin_vel,
        }
    }

    /// Position component of the shift (`old_origin - new_origin`).
    #[inline]
    pub fn shift_pos(&self) -> glam::DVec3 {
        self.shift_pos
    }

    /// Velocity component of the shift (`old_origin - new_origin`).
    #[inline]
    pub fn shift_vel(&self) -> glam::DVec3 {
        self.shift_vel
    }

    /// Re-express a `(position, velocity)` pair stored in the *old*
    /// integ-frame's coordinates into the *new* integ-frame's
    /// coordinates by adding the shift component-wise.
    ///
    /// The pair is a pure translation between two co-aligned inertial
    /// frames, so the same translation applies to both stored
    /// `TranslationalState`-style payloads (the body's integ-frame
    /// trans state) and the body-frame entity's `FrameTrans`-style
    /// parent-relative payload — the per-frame rewrite at the reparent
    /// site can call this helper for both.
    #[inline]
    pub fn apply(
        &self,
        position: glam::DVec3,
        velocity: glam::DVec3,
    ) -> (glam::DVec3, glam::DVec3) {
        (position + self.shift_pos, velocity + self.shift_vel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, DVec3};

    /// `stage_attach_combine` must reproduce the kernel's output for a
    /// "no relative motion" merge: identical state on both bodies →
    /// composite has the same translational + rotational state.
    #[test]
    fn stage_attach_combine_preserves_state_on_soft_merge() {
        let parent_mass = MassProperties::with_inertia(
            100.0,
            DMat3::from_diagonal(DVec3::new(50.0, 50.0, 50.0)),
            DVec3::ZERO,
        );
        let child_mass = MassProperties::with_inertia(
            50.0,
            DMat3::from_diagonal(DVec3::new(20.0, 20.0, 20.0)),
            DVec3::ZERO,
        );
        let combined_mass = MassProperties::with_inertia(
            150.0,
            DMat3::from_diagonal(DVec3::new(70.0, 70.0, 70.0)),
            DVec3::ZERO,
        );

        let v = DVec3::new(0.0, 7600.0, 0.0);
        let w = DVec3::new(0.0, -1.13e-3, 0.0);
        let pos = DVec3::new(7e6, 0.0, 0.0);
        let q = JeodQuat::identity();

        let out = stage_attach_combine(StageAttachInputs {
            parent_position: pos,
            parent_velocity: v,
            parent_quaternion: q,
            parent_ang_vel_body: w,
            parent_mass,
            orig_parent_cm_struct: DVec3::ZERO,
            parent_t_inertial_struct: DMat3::IDENTITY,

            child_position: pos,
            child_velocity: v,
            child_quaternion: q,
            child_ang_vel_body: w,
            child_mass,

            combined_mass,
        });
        assert!((out.velocity - v).length() < 1e-9);
        assert!((out.ang_vel_body - w).length() < 1e-9);
    }

    /// Linear momentum conservation: equal-mass merge with relative
    /// translational velocity yields the average velocity, plus an
    /// induced spin matching the kernel's ω = I⁻¹ · L solution.
    #[test]
    fn stage_attach_combine_conserves_linear_and_angular_momentum() {
        let parent_mass = MassProperties::with_inertia(
            1000.0,
            DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
            DVec3::ZERO,
        );
        let child_mass = MassProperties::with_inertia(
            1000.0,
            DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
            DVec3::ZERO,
        );
        let combined_mass = MassProperties::with_inertia(
            2000.0,
            DMat3::from_diagonal(DVec3::new(2000.0, 2000.0, 2000.0)),
            DVec3::new(1.0, 0.0, 0.0),
        );

        let q = JeodQuat::identity();
        let out = stage_attach_combine(StageAttachInputs {
            parent_position: DVec3::ZERO,
            parent_velocity: DVec3::ZERO,
            parent_quaternion: q,
            parent_ang_vel_body: DVec3::ZERO,
            parent_mass,
            orig_parent_cm_struct: DVec3::ZERO,
            parent_t_inertial_struct: DMat3::IDENTITY,

            child_position: DVec3::new(2.0, 0.0, 0.0),
            child_velocity: DVec3::new(0.0, 1.0, 0.0),
            child_quaternion: q,
            child_ang_vel_body: DVec3::ZERO,
            child_mass,

            combined_mass,
        });
        let expected_v = DVec3::new(0.0, 0.5, 0.0);
        let expected_w = DVec3::new(0.0, 0.0, 0.5);
        assert!((out.velocity - expected_v).length() < 1e-9);
        assert!((out.ang_vel_body - expected_w).length() < 1e-9);
    }

    /// Detach capture: the subtree's instantaneous state should round-trip
    /// through `stage_detach_capture` byte-identically.
    #[test]
    fn stage_detach_capture_round_trips() {
        let pos = DVec3::new(7e6, 1.0, -3.0);
        let vel = DVec3::new(0.0, 7672.0, 0.0);
        let q = JeodQuat::identity();
        let omega = DVec3::new(0.001, -0.002, 0.0011);
        let captured = stage_detach_capture(pos, vel, q, omega);
        assert_eq!(captured.composite_position.raw_si(), pos);
        assert_eq!(captured.composite_velocity.raw_si(), vel);
        assert_eq!(captured.composite_ang_vel_body, omega);
    }

    /// Captured state propagates ballistically — position drifts at
    /// `velocity`, attitude rotates under `ang_vel`, velocity and
    /// `ang_vel` are unchanged.
    #[test]
    fn stage_detach_capture_propagates_ballistically() {
        let pos = DVec3::new(7e6, 0.0, 0.0);
        let vel = DVec3::new(0.0, 7672.0, 0.0);
        let q = JeodQuat::identity();
        let omega = DVec3::ZERO;
        let mut captured = stage_detach_capture(pos, vel, q, omega);
        let dt = 60.0;
        captured.step_ballistic(dt);
        assert!((captured.composite_position.raw_si() - (pos + vel * dt)).length() < 1e-9);
        assert!((captured.composite_velocity.raw_si() - vel).length() < 1e-9);
        assert!((captured.composite_ang_vel_body - omega).length() < 1e-9);
    }

    /// Fail-loudly: a non-unit quaternion must panic, not silently
    /// produce a `BodyAttitude` whose witness was never validated.
    #[test]
    #[should_panic(expected = "not unit-norm")]
    fn stage_detach_capture_panics_on_non_unit_quat() {
        let bad_q = JeodQuat::new(2.0, 0.0, 0.0, 0.0);
        let _ = stage_detach_capture(DVec3::ZERO, DVec3::ZERO, bad_q, DVec3::ZERO);
    }

    /// `between_integ_origins` returns the additive shift `old - new`,
    /// and `apply` adds that shift to a stored `(position, velocity)`
    /// pair. The composition of "subtract origin" → "add origin" is
    /// the typed translation between two non-rotating inertial frames.
    #[test]
    fn cross_integ_frame_state_shift_translates_state_between_origins() {
        let old_origin_pos = DVec3::new(1.5e11, 2.0, -3.0);
        let old_origin_vel = DVec3::new(0.1, 30_000.0, 0.0);
        let new_origin_pos = DVec3::new(1.0e11, -4.0, 2.5);
        let new_origin_vel = DVec3::new(-0.05, 25_000.0, 0.5);
        let shift = CrossIntegFrameStateShift::between_integ_origins(
            old_origin_pos,
            old_origin_vel,
            new_origin_pos,
            new_origin_vel,
        );
        assert_eq!(shift.shift_pos(), old_origin_pos - new_origin_pos);
        assert_eq!(shift.shift_vel(), old_origin_vel - new_origin_vel);

        // A body whose stored state is `(p_old, v_old)` in the OLD
        // frame represents the same physical pose as
        // `(p_old + (old_origin - new_origin), v_old + (old_origin_vel
        //  - new_origin_vel))` in the NEW frame.
        let p_old = DVec3::new(7e6, 0.0, 0.0);
        let v_old = DVec3::new(0.0, 7500.0, 0.0);
        let (p_new, v_new) = shift.apply(p_old, v_old);
        // Cross-check: the absolute root-inertial pose should be
        // identical no matter which integ-frame we read it from.
        assert!((p_old + old_origin_pos - (p_new + new_origin_pos)).length() < 1e-9);
        assert!((v_old + old_origin_vel - (v_new + new_origin_vel)).length() < 1e-9);
    }

    /// Same-origin → zero shift. Folds the same-integ-frame case into a
    /// bit-identical no-op without the caller having to special-case it.
    #[test]
    fn cross_integ_frame_state_shift_is_no_op_for_matching_origins() {
        let origin_pos = DVec3::new(1.5e11, 0.0, 0.0);
        let origin_vel = DVec3::new(0.0, 30_000.0, 0.0);
        let shift = CrossIntegFrameStateShift::between_integ_origins(
            origin_pos, origin_vel, origin_pos, origin_vel,
        );
        assert_eq!(shift.shift_pos(), DVec3::ZERO);
        assert_eq!(shift.shift_vel(), DVec3::ZERO);
        let p = DVec3::new(7e6, 0.0, 0.0);
        let v = DVec3::new(0.0, 7500.0, 0.0);
        let (p_new, v_new) = shift.apply(p, v);
        assert_eq!(p_new, p);
        assert_eq!(v_new, v);
    }
}
