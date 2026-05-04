//! Frame-attached body integration kernel.
//!
//! Port of JEOD's `DynBody::attach_to_frame` integration path
//! (`models/dynamics/dyn_body/src/dyn_body_integration.cc:282-342` and
//! the attachment-state initialization in
//! `models/dynamics/dyn_body/src/dyn_body_attach.cc:271-379`). When a
//! `DynBody` is attached to a parent reference frame (rather than to
//! another body), JEOD bypasses the translational + rotational
//! integrators and instead derives the body's structure-frame state
//! each tick from the parent reference frame's current state plus the
//! attach offset:
//!
//! ```text
//! X_struct(t) = X_parent_frame(t) ⊕ X_attach_offset
//! ```
//!
//! where `⊕` is the rigid-body composition implemented by
//! [`crate::propagation::propagate_forward`] — the same kernel that
//! drives mass-tree kinematic propagation. The composition is exact for
//! a structure rigidly fixed to the parent frame; both linear and
//! angular velocity follow from the parent's `ω × r` rigid-body
//! relation.
//!
//! ### Why this lives in a separate kernel from the mass-tree walk
//!
//! Mass-tree kinematic propagation (`kinematic_propagation.rs` /
//! `propagate_state_via_storage`) composes through structure points
//! belonging to a parent **body** that itself integrates. Frame
//! attachment composes through a parent **reference frame** whose
//! state is already authoritative (driven by ephemeris, planet
//! rotation, or kinematic joints — the simulation's frame tree is the
//! source of truth). The arithmetic is the same `propagate_forward`
//! call, but the upstream input is structurally distinct: `FrameTree`
//! versus `MassTree`. Keeping the two kernels apart lets each adapter
//! (Bevy `staging_system`, runner `Simulation::step_internal`) read
//! from the right structure without an `enum`-shaped fork inside the
//! per-link math.
//!
//! ### What this kernel does **not** do
//!
//! - It does not validate that the parent frame is non-inertial. JEOD
//!   permits inertial parents too (e.g., `attach_to_frame("Earth.pfix")`
//!   is rotating; `attach_to_frame("Earth.inertial")` is not — both are
//!   accepted). Validation of misuse (attaching to your own body frame,
//!   creating a cycle, attaching while already attached) belongs to the
//!   caller — see `Simulation::attach_to_frame` for the runtime gate.
//! - It does not reset integrator history. The runner / Bevy adapter
//!   call sites are responsible for clearing GJ / ABM4 history so the
//!   transition from integrated to frame-attached doesn't leave stale
//!   predictor state behind (JEOD `reset_integrators()` at
//!   `dyn_body_attach.cc:860,871`). The pattern matches what
//!   `Simulation::attach` already does for mass-tree attaches.

use crate::mass_body::MassPointState;
use crate::propagation::propagate_forward;
use jeod_frames::RefFrameState;

/// Inputs to [`derive_frame_attached_state`].
///
/// The parent ref-frame state must be expressed in caller-chosen
/// inertial coordinates — the kernel composes it rigidly with the
/// captured offset and returns a state in the *same* coordinates.
/// The runner passes parent state in root-inertial coordinates (read
/// from `FrameTree::compute_relative_state(root, parent_frame_id)`)
/// and lowers the kernel's output through the body's `IntegOrigin`
/// only on writeback to the integration-frame storage; the Bevy
/// adapter follows the same pattern via the parent frame entity's
/// relative state.
#[derive(Debug, Clone, Copy)]
pub struct FrameAttachInputs {
    /// Parent reference frame's current inertial state (position,
    /// velocity, attitude, angular velocity), in the caller's chosen
    /// inertial coordinates. The kernel does not apply any
    /// integration-origin shift — that lives in the call site, after
    /// the kernel returns the body's composite-body state in those
    /// same coordinates.
    pub parent_frame: RefFrameState,
    /// Rigid-body offset from the parent frame to the attached body's
    /// composite-body frame: `position` is the attach point in
    /// parent-frame coordinates, `t_parent_this` is the rotation from
    /// parent-frame axes to body-frame axes. Mirrors the
    /// `RefFrameState` that JEOD's
    /// `frame_attach.initialize_attachment(parent, X_pframe_to_struct)`
    /// captures at attach time.
    pub attach_offset: MassPointState,
}

/// Compute the attached body's composite-body inertial state from the
/// parent reference frame's current state and the (frozen-at-attach)
/// rigid-body offset.
///
/// Pure function: no side effects, no I/O, no mutation. Call sites are
/// the per-step "frame-attached body update" pass in
/// `Simulation::step_internal` (runner) and, in the Bevy adapter, the
/// per-tick `propagate_frame_attached_state_system` (pre-integration)
/// plus its post-integration twin
/// `propagate_frame_attached_state_post_integration_system`. The two
/// Bevy registrations mirror the runner's pre- and post-integration
/// invocations so derived-state consumers always observe a body whose
/// state reflects the parent frame's just-finished intra-step updates.
///
/// Implementation is the existing `propagate_forward` kernel — the
/// algebra is bit-identical to the mass-tree kinematic walk's per-link
/// composition. The only difference is which structure (`FrameTree`
/// vs. `MassTree`) the upstream parent state was read from.
pub fn derive_frame_attached_state(input: FrameAttachInputs) -> RefFrameState {
    propagate_forward(&input.parent_frame, &input.attach_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use jeod_frames::{RefFrameRot, RefFrameTrans};
    use jeod_math::JeodQuat;

    /// Identity offset (zero translation, identity rotation, zero
    /// angular velocity contribution) reproduces the parent frame's
    /// state verbatim. Sanity check that the kernel composes the
    /// expected null operation.
    #[test]
    fn identity_offset_returns_parent_state() {
        let parent = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7600.0, 0.0),
            },
            rot: RefFrameRot {
                q_parent_this: JeodQuat::identity(),
                t_parent_this: JeodQuat::identity().left_quat_to_transformation(),
                ang_vel_this: DVec3::new(0.0, 0.0, 7.292115e-5),
            },
        };
        let offset = MassPointState::default();
        let derived = derive_frame_attached_state(FrameAttachInputs {
            parent_frame: parent,
            attach_offset: offset,
        });
        assert!((derived.trans.position - parent.trans.position).length() < 1e-9);
        assert!((derived.trans.velocity - parent.trans.velocity).length() < 1e-9);
        assert!((derived.rot.ang_vel_this - parent.rot.ang_vel_this).length() < 1e-9);
    }

    /// A body attached at a fixed offset from a rotating parent picks
    /// up an `ω × r` velocity contribution. Test with the canonical
    /// Earth pfix rotation rate to mirror SIM_ref_attach's
    /// "attach_to_frame(Earth.pfix)" use case at order-of-magnitude.
    #[test]
    fn pfix_attached_body_picks_up_omega_cross_r_velocity() {
        let omega_earth = 7.292115e-5_f64;
        // Parent frame at planet center, rotating at Earth's sidereal rate.
        let parent = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            rot: RefFrameRot {
                q_parent_this: JeodQuat::identity(),
                t_parent_this: JeodQuat::identity().left_quat_to_transformation(),
                ang_vel_this: DVec3::new(0.0, 0.0, omega_earth),
            },
        };
        // Body attached at the equator, 1000 m east of the prime
        // meridian (along +x in pfix).
        let r = 6.371e6;
        let offset = MassPointState {
            position: DVec3::new(r, 0.0, 0.0),
            t_parent_this: JeodQuat::identity().left_quat_to_transformation(),
        };
        let derived = derive_frame_attached_state(FrameAttachInputs {
            parent_frame: parent,
            attach_offset: offset,
        });
        // Position equals the offset (parent at origin, identity rotation).
        assert!((derived.trans.position - DVec3::new(r, 0.0, 0.0)).length() < 1e-9);
        // Velocity is ω × r = (0, 0, ω) × (r, 0, 0) = (0, ω·r, 0).
        let expected_v = DVec3::new(0.0, omega_earth * r, 0.0);
        assert!(
            (derived.trans.velocity - expected_v).length() < 1e-6,
            "got {:?} expected {:?}",
            derived.trans.velocity,
            expected_v
        );
        // Angular velocity matches the parent (rigid attachment).
        assert!((derived.rot.ang_vel_this - parent.rot.ang_vel_this).length() < 1e-12);
    }

    /// Detach is exact: applying [`crate::propagation::propagate_reverse`]
    /// to the derived state with the same offset must round-trip back
    /// to the parent state. Pins the algebraic identity that lets a
    /// detached body resume integration from its instantaneous
    /// composite state without losing fidelity to the parent's
    /// rigid-body relation.
    #[test]
    fn forward_propagate_round_trips_via_reverse() {
        use crate::propagation::propagate_reverse;
        let q_parent =
            JeodQuat::left_quat_from_eigen_rotation(0.5, DVec3::new(0.2, 0.7, 1.0).normalize());
        let parent = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(7e6, 1e5, -3e4),
                velocity: DVec3::new(7300.0, -50.0, 13.0),
            },
            rot: RefFrameRot {
                q_parent_this: q_parent,
                t_parent_this: q_parent.left_quat_to_transformation(),
                ang_vel_this: DVec3::new(1e-4, -2e-4, 3e-4),
            },
        };
        let q_offset =
            JeodQuat::left_quat_from_eigen_rotation(0.3, DVec3::new(1.0, -0.4, 0.2).normalize());
        let offset = MassPointState {
            position: DVec3::new(2.5, -1.0, 0.5),
            t_parent_this: q_offset.left_quat_to_transformation(),
        };
        let derived = derive_frame_attached_state(FrameAttachInputs {
            parent_frame: parent,
            attach_offset: offset,
        });
        let recovered = propagate_reverse(&derived, &offset);
        assert!((recovered.trans.position - parent.trans.position).length() < 1e-6);
        assert!((recovered.trans.velocity - parent.trans.velocity).length() < 1e-6);
        assert!((recovered.rot.ang_vel_this - parent.rot.ang_vel_this).length() < 1e-9);
    }
}
