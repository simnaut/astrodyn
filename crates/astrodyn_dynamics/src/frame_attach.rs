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
use astrodyn_frames::RefFrameState;

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
    /// **structural** frame: `position` is the structure-frame origin
    /// in parent-frame coordinates, `t_parent_this` is the rotation
    /// from parent-frame axes to structure-frame axes. Mirrors the
    /// `RefFrameState` that JEOD's
    /// `frame_attach.initialize_attachment(parent, X_pframe_to_struct)`
    /// captures at attach time
    /// (`models/dynamics/dyn_body/src/dyn_body_attach.cc:293-300, 367-378`).
    pub attach_offset: MassPointState,
    /// Body's structure→composite-body offset, i.e. the composite CoM
    /// position and orientation in the body's structural frame. This
    /// is JEOD's `mass.composite_properties` (position is the
    /// composite CoM in struct coords; `t_parent_this` is the
    /// structure→body rotation, normally identity).
    ///
    /// The kernel applies a second rigid-body composition after the
    /// parent⊕attach_offset step to derive the composite-body inertial
    /// state from the structure-inertial state. Mirrors JEOD's
    /// `propagate_state_from_structure` →
    /// `compute_derived_state_forward(structure, mass.composite_properties,
    /// composite_body)` in `dyn_body_propagate_state.cc:564-642`.
    /// Without this step, callers that capture a struct-frame offset
    /// in the parent (the JEOD convention) would land the body
    /// `|composite_properties.position|` away from the correct
    /// composite-body inertial pose — exactly the surface-attach
    /// residual the matrix and named-point overloads exhibit when
    /// the body's structure → CoM offset is nonzero.
    ///
    /// Pass `MassPointState::default()` (zero offset, identity
    /// rotation) for atomic point-mass bodies whose CoM coincides
    /// with the structural origin; the second `propagate_forward`
    /// step is then a numerical no-op.
    pub composite_offset: MassPointState,
}

/// Compute the attached body's composite-body inertial state from the
/// parent reference frame's current state, the (frozen-at-attach)
/// rigid-body parent→structure offset, and the body's structure→composite
/// CoM offset.
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
/// Two-step composition mirroring JEOD:
/// 1. `parent_frame ⊕ attach_offset` → structure-frame inertial state,
///    matching `dyn_body_integration.cc:325-332`'s rebuild of
///    `X_iframe_to_struct = X_iframe_to_pframe ⊕ attach_offset`.
/// 2. `structure_state ⊕ composite_offset` → composite-body inertial
///    state, matching `dyn_body_propagate_state.cc:571`'s
///    `compute_derived_state_forward(structure, mass.composite_properties,
///    composite_body)` call.
///
/// Both steps are the existing `propagate_forward` kernel; only the
/// upstream input differs (frame tree vs. mass tree), and the second
/// step is a numerical no-op when `composite_offset` is the default.
// JEOD_INV: DB.31 — frame-attach derives composite from captured struct
// offset (struct → composite CoM correction at capture time). Mirrors
// `dyn_body_propagate_state.cc:571` (`propagate_state_from_structure`'s
// `compute_derived_state_forward(structure, mass.composite_properties,
// composite_body)` call).
pub fn derive_frame_attached_state(input: FrameAttachInputs) -> RefFrameState {
    let structure_state = propagate_forward(&input.parent_frame, &input.attach_offset);
    propagate_forward(&structure_state, &input.composite_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_frames::{RefFrameRot, RefFrameTrans};
    use astrodyn_math::JeodQuat;
    use glam::DVec3;

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
            composite_offset: MassPointState::default(),
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
            composite_offset: MassPointState::default(),
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

    /// Non-zero `composite_offset` shifts the kernel's output by
    /// `T_inertial_struct^T · composite_offset.position` relative to the
    /// pure-struct case. Pins JEOD's `propagate_state_from_structure`
    /// → `compute_derived_state_forward(structure, composite_properties,
    /// composite_body)` step (`dyn_body_propagate_state.cc:571`):
    /// the captured offset is the parent → struct pose, and the
    /// composite-body pose is derived from struct via the body's
    /// structure → CoM offset. SIM_dyncomp's mass fixture uses
    /// `composite_properties.position = (-3, -1.5, 4)` (5.22 m
    /// magnitude); a missing second propagate_forward step lands the
    /// body that far off JEOD's composite-body trajectory on every
    /// surface-attach event.
    #[test]
    fn composite_offset_applies_struct_to_com_correction() {
        // Parent at the origin with identity attitude — so the
        // captured `attach_offset` lands directly as the body's
        // structure-frame state in inertial.
        let parent = RefFrameState::default();
        // Struct origin sits at (1e6, 0, 0) in inertial with identity
        // attitude; mirrors the matrix-attach overload's input.
        let attach_offset = MassPointState {
            position: DVec3::new(1.0e6, 0.0, 0.0),
            t_parent_this: JeodQuat::identity().left_quat_to_transformation(),
        };
        // Body's CoM is offset by (-3, -1.5, 4) m from the struct
        // origin in struct coordinates — exactly JEOD's
        // `set_mass_iss()` ISS fixture.
        let composite_offset = MassPointState {
            position: DVec3::new(-3.0, -1.5, 4.0),
            t_parent_this: JeodQuat::identity().left_quat_to_transformation(),
        };

        let derived = derive_frame_attached_state(FrameAttachInputs {
            parent_frame: parent,
            attach_offset,
            composite_offset,
        });

        // Identity attitudes throughout, so the kernel's composite
        // position is the struct position plus the CoM offset
        // verbatim.
        let expected = attach_offset.position + composite_offset.position;
        assert!(
            (derived.trans.position - expected).length() < 1e-12,
            "expected {expected:?}, got {:?}",
            derived.trans.position
        );

        // Sanity: dropping the composite offset back to default
        // returns the body to the captured struct pose, demonstrating
        // the magnitude of the correction (5.22 m for this fixture).
        let no_correction = derive_frame_attached_state(FrameAttachInputs {
            parent_frame: parent,
            attach_offset,
            composite_offset: MassPointState::default(),
        });
        let delta = (derived.trans.position - no_correction.trans.position).length();
        let expected_delta = composite_offset.position.length();
        assert!(
            (delta - expected_delta).abs() < 1e-12,
            "CoM correction magnitude should be {expected_delta} m, got {delta}"
        );
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
            composite_offset: MassPointState::default(),
        });
        let recovered = propagate_reverse(&derived, &offset);
        assert!((recovered.trans.position - parent.trans.position).length() < 1e-6);
        assert!((recovered.trans.velocity - parent.trans.velocity).length() < 1e-6);
        assert!((recovered.rot.ang_vel_this - parent.rot.ang_vel_this).length() < 1e-9);
    }
}
