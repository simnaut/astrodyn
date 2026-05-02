//! Mass-tree attach: combine two free bodies' kinematic states into a
//! single rigid body via JEOD's "magical" merge algorithm.
//!
//! Port of `DynBody::attach_child` lines 967–1106 of
//! `models/dynamics/dyn_body/src/dyn_body_attach.cc` (JEOD v5.4). Pure
//! function — no I/O, no mutation of the inputs.
//!
//! ### What the algorithm does
//!
//! 1. Conservation of linear momentum about the integration-frame origin
//!    gives the new combined-body velocity.
//! 2. The composite-body inertial position is shifted by
//!    `T_struct_to_inertial · (new_cm − orig_cm)` to track the
//!    structurally-fixed CoM.
//! 3. Conservation of angular momentum about the merged composite CoM
//!    gives the new combined-body angular velocity. The angular momentum
//!    sums four terms: each sub-body's spin (`I·ω`) plus each sub-body's
//!    moment-of-relative-translational-momentum (`r × p`) about the new
//!    CoM.
//! 4. The composite attitude is unchanged (per JEOD's comment "the new
//!    parent body frame is colinear with the original body frame, just
//!    offset and moving").
//!
//! ### Why it is non-physical
//!
//! Real spacecraft docking dissipates relative-translational velocity
//! through compliance and contact forces; mission-engineering practice
//! is to control closing rate to ~cm/s before contact precisely so the
//! `r × p` terms stay small. JEOD's implementation makes no attempt to
//! reject relative velocity — whatever the sub-trees were doing at the
//! instant before merger is absorbed into the merged body's spin. The
//! original C++ comment acknowledges this:
//!
//! > "Shift to a reference frame in which v_t = 0 and r_t = 0 (due to
//! > 'magical' migration of child, this is a non-physical problem, and
//! > angular momentum cannot be conserved in all frames simultaneously,
//! > this is the best choice)."
//!
//! For coarse mass-tree exercises (`SIM_Apollo`) this is fine because
//! the rotational state is incidental. For trajectory cross-validation
//! we have to model the same impulsive update so our composite-body
//! state matches what JEOD records.

use crate::mass::MassProperties;
use glam::{DMat3, DVec3};
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};

/// Inputs to [`combine_states_at_attach`].
///
/// All states are in the *integration frame* (typically Earth.inertial).
/// Both `parent` and `child` describe their *composite_body* frames —
/// the inertial state of the body's composite CoM and the orientation
/// of its principal-inertia axes.
#[derive(Debug, Clone, Copy)]
pub struct AttachCombineInputs {
    /// Parent's pre-attach composite-body inertial state.
    pub parent_composite: RefFrameState,
    /// Parent's pre-attach composite mass properties (mass, inertia, CoM).
    pub parent_mass: MassProperties,
    /// Parent's structure-frame transform — `t_inertial_to_struct`.
    /// Used to rotate the structure-frame CoM-shift vector
    /// (`new_cm − orig_cm`) into the inertial frame.
    pub parent_t_inertial_struct: DMat3,

    /// Child's pre-attach composite-body inertial state.
    pub child_composite: RefFrameState,
    /// Child's pre-attach composite mass properties.
    pub child_mass: MassProperties,

    /// Post-attach composite mass properties of the *combined* body
    /// (caller computes this by running the topology-only mass-tree
    /// update first; `position` is the new CoM in parent-structure frame
    /// and `inverse_inertia` is the new combined inertia).
    pub combined_mass: MassProperties,
    /// Pre-attach parent CoM position in parent-structure frame
    /// (i.e. `parent_mass.position` *before* the tree update —
    /// `combined_mass.position` is the post-attach value).
    pub orig_parent_cm_struct: DVec3,
}

/// Output of [`combine_states_at_attach`].
#[derive(Debug, Clone, Copy)]
pub struct AttachCombineOutputs {
    /// Post-attach composite-body inertial state of the merged root.
    /// `attitude` is unchanged from the parent's pre-attach attitude;
    /// `position` is shifted by the inertial CoM-delta; `velocity` is
    /// the mass-weighted average; `ang_vel_this` is the angular-momentum
    /// solution in the new combined body frame (= parent body frame).
    pub composite_state: RefFrameState,
}

/// Combine two free-body states into one rigid body, conserving linear
/// momentum (about the integration-frame origin) and angular momentum
/// (about the new combined CoM). See module docs for the algorithm and
/// JEOD source mapping.
pub fn combine_states_at_attach(input: AttachCombineInputs) -> AttachCombineOutputs {
    let parent = &input.parent_composite;
    let child = &input.child_composite;

    let m_p = input.parent_mass.mass;
    let m_c = input.child_mass.mass;
    let inv_m_t = input.combined_mass.inverse_mass;
    let inv_i_t = input.combined_mass.inverse_inertia;

    let v_p = parent.trans.velocity;
    let v_c = child.trans.velocity;
    let w_p = parent.rot.ang_vel_this;
    let w_c = child.rot.ang_vel_this;

    // ── Step 4 — linear velocity from linear-momentum conservation ──
    // p_total = m_p v_p + m_c v_c → v_t = p_total / m_total.
    let v_t = (v_p * m_p + v_c * m_c) * inv_m_t;

    // ── Step 3 — composite-body position shift ──
    // cm_delta_struct = new_cm − orig_cm  (in parent-structure frame)
    // cm_delta_inrtl  = T_struct_to_inertial · cm_delta_struct
    //                 = (T_inertial_to_struct)ᵀ · cm_delta_struct
    let cm_delta_struct = input.combined_mass.position - input.orig_parent_cm_struct;
    let cm_delta_inertial = input.parent_t_inertial_struct.transpose() * cm_delta_struct;
    let new_composite_position = parent.trans.position + cm_delta_inertial;

    // ── Step 5 — angular-momentum conservation in inertial axes ──
    // L_total = (r_p × p_p_rel) + I_p ω_p + (r_c × p_c_rel) + I_c ω_c
    // where r_x is the sub-body's pre-attach composite position relative
    // to the new CoM, and p_x_rel is its momentum in the merged-CoM frame.

    // Spin angular momenta in respective body frames.
    let l_p_pbdy = input.parent_mass.inertia * w_p;
    let l_c_cbdy = input.child_mass.inertia * w_c;

    // Child's spin angular momentum in inertial axes.
    //   T_child_to_inertial · L_cbody = T_parent_thisᵀ · L_cbody
    let l_c_inertial = child.rot.t_parent_this.transpose() * l_c_cbdy;

    // Child's pre-attach inertial composite-CoM position relative to the
    // new combined CoM (post-shift).
    let r_c_rel = child.trans.position - new_composite_position;

    // Sub-body velocities relative to the merged-CoM rest frame.
    let v_c_rel = v_c - v_t;
    let v_p_rel = v_p - v_t;

    // Moments of relative linear momentum about the new CoM.
    //   L_c_vel = r_c × (m_c v_c_rel)
    //   L_p_vel = r_p × (m_p v_p_rel)
    // For the parent, r_p_rel = parent_pre_pos − new_pos = −cm_delta_inertial,
    // so L_p_vel = −cm_delta_inertial × p_p_rel = p_p_rel × cm_delta_inertial.
    let l_c_vel = r_c_rel.cross(v_c_rel * m_c);
    let l_p_vel = (v_p_rel * m_p).cross(cm_delta_inertial);

    // Sum the three inertial-axes contributions.
    let l_inertial = l_c_vel + l_p_vel + l_c_inertial;

    // Convert to the parent body frame (which equals the new combined
    // body frame — the merged body inherits the parent's attitude per
    // JEOD's "colinear" comment).
    let l_in_pbody = parent.rot.t_parent_this * l_inertial;

    // Add the parent's spin (already in parent body frame).
    let l_total_pbody = l_in_pbody + l_p_pbdy;

    // New angular velocity in the body frame.
    let ang_vel_this = inv_i_t * l_total_pbody;

    AttachCombineOutputs {
        composite_state: RefFrameState {
            trans: RefFrameTrans {
                position: new_composite_position,
                velocity: v_t,
            },
            rot: RefFrameRot {
                q_parent_this: parent.rot.q_parent_this,
                t_parent_this: parent.rot.t_parent_this,
                ang_vel_this,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A child with the same velocity, attitude, and ang_vel as the parent
    /// must produce a combined body with those same values (no spurious
    /// kicks when the merge is "soft").
    #[test]
    fn no_relative_motion_keeps_state() {
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
        // The combined CoM at the same point as the parent's (child also
        // at origin in struct frame, equal masses → CoM at origin scaled).
        // Easier path: child at (1,0,0) struct, parent at origin →
        // combined at (1*50)/(150) = 0.333 struct. We need to compute
        // post-attach mass props consistent with this. For the no-motion
        // case we sidestep by giving both bodies the same velocity etc.;
        // the test only checks v_t = v, w_t = w_p (when child also has
        // that w in inertial — i.e., when child's ang_vel_in_pbody is
        // the same vector and the rotation is identity).
        let combined_mass = MassProperties::with_inertia(
            150.0,
            DMat3::from_diagonal(DVec3::new(70.0, 70.0, 70.0)),
            DVec3::ZERO, // in parent struct frame
        );

        let v = DVec3::new(0.0, 7600.0, 0.0);
        let w = DVec3::new(0.0, -1.13e-3, 0.0);
        let parent = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: v,
            },
            rot: RefFrameRot {
                ang_vel_this: w,
                ..Default::default()
            },
        };
        let child = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: v,
            },
            rot: RefFrameRot {
                ang_vel_this: w,
                ..Default::default()
            },
        };
        let out = combine_states_at_attach(AttachCombineInputs {
            parent_composite: parent,
            parent_mass,
            parent_t_inertial_struct: DMat3::IDENTITY,
            child_composite: child,
            child_mass,
            combined_mass,
            orig_parent_cm_struct: DVec3::ZERO,
        });
        assert!((out.composite_state.trans.velocity - v).length() < 1e-9);
        assert!((out.composite_state.rot.ang_vel_this - w).length() < 1e-9);
    }

    /// Two bodies with relative translational velocity at non-zero offset
    /// from the new CoM produce a non-zero combined angular velocity (the
    /// "spurious" kick we are deliberately replicating).
    #[test]
    fn relative_velocity_at_offset_induces_spin() {
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
        // Equal masses → combined CoM halfway between. With child placed
        // 2 m along +x in parent struct, combined CoM at (1, 0, 0).
        let combined_mass = MassProperties::with_inertia(
            2000.0,
            DMat3::from_diagonal(DVec3::new(2000.0, 2000.0, 2000.0)),
            DVec3::new(1.0, 0.0, 0.0), // new CoM in parent struct
        );

        let parent = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            ..Default::default()
        };
        // Child at +2 m along x_struct, moving at +1 m/s along y.
        let child = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(2.0, 0.0, 0.0),
                velocity: DVec3::new(0.0, 1.0, 0.0),
            },
            ..Default::default()
        };
        let out = combine_states_at_attach(AttachCombineInputs {
            parent_composite: parent,
            parent_mass,
            parent_t_inertial_struct: DMat3::IDENTITY,
            child_composite: child,
            child_mass,
            combined_mass,
            orig_parent_cm_struct: DVec3::ZERO,
        });
        // Expected: v_t = (0 + 1*1000) / 2000 along y = 0.5 m/s
        // r_c_rel × m_c v_c_rel = (1,0,0) × (0, 500, 0) = (0,0,500)
        // r_p_rel × m_p v_p_rel = (-1,0,0) × (0, -500, 0) = (0,0,500)
        // L_total_inertial = (0,0,1000)
        // ω = I^-1 · L = (0, 0, 0.5) rad/s
        let expected_v = DVec3::new(0.0, 0.5, 0.0);
        let expected_w = DVec3::new(0.0, 0.0, 0.5);
        assert!(
            (out.composite_state.trans.velocity - expected_v).length() < 1e-9,
            "got {:?}",
            out.composite_state.trans.velocity
        );
        assert!(
            (out.composite_state.rot.ang_vel_this - expected_w).length() < 1e-9,
            "got {:?}",
            out.composite_state.rot.ang_vel_this
        );
    }
}
