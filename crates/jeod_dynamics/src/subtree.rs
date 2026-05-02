//! Composite-body kinematic state of a free-flying mass-tree subtree.
//!
//! Used by mass-tree consumers (e.g. `jeod_runner::Simulation`) to track
//! the inertial state of a subtree that has been detached from its
//! parent's tree and is coasting ballistically (no force, no torque)
//! before being re-attached. Pure rigid-body kinematics — no Simulation
//! or runner dependency.
//!
//! Relocated from `jeod_runner` in issue #253 to live alongside the
//! [`crate::advance_left_quat_body_rate`] helper this type's
//! `step_ballistic` method delegates to.

use glam::DVec3;
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_math::JeodQuat;

/// Composite-body inertial state of a free-flying mass-tree subtree
/// (i.e. a tree root that is not the integrated body). All fields are
/// in the simulation's integration frame (typically Earth.inertial).
#[derive(Debug, Clone, Copy)]
pub struct DetachedSubtreeState {
    /// Inertial position of the subtree's composite CoM.
    pub composite_position: DVec3,
    /// Inertial velocity of the subtree's composite CoM.
    pub composite_velocity: DVec3,
    /// Inertial-to-body rotation (`q_parent_this`, JEOD scalar-first
    /// left-quat convention — same orientation semantics as
    /// `RotationalState::quaternion`).
    pub composite_attitude: JeodQuat,
    /// Angular velocity in the subtree's body frame.
    pub composite_ang_vel_body: DVec3,
}

impl DetachedSubtreeState {
    /// Convert to a [`RefFrameState`] for use with the propagation
    /// helpers in this crate.
    pub fn to_ref_frame_state(&self) -> RefFrameState {
        RefFrameState {
            trans: RefFrameTrans {
                position: self.composite_position,
                velocity: self.composite_velocity,
            },
            rot: RefFrameRot {
                q_parent_this: self.composite_attitude,
                t_parent_this: self.composite_attitude.left_quat_to_transformation(),
                ang_vel_this: self.composite_ang_vel_body,
            },
        }
    }

    /// Construct from a [`RefFrameState`].
    pub fn from_ref_frame_state(state: &RefFrameState) -> Self {
        Self {
            composite_position: state.trans.position,
            composite_velocity: state.trans.velocity,
            composite_attitude: state.rot.q_parent_this,
            composite_ang_vel_body: state.rot.ang_vel_this,
        }
    }

    /// Advance the subtree ballistically by `dt` seconds (no force, no
    /// torque). Position drifts at `composite_velocity`; attitude
    /// rotates at `composite_ang_vel_body` via
    /// [`crate::advance_left_quat_body_rate`] (which owns the JEOD
    /// left-multiply convention so this site can't get it wrong);
    /// velocity and ang_vel are unchanged.
    pub fn step_ballistic(&mut self, dt: f64) {
        self.composite_position += self.composite_velocity * dt;
        self.composite_attitude = crate::advance_left_quat_body_rate(
            self.composite_attitude,
            self.composite_ang_vel_body,
            dt,
        );
    }
}
