//! Composite-body kinematic state of a free-flying mass-tree subtree.
//!
//! Used by mass-tree consumers (e.g. `jeod_runner::Simulation`) to track
//! the inertial state of a subtree that has been detached from its
//! parent's tree and is coasting ballistically (no force, no torque)
//! before being re-attached. Pure rigid-body kinematics — no Simulation
//! or runner dependency.
//!
//! Relocated from `jeod_runner` in issue #253. The attitude field is a
//! [`BodyAttitude<SelfRef>`] (issue #252) — the wrapper owns JEOD's
//! `q̇ = -½(ω ⊗ q)` left-multiply convention so `step_ballistic`
//! cannot accidentally swap operand order, and the type system makes
//! the wrong order unrepresentable at the call site.

use glam::DVec3;
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_math::JeodQuat;
use jeod_quantities::aliases::AngularVelocity;
use jeod_quantities::body_attitude::BodyAttitude;
use jeod_quantities::frame::{BodyFrame, SelfRef};
use jeod_quantities::quat::NormalizedQuat;

/// Composite-body inertial state of a free-flying mass-tree subtree
/// (i.e. a tree root that is not the integrated body). All fields are
/// in the simulation's integration frame (typically Earth.inertial).
#[derive(Debug, Clone, Copy)]
pub struct DetachedSubtreeState {
    /// RootInertial position of the subtree's composite CoM.
    pub composite_position: DVec3,
    /// RootInertial velocity of the subtree's composite CoM.
    pub composite_velocity: DVec3,
    /// RootInertial-to-body attitude. Wrapped in [`BodyAttitude`] so the
    /// JEOD left-multiply integration convention (`q̇ = -½(ω ⊗ q)`) is
    /// type-enforced — there is no public `multiply` to swap operand
    /// order on. See issue #252.
    pub composite_attitude: BodyAttitude<SelfRef>,
    /// Angular velocity in the subtree's body frame.
    pub composite_ang_vel_body: DVec3,
}

impl DetachedSubtreeState {
    /// Convert to a [`RefFrameState`] for use with the propagation
    /// helpers in this crate.
    pub fn to_ref_frame_state(&self) -> RefFrameState {
        let q = self.composite_attitude.to_jeod_quat();
        RefFrameState {
            trans: RefFrameTrans {
                position: self.composite_position,
                velocity: self.composite_velocity,
            },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: self
                    .composite_attitude
                    .as_witness()
                    .left_quat_to_transformation(),
                ang_vel_this: self.composite_ang_vel_body,
            },
        }
    }

    /// Construct from a [`RefFrameState`]. The caller must have
    /// renormalized `state.rot.q_parent_this` to within
    /// [`NormalizedQuat::DEFAULT_TOLERANCE`] of unit norm — this is
    /// satisfied by the integrators' end-of-step `normalize_integ` and
    /// by the frame-tree assembly code.
    pub fn from_ref_frame_state(state: &RefFrameState) -> Self {
        let q = NormalizedQuat::new(state.rot.q_parent_this).unwrap_or_else(|err| {
            panic!(
                "DetachedSubtreeState::from_ref_frame_state: q_parent_this is not unit-norm: {err}"
            )
        });
        Self {
            composite_position: state.trans.position,
            composite_velocity: state.trans.velocity,
            composite_attitude: BodyAttitude::from_witness(q),
            composite_ang_vel_body: state.rot.ang_vel_this,
        }
    }

    /// Wrap a raw inertial-frame [`JeodQuat`] as the composite attitude.
    /// The caller asserts the quaternion has inertial-to-body semantics
    /// and is unit-norm to within
    /// [`NormalizedQuat::DEFAULT_TOLERANCE`]; otherwise this panics.
    /// Used by detach call sites that have a raw `RefFrameRot`
    /// quaternion in hand and are constructing the subtree state
    /// directly (rather than via [`Self::from_ref_frame_state`]).
    pub fn attitude_from_raw_jeod_quat(q: JeodQuat) -> BodyAttitude<SelfRef> {
        let witness = NormalizedQuat::new(q).unwrap_or_else(|err| {
            panic!("DetachedSubtreeState::attitude_from_raw_jeod_quat: q is not unit-norm: {err}")
        });
        BodyAttitude::from_witness(witness)
    }

    /// Advance the subtree ballistically by `dt` seconds (no force, no
    /// torque). Position drifts at `composite_velocity`; attitude
    /// rotates at `composite_ang_vel_body` via
    /// [`BodyAttitude::advance_under_body_rate`] (which owns the JEOD
    /// left-multiply convention — the type system makes the wrong
    /// operand order unrepresentable here, see issue #252); velocity
    /// and ang_vel are unchanged.
    pub fn step_ballistic(&mut self, dt: f64) {
        self.composite_position += self.composite_velocity * dt;
        let omega: AngularVelocity<BodyFrame<SelfRef>> =
            AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(self.composite_ang_vel_body);
        self.composite_attitude = self.composite_attitude.advance_under_body_rate(omega, dt);
    }
}
