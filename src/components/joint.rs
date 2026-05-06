//! Bevy `Component` newtypes for kinematically prescribed joints —
//! constant-rate, sinusoidal, closure, and multi-DOF spec wrappers
//! consumed by the matching joint-kinematics driver systems.

use bevy::prelude::*;
use glam::DVec3;

use super::frame_tree::{FrameAngVelC, FrameRotC, FrameTransC};
#[allow(unused_imports)] // intra-doc-link resolution
use super::gravity::PlanetFixedRotationC;

/// Declarative spec for a kinematically driven single-axis joint.
///
/// Place this component on a *frame entity* (one carrying the full
/// [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] triplet) to have
/// [`crate::systems::joint_kinematics_system`] drive the entity's
/// rotation about its parent frame each tick. The rotation angle
/// follows `θ(t) = initial_angle_rad + rate_rad_per_s · t`, applied
/// about `axis_in_parent` (a unit vector in the parent frame).
///
/// Mirrors [`jeod_sim::JointKinematicsSpec`] one-to-one. The component
/// is the analog of [`PlanetFixedRotationC`] generalised to an
/// arbitrary user-declared axis: where pfix entities are spun by
/// [`crate::systems::planet_fixed_rotation_system`] under an Earth-/
/// Mars-/Moon-rotation kernel, joint entities are spun by
/// [`crate::systems::joint_kinematics_system`] under a constant-rate
/// kernel.
///
/// "Kinematic" here means: the angle (and therefore rotation and
/// angular velocity) is an *input* — there is no torque, inertia, or
/// momentum exchange. Joint dynamics (free-swinging joints,
/// constraint-derived joint forces, inverse dynamics) are explicitly
/// out of scope; see the deferred-dynamics meta.
///
/// # Frame-tree contract
///
/// Frame-tree consumers ([`crate::frame_param::RelativeFrameState`])
/// treat every frame entity as carrying the full
/// [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] triplet — a node
/// missing any of the three would make a hierarchy walk that crosses
/// the joint observe an undefined translation, rotation, or angular
/// velocity. This component therefore auto-inserts all three via
/// `#[require]`. A single-axis joint is by definition a pure rotation
/// about a fixed axis at a fixed point in the parent frame, so the
/// default [`FrameTransC`] (zero offset, zero relative velocity) is
/// the physically correct value for an articulated joint frame and
/// callers do not need to spawn it explicitly.
///
/// # Example
///
/// ```ignore
/// // A solar-array joint that spins at 6 °/min about the +Y axis,
/// // starting at θ = 0. FrameTransC / FrameRotC / FrameAngVelC are
/// // auto-inserted via the #[require] attribute on JointKinematicsC.
/// commands.spawn((
///     JointKinematicsC(JointKinematicsSpec {
///         axis_in_parent: DVec3::Y,
///         rate_rad_per_s: 6.0_f64.to_radians() / 60.0,
///         initial_angle_rad: 0.0,
///     }),
///     ChildOf(parent_frame_entity),
/// ));
/// ```
///
/// Per the design doc Section 15.1, articulated sub-trees declare a
/// chain of joint frame entities under a body frame; each joint frame
/// carries this component and the resulting `FrameTransC` /
/// `FrameRotC` / `FrameAngVelC` flow into the same
/// [`crate::frame_param::RelativeFrameState`] consumers that read
/// planet-fixed rotations.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct JointKinematicsC(pub jeod_sim::JointKinematicsSpec);

impl JointKinematicsC {
    /// Convenience constructor: build a joint spec from raw axis / rate
    /// / initial-angle values.
    #[inline]
    pub fn new(axis_in_parent: DVec3, rate_rad_per_s: f64, initial_angle_rad: f64) -> Self {
        Self(jeod_sim::JointKinematicsSpec {
            axis_in_parent,
            rate_rad_per_s,
            initial_angle_rad,
        })
    }
}

impl From<jeod_sim::JointKinematicsSpec> for JointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::JointKinematicsSpec) -> Self {
        Self(spec)
    }
}

/// Declarative spec for a kinematically driven single-axis joint whose
/// angle is a sinusoidal function of time
/// (`θ(t) = offset + amplitude · sin(ω · t + phase)`).
///
/// Sibling component to [`JointKinematicsC`] for the periodic-articulation
/// case — solar-array dither, antenna scan, gimbal sweep — that the
/// constant-rate spec cannot express. The driving system writes the
/// same [`FrameRotC`] / [`FrameAngVelC`] storage as
/// [`JointKinematicsC`], so a downstream consumer that walks the
/// frame tree sees a uniform rotation snapshot regardless of which
/// kinematic style drives the joint.
///
/// The `#[require]` triplet matches [`JointKinematicsC`] so spawning a
/// joint frame entity carrying this component automatically materializes
/// the [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] frame-tree
/// triplet, so `RelativeFrameState` walks across the joint remain
/// well-defined.
///
/// Wraps [`jeod_sim::SinusoidalJointKinematicsSpec`] one-to-one. Mission
/// code that needs richer kinematic styles than constant-rate /
/// sinusoidal / closure (e.g., piecewise-linear angular splines) reaches
/// for a custom system; the kinematic-only spec catalogue exposed here
/// covers the periodic / loop-closing / multi-DOF cases.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct SinusoidalJointKinematicsC(pub jeod_sim::SinusoidalJointKinematicsSpec);

impl From<jeod_sim::SinusoidalJointKinematicsSpec> for SinusoidalJointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::SinusoidalJointKinematicsSpec) -> Self {
        Self(spec)
    }
}

/// Declarative spec for a *closure* joint — one pinned to a fixed
/// rotation about a single axis with no time dependence.
///
/// The kinematic-only degenerate case useful for closing kinematic
/// loops where one joint's pose is constrained at declaration time
/// rather than driven through `θ(t)`. The system writes a constant
/// `FrameRotC` and zero `FrameAngVelC` every tick, so the joint
/// frame's contribution to a `RelativeFrameState` walk is the same
/// every step (cheap; the per-tick reassignment is the same value
/// each time).
///
/// Wraps [`jeod_sim::ClosureJointKinematicsSpec`] one-to-one and
/// auto-inserts the frame-tree triplet via `#[require]`, matching
/// [`JointKinematicsC`].
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct ClosureJointKinematicsC(pub jeod_sim::ClosureJointKinematicsSpec);

impl From<jeod_sim::ClosureJointKinematicsSpec> for ClosureJointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::ClosureJointKinematicsSpec) -> Self {
        Self(spec)
    }
}

/// Declarative spec for a multi-DOF kinematic joint — up to
/// [`jeod_sim::MAX_MULTI_DOF_AXES`] single-axis stages composed into
/// one chain.
///
/// Each stage is a `SingleDofKinematics` variant
/// (`ConstantRate`/`Sinusoidal`/`Closure`) that rotates about its
/// declared axis in the *intermediate frame produced by the
/// preceding stages*. Stages must be a contiguous prefix of the
/// fixed-size axes array; the kernel asserts this.
///
/// The semantic equivalence is deliberate: a multi-DOF joint with N
/// stages on a single entity produces the same `(rotation, angular
/// velocity)` snapshot as a chain of N single-DOF joint entities
/// linked by `ChildOf`. Mission code picks whichever shape is more
/// ergonomic — a long arm benefits from N entities (each with its
/// own name + frame-tree slot for inspection); a tightly-coupled 2-3
/// DOF gimbal benefits from one entity.
///
/// Wraps [`jeod_sim::MultiDofJointKinematicsSpec`] one-to-one and
/// auto-inserts the frame-tree triplet via `#[require]`.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct MultiDofJointKinematicsC(pub jeod_sim::MultiDofJointKinematicsSpec);

impl From<jeod_sim::MultiDofJointKinematicsSpec> for MultiDofJointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::MultiDofJointKinematicsSpec) -> Self {
        Self(spec)
    }
}
