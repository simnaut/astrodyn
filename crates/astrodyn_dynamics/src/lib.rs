//! Rigid-body dynamics: state, mass properties, forces, and integration.
//!
//! This crate is the JEOD-faithful port of the dynamics half of
//! `models/dynamics/`. It owns the per-vehicle state types
//! ([`TranslationalState`], [`RotationalState`], [`SixDofState`]), the rigid-
//! body mass model ([`MassProperties`], [`MassBody`], [`MassPoint`],
//! [`MassTree`], [`point_mass_inertia`]), the force-collection pipeline
//! ([`collect_forces`], [`ForceContributions`], [`TotalForce`],
//! [`GravityAcceleration`], [`compute_translational_acceleration`],
//! [`compute_translational_derivatives`], [`compute_frame_derivatives`],
//! [`compute_t_inertial_struct`]), and a battery of integrators
//! ([`rk4_translational_step`], [`rk4_sixdof_step`], adaptive RKF4(5) in
//! [`rkf45`], multistep [`abm4`]/Adams–Bashforth–Moulton, and
//! [`gauss_jackson`]).
//!
//! Initial-condition machinery in [`body_init`] mirrors JEOD's
//! `BodyAction` subclasses: [`init_from_orbital_elements`],
//! [`init_from_mean_anomaly`], [`init_from_time_periapsis`],
//! [`init_from_lvlh`], [`init_from_ned`], and [`compute_ned_rotation`].
//! Frame propagation lives in [`propagation`]: [`propagate_body_frames`],
//! [`propagate_forward`], [`propagate_reverse`]. Holonomic constraints
//! ([`HolonomicConstraint`], [`PendulumConstraint`], [`BaumgarteSolver`],
//! [`apply_constraint`]) port the constraint-stabilization layer used by
//! tethers and articulated structures. Declaratively driven kinematic
//! joints — single-axis frame-rotation drivers used by articulated
//! sub-trees that consume a prescribed angular trajectory rather than
//! integrating one — live in [`kinematic_joint`]
//! ([`JointKinematicsSpec`], [`evaluate_joint_kinematics`]).
//! Root-to-leaf state propagation through kinematic-child chains lives
//! in [`kinematic_propagation`] ([`compute_kinematic_child_state`],
//! [`compute_kinematic_child_state_typed`],
//! [`derive_kinematic_child_from_states`], [`KinematicChildInputs`],
//! [`KinematicChildOutputs`]).
//!
//! JEOD source: `models/dynamics/dyn_body/`, `models/dynamics/mass/`,
//! `models/dynamics/body_action/`, and `models/utils/integration/`. Pure
//! Rust, zero Bevy dependency — orchestration lives in `astrodyn` and ECS
//! wiring lives in the `astrodyn_bevy` root crate.
//!
//! ## Example
//!
//! Compute the parallel-axis (Steiner) inertia contribution of a 10 kg
//! point mass offset 2 m along +X from the reference point. The
//! resulting matrix has `0` on the (0,0) diagonal (no inertia about the
//! axis through the offset) and `mass * r^2 = 40` on the perpendicular
//! diagonal entries:
//!
//! ```
//! use astrodyn_dynamics::point_mass_inertia;
//! use glam::DVec3;
//!
//! let i = point_mass_inertia(10.0, DVec3::new(2.0, 0.0, 0.0));
//! assert!(i.x_axis.x.abs() < 1e-12);
//! assert!((i.y_axis.y - 40.0).abs() < 1e-12);
//! assert!((i.z_axis.z - 40.0).abs() < 1e-12);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod abm4;
pub mod attach;
pub mod body_init;
pub mod constraints;
pub mod forces;
pub mod frame_attach;
pub mod gauss_jackson;
pub mod integration;
pub mod kinematic_joint;
pub mod kinematic_propagation;
pub mod lsode;
pub mod mass;
pub mod mass_body;
pub mod mass_storage;
pub mod propagation;
pub mod rkf45;
pub mod rotational;
pub mod state;
pub mod subtree;
pub mod wrench;

pub use abm4::{abm4_translational_step, Abm4State};
pub use attach::{combine_states_at_attach, AttachCombineInputs, AttachCombineOutputs};
pub use body_init::{
    compute_ned_rotation, init_from_altitudes_time_periapsis, init_from_altitudes_true_anomaly,
    init_from_lvlh, init_from_mean_anomaly, init_from_ned, init_from_orbital_elements,
    init_from_semi_latus_rectum_true_anomaly, init_from_time_periapsis,
};
pub use constraints::{apply_constraint, BaumgarteSolver, HolonomicConstraint, PendulumConstraint};
pub use forces::{
    collect_forces, compute_frame_derivatives, compute_t_inertial_struct,
    compute_translational_acceleration, compute_translational_derivatives, DynamicsConfig,
    ForceContributions, FrameDerivatives, GravityAcceleration, TotalForce,
};
pub use frame_attach::{derive_frame_attached_state, FrameAttachInputs};
pub use gauss_jackson::config::GaussJacksonConfig;
pub use gauss_jackson::{GaussJacksonState, IntegratorResult};
pub use integration::{rk4_sixdof_step, rk4_translational_step, IntegratorType};
pub use kinematic_joint::{
    evaluate as evaluate_joint_kinematics, evaluate_closure as evaluate_closure_kinematics,
    evaluate_model as evaluate_joint_kinematics_model,
    evaluate_multi_dof as evaluate_multi_dof_kinematics,
    evaluate_sinusoidal as evaluate_sinusoidal_kinematics, ClosureJointKinematicsSpec,
    JointKinematicsModel, JointKinematicsSpec, MultiDofJointKinematicsSpec, SingleDofKinematics,
    SinusoidalJointKinematicsSpec, MAX_MULTI_DOF_AXES,
};
pub use kinematic_propagation::{
    compute_kinematic_child_state, compute_kinematic_child_state_dquat,
    compute_kinematic_child_state_typed, derive_kinematic_child_from_states, KinematicChildInputs,
    KinematicChildOutputs,
};
pub use lsode::{
    lsode_translational_step, CorrectorMethod, IntegrationMethod, LsodeConfig, LsodeState,
};
pub use mass::{
    MassProperties, MassPropertiesTyped, INERTIA_CONSISTENCY_TOL, MAX_SAFE_MASS_KG,
    MIN_SAFE_MASS_KG,
};
pub use mass_body::{
    point_mass_inertia, MassBody, MassBodyId, MassPoint, MassPointState, MassTree,
};
pub use mass_storage::{
    compute_node_composite, finalize_child_in_parent_frame, recompute_composites_via_storage,
    MassNodeOutputs, MassNodeView, MassStorage,
};
pub use propagation::{propagate_body_frames, propagate_forward, propagate_reverse};
pub use rkf45::{
    rkf45_adaptive_sixdof_step, rkf45_adaptive_translational_step, rkf45_sixdof_step,
    rkf45_translational_step, AdaptiveConfig, AdaptiveResult,
};
pub use rotational::{
    compute_left_quat_deriv, compute_left_quat_deriv_typed, compute_rotational_acceleration,
    compute_rotational_acceleration_typed, normalize_integ, RotationalState, RotationalStateTyped,
    SixDofState, SixDofStateTyped,
};
pub use state::TranslationalState;
pub use subtree::DetachedSubtreeState;
pub use wrench::{shift_wrench_to_parent, shift_wrench_to_parent_typed, Wrench};
