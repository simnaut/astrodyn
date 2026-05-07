//! Glue: kinematic-joint drivers and stacked-spec exclusivity guards.
//!
//! The four constant-rate / sinusoidal / closure / multi-DOF driver
//! systems run inside [`AstrodynSet::EphemerisUpdate`](crate::AstrodynSet::EphemerisUpdate)
//! (writing `FrameRotC` / `FrameAngVelC` on joint frame entities).
//! The stacked-spec exclusivity guard runs in `PostStartup` and via
//! `on_insert` lifecycle hooks installed at plugin build time so any
//! attempt to land two spec components on one entity panics loudly
//! rather than letting one driver silently drop the entity.

use bevy::prelude::*;

#[allow(unused_imports)] // intra-doc-link resolution
use super::super::ephemeris_update::planet_fixed_rotation_system;
use crate::components::*;
use crate::SimulationTimeR;

/// Drives kinematically prescribed joint frames each tick.
///
/// For every entity carrying a [`JointKinematicsC`] spec, the joint
/// angle at the current simulation time is `θ(t) = initial + rate · t`,
/// where `t` is the tick's `tai_seconds` (the elapsed-since-epoch time
/// scale `time_advance_system` already advances every step). The
/// system writes:
///
/// - [`FrameRotC::q_parent_this`] = left-transformation quaternion
///   `parent → this` for the rotation about the spec's
///   `axis_in_parent` by `θ(t)`,
/// - [`FrameRotC::t_parent_this`] = the corresponding 3×3 transformation
///   matrix (cache),
/// - [`FrameAngVelC::0`] = `rate · axis_in_parent` (the angular
///   velocity in this-frame coordinates — the rotation axis is the
///   eigenvector of the rotation, so it's invariant between parent
///   and this frames).
///
/// This is the analog of [`planet_fixed_rotation_system`] for arbitrary
/// user-declared joint axes: planet-fixed frames spin at JEOD's
/// Earth/Mars/Moon rotation rates about the planet pole;
/// joint frames spin at a mission-declared `rate_rad_per_s` about an
/// arbitrary `axis_in_parent`. Both write the same `FrameRotC` /
/// `FrameAngVelC` storage, so any downstream consumer that reads
/// frame-tree state through [`crate::components::FrameRotC`] /
/// [`crate::components::FrameAngVelC`] (or through a future
/// `RelativeFrameState` SystemParam) sees the joint kinematics
/// uniformly with planet-fixed kinematics.
///
/// Scheduled in [`crate::AstrodynSet::EphemerisUpdate`] alongside
/// `planet_fixed_rotation_system` so the joint frame's rotation /
/// angular velocity are current before any consumer that walks the
/// frame tree (gravity, derived state, integration) reads them.
///
/// "Kinematic" means the angle is an *input*, not an integrated
/// state — there is no torque, inertia, or momentum. Joint dynamics
/// (free-swinging joints, IK, constraint-derived joint forces) are
/// out of scope; see the deferred-dynamics meta.
///
/// The `Without<...>` filters on the three sibling kinematic-spec
/// components are a *parallelism signal* for Bevy's scheduler — they
/// make this query structurally disjoint from the sinusoidal /
/// closure / multi-DOF drivers so the four systems can dispatch in
/// parallel under `AstrodynSet::EphemerisUpdate` without a runtime borrow
/// conflict on `FrameRotC` / `FrameAngVelC`. They are *not* the
/// correctness mechanism that rejects stacked-spec entities.
///
/// Stacked-spec rejection is enforced by the per-component
/// `on_insert` hooks installed via
/// [`register_joint_kinematics_exclusivity_hooks`]: inserting a
/// second kinematic-spec component on an entity that already carries
/// one panics immediately, naming the entity and both spec
/// components, before any driver query ever runs. The PostStartup
/// [`validate_joint_kinematics_exclusivity`] pass is defense in
/// depth — it catches stacking patterns that bypass the hook order
/// (e.g., a `Bundle` whose components arrive in the same archetype
/// move, or future spec components added without registering a hook)
/// and aggregates every offender into a single startup-time panic.
#[allow(clippy::type_complexity)]
pub fn joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (&JointKinematicsC, &mut FrameRotC, &mut FrameAngVelC),
        (
            Without<SinusoidalJointKinematicsC>,
            Without<ClosureJointKinematicsC>,
            Without<MultiDofJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) = astrodyn::evaluate_joint_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// Drives sinusoidal kinematic joint frames each tick.
///
/// Sibling of [`joint_kinematics_system`] that handles
/// [`SinusoidalJointKinematicsC`]-tagged frame entities. Reads the
/// same `tai_seconds` clock and writes the same
/// [`FrameRotC`] / [`FrameAngVelC`] storage, so downstream consumers
/// that walk the frame tree see uniform output across the kinematic
/// styles.
///
/// Scheduled in [`crate::AstrodynSet::EphemerisUpdate`] alongside
/// `planet_fixed_rotation_system` and `joint_kinematics_system` —
/// the joint frame's rotation / angular velocity must be current
/// before any consumer that walks the frame tree (gravity, derived
/// state, integration) reads them.
///
/// The `Without<...>` filters mirror the contract documented on
/// [`joint_kinematics_system`]: they are a parallelism signal that
/// keeps the four kinematic-spec drivers pairwise-disjoint at the
/// query level. The correctness mechanism that rejects stacked-spec
/// entities is the on_insert hooks installed by
/// [`register_joint_kinematics_exclusivity_hooks`] (panic at
/// insertion); [`validate_joint_kinematics_exclusivity`] is
/// PostStartup defense in depth.
#[allow(clippy::type_complexity)]
pub fn sinusoidal_joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (
            &SinusoidalJointKinematicsC,
            &mut FrameRotC,
            &mut FrameAngVelC,
        ),
        (
            Without<JointKinematicsC>,
            Without<ClosureJointKinematicsC>,
            Without<MultiDofJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) =
            astrodyn::evaluate_sinusoidal_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// Drives closure (fixed-pose) kinematic joint frames each tick.
///
/// Sibling of [`joint_kinematics_system`] that handles
/// [`ClosureJointKinematicsC`]-tagged frame entities. The output is
/// constant in time, so the system writes the same `FrameRotC` /
/// `FrameAngVelC` value every step. Scheduled in
/// [`crate::AstrodynSet::EphemerisUpdate`] alongside
/// `joint_kinematics_system` so the closure-pinned frame's rotation
/// is materialized before any frame-tree consumer reads it.
///
/// The `Without<...>` filters mirror the contract documented on
/// [`joint_kinematics_system`]: they are a parallelism signal that
/// keeps the four kinematic-spec drivers pairwise-disjoint at the
/// query level. The correctness mechanism that rejects stacked-spec
/// entities is the on_insert hooks installed by
/// [`register_joint_kinematics_exclusivity_hooks`] (panic at
/// insertion); [`validate_joint_kinematics_exclusivity`] is
/// PostStartup defense in depth.
#[allow(clippy::type_complexity)]
pub fn closure_joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (&ClosureJointKinematicsC, &mut FrameRotC, &mut FrameAngVelC),
        (
            Without<JointKinematicsC>,
            Without<SinusoidalJointKinematicsC>,
            Without<MultiDofJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) = astrodyn::evaluate_closure_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// Drives multi-DOF kinematic joint frames each tick.
///
/// Sibling of [`joint_kinematics_system`] that handles
/// [`MultiDofJointKinematicsC`]-tagged frame entities. Each entity
/// carries an N-stage chain (`N <= MAX_MULTI_DOF_AXES`); the kernel
/// folds the per-stage `(rotation, ang_vel)` contributions through
/// `RefFrameState::incr_right` so the output is bit-identical to a
/// chain of N single-DOF joint entities walked through the frame
/// tree. Scheduled in [`crate::AstrodynSet::EphemerisUpdate`] for the
/// same reason as the other joint-kinematics systems.
///
/// The `Without<...>` filters mirror the contract documented on
/// [`joint_kinematics_system`]: they are a parallelism signal that
/// keeps the four kinematic-spec drivers pairwise-disjoint at the
/// query level. The correctness mechanism that rejects stacked-spec
/// entities is the on_insert hooks installed by
/// [`register_joint_kinematics_exclusivity_hooks`] (panic at
/// insertion); [`validate_joint_kinematics_exclusivity`] is
/// PostStartup defense in depth.
#[allow(clippy::type_complexity)]
pub fn multi_dof_joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (&MultiDofJointKinematicsC, &mut FrameRotC, &mut FrameAngVelC),
        (
            Without<JointKinematicsC>,
            Without<SinusoidalJointKinematicsC>,
            Without<ClosureJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) =
            astrodyn::evaluate_multi_dof_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// PostStartup-time guard that asserts at most one of the four
/// joint-kinematic spec components is present on any single entity.
///
/// The four joint-kinematic drivers
/// ([`joint_kinematics_system`], [`sinusoidal_joint_kinematics_system`],
/// [`closure_joint_kinematics_system`], [`multi_dof_joint_kinematics_system`])
/// each carry `Without<...>` filters for the other three spec
/// components so Bevy's scheduler can dispatch them in parallel under
/// `AstrodynSet::EphemerisUpdate` without contending for `FrameRotC` /
/// `FrameAngVelC`. That filter discipline turns an entity that
/// accidentally carries two specs into a *silent drop* from every
/// driver — its `FrameRotC` would never be written and the joint
/// frame would advertise stale (or default-identity) state to every
/// downstream `RelativeFrameState` walk.
///
/// Per the project's fail-loud rule, that misconfiguration must
/// panic at the earliest detection point with a diagnostic that
/// names the offending entity and the specs it carries. The primary
/// guard is the per-component `on_insert` hook installed by
/// [`register_joint_kinematics_exclusivity_hooks`], which fires at
/// insertion time and catches every stacking pattern (Startup,
/// FixedUpdate, observers, …). This `PostStartup` validator is
/// defense in depth: it walks every entity that already carries at
/// least one kinematic spec once before the first `FixedUpdate` tick
/// and emits a *single* aggregated panic message that lists every
/// offending entity at once. The `on_insert` hooks panic on the
/// first stacked insertion they observe, which is right for runtime
/// but less informative when the user declares several stacked
/// entities together at startup; the aggregated startup pass keeps
/// that path actionable.
///
/// The four spec components are declarative alternatives — a joint
/// is *either* constant-rate, *or* sinusoidal, *or* a closure pose,
/// *or* a multi-DOF chain — so stacking two of them has no
/// meaningful semantics. If a future kinematic style needs to
/// compose with an existing one, that composition belongs in a new
/// dedicated spec (e.g., extend `SingleDofKinematics` and route
/// through `MultiDofJointKinematicsC`), not in two parallel
/// drivers racing for the same storage.
///
/// # Panics
/// Panics if any entity carries more than one of `JointKinematicsC`,
/// [`SinusoidalJointKinematicsC`], [`ClosureJointKinematicsC`], or
/// [`MultiDofJointKinematicsC`]. The message lists every offending
/// entity together with the specs it carries.
#[allow(clippy::type_complexity)]
pub fn validate_joint_kinematics_exclusivity(
    query: Query<
        (
            Entity,
            Has<JointKinematicsC>,
            Has<SinusoidalJointKinematicsC>,
            Has<ClosureJointKinematicsC>,
            Has<MultiDofJointKinematicsC>,
        ),
        Or<(
            With<JointKinematicsC>,
            With<SinusoidalJointKinematicsC>,
            With<ClosureJointKinematicsC>,
            With<MultiDofJointKinematicsC>,
        )>,
    >,
) {
    let mut offenders: Vec<String> = Vec::new();
    for (entity, has_const, has_sin, has_close, has_multi) in &query {
        let count = usize::from(has_const)
            + usize::from(has_sin)
            + usize::from(has_close)
            + usize::from(has_multi);
        if count > 1 {
            let mut names: Vec<&'static str> = Vec::new();
            if has_const {
                names.push("JointKinematicsC");
            }
            if has_sin {
                names.push("SinusoidalJointKinematicsC");
            }
            if has_close {
                names.push("ClosureJointKinematicsC");
            }
            if has_multi {
                names.push("MultiDofJointKinematicsC");
            }
            offenders.push(format!("{entity:?} carries [{}]", names.join(", ")));
        }
    }
    assert!(
        offenders.is_empty(),
        "Joint-kinematics spec components are mutually exclusive — each frame entity \
         must carry at most one of JointKinematicsC, SinusoidalJointKinematicsC, \
         ClosureJointKinematicsC, MultiDofJointKinematicsC. Offending entities: {}. \
         Fix: pick a single kinematic style per joint frame; for composed motions \
         use MultiDofJointKinematicsC with a chain of SingleDofKinematics stages.",
        offenders.join("; ")
    );
}

/// Format a "stacked specs" panic diagnostic for a single offending
/// entity. Centralized so the `on_insert` hooks below and any future
/// detection site share one message shape — a mission engineer reading
/// the panic always sees the same actionable instructions regardless
/// of which path tripped the check.
fn format_stacked_specs_panic(entity: Entity, names: &[&'static str]) -> String {
    format!(
        "Joint-kinematics spec components are mutually exclusive — each frame entity \
         must carry at most one of JointKinematicsC, SinusoidalJointKinematicsC, \
         ClosureJointKinematicsC, MultiDofJointKinematicsC. Offending entity: \
         {entity:?} carries [{}]. \
         Fix: pick a single kinematic style per joint frame; for composed motions \
         use MultiDofJointKinematicsC with a chain of SingleDofKinematics stages.",
        names.join(", ")
    )
}

/// Shared body of the four joint-kinematics `on_insert` hooks. Reads
/// every spec flag off the entity's post-insertion archetype, counts
/// how many distinct specs are present, and panics with
/// [`format_stacked_specs_panic`] if more than one is. `self_name`
/// is the spec component whose hook is firing — included in the
/// panic so a mission engineer reading the backtrace sees which
/// insertion attempt tripped the check.
///
/// `on_insert` runs after the bundle's components are already added
/// to the entity's archetype, so the four `contains::<...>` reads
/// on the `DeferredWorld` reflect the full post-insertion state.
fn check_stacked_specs(
    world: bevy::ecs::world::DeferredWorld<'_>,
    entity: Entity,
    self_name: &'static str,
) {
    let entity_ref = world.get_entity(entity).expect(
        "joint-kinematics on_insert hook: entity must exist when its component is inserted",
    );
    let has_const = entity_ref.contains::<JointKinematicsC>();
    let has_sin = entity_ref.contains::<SinusoidalJointKinematicsC>();
    let has_close = entity_ref.contains::<ClosureJointKinematicsC>();
    let has_multi = entity_ref.contains::<MultiDofJointKinematicsC>();
    let count = usize::from(has_const)
        + usize::from(has_sin)
        + usize::from(has_close)
        + usize::from(has_multi);
    if count <= 1 {
        return;
    }
    let mut names: Vec<&'static str> = Vec::new();
    if has_const {
        names.push("JointKinematicsC");
    }
    if has_sin {
        names.push("SinusoidalJointKinematicsC");
    }
    if has_close {
        names.push("ClosureJointKinematicsC");
    }
    if has_multi {
        names.push("MultiDofJointKinematicsC");
    }
    panic!(
        "{} (triggered while inserting {self_name})",
        format_stacked_specs_panic(entity, &names)
    );
}

// `ComponentHook` is `fn(DeferredWorld, HookContext)` — a plain
// function pointer with no captures. Each spec component therefore
// gets its own dedicated `fn` item that forwards to
// `check_stacked_specs` with a hard-coded `self_name`.

fn on_insert_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "JointKinematicsC");
}

fn on_insert_sinusoidal_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "SinusoidalJointKinematicsC");
}

fn on_insert_closure_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "ClosureJointKinematicsC");
}

fn on_insert_multi_dof_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "MultiDofJointKinematicsC");
}

/// Register `on_insert` hooks on every joint-kinematics spec component
/// so any insertion that lands a second spec on an entity panics
/// immediately.
///
/// The `PostStartup` validator
/// ([`validate_joint_kinematics_exclusivity`]) is a startup-time
/// safety net: it walks the world *once* before the first
/// `FixedUpdate` tick and catches misconfigurations declared in
/// `Startup` systems. It cannot observe entities spawned after
/// `PostStartup` — for example, a `Commands::spawn(...)` issued from
/// a `FixedUpdate` user system, an `Update` system, or an event
/// handler. Without a runtime guard those late spawns slip past every
/// driver's `Without<...>` filter and silently propagate stale
/// `FrameRotC` / `FrameAngVelC`, which the project's fail-loud rule
/// forbids.
///
/// Bevy 0.18 component lifecycle hooks (`on_insert`) close that gap:
/// every kinematic-spec insertion — `spawn`, `insert`, or `replace`
/// — fires its hook before the next system observes the new
/// component, so a bad insertion panics at the insertion site rather
/// than silently propagating bad state. The hook reads the entity's
/// post-insertion archetype to count how many of the four spec
/// components are present and panics with the same diagnostic shape
/// as [`validate_joint_kinematics_exclusivity`] if more than one is.
///
/// Idempotent re-registration of the *same* spec component (insert A
/// onto an entity that already has A) does not trip the hook: the
/// count of distinct kinematic specs is unchanged. Only stacking
/// distinct specs panics.
///
/// `AstrodynPlugin::build` calls this once during plugin setup. Tests
/// that exercise the joint-kinematics pipeline without `AstrodynPlugin`
/// can call this directly to install the same guard.
pub fn register_joint_kinematics_exclusivity_hooks(app: &mut App) {
    let world = app.world_mut();
    world
        .register_component_hooks::<JointKinematicsC>()
        .on_insert(on_insert_joint_kinematics_c);
    world
        .register_component_hooks::<SinusoidalJointKinematicsC>()
        .on_insert(on_insert_sinusoidal_joint_kinematics_c);
    world
        .register_component_hooks::<ClosureJointKinematicsC>()
        .on_insert(on_insert_closure_joint_kinematics_c);
    world
        .register_component_hooks::<MultiDofJointKinematicsC>()
        .on_insert(on_insert_multi_dof_joint_kinematics_c);
}
