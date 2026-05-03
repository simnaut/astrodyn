//! Bevy system for composite-rigid-body wrench aggregation.
//!
//! The Bevy half of the composite-rigid-body wrench pipeline: walks
//! the `MassChildOf` tree after
//! [`force_collection_system`](crate::systems::force_collection_system)
//! has populated each body's `TotalForceC` and propagates every child's
//! `(force, torque)` into its root's totals via the parallel-axis arm
//! ([`jeod_sim::shift_wrench_to_parent`] from `jeod_dynamics`).
//!
//! # Per the three-layer rule
//!
//! - The pure math (`shift_wrench_to_parent`) lives in `jeod_dynamics`.
//! - The orchestration walk (`aggregate_wrenches_via_storage`) lives
//!   in `jeod_sim`.
//! - This module is the thin Bevy glue: it builds the
//!   [`MassTreeView`], assembles the
//!   per-edge geometry from `MassChildOf` + live composite
//!   `MassPropertiesC`, runs the kernel, and writes the per-root
//!   aggregated result back into the root's `TotalForceC` /
//!   `FrameDerivativesC`.
//!
//! # Schedule
//!
//! Runs in `JeodSet::ForceCollection`, **after**
//! [`force_collection_system`](crate::systems::force_collection_system).
//! [`composite_mass_system`](crate::mass_tree::composite_mass_system)
//! must have already run earlier in the tick (it does, scheduled
//! `before(JeodSet::EphemerisUpdate)`) so the per-entity
//! `MassPropertiesC.position` is the live composite CoM, which the
//! aggregation walk's `pcm_to_ccm = child.composite_wrt_pstr.position
//! − parent.composite.position` arithmetic depends on.
//!
//! # Children remain kinematic
//!
//! Under the composite-rigid-body model only the root integrates.
//! After the aggregation walk runs, **non-root children's
//! `TotalForceC` and `FrameDerivativesC` are zeroed** so the existing
//! [`integration_system`](crate::systems::integration_system) does not
//! double-count their contributions. Children that still carry
//! `DynamicsConfigC` will integrate with zero external force / torque;
//! the kinematic propagation that derives child poses from the root
//! (the design-doc `propagate_state_from_root_system`) is a separate
//! follow-up.
//!
//! # Frame conventions inside the system
//!
//! All aggregation arithmetic happens in the **per-entity structural
//! frame**, mirroring JEOD `dyn_body_collect.cc:138-202`. JEOD walks
//! every chain shifting each child's `(force, torque)` into the
//! *parent's* structural frame via `T_parent_this^T` (the per-link
//! attach rotation's inverse) plus the parallel-axis arm `pcm_to_ccm`,
//! and only at the root does it rotate the aggregated total back to
//! inertial / body for integration.
//!
//! Concretely:
//!
//! - **Entry boundary** (per non-root entity): the live
//!   `TotalForceC.force` is `Force<RootInertial>` and `TotalForceC.torque`
//!   is `Torque<BodyFrame<SelfRef>>`. Both are converted to **this
//!   entity's structural frame** before being handed to the kernel —
//!   `force_struct = T_inertial_struct · force_inertial` and
//!   `torque_struct = T_struct_body^T · torque_body`,
//!   where `T_inertial_struct = T_struct_body^T · T_inertial_body` is
//!   the same composition `force_collection_system` already uses, and
//!   `T_struct_body` comes from the entity's `StructuralTransformC`
//!   (defaults to identity when absent).
//! - **Per-link shift** (kernel): with both ends in their respective
//!   structural frames, the kernel uses the real
//!   `t_parent_child = MassChildOf.t_parent_child` so child-struct
//!   components correctly rotate into parent-struct via
//!   `t_parent_child^T`, and the parallel-axis arm
//!   `r = pcm_to_ccm` (already in parent struct) plus the now-
//!   parent-struct force gives a `r × F` torque also in parent struct.
//!   This is exactly JEOD lines 152-185.
//! - **Exit boundary** (root): the aggregated total lives in the
//!   root's structural frame. Convert back so the root's
//!   `TotalForceC` keeps its `Force<RootInertial>` /
//!   `Torque<BodyFrame<SelfRef>>` phantoms —
//!   `force_inertial = T_inertial_struct^T · force_struct` and
//!   `torque_body = T_struct_body · torque_struct`.
//!   This matches `force_collection_system`'s root-exit rotation
//!   (JEOD lines 219-252).
//!
//! Identity-attitude chains (no rotation anywhere) collapse every
//! transform to `IDENTITY` and the math reduces to bit-exact addition;
//! rotated chains (parent or any link non-identity) get the same
//! result JEOD does because every per-link rotation matches.

use bevy::prelude::*;
use glam::{DMat3, DVec3};
use std::collections::{HashMap, HashSet};

use jeod_sim::{aggregate_wrenches_via_storage, EdgeGeometry, Wrench};

use crate::components::{
    Abm4StateC, DynamicsConfigC, FrameDerivativesC, GaussJacksonStateC, GravityAccelerationC,
    KinematicChildC, MassChildOf, MassPropertiesC, RotationalStateC, StructuralTransformC,
    TotalForceC,
};
use crate::mass_tree::MassTreeView;

/// Clear multi-step integrator (Gauss-Jackson, ABM4) predictor /
/// corrector history for an entity that's about to resume root-side
/// integration after being a kinematic child of a `MassChildOf`
/// chain. Single-step integrators (RK4, RKF4(5)) carry no per-step
/// history — `jeod_sim::reset_integrators` no-ops their absent state,
/// so this is safe to call unconditionally at every detach site.
///
/// Mirrors `staging_system`'s sister reset for the legacy
/// `AttachEvent` / `DetachEvent` path; the detach codepath here is
/// the ECS-native equivalent on `MassChildOf` rewires.
fn reset_multi_step_history(
    entity: Entity,
    gj_q: &mut Query<&mut GaussJacksonStateC>,
    abm_q: &mut Query<&mut Abm4StateC>,
) {
    let mut gj = gj_q.get_mut(entity).ok();
    let mut abm = abm_q.get_mut(entity).ok();
    jeod_sim::reset_integrators(
        gj.as_mut().map(|g| &mut g.0),
        abm.as_mut().map(|a| &mut a.0),
    );
}

/// Compute `T_inertial_struct = T_struct_body^T · T_inertial_body` for
/// a single entity. `T_struct_body` defaults to identity when the
/// entity has no `StructuralTransformC` (single-body vehicles); the
/// inertial→body rotation defaults to identity when the entity has no
/// `RotationalStateC` (typical kinematic child / 3-DOF body). Mirrors
/// the same composition `force_collection_system` does for the root.
fn t_inertial_struct(
    entity: Entity,
    rot_q: &Query<&RotationalStateC>,
    struct_q: &Query<&StructuralTransformC>,
) -> DMat3 {
    let t_inertial_body = rot_q.get(entity).map_or(DMat3::IDENTITY, |r| {
        r.0.q_inertial_body
            .as_witness()
            .left_quat_to_transformation()
    });
    let t_struct_body = struct_q
        .get(entity)
        .map_or(DMat3::IDENTITY, |s| *s.0.matrix_ref());
    // T_inertial_struct = T_struct_body^T · T_inertial_body.
    // Same identity `force_collection_system` uses (jeod_sim::compute_t_inertial_struct).
    t_struct_body.transpose() * t_inertial_body
}

/// Aggregate per-body external force / torque up every `MassChildOf`
/// chain and write the result into each root's
/// [`TotalForceC`] / [`FrameDerivativesC`]. Non-root children's
/// `TotalForceC` and `FrameDerivativesC` are zeroed so the integration
/// system does not double-integrate the same contributions.
///
/// Fast-paths to a no-op when no entity carries [`MassChildOf`] — the
/// system is free for the single-body case (no chains, no aggregation,
/// nothing to write). The fast-path check uses
/// `parents_q.is_empty()` so the cost is one query iteration over
/// the empty set.
///
/// # Order in `JeodSet::ForceCollection`
///
/// Schedule this system **after**
/// [`force_collection_system`](crate::systems::force_collection_system)
/// within `JeodSet::ForceCollection`. Both must run before
/// `JeodSet::Integration`. The
/// [`composite_mass_system`](crate::mass_tree::composite_mass_system)
/// runs earlier in the tick (before `JeodSet::EphemerisUpdate`) so the
/// per-entity composite CoM (`MassPropertiesC.position`) is already
/// the post-Steiner value the parallel-axis arm consumes.
// JEOD_INV: DB.16 — child forces propagated to parent recursively (composite-rigid-body upward walk)
// JEOD_INV: DB.17 — only the root's TotalForce/FrameDerivatives carry the whole-composite total (children zeroed)
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn wrench_aggregation_system(
    mut commands: Commands,
    mass_q: Query<(Entity, &MassPropertiesC)>,
    parents_q: Query<(Entity, &MassChildOf)>,
    kinematic_q: Query<Entity, With<KinematicChildC>>,
    names_q: Query<&Name>,
    rot_q: Query<&RotationalStateC>,
    struct_q: Query<&StructuralTransformC>,
    grav_q: Query<&GravityAccelerationC>,
    dyn_cfg_q: Query<&DynamicsConfigC>,
    mut totals_q: Query<(Entity, &mut TotalForceC)>,
    mut derivs_q: Query<(Entity, &mut FrameDerivativesC)>,
    // Multi-step integrator state — reset on detach (kinematic →
    // root) so a body resuming integration after the chain is torn
    // down does not step against stale predictor / corrector
    // history. JEOD_INV: IG.37 — multi-step integrator history must
    // be reset on topology change.
    mut gj_q: Query<&mut GaussJacksonStateC>,
    mut abm_q: Query<&mut Abm4StateC>,
) {
    // Fast path: no MassChildOf edges in the world means no chains —
    // every entity is its own root and the existing per-entity
    // `force_collection_system` output is already correct. Still need
    // to clear stale `KinematicChildC` markers from a previous tick
    // where edges existed (e.g. mass tree was just torn down via
    // detach), or `integration_system`'s `Without<KinematicChildC>`
    // filter would keep the entity frozen forever.
    //
    // Per JEOD_INV: IG.37, every body that resumes integration after
    // a topology change must clear its multi-step integrator
    // (Gauss-Jackson, ABM4) predictor / corrector history; the
    // pre-detach history was built against the now-defunct chain's
    // composite dynamics and would diverge on the next step until
    // the predictor catches up. RK4 / RKF4(5) bodies carry no per-
    // step history — `reset_integrators` no-ops their absent state.
    // JEOD_INV: DB.17 — kinematic-child marker cleared when the tree is gone
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    if parents_q.is_empty() {
        for entity in kinematic_q.iter() {
            reset_multi_step_history(entity, &mut gj_q, &mut abm_q);
            commands.entity(entity).remove::<KinematicChildC>();
        }
        return;
    }

    // 1. Build the view (same shape as `composite_mass_system`).
    let view = MassTreeView::from_queries(&mass_q, &parents_q, &names_q);
    if view.is_empty() {
        return;
    }

    // 1a. Validate frame discipline across every chain. The
    //     structural-frame walk needs each entity's
    //     `T_inertial_struct = T_struct_body^T · T_inertial_body` to
    //     be correct, which means each entity must carry the
    //     `RotationalStateC` that supplies `T_inertial_body`.
    //     Defaulting that rotation to `IDENTITY` when the component
    //     is missing would silently reinterpret the entity's
    //     `Force<RootInertial>` as if it were already in the entity's
    //     structural frame, propagating the wrong components
    //     up the chain — a half-baked failure mode forbidden by
    //     CLAUDE.md "Fail Loudly".
    //
    //     We only enforce this when the chain *contains a non-
    //     trivial rotation*: a chain with every `MassChildOf
    //     .t_parent_child = IDENTITY` and every member's
    //     `RotationalStateC` either absent or identity has
    //     `T_inertial_struct = I` everywhere by construction, so the
    //     missing-component default is bit-correct. Adding the
    //     panic only where the rotation actually matters keeps the
    //     simple single-orbit / 3-DOF point-mass topologies free of
    //     a `RotationalStateC` requirement they don't need.
    //
    //     The complete fix — kinematic propagation that derives a
    //     missing child attitude from its parent's attitude composed
    //     with `MassChildOf.t_parent_child`'s rotational component —
    //     is the design-doc Section 15.3
    //     `propagate_state_from_root_system`, tracked as a follow-up
    //     to this PR (see issue #272).
    let chain_has_rotation = parents_q
        .iter()
        .any(|(_, link)| !link.t_parent_child.abs_diff_eq(DMat3::IDENTITY, 1e-12))
        || view.iter_entities().any(|e| {
            rot_q.get(e).is_ok_and(|r| {
                let t =
                    r.0.q_inertial_body
                        .as_witness()
                        .left_quat_to_transformation();
                !t.abs_diff_eq(DMat3::IDENTITY, 1e-12)
            })
        });
    // JEOD_INV: DB.16 — child forces propagated to parent recursively (frame discipline guard for the structural walk)
    if chain_has_rotation {
        for entity in view.iter_entities() {
            assert!(
                rot_q.get(entity).is_ok(),
                "wrench_aggregation_system: entity {entity:?} is in a `MassChildOf` chain that \
                 contains a non-identity rotation (parent attitude or attach `t_parent_child`), \
                 but it has no `RotationalStateC` component. The structural-frame wrench walk \
                 needs every chain member's `T_inertial_body` to be correct; defaulting it to \
                 IDENTITY for this entity would silently treat its `Force<RootInertial>` as if \
                 it were already in the entity's structural frame and aggregate a wrong \
                 composite force.\n\nFix one of:\n  1. Add `RotationalStateC` to entity \
                 {entity:?} (matching its parent's attitude — the typical case for a rigidly \
                 attached appendage that does not yet have its own integrator).\n  2. Make every \
                 link in the chain identity-attitude (no `MassChildOf.t_parent_child` \
                 rotations, no parent `q_inertial_body` rotations) so the IDENTITY default is \
                 correct.\n  3. Wait for the kinematic-propagation system (design-doc Section \
                 15.3 `propagate_state_from_root_system`, follow-up to issue #272) to derive \
                 the missing child attitude from the parent's attitude composed with the \
                 attach rotation."
            );
        }
    }

    // 2. Build per-edge geometry directly from `MassChildOf` + the
    //    live composite `MassPropertiesC`. `pcm_to_ccm` and the
    //    per-link `t_parent_child` are JEOD-faithful — see the
    //    module-level "Frame conventions" doc for why the kernel
    //    needs the *real* `t_parent_child` (not identity) when the
    //    walk happens in structural frames.
    //
    //    JEOD `dyn_body_collect.cc:181`:
    //        pcm_to_ccm = composite_wrt_pstr.position − parent.composite.position
    //    where composite_wrt_pstr.position derives from
    //        MassChildOf.offset + t_parent_child^T · child.composite.position
    //    (`composite.position` = child's composite CoM in *its own*
    //    structural frame; rotated into parent's structural frame
    //    by `t_parent_child^T`, then offset to the parent struct
    //    origin via `MassChildOf.offset`).
    let mut edges: HashMap<Entity, EdgeGeometry> = HashMap::new();
    for (child, link) in parents_q.iter() {
        let parent = link.parent;
        let parent_composite_pos = mass_q
            .get(parent)
            .map(|(_, m)| m.0.center_of_mass.raw_si())
            .unwrap_or(DVec3::ZERO);
        let child_composite_pos = mass_q
            .get(child)
            .map(|(_, m)| m.0.center_of_mass.raw_si())
            .unwrap_or(DVec3::ZERO);
        // `t_parent_child` takes parent-frame components → child-frame
        // components, so its transpose maps the child-frame position
        // back to parent-frame components.
        let child_pos_in_parent_struct =
            link.t_parent_child.transpose() * child_composite_pos + link.offset;
        let pcm_to_ccm = child_pos_in_parent_struct - parent_composite_pos;
        edges.insert(
            child,
            EdgeGeometry {
                pcm_to_ccm,
                t_parent_child: link.t_parent_child,
            },
        );
    }

    // 3. Build per-entity wrenches in **the entity's own structural
    //    frame**. JEOD walks every chain in structural frames so the
    //    per-link `t_parent_child^T` rotation correctly converts
    //    child-struct components into parent-struct components and
    //    the parallel-axis arm `r = pcm_to_ccm` (in parent struct)
    //    composes with a parent-struct force to produce a parent-
    //    struct torque. Doing the walk in inertial would force the
    //    per-link rotation to identity and silently produce wrong
    //    results for any chain with a non-identity attach rotation
    //    or a non-identity parent attitude.
    //
    //    Conversion at this entry boundary is the same composition
    //    `force_collection_system` already uses for the root:
    //      T_inertial_struct = T_struct_body^T · T_inertial_body
    //      force_struct  = T_inertial_struct · force_inertial
    //      torque_struct = T_struct_body^T · torque_body
    //    The defaults (`T_struct_body = I` when no `StructuralTransformC`,
    //    `T_inertial_body = I` when no `RotationalStateC`) collapse
    //    every transform to identity for single-body vehicles and
    //    identity-attitude chains — bit-exact with the previous
    //    inertial-frame walk for those cases.
    let mut wrenches: HashMap<Entity, Wrench> = HashMap::new();
    for (entity, total) in totals_q.iter() {
        if !view.contains(entity) {
            continue;
        }
        let force_inertial = total.0.force.raw_si();
        let torque_body = total.0.torque.raw_si();
        let t_inertial_struct = t_inertial_struct(entity, &rot_q, &struct_q);
        let t_struct_body = struct_q
            .get(entity)
            .map_or(DMat3::IDENTITY, |s| *s.0.matrix_ref());
        let force_struct = t_inertial_struct * force_inertial;
        // T_struct_body takes struct → body, so its transpose takes
        // body → struct (vector components). Mirrors JEOD line 250
        // (`Vector3::transform(composite_properties.T_parent_this, ..)`)
        // run in reverse for the child-side entry.
        let torque_struct = t_struct_body.transpose() * torque_body;
        wrenches.insert(entity, Wrench::new(force_struct, torque_struct));
    }

    // 4. Aggregate up every chain. The kernel walks each child→parent
    //    edge applying `shift_wrench_to_parent(f_child, tau_child,
    //    pcm_to_ccm, t_parent_child)`, which under the JEOD convention
    //    rotates the child's wrench from child-struct into
    //    parent-struct via `t_parent_child^T` and adds
    //    `pcm_to_ccm × f_pstr` for the parallel-axis arm. Returns a
    //    `HashMap<root, Wrench>` whose force/torque components live in
    //    the *root's* structural frame.
    let aggregated: HashMap<Entity, Wrench> =
        aggregate_wrenches_via_storage(&view, &wrenches, &edges);

    // 5. Identify roots once for the writeback pass.
    let roots: HashSet<Entity> = view.iter_roots().collect();

    // 6. Mark non-root nodes as `KinematicChildC` so
    //    `integration_system`'s `Without<KinematicChildC>` filter
    //    skips them. JEOD's composite-rigid-body model integrates
    //    only the root; without this marker the integration system
    //    would still advance every entity carrying
    //    `DynamicsConfigC + TranslationalStateC + GravityControlsC`
    //    under gravity at every RK stage, even though we just zeroed
    //    its `TotalForceC`. JEOD_INV: DB.17 — only the root
    //    integrates.
    //
    //    Conversely, any entity carrying `KinematicChildC` from a
    //    previous tick that is now a root (mass tree was rewired)
    //    must have the marker removed so it resumes integrating.
    let mut should_be_kinematic: HashSet<Entity> = HashSet::new();
    for entity in view.iter_entities() {
        if !roots.contains(&entity) {
            should_be_kinematic.insert(entity);
        }
    }
    // Add markers to entities that should be kinematic but aren't
    // already. Insertion is idempotent in Bevy (re-inserting the same
    // unit struct does nothing), but we filter to avoid the change-
    // detection tick churn on stable chains.
    for entity in &should_be_kinematic {
        if kinematic_q.get(*entity).is_err() {
            commands.entity(*entity).insert(KinematicChildC);
        }
    }
    // Remove markers from entities that are no longer kinematic
    // children (e.g. mass tree was rewired or torn down). Demoting
    // back to a root re-enables integration for that body, so its
    // multi-step integrator history must be cleared first per
    // JEOD_INV: IG.37 — otherwise the next integration step would
    // run the predictor against pre-detach history that no longer
    // matches the body's standalone dynamics.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    for entity in kinematic_q.iter() {
        if !should_be_kinematic.contains(&entity) {
            reset_multi_step_history(entity, &mut gj_q, &mut abm_q);
            commands.entity(entity).remove::<KinematicChildC>();
        }
    }

    // 7. Write `TotalForceC` per entity:
    //    - Roots: aggregated struct-frame total → inertial force,
    //      body torque (mirrors `force_collection_system`'s root
    //      exit; JEOD lines 219-252).
    //    - Non-roots: zero.
    for (entity, mut tf) in totals_q.iter_mut() {
        if !view.contains(entity) {
            continue;
        }
        if roots.contains(&entity) {
            let agg = aggregated
                .get(&entity)
                .copied()
                .unwrap_or_else(Wrench::zero);
            // Root exit boundary: struct → inertial for force,
            // struct → body for torque.
            //
            //   force_inertial = T_inertial_struct^T · force_struct
            //                  = T_struct_body · T_inertial_body^T · force_struct
            //   (JEOD line 219-221: `transform_transpose(structure.state.rot.T_parent_this, …, …_inrtl)`).
            //   torque_body    = T_struct_body · torque_struct
            //   (JEOD line 250: `transform(composite_properties.T_parent_this, …, …_body)`).
            let t_inertial_struct = t_inertial_struct(entity, &rot_q, &struct_q);
            let t_struct_body = struct_q
                .get(entity)
                .map_or(DMat3::IDENTITY, |s| *s.0.matrix_ref());
            let force_inertial = t_inertial_struct.transpose() * agg.force;
            let torque_body = t_struct_body * agg.torque;
            // allowed: wrench-aggregation kernel boundary; `agg.force`
            // arrives as a raw `DVec3` in the root's structural
            // frame from the kernel walk, then rotated to inertial
            // here. Re-wrapping is the canonical re-entry into the
            // typed surface (mirrors `force_collection_system`'s
            // root-exit boundary write).
            tf.0.force = jeod_sim::Force::<jeod_sim::RootInertial>::from_raw_si(force_inertial);
            // allowed: same wrench-kernel boundary; `torque_body`
            // is the structural→body rotation of the kernel's
            // root-struct-frame torque sum, in raw `DVec3`.
            tf.0.torque = jeod_sim::Torque::<jeod_sim::BodyFrame<jeod_sim::SelfRef>>::from_raw_si(
                torque_body,
            );
        } else {
            // allowed: zeroing a typed accumulator; raw zero is
            // unambiguous in any frame phantom.
            tf.0.force = jeod_sim::Force::<jeod_sim::RootInertial>::from_raw_si(DVec3::ZERO);
            // allowed: same.
            tf.0.torque = jeod_sim::Torque::<jeod_sim::BodyFrame<jeod_sim::SelfRef>>::from_raw_si(
                DVec3::ZERO,
            );
        }
    }

    // 8. Recompute `FrameDerivativesC` for the root from the new
    //    `TotalForceC`, and zero it for children. Mirrors the
    //    end-of-step write in `force_collection_system` so downstream
    //    integrators read consistent values.
    let mut updated_totals: HashMap<Entity, jeod_sim::TotalForce> = HashMap::new();
    for (entity, tf) in totals_q.iter() {
        if view.contains(entity) {
            updated_totals.insert(entity, tf.0.to_untyped());
        }
    }

    for (entity, mut fd) in derivs_q.iter_mut() {
        if !view.contains(entity) {
            continue;
        }
        if roots.contains(&entity) {
            if dyn_cfg_q.get(entity).is_err() {
                continue;
            }
            let mass = mass_q.get(entity).ok().map(|(_, m)| m.0.to_untyped());
            let grav_accel = grav_q
                .get(entity)
                .map_or(DVec3::ZERO, |g| g.0.grav_accel.raw_si());
            let total = updated_totals.get(&entity).copied().unwrap_or_default();
            let rot = rot_q.get(entity).ok().map(|r| r.0.to_untyped());
            let new_derivs = if let (Some(rot), Some(m)) = (rot, mass) {
                jeod_sim::compute_frame_derivatives(
                    &total,
                    m.inverse_mass,
                    grav_accel,
                    &m.inertia,
                    &m.inverse_inertia,
                    rot.ang_vel_body,
                )
            } else if let Some(m) = mass {
                jeod_sim::compute_translational_derivatives(total.force, m.inverse_mass, grav_accel)
            } else {
                jeod_sim::FrameDerivatives {
                    trans_accel: grav_accel,
                    rot_accel: DVec3::ZERO,
                }
            };
            // allowed: typed↔untyped kernel boundary; the
            // `compute_frame_derivatives` kernel returns a raw
            // `FrameDerivatives` and re-wrapping is the canonical
            // boundary pattern (mirrors `force_collection_system`).
            fd.0 = jeod_sim::FrameDerivativesTyped::<jeod_sim::RootInertial, jeod_sim::SelfRef>::from_untyped_unchecked(
                &new_derivs,
            );
        } else {
            fd.0 = jeod_sim::FrameDerivativesTyped::<jeod_sim::RootInertial, jeod_sim::SelfRef>::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{
        DynamicsConfigC, ExternalForceC, ExternalTorqueC, FrameDerivativesC, MassChildOf,
        MassPropertiesC, TotalForceC,
    };
    use crate::mass_tree::composite_mass_system;
    use crate::systems::force_collection_system;
    use jeod_sim::MassProperties;

    fn add_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app
    }

    /// Construct a typed inertial-frame [`ExternalForceC`] from a raw
    /// `DVec3`. Test fixtures need to mint typed forces from raw inputs
    /// — the canonical typed APIs (`F64Ext::n()`, `Position::new(...)`,
    /// etc.) operate on per-component scalars, which is awkward when
    /// the test's intent is "set this exact `DVec3` as the external
    /// force". Centralising the lift here keeps the `// allowed:`
    /// boundary annotation in one place.
    fn ext_force_in_root_inertial(v: DVec3) -> ExternalForceC {
        // allowed: test-fixture constructor lifts a raw DVec3 into the
        // typed `Force<RootInertial>` accumulator; mirror of the
        // canonical insertion-time bridge in src/components.rs's
        // `ExternalForceC::From<...>`-equivalent test usage.
        ExternalForceC(jeod_sim::Force::<jeod_sim::RootInertial>::from_raw_si(v))
    }

    /// Construct a typed body-frame [`ExternalTorqueC`] from a raw
    /// `DVec3`. Same rationale as [`ext_force_in_root_inertial`].
    fn ext_torque_in_body(v: DVec3) -> ExternalTorqueC {
        // allowed: test-fixture constructor — lifts a raw DVec3 into typed Torque<BodyFrame<SelfRef>> for spawn args.
        let t = jeod_sim::Torque::<jeod_sim::BodyFrame<jeod_sim::SelfRef>>::from_raw_si(v);
        ExternalTorqueC(t)
    }

    fn run_pipeline(app: &mut App) {
        // Run the same per-tick sequence the real plugin schedules:
        // composite recomputation → force collection → wrench aggregation.
        // Each is a single system call so we don't drag in the full
        // FixedUpdate machinery for unit tests.
        app.add_systems(
            Update,
            (
                composite_mass_system,
                force_collection_system.after(composite_mass_system),
                wrench_aggregation_system.after(force_collection_system),
            ),
        );
        app.update();
    }

    #[test]
    fn no_chains_is_noop() {
        // Single body, no MassChildOf — wrench system fast-paths and
        // leaves the per-entity TotalForceC alone.
        let mut app = add_test_app();
        let core = MassProperties::new(10.0);
        let f = DVec3::new(1.0, 2.0, 3.0);
        let entity = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(core),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ext_force_in_root_inertial(f),
                ExternalTorqueC::default(),
            ))
            .id();
        run_pipeline(&mut app);

        let tf = app
            .world()
            .get::<TotalForceC>(entity)
            .unwrap()
            .0
            .to_untyped();
        assert_eq!(tf.force, f, "single body external force passes through");
    }

    #[test]
    fn child_force_appears_at_root_with_cross_term() {
        // Parent at origin, child at offset (1,0,0). Child carries a
        // pure +y external force. After force collection +
        // wrench aggregation:
        //   - root.force_inertial = +y (free vector preserved across
        //     identity-attitude chains).
        //   - root.torque_body = pcm_to_ccm × F (identity attitude: body=struct=inertial).
        //
        // With parent mass 10 and child mass 5, composite CoM lives at
        //   (10·0 + 5·1)/15 = 1/3 along x.
        //   pcm_to_ccm = (1,0,0) − (1/3,0,0) = (2/3, 0, 0).
        //   r × F = (2/3,0,0) × (0,1,0) = (0, 0, 2/3).
        let mut app = add_test_app();
        let parent = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(10.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();
        let child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ext_force_in_root_inertial(DVec3::new(0.0, 1.0, 0.0)),
                ExternalTorqueC::default(),
            ))
            .id();

        run_pipeline(&mut app);

        let root_tf = app
            .world()
            .get::<TotalForceC>(parent)
            .unwrap()
            .0
            .to_untyped();
        let child_tf = app
            .world()
            .get::<TotalForceC>(child)
            .unwrap()
            .0
            .to_untyped();

        let two_thirds = 2.0 / 3.0;
        let root_force_err = (root_tf.force - DVec3::new(0.0, 1.0, 0.0)).length();
        let root_torque_err = (root_tf.torque - DVec3::new(0.0, 0.0, two_thirds)).length();
        assert!(root_force_err < 1e-12, "root force {:?}", root_tf.force);
        assert!(root_torque_err < 1e-12, "root torque {:?}", root_tf.torque);

        // Child must have been zeroed so it doesn't double-integrate.
        assert_eq!(child_tf.force, DVec3::ZERO, "child force zeroed");
        assert_eq!(child_tf.torque, DVec3::ZERO, "child torque zeroed");
    }

    #[test]
    fn pure_child_torque_aggregates_to_root() {
        // Identical setup but with a child external torque only.
        // No force → no parallel-axis cross term; root torque equals
        // (rotated) child torque (identity attitude here, so no
        // rotation: torque components pass through).
        let mut app = add_test_app();
        let parent = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(10.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();
        let child_torque = DVec3::new(0.5, -0.25, 1.0);
        let child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ext_torque_in_body(child_torque),
            ))
            .id();

        run_pipeline(&mut app);

        let root_tf = app
            .world()
            .get::<TotalForceC>(parent)
            .unwrap()
            .0
            .to_untyped();
        assert_eq!(root_tf.force, DVec3::ZERO);
        let err = (root_tf.torque - child_torque).length();
        assert!(
            err < 1e-12,
            "root torque {:?}, expected {:?}",
            root_tf.torque,
            child_torque
        );

        let child_tf = app
            .world()
            .get::<TotalForceC>(child)
            .unwrap()
            .0
            .to_untyped();
        assert_eq!(child_tf.force, DVec3::ZERO);
        assert_eq!(child_tf.torque, DVec3::ZERO);
    }

    /// A `MassChildOf` chain with a non-identity attach rotation
    /// must have `RotationalStateC` on every chain member —
    /// otherwise `T_inertial_body` defaults to IDENTITY for the
    /// missing entity and the structural-frame walk silently treats
    /// the entity's `Force<RootInertial>` as if it were already in
    /// the entity's structural frame, propagating the wrong
    /// components up the chain. This is precisely the half-baked
    /// failure mode forbidden by CLAUDE.md "Fail Loudly".
    ///
    /// Until the kinematic-propagation system (design-doc Section
    /// 15.3 `propagate_state_from_root_system`, follow-up to issue
    /// #272) lands and can derive a missing child attitude from
    /// the parent's attitude composed with the attach rotation,
    /// `wrench_aggregation_system` requires every chain member to
    /// carry its own `RotationalStateC` whenever the chain has any
    /// non-identity rotation.
    #[test]
    #[should_panic(expected = "is in a `MassChildOf` chain that contains a non-identity rotation")]
    fn child_with_attach_rotation_and_no_rotational_state_panics() {
        let mut app = add_test_app();
        let parent = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(10.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();
        let t_pc = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let _child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::with_rotation(parent, DVec3::new(1.0, 0.0, 0.0), t_pc),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ext_force_in_root_inertial(DVec3::new(5.0, 0.0, 0.0)),
                ExternalTorqueC::default(),
            ))
            .id();

        run_pipeline(&mut app);
    }

    /// Same chain shape (parent + rotated-attach child), but every
    /// chain member now carries `RotationalStateC = identity`. The
    /// chain still has rotation (the attach `t_parent_child` is
    /// non-identity), so the walk must use the per-link rotation
    /// correctly; with explicit identity attitudes everywhere the
    /// child's force is correctly interpreted as already in its
    /// structural frame (since that frame coincides with inertial
    /// at identity attitude), then rotated through the attach
    /// rotation into the parent's structural frame.
    ///
    /// JEOD-derived analytical answer:
    /// - Setup as in the panic case: parent mass 10 at origin; child
    ///   mass 5 attached at offset (1,0,0) with `t_pc` = R(parent →
    ///   child) = +90° about +Z (parent's +x → child's +y).
    /// - Child carries `Force<RootInertial>` = (5, 0, 0). With
    ///   `RotationalStateC = identity`, child struct == child body
    ///   == inertial; the entry boundary leaves the components
    ///   untouched.
    /// - Per-link shift: `t_pc^T · (5,0,0)_child = (0, -5, 0)_parent`.
    /// - `pcm_to_ccm = (2/3, 0, 0)` (composite of mass-10 root + mass-5
    ///   child whose composite sits at parent struct offset (1,0,0)).
    /// - Parallel-axis torque: `(2/3,0,0) × (0,-5,0) = (0, 0, -10/3)`.
    /// - Root attitude identity, so root struct == root body == inertial:
    ///   force_inertial = (0, -5, 0), torque_body = (0, 0, -10/3).
    #[test]
    fn child_with_attach_rotation_aggregates_correctly_when_attitude_is_explicit() {
        use jeod_sim::RotationalState;

        let mut app = add_test_app();
        let identity_rot = RotationalState::default();

        let parent = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(10.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                RotationalStateC::from(identity_rot),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();
        let t_pc = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let _child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::with_rotation(parent, DVec3::new(1.0, 0.0, 0.0), t_pc),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                RotationalStateC::from(identity_rot),
                ext_force_in_root_inertial(DVec3::new(5.0, 0.0, 0.0)),
                ExternalTorqueC::default(),
            ))
            .id();

        run_pipeline(&mut app);

        let root_tf = app
            .world()
            .get::<TotalForceC>(parent)
            .unwrap()
            .0
            .to_untyped();
        let expected_force = DVec3::new(0.0, -5.0, 0.0);
        let expected_torque = DVec3::new(0.0, 0.0, -10.0 / 3.0);
        let f_err = (root_tf.force - expected_force).length();
        let t_err = (root_tf.torque - expected_torque).length();
        assert!(
            f_err < 1e-12,
            "root force {:?}, expected {:?}",
            root_tf.force,
            expected_force
        );
        assert!(
            t_err < 1e-12,
            "root torque {:?}, expected {:?}",
            root_tf.torque,
            expected_torque
        );
    }

    #[test]
    fn parent_and_two_children_sum() {
        // Parent (mass 10) + two children (mass 5) at ±y offsets.
        // Both children push in +x; the parallel-axis torques cancel.
        // Parent itself has a +z force.
        let mut app = add_test_app();
        let parent = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(10.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ext_force_in_root_inertial(DVec3::new(0.0, 0.0, 1.0)),
                ExternalTorqueC::default(),
            ))
            .id();
        let _a = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::new(parent, DVec3::new(0.0, 1.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ext_force_in_root_inertial(DVec3::new(1.0, 0.0, 0.0)),
                ExternalTorqueC::default(),
            ))
            .id();
        let _b = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::new(parent, DVec3::new(0.0, -1.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ext_force_in_root_inertial(DVec3::new(1.0, 0.0, 0.0)),
                ExternalTorqueC::default(),
            ))
            .id();

        run_pipeline(&mut app);

        let root_tf = app
            .world()
            .get::<TotalForceC>(parent)
            .unwrap()
            .0
            .to_untyped();
        let expected_force = DVec3::new(2.0, 0.0, 1.0);
        let expected_torque = DVec3::ZERO;
        let f_err = (root_tf.force - expected_force).length();
        let t_err = (root_tf.torque - expected_torque).length();
        assert!(f_err < 1e-12, "force {:?}", root_tf.force);
        assert!(t_err < 1e-12, "torque {:?}", root_tf.torque);
    }

    /// Frame-discipline regression: when the parent attitude is
    /// non-identity (root has a real `RotationalStateC` whose
    /// `q_inertial_body` is not identity), the wrench-shift
    /// cross-product must use the parent's structural-frame `r` and a
    /// parent-structural-frame `F` — equivalently, the inertial-frame
    /// `F` must be rotated into the parent's structural frame before
    /// the cross-product. An inertial-frame walk that crossed
    /// `pcm_to_ccm` (in parent struct) with the inertial-frame force
    /// directly would be bit-correct only at identity attitude and
    /// silently wrong otherwise.
    ///
    /// JEOD-derived analytical answer for this scenario, in the
    /// **parent's structural frame**:
    ///
    /// ```text
    /// parent attitude: passive +30° about Z. The constructor
    ///   `JeodQuat::left_quat_from_eigen_rotation(angle, axis)`
    /// produces a quaternion `q` for which
    ///   `T_inertial_body = q.left_quat_to_transformation()`
    /// is the passive matrix
    ///   T_inertial_body · (1,0,0) = (cos 30°, −sin 30°, 0)
    /// (the inertial x-axis, expressed in body coords after the body
    /// frame has been rotated +30° about its Z axis).
    ///
    /// child mass 5 attached at parent struct offset (1,0,0) with
    /// identity attach rotation; child carries the same
    /// RotationalStateC as the parent so the chain is physically
    /// consistent (parent struct == child struct under identity
    /// attach).
    ///
    /// composite CoM (parent struct) = (10·0 + 5·1)/15 = (1/3,0,0)
    /// pcm_to_ccm = (2/3, 0, 0)
    /// external force on the child = (1, 0, 0) inertial.
    ///
    /// entry: convert to child struct (= parent struct here):
    ///   T_inertial_struct = T_struct_body^T · T_inertial_body
    ///                     = I · T_inertial_body
    ///   force_struct = T_inertial_body · (1,0,0)
    ///                = (cos 30°, −sin 30°, 0).
    ///
    /// kernel cross-product in parent struct:
    ///   r × F_pstr = (2/3,0,0) × (cos 30°, −sin 30°, 0)
    ///              = (0, 0, 2/3 · (−sin 30°))
    ///              = (0, 0, −1/3).
    ///
    /// root exit: rotate force_pstr to inertial (T_inertial_struct^T):
    ///   T_inertial_body^T · (cos 30°, −sin 30°, 0) = (1, 0, 0)  ✓
    /// rotate torque_pstr to body (T_struct_body):
    ///   I · (0, 0, −1/3) = (0, 0, −1/3).
    /// ```
    ///
    /// The previous inertial-frame walk would have computed
    /// `r × F_inrtl = (2/3,0,0) × (1,0,0) = (0,0,0)` — silently
    /// dropping the parallel-axis torque entirely. The nonzero torque
    /// this test now demands is the load-bearing signal that frame
    /// discipline survives a non-identity parent attitude.
    #[test]
    fn rotated_parent_attitude_routes_cross_term_through_parent_struct() {
        use jeod_sim::RotationalState;

        let mut app = add_test_app();

        let parent_q = jeod_sim::JeodQuat::left_quat_from_eigen_rotation(
            std::f64::consts::FRAC_PI_6, // 30°
            DVec3::Z,
        );
        let parent_rot_state = RotationalState {
            quaternion: parent_q,
            ang_vel_body: DVec3::ZERO,
        };

        let parent = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(10.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                RotationalStateC::from(parent_rot_state),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();
        // Child at offset (1,0,0), identity attach, attitude matches
        // parent so the chain is physically consistent
        // (q_inertial_body = R_z(30°) at every link).
        let _child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                RotationalStateC::from(parent_rot_state),
                ext_force_in_root_inertial(DVec3::new(1.0, 0.0, 0.0)),
                ExternalTorqueC::default(),
            ))
            .id();

        run_pipeline(&mut app);

        let root_tf = app
            .world()
            .get::<TotalForceC>(parent)
            .unwrap()
            .0
            .to_untyped();

        // Free-vector force preserved at the root: T_inertial_body^T ·
        // T_inertial_body · (1,0,0) = (1, 0, 0).
        let expected_force = DVec3::new(1.0, 0.0, 0.0);
        // Parallel-axis torque, in body frame (= struct frame for
        // T_struct_body = identity): (0, 0, −1/3).
        let expected_torque = DVec3::new(0.0, 0.0, -1.0 / 3.0);
        let f_err = (root_tf.force - expected_force).length();
        let t_err = (root_tf.torque - expected_torque).length();
        assert!(
            f_err < 1e-12,
            "root force {:?}, expected {:?}",
            root_tf.force,
            expected_force
        );
        assert!(
            t_err < 1e-12,
            "root torque {:?}, expected {:?}",
            root_tf.torque,
            expected_torque
        );
    }

    /// Kinematic-children integration gate: children of `MassChildOf`
    /// chains must NOT drift under gravity across multiple integration
    /// steps. Without the `KinematicChildC` marker, zeroing children's
    /// `TotalForceC` is not enough — `integration_system` recomputes
    /// gravity at every RK sub-stage from `GravityControlsC` and would
    /// advance the child's `TranslationalStateC` regardless. This test
    /// stands up a real Earth gravity source, runs the full
    /// FixedUpdate pipeline through several steps (force collection,
    /// wrench aggregation, integration), and asserts the child's
    /// translational state stays at the spawn-time value.
    #[test]
    fn child_translational_state_does_not_drift_under_gravity() {
        use crate::PlanetBundle;
        use bevy::time::Fixed;
        use jeod_sim::recipes::{constants, orbital_elements, vehicle};
        use jeod_sim::{GravityControl, IntegratorType, TranslationalState, VehicleBuilder, EARTH};
        use std::time::Duration;

        const DT: f64 = 1.0;

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        // allowed: test-fixture FixedUpdate timestep; mirrors the same
        // construction every `tests/bevy_parity*.rs` integration test
        // already does. The escape-hatch script guards per-step
        // typed-quantities bypasses in production code paths, not
        // one-shot test-app setup of the Bevy `Time<Fixed>` resource.
        app.insert_resource(Time::<Fixed>::from_seconds(DT));
        app.add_plugins(crate::JeodPlugin);

        // Earth point-mass source.
        let earth = app
            .world_mut()
            .spawn(PlanetBundle::point_mass("Earth", &EARTH))
            .id();

        // Parent: a 3-DOF point-mass orbital body at ISS-like
        // initial conditions, integrated under spherical gravity.
        let parent_cfg = VehicleBuilder::new()
            .from_orbital_elements(orbital_elements::iss(), constants::mu_ggm05c())
            .three_dof_point_mass(vehicle::iss_mass())
            .with_integrator(IntegratorType::Rk4)
            .gravity(GravityControl::new_spherical(0_usize, false))
            .build();
        let parent = {
            // Lift `VehicleConfig::spawn_bevy` (defined on
            // `VehicleConfigBevyExt` in `crate::lib.rs`) into scope
            // for this one call. Importing the trait at the test
            // module's top would conflict with name resolution
            // elsewhere; localizing the `use` keeps it surgical.
            use crate::VehicleConfigBevyExt;
            let mut cmds = app.world_mut().commands();
            let p = parent_cfg.spawn_bevy(&mut cmds, &[earth]);
            app.world_mut().flush();
            p
        };

        // Child: a point-mass with a `MassChildOf` link to the
        // parent. Spawn it with the *parent's* initial position so
        // we can detect drift as a non-zero delta from that spawn
        // value. (In production, kinematic propagation would set
        // the child's pose every step from the root; for this
        // regression test we only need to confirm the integrator
        // does not move it.)
        let parent_pos = app
            .world()
            .get::<crate::TranslationalStateC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .position;
        let child = app
            .world_mut()
            .spawn((
                Name::new("child"),
                MassPropertiesC::from(MassProperties::new(100.0)),
                MassChildOf::new(parent, DVec3::new(0.5, 0.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                crate::TranslationalStateC::from(TranslationalState {
                    position: parent_pos,
                    velocity: DVec3::ZERO,
                }),
                crate::GravityControlsC(jeod_sim::GravityControls::<Entity> {
                    controls: vec![GravityControl::new_spherical(earth, false)],
                }),
                crate::GravityAccelerationC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // Run several FixedUpdate cycles. The standard pattern in
        // tests/bevy_parity.rs:108-113: advance `Time<Fixed>` by DT,
        // then run the `FixedUpdate` schedule directly. Avoids
        // depending on `app.update()`'s implicit virtual-time
        // advancement (which `MinimalPlugins` does not deliver
        // out-of-the-box).
        for _ in 0..5 {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(DT));
            app.world_mut().run_schedule(FixedUpdate);
        }

        // Child must still be at its spawn position. With the bug,
        // gravity would have integrated it ~9.8/2 m in the first
        // step alone (4.9 m), with growing drift each subsequent
        // step.
        let child_pos = app
            .world()
            .get::<crate::TranslationalStateC>(child)
            .unwrap()
            .0
            .to_untyped()
            .position;
        let drift = (child_pos - parent_pos).length();
        // The child should not have moved under integration. Allow
        // numerical noise but fail loudly on any meaningful drift.
        assert!(
            drift < 1e-6,
            "kinematic child drifted {drift:.3e} m under gravity over 5 steps; \
             expected ~0 (KinematicChildC marker should keep integration_system \
             from advancing it)"
        );

        // Sanity: the parent (root) DID integrate. If neither moved,
        // the test would silently pass even with a broken
        // integration system.
        let parent_pos_after = app
            .world()
            .get::<crate::TranslationalStateC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .position;
        let parent_drift = (parent_pos_after - parent_pos).length();
        assert!(
            parent_drift > 1.0,
            "parent did not integrate (drift {parent_drift:.3e} m); test setup broken"
        );

        // The child must carry `KinematicChildC` after the first
        // tick — pin the marker contract directly.
        assert!(
            app.world().entity(child).contains::<KinematicChildC>(),
            "child {child:?} should carry KinematicChildC after wrench aggregation"
        );
    }

    /// Derivative-class SRP/thermal regression: a kinematic child
    /// of a `MassChildOf` chain that carries `FlatPlateConfigC` with
    /// `DerivativeFirstOrder` / `DerivativeRk4` integration order
    /// must still get plate temperature updates and a non-zero
    /// `RadiationForceC` once per step. The derivative-class SRP/
    /// thermal recompute lives inside `integration_system`'s
    /// `Without<KinematicChildC>`-filtered loop, so without a
    /// fallback in `flat_plate_srp_system` the kinematic child's
    /// `RadiationForceC` would stay at the per-tick zero and its
    /// plate temperatures would never advance. The fix reroutes
    /// kinematic children through the `Scheduled` arm here (single-
    /// shot Euler) — the appendage's thermal state and force stay
    /// live; only the orbital state is gated by the chain's
    /// composite-rigid-body integration.
    #[test]
    fn kinematic_child_with_derivative_srp_still_gets_force_and_temp_update() {
        use crate::components::{
            FlatPlateConfigC, RadiationForceC, SunMarker, TranslationalStateC,
        };
        use jeod_sim::{
            FlatPlate, FlatPlateParams, FlatPlateState, FlatPlateThermal, MassProperties,
            ThermalIntegrationOrder, TranslationalState,
        };

        let mut app = add_test_app();

        // Sun ~1 AU along +x. Pure inertial position; no gravity / mass.
        app.world_mut().spawn((
            SunMarker,
            TranslationalStateC::from(TranslationalState {
                position: DVec3::new(1.496e11, 0.0, 0.0),
                velocity: DVec3::ZERO,
            }),
        ));

        // Parent root (no SRP). Just somewhere outside the Sun's
        // collapse radius.
        let parent = app
            .world_mut()
            .spawn((
                Name::new("parent"),
                MassPropertiesC::from(MassProperties::new(10.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
                TranslationalStateC::from(TranslationalState {
                    position: DVec3::new(7.0e6, 0.0, 0.0),
                    velocity: DVec3::ZERO,
                }),
            ))
            .id();

        // Child appendage with derivative-class FlatPlateConfigC. Plate
        // pointed at the Sun (normal +X = away from Sun, so flux hits
        // the back: actually we want it facing the Sun, so normal -X).
        // FlatPlate normal convention: outward-facing; SRP only
        // contributes when -normal · flux_hat > 0, i.e. normal points
        // into the Sun.
        let plates = vec![(
            FlatPlate {
                area: 10.0,
                normal: -DVec3::X, // facing the Sun
                position: DVec3::ZERO,
            },
            FlatPlateParams {
                albedo: 0.3,
                diffuse: 0.3,
            },
            FlatPlateThermal {
                emissivity: 0.5,
                heat_capacity_per_area: 50.0,
                thermal_power_dump: 0.0,
            },
        )];
        let initial_temp = 270.0;
        let child = app
            .world_mut()
            .spawn((
                Name::new("child_appendage"),
                MassPropertiesC::from(MassProperties::new(1.0)),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
                TranslationalStateC::from(TranslationalState {
                    position: DVec3::new(7.0e6, 0.0, 0.0),
                    velocity: DVec3::ZERO,
                }),
                RadiationForceC::default(),
                FlatPlateConfigC(FlatPlateState {
                    plates,
                    temperatures: vec![initial_temp],
                    t_pow4_cached: vec![initial_temp.powi(4)],
                    integration_order: ThermalIntegrationOrder::DerivativeRk4,
                    ..Default::default()
                }),
            ))
            .id();

        // Run the wrench-aggregation pipeline + the SRP system.
        // Order: composite_mass → flat_plate_srp → force_collection →
        // wrench_aggregation. We need a non-zero `dt` for the
        // Scheduled-arm Euler integration to actually move
        // temperatures.
        use bevy::time::Fixed;
        use std::time::Duration;
        const DT: f64 = 1.0;
        // allowed: test-fixture FixedUpdate timestep; same pattern as
        // `child_translational_state_does_not_drift_under_gravity`
        // and the wider `tests/bevy_parity*.rs` integration tests.
        app.insert_resource(Time::<Fixed>::from_seconds(DT));
        app.add_systems(
            Update,
            (
                composite_mass_system,
                crate::systems::flat_plate_srp_system.after(composite_mass_system),
                force_collection_system.after(crate::systems::flat_plate_srp_system),
                wrench_aggregation_system.after(force_collection_system),
            ),
        );
        // First tick: `flat_plate_srp_system` runs *before*
        // `wrench_aggregation_system` adds `KinematicChildC` (the
        // marker comes online at the end of the tick), so the SRP
        // path takes the derivative arm and stashes `stage_inputs`.
        // Second tick is the load-bearing one — by then the child
        // carries `KinematicChildC` and the SRP path must take the
        // Scheduled fallback.
        for _ in 0..2 {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(DT));
            app.update();
        }

        // The child must carry `KinematicChildC` (sanity).
        assert!(
            app.world().entity(child).contains::<KinematicChildC>(),
            "child must be marked KinematicChildC after first tick"
        );

        // RadiationForceC on the child should be non-zero —
        // `flat_plate_srp_system`'s kinematic-child fallback ran the
        // Scheduled arm, which writes the SRP force directly here.
        // Without the fallback, the derivative-class arm would
        // stash `stage_inputs` and leave `RadiationForceC` at zero
        // (the integration system that consumes `stage_inputs` is
        // filtered out for kinematic children).
        let rf = app
            .world()
            .get::<RadiationForceC>(child)
            .expect("child should have RadiationForceC");
        let force_mag = rf.force.length();
        assert!(
            force_mag > 1e-9,
            "kinematic child with derivative-class FlatPlateConfigC should still see SRP force; \
             got |force|={force_mag:.3e} N (the per-step zero clear was never overwritten — \
             flat_plate_srp_system's kinematic-child fallback did not fire)"
        );

        // Plate temperatures should also have changed — the Euler
        // step fed `temp_dots` into `integrate_temperatures` only on
        // the Scheduled arm. The temperature delta direction depends
        // on the absorbed flux vs emitted blackbody, but it must be
        // *something* under direct sunlight on a non-zero-emissivity
        // plate at 270 K.
        let fc = app
            .world()
            .get::<FlatPlateConfigC>(child)
            .expect("child should have FlatPlateConfigC");
        let temp_delta = (fc.0.temperatures[0] - initial_temp).abs();
        assert!(
            temp_delta > 1e-6,
            "kinematic child plate temperature should advance under Euler integration; \
             got Δ={temp_delta:.3e} K (still at {} K — flat_plate_srp_system's kinematic-child \
             fallback did not run the temperature integration)",
            fc.0.temperatures[0]
        );
    }

    /// IG.37 regression on the ECS-native `MassChildOf` rewire path:
    /// when a body is detached from its parent (the `MassChildOf`
    /// link is removed), `wrench_aggregation_system` demotes it from
    /// kinematic-child back to root by stripping `KinematicChildC`.
    /// Before that demote, every multi-step integrator
    /// (Gauss-Jackson, ABM4) on the entity must have its predictor /
    /// corrector history cleared — otherwise the next integration
    /// step would consume stale history that no longer matches the
    /// body's standalone composite dynamics, producing a transient
    /// trajectory divergence until the predictor catches up. JEOD's
    /// `dyn_body_attach.cc::reset_integrators()` (lines 860, 871)
    /// and `dyn_body_detach.cc:271-273` already guard the
    /// `AttachEvent` / `DetachEvent` legacy path; this test pins the
    /// equivalent guard on the ECS-native `MassChildOf` rewire.
    #[test]
    fn detached_child_gj_history_resets_before_resuming_integration() {
        use crate::components::{GaussJacksonStateC, IntegratorTypeC};
        use jeod_sim::{
            GaussJacksonConfig, GaussJacksonState, IntegratorType, MassProperties,
            TranslationalState,
        };

        let mut app = add_test_app();

        // Parent (a real root): doesn't carry GJ state, just present so the
        // child has something to be a `MassChildOf`.
        let parent_mass = MassProperties::new(10.0);
        let parent = app
            .world_mut()
            .spawn((
                Name::new("parent"),
                MassPropertiesC::from(parent_mass),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // Child: carries a GJ state. Spawned attached to `parent` via
        // `MassChildOf`, which `wrench_aggregation_system` will mark
        // `KinematicChildC` on the first tick.
        let gj_cfg = GaussJacksonConfig::default();
        let mut primed_gj = GaussJacksonState::new(gj_cfg);
        // Force the GJ state out of priming so we can detect the
        // reset by observing `is_priming() == true` afterward. The
        // production codepath would normally need ~200 steps; we
        // shortcut by faking a stale topology marker which
        // `reset_for_topology_change` clears the same way as a real
        // detach. Using `mark_topology_dirty` here is the smallest
        // observable proxy for "non-default GJ state": its inverse
        // (`is_topology_dirty == false`) is exactly what
        // `reset_for_topology_change` guarantees on exit.
        primed_gj.mark_topology_dirty();
        let child = app
            .world_mut()
            .spawn((
                Name::new("child"),
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
                IntegratorTypeC(IntegratorType::GaussJackson(gj_cfg)),
                GaussJacksonStateC(primed_gj),
                crate::TranslationalStateC::from(TranslationalState::default()),
            ))
            .id();

        // Tick once — `wrench_aggregation_system` sees the chain and
        // marks the child `KinematicChildC`. The GJ state should be
        // untouched here (the child is not demoting back to root yet
        // — just being identified as kinematic).
        run_pipeline(&mut app);
        assert!(
            app.world().entity(child).contains::<KinematicChildC>(),
            "child must carry KinematicChildC while attached"
        );
        assert!(
            app.world()
                .get::<GaussJacksonStateC>(child)
                .unwrap()
                .0
                .is_topology_dirty(),
            "GJ state should remain untouched while child is still kinematic"
        );

        // Now tear down the chain: remove `MassChildOf` from the
        // child. Next tick of `wrench_aggregation_system` must
        // demote the child back to root and clear its GJ history
        // (IG.37) before stripping `KinematicChildC`.
        app.world_mut().entity_mut(child).remove::<MassChildOf>();
        app.update();
        // Bevy commands flush is implicit at the end of `app.update()`
        // — re-run once more to surface the deferred `Commands`
        // mutations from `wrench_aggregation_system` against the
        // `&mut GaussJacksonStateC` the system queried this same
        // tick. The system applies the reset directly via the
        // `gj_q.get_mut` query, so the change is visible after the
        // first post-detach update — but we run an extra update for
        // safety so any post-flush `KinematicChildC` removal
        // settles.
        app.update();

        assert!(
            !app.world().entity(child).contains::<KinematicChildC>(),
            "child must be demoted to root once MassChildOf is removed"
        );
        let gj_after = &app.world().get::<GaussJacksonStateC>(child).unwrap().0;
        assert!(
            !gj_after.is_topology_dirty(),
            "detach must clear topology_dirty (IG.37): GaussJacksonStateC.is_topology_dirty \
             still true after MassChildOf removal"
        );
        assert!(
            gj_after.is_priming(),
            "detach must reset GJ history to priming (IG.37): GaussJacksonStateC \
             still reports is_priming() == false after MassChildOf removal — its \
             stale predictor history would be applied to the body's standalone \
             dynamics on the next integration step"
        );
    }
}
