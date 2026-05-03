//! Bevy system for composite-rigid-body wrench aggregation.
//!
//! Closes the Bevy half of issue [#272]: walks the `MassChildOf` tree
//! after [`force_collection_system`](crate::systems::force_collection_system)
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
//!   [`MassTreeView`](crate::mass_tree::MassTreeView), assembles the
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
//! # Children remain kinematic in this PR
//!
//! Per [#272] the composite-rigid-body model integrates only the root.
//! After the aggregation walk runs, **non-root children's
//! `TotalForceC` and `FrameDerivativesC` are zeroed** so the existing
//! [`integration_system`](crate::systems::integration_system) does not
//! double-count their contributions. Children that still carry
//! `DynamicsConfigC` will integrate with zero external force / torque;
//! the kinematic propagation that derives child poses from the root
//! (the design-doc `propagate_state_from_root_system`) is a separate
//! sub-issue and not part of this PR.
//!
//! # Frame conventions inside the system
//!
//! All aggregation arithmetic happens in the **inertial (root)** frame:
//!
//! - `TotalForceC.force` is read directly as inertial-frame force (its
//!   phantom is `Force<RootInertial>`).
//! - `TotalForceC.torque` is read in body frame and rotated into
//!   inertial via the body's `RotationalStateC.q_inertial_body`. When
//!   no `RotationalStateC` is present (the typical kinematic-child
//!   case), the rotation defaults to identity — i.e., the body-frame
//!   torque is treated as inertial-frame components. This is correct
//!   for chains where every link has identity attitude relative to
//!   the root; for rotated chains, callers must add a
//!   `RotationalStateC` whose quaternion encodes the chain's
//!   accumulated rotation. The kinematic propagation that would do
//!   this automatically lives in a follow-up sub-issue
//!   (design-doc Section 15.3 `propagate_state_from_root_system`).
//! - The parallel-axis cross-term `r × F` uses
//!   `r = pcm_to_ccm` (parent CoM → child CoM in parent's structural
//!   frame). Because force is a free vector, `r × F` produces a
//!   torque whose components are also in the parent's structural
//!   frame; with identity-attitude chains, parent struct = inertial,
//!   so no rotation is needed. With rotated chains, the cross-term
//!   carries small frame error proportional to the chain rotation —
//!   again, a follow-up's responsibility.
//!
//! At the root exit boundary, the aggregated inertial-frame torque is
//! rotated back into the root's body frame via the root's own
//! `T_inertial_body` and written to `TotalForceC.torque`.
//!
//! [#272]: https://github.com/simnaut/bevy_jeod/issues/272

use bevy::prelude::*;
use glam::{DMat3, DVec3};
use std::collections::{HashMap, HashSet};

use jeod_sim::{aggregate_wrenches_via_storage, EdgeGeometry, Wrench};

use crate::components::{
    DynamicsConfigC, FrameDerivativesC, GravityAccelerationC, MassChildOf, MassPropertiesC,
    RotationalStateC, StructuralTransformC, TotalForceC,
};
use crate::mass_tree::MassTreeView;

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
    mass_q: Query<(Entity, &MassPropertiesC)>,
    parents_q: Query<(Entity, &MassChildOf)>,
    names_q: Query<&Name>,
    rot_q: Query<&RotationalStateC>,
    _struct_q: Query<&StructuralTransformC>,
    grav_q: Query<&GravityAccelerationC>,
    dyn_cfg_q: Query<&DynamicsConfigC>,
    mut totals_q: Query<(Entity, &mut TotalForceC)>,
    mut derivs_q: Query<(Entity, &mut FrameDerivativesC)>,
) {
    // Fast path: no MassChildOf edges in the world means no chains —
    // every entity is its own root and the existing per-entity
    // `force_collection_system` output is already correct.
    if parents_q.is_empty() {
        return;
    }

    // 1. Build the view (same shape as `composite_mass_system`).
    let view = MassTreeView::from_queries(&mass_q, &parents_q, &names_q);
    if view.is_empty() {
        return;
    }

    // 2. Build per-edge geometry directly from `MassChildOf` + the
    //    live composite `MassPropertiesC`.
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

    // 3. Build per-entity wrenches in **inertial** frame (force) and
    //    in **inertial** frame (torque, rotated from body via the
    //    body's `T_inertial_body`). Working in inertial means force
    //    is a free vector that doesn't change components at any link
    //    — the kernel's `t_parent_child^T · F` is identity for
    //    inertial-frame inputs (since the kernel is frame-agnostic
    //    and we're reusing the structural-frame walk in a different
    //    common frame). The cross-term `r × F` uses `r = pcm_to_ccm`
    //    (in parent's structural frame); for chains where every
    //    link's struct = inertial (identity attach rotations and
    //    identity attitudes) this is bit-exact. For rotated chains
    //    the cross-term carries the same first-order error as the
    //    `t_parent_child` rotation — see the module doc for the
    //    follow-up that closes this gap.
    //
    //    To make the kernel's per-link rotation a no-op for force
    //    (free-vector preservation), we pass `t_parent_child =
    //    identity` to the kernel and rely on the `pcm_to_ccm`
    //    geometry alone for the parallel-axis arm. The kernel's
    //    docstring documents the identity-rotation case as the
    //    canonical "shift only" mode.
    let mut wrenches: HashMap<Entity, Wrench> = HashMap::new();
    for (entity, total) in totals_q.iter() {
        if !view.contains(entity) {
            continue;
        }
        let force_inertial = total.0.force.raw_si();
        let torque_body = total.0.torque.raw_si();
        // Body → inertial via T_inertial_body. When no
        // RotationalStateC is present (typical kinematic child),
        // T_inertial_body = identity — body-frame components are
        // taken as inertial components. See the module doc for
        // when this is correct.
        let t_inertial_body = rot_q.get(entity).map_or(DMat3::IDENTITY, |r| {
            r.0.q_inertial_body
                .as_witness()
                .left_quat_to_transformation()
        });
        // T_inertial_body takes inertial → body, so its transpose
        // takes body → inertial (vector components).
        let torque_inertial = t_inertial_body.transpose() * torque_body;
        wrenches.insert(entity, Wrench::new(force_inertial, torque_inertial));
    }

    // Override per-edge `t_parent_child` to identity so the kernel's
    // rotation step is a no-op. We're working in inertial frame for
    // force / torque components, so per-link rotation should be
    // identity; the parallel-axis arm via `pcm_to_ccm` is the only
    // geometric effect we keep. Constructing a fresh edge map with
    // identity rotation makes the intent explicit at the call site.
    let inertial_edges: HashMap<Entity, EdgeGeometry> = edges
        .iter()
        .map(|(e, g)| {
            (
                *e,
                EdgeGeometry {
                    pcm_to_ccm: g.pcm_to_ccm,
                    t_parent_child: DMat3::IDENTITY,
                },
            )
        })
        .collect();

    // 4. Aggregate up every chain. Returns a `HashMap<root, Wrench>`
    //    in the *root's* inertial frame for force, inertial frame for
    //    torque (still about the root's composite CoM).
    let aggregated: HashMap<Entity, Wrench> =
        aggregate_wrenches_via_storage(&view, &wrenches, &inertial_edges);

    // 5. Identify roots once for the writeback pass.
    let roots: HashSet<Entity> = view.iter_roots().collect();

    // 6. Write `TotalForceC` per entity:
    //    - Roots: aggregated inertial force / inertial-rotated-to-body torque.
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
            // Inertial → body for the root: T_inertial_body · tau_inertial.
            let t_inertial_body = rot_q.get(entity).map_or(DMat3::IDENTITY, |r| {
                r.0.q_inertial_body
                    .as_witness()
                    .left_quat_to_transformation()
            });
            let torque_body = t_inertial_body * agg.torque;
            // allowed: wrench-aggregation kernel boundary; the
            // aggregated `agg.force` lives in inertial-frame
            // `DVec3` storage from the kernel walk and the typed
            // wrap is the canonical re-entry into the typed surface
            // (mirrors `force_collection_system`'s boundary write).
            tf.0.force = jeod_sim::Force::<jeod_sim::RootInertial>::from_raw_si(agg.force);
            // allowed: same wrench-kernel boundary; `torque_body`
            // is the inertial→body rotation of the kernel's
            // inertial-frame torque sum, in raw `DVec3`.
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

    // 7. Recompute `FrameDerivativesC` for the root from the new
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

    #[test]
    fn child_with_attach_offset_and_rotation_preserves_inertial_force() {
        // Child attached to parent with a non-identity `t_parent_child`
        // (90° about +Z attach), at offset (1,0,0). Apply an
        // **inertial-frame** external force on the child of (5,0,0).
        // Force is a free vector — its inertial components are
        // preserved through the aggregation walk regardless of attach
        // rotation. The cross-term uses `pcm_to_ccm` (in parent
        // struct) and the inertial-frame force; with identity root
        // attitude the parent struct = inertial.
        //
        // Setup: parent mass 10 at origin; child mass 5 attached at
        //   offset (1,0,0) with attach rotation R_z(90°) (parent +x
        //   maps to child +y in the JEOD `T_parent_this` convention).
        // Composite CoM in parent struct: child.center_of_mass=(0,0,0)
        //   so child_pos_in_parent_struct = t_pc^T · 0 + (1,0,0)
        //   = (1,0,0). Composite CoM = (10·0 + 5·1)/15 = (1/3,0,0).
        //   pcm_to_ccm = (1,0,0) − (1/3,0,0) = (2/3, 0, 0).
        // r × F = (2/3,0,0) × (5,0,0) = (0,0,0) (parallel).
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

        let root_tf = app
            .world()
            .get::<TotalForceC>(parent)
            .unwrap()
            .0
            .to_untyped();
        let f_err = (root_tf.force - DVec3::new(5.0, 0.0, 0.0)).length();
        assert!(
            f_err < 1e-12,
            "root inertial force {:?} should be (5,0,0) (free vector preserved)",
            root_tf.force
        );
        // Force parallel to r ⇒ zero cross-term.
        let t_err = (root_tf.torque - DVec3::ZERO).length();
        assert!(t_err < 1e-12, "root torque {:?}", root_tf.torque);
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
}
