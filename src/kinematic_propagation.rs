//! Bevy system for kinematic state propagation root → leaves.
//!
//! For every kinematic child (an entity carrying [`KinematicChildC`]
//! and [`MassChildOf`]) this system derives the child's instantaneous
//! [`RotationalStateC`] and [`TranslationalStateC`] from its parent's
//! state composed with the link's attach geometry, mirroring JEOD's
//! `DynBody::propagate_state_from_structure` (see
//! [`models/dynamics/dyn_body/src/dyn_body_propagate_state.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_propagate_state.cc)).
//!
//! # Per the three-layer rule
//!
//! - The pure math kernel
//!   ([`jeod_sim::compute_kinematic_child_state`]) lives in
//!   `jeod_dynamics`.
//! - The orchestration walk
//!   ([`jeod_sim::propagate_state_via_storage`]) lives in `jeod_sim`.
//! - This module is the thin Bevy glue: it builds the
//!   [`MassTreeView`], assembles the per-node [`KinematicNodeState`]
//!   and per-edge [`KinematicEdge`] from `MassChildOf` + live
//!   [`MassPropertiesC`] + [`StructuralTransformC`], runs the kernel,
//!   and writes the per-child outputs back into [`RotationalStateC`]
//!   and [`TranslationalStateC`].
//!
//! # Schedule
//!
//! The plugin schedules **two** kinematic-propagation passes per tick,
//! mirroring the runner's stage 3b / 8d sweeps in
//! `crates/jeod_runner/src/simulation/step/mod.rs`:
//!
//! - Pre-integration (this fn) — runs in
//!   [`JeodSet::ForceCollection`](crate::JeodSet::ForceCollection),
//!   **after**
//!   [`composite_mass_system`](crate::mass_tree::composite_mass_system)
//!   (which writes the live composite CoM into each entity's
//!   [`MassPropertiesC`]) and **before**
//!   [`wrench_aggregation_system`](crate::wrench::wrench_aggregation_system)
//!   (so the upward wrench walk reads the fresh per-chain attitudes
//!   the propagation has just installed). The wrench-aggregation guard
//!   that requires `RotationalStateC` on every member of a rotated
//!   chain becomes pure defense-in-depth once this system runs in
//!   front of it: every kinematic child gets its `RotationalStateC`
//!   written before the guard checks for it.
//! - Post-integration ([`propagate_state_from_root_post_integration_system`])
//!   — runs in [`JeodSet::Integration`](crate::JeodSet::Integration),
//!   after `frame_switch_system` and after the post-integration
//!   frame-attached propagation, and before
//!   [`JeodSet::DerivedState`](crate::JeodSet::DerivedState). Without
//!   this pass, kinematic descendants of a freshly-integrated (or
//!   freshly frame-attached) root would lag by one tick — derived
//!   states (`orbital_elements_system`, `geodetic_system`, …) would
//!   observe the *previous* tick's parent state composed with the
//!   link.
//!
//! # Roots are read-only
//!
//! Root bodies (entities without [`MassChildOf`]) integrate their own
//! state via the integration system; this system never writes their
//! [`RotationalStateC`] / [`TranslationalStateC`]. Only entities
//! marked [`KinematicChildC`] receive derived writes, matching the
//! gating contract that
//! [`wrench_aggregation_system`](crate::wrench::wrench_aggregation_system)
//! installs.
//!
//! # Composite-CoM offset semantics
//!
//! The kernel needs each node's composite CoM expressed in **the
//! node's own structural frame** (`mass.composite_properties.position`
//! in JEOD parlance). After [`composite_mass_system`](crate::mass_tree::composite_mass_system) has run, each
//! entity's [`MassPropertiesC`] *is* its composite, and
//! `MassPropertiesC.center_of_mass` is exactly that quantity. We read
//! it directly here without re-running the composition kernel.

use bevy::ecs::system::ParamSet;
use bevy::prelude::*;
use glam::DVec3;
use std::collections::HashMap;

use jeod_sim::{
    propagate_state_via_storage, KinematicEdge, KinematicNodeState, MassStorage,
    RotationalStateTyped, SelfRef, TranslationalStateTyped,
};

use crate::components::{
    KinematicChildC, MassChildOf, MassPropertiesC, RotationalStateC, StructuralTransformC,
    TranslationalStateC,
};
use crate::mass_tree::MassTreeView;

/// Walk the [`MassChildOf`] mass tree pre-order from each root and
/// derive every [`KinematicChildC`]-marked entity's
/// [`RotationalStateC`] / [`TranslationalStateC`] from its parent's
/// state composed with the link's `MassChildOf` rotation + offset.
///
/// Roots and entities without [`MassChildOf`] are untouched — they
/// integrate under their own dynamics and this system must not
/// stomp the integrator's output.
///
/// Fast-paths to a no-op when no entity carries [`MassChildOf`] (no
/// chains in the world means nothing to propagate). The fast-path
/// check uses `parents_q.is_empty()` so the cost is one query
/// iteration over the empty set.
///
/// # Order in `JeodSet::ForceCollection`
///
/// Schedule this system **after**
/// [`composite_mass_system`](crate::mass_tree::composite_mass_system)
/// and **before**
/// [`wrench_aggregation_system`](crate::wrench::wrench_aggregation_system).
/// The plugin wires this ordering automatically; tests that compose
/// a custom subset of systems must replicate it.
///
/// # Panics
///
/// Panics with a "Fail Loudly" diagnostic when the kernel detects a
/// non-orthonormal composed rotation (degenerate
/// `MassChildOf.t_parent_child` upstream), or when the
/// orchestration walk finds an orphaned subtree
/// (a `MassChildOf` chain that doesn't terminate at a root in the
/// world). Both cases indicate a misconfiguration in mission code,
/// not a runtime physics divergence.
// JEOD_INV: DB.13 — kinematic state propagation routed through structural frames (parent → struct → link → child struct → child body)
// JEOD_INV: DB.17 — only the root integrates; non-root state is derived each step
#[allow(clippy::type_complexity)]
pub fn propagate_state_from_root_system(
    mass_q: Query<(Entity, &MassPropertiesC)>,
    parents_q: Query<(Entity, &MassChildOf)>,
    kinematic_q: Query<Entity, With<KinematicChildC>>,
    names_q: Query<&Name>,
    struct_q: Query<&StructuralTransformC>,
    // The rotational / translational state is read across the whole
    // mass tree (every node feeds into the per-tick `nodes` map) and
    // mutably written through the marker-gated entry of the same
    // ParamSet. ParamSet serializes the borrows so a single system
    // can hold both an unfiltered read view and a marker-gated write
    // view of the same components — without it Bevy's borrow checker
    // would refuse the conflicting access (`B0001`).
    mut state_qs: ParamSet<(
        Query<(&RotationalStateC, &TranslationalStateC)>,
        Query<(&mut RotationalStateC, &mut TranslationalStateC), With<KinematicChildC>>,
    )>,
) {
    // 1. Fast path: no MassChildOf edges → nothing to propagate.
    if parents_q.is_empty() {
        return;
    }

    // 2. Build the topological view (same shape as
    //    composite_mass_system / wrench_aggregation_system).
    let view = MassTreeView::from_queries(&mass_q, &parents_q, &names_q);
    if view.is_empty() {
        return;
    }

    // 3. Materialize the per-node state map. Every entity in the
    //    storage must have an entry — including non-root children
    //    whose `rot` / `trans` are about to be overwritten, because
    //    the kernel still reads their `t_struct_body` and
    //    `composite_in_struct` fields when routing through the
    //    child's structural frame.
    //
    //    Roots' RotationalStateC / TranslationalStateC seed the
    //    walk; non-roots' values may be stale this tick (they were
    //    last written by the previous tick's propagation system, or
    //    by the body-init bundle on first tick) — the walk will
    //    overwrite them.
    let mut nodes: HashMap<Entity, KinematicNodeState> = HashMap::with_capacity(view.len());
    {
        let read_q = state_qs.p0();
        for entity in view.iter_entities() {
            let (rot_untyped, trans_untyped) = match read_q.get(entity) {
                Ok((r, t)) => (r.0.to_untyped(), t.0.to_untyped()),
                Err(_) => Default::default(),
            };
            let t_struct_body = struct_q
                .get(entity)
                .map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
            // After composite_mass_system has run, MassPropertiesC.center_of_mass
            // is the *composite* CoM in this body's structural frame —
            // exactly what the kernel calls `composite_in_struct`. Single-
            // body vehicles with no children have composite == core, so
            // this is also the core CoM in those cases (matching JEOD's
            // `composite_properties.position` for atomic bodies).
            let composite_in_struct = mass_q
                .get(entity)
                .map(|(_, m)| m.0.center_of_mass.raw_si())
                .unwrap_or(DVec3::ZERO);
            nodes.insert(
                entity,
                KinematicNodeState {
                    rot: rot_untyped,
                    trans: trans_untyped,
                    t_struct_body,
                    composite_in_struct,
                },
            );
        }
    }

    // 4. Build the per-edge KinematicEdge map directly from
    //    MassChildOf — same shape as wrench_aggregation_system's
    //    edge map but with the parent-CoM term removed (the kinematic
    //    walk computes pcm_to_ccm internally; this map only needs
    //    the raw link rotation + offset).
    let mut edges: HashMap<Entity, KinematicEdge> = HashMap::with_capacity(view.len());
    for (child, link) in parents_q.iter() {
        edges.insert(
            child,
            KinematicEdge {
                t_parent_child: link.t_parent_child,
                offset_in_pstr: link.offset,
            },
        );
    }

    // 5. Run the orchestration walk. Pre-order, root → leaves; every
    //    non-root child's outputs are derived from the just-written
    //    parent state.
    let derived = propagate_state_via_storage(&view, &nodes, &edges);

    // 6. Write back into every entity tagged KinematicChildC. Roots
    //    and standalone bodies are untouched — they integrate under
    //    their own dynamics, and overwriting their state would stomp
    //    the integrator output.
    //
    //    `KinematicChildC` is owned by `wrench_aggregation_system`
    //    (it inserts the marker on every non-root mass-tree node and
    //    removes it on demotion to a root). When the marker is
    //    out-of-sync this tick (e.g. an entity newly attached but
    //    the wrench system hasn't run yet to insert
    //    `KinematicChildC`), we fall back to writing every non-root
    //    in the mass tree. That's safe because:
    //      - On the first tick after attach, the wrench system runs
    //        immediately after this one and will insert the marker;
    //        the next tick's propagation will write the same value
    //        through the marker-gated query.
    //      - Skipping the write here would leave the new child's
    //        state at whatever the integrator most recently wrote
    //        (typically pre-attach garbage) until the wrench system
    //        catches up — silent staleness, the failure mode this
    //        whole system is meant to fix.
    //    So we use both paths: (a) marker-gated writeback for the
    //    steady-state case (most ticks), and (b) a fallback for any
    //    non-root that's missing the marker but is in the mass tree.
    //
    //    JEOD_INV: DB.17 — only the root's RotationalState/TranslationalState
    //    is integrated; every other chain member is derived here.
    let kinematic_set: std::collections::HashSet<Entity> = kinematic_q.iter().collect();
    let mut writeback_q = state_qs.p1();
    for (entity, state) in &derived {
        // Roots are seeded from the integrator's output and re-written
        // verbatim; skip them here to avoid a redundant write that
        // would advance change-detection ticks for no reason.
        if MassStorage::parent(&view, *entity).is_none() {
            continue;
        }
        // Marker-gated writeback covers the steady-state case where
        // `wrench_aggregation_system` has already tagged the entity.
        // First-tick-after-attach (marker not yet installed) is
        // handled by the next tick's propagation: `wrench_aggregation_system`
        // runs immediately after this system in
        // `JeodSet::ForceCollection`, observes the `MassChildOf`
        // edge, and inserts `KinematicChildC` via Commands. The
        // marker becomes visible on the *next* schedule run, at
        // which point the steady-state writeback covers the entity.
        // The single-frame staleness window is bounded by one tick.
        if kinematic_set.contains(entity) {
            if let Ok((mut rot_c, mut trans_c)) = writeback_q.get_mut(*entity) {
                // allowed: kinematic-propagation kernel boundary —
                // `state.rot` / `state.trans` arrive as raw
                // `RotationalState` / `TranslationalState` from the
                // kernel walk in `jeod_sim::propagate_state_via_storage`,
                // and re-wrapping them as `RotationalStateTyped<SelfRef>` /
                // `TranslationalStateTyped<PlanetInertial<SelfPlanet>>`
                // is the canonical re-entry into the typed surface
                // (mirrors `wrench_aggregation_system`'s root-exit
                // boundary writes through `from_raw_si`). The
                // translational tag matches `TranslationalStateC`'s
                // wildcard-`<PlanetInertial<SelfPlanet>>` storage post
                // #263, not `<RootInertial>`: the body lives in its
                // integration frame, and the `<RootInertial>` lift is
                // applied at *shift sites* via `to_inertial(&origin)`
                // — never silently here.
                rot_c.0 = RotationalStateTyped::<SelfRef>::from_untyped_unchecked(&state.rot);
                trans_c.0 =
                    // allowed: kernel boundary (see rotational sibling write above for the full rationale).
                    TranslationalStateTyped::<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>::from_untyped_unchecked(
                        &state.trans,
                    );
            }
        }
    }
}

/// Post-integration twin of [`propagate_state_from_root_system`].
///
/// Re-runs the same root → leaves kinematic walk after
/// `integration_system` + `sync_body_to_frame_system` +
/// `frame_switch_system` have landed, so a kinematic child whose root
/// was just integrated reflects the same-tick parent state rather than
/// the previous tick's. Mirrors stage 8d of the runner's
/// `Simulation::step_internal` (see
/// `crates/jeod_runner/src/simulation/step/mod.rs`), which calls
/// `propagate_kinematic_state` both before and after integration.
///
/// Without this pass the kinematic descendants of a freshly-integrated
/// (or freshly frame-attached) root would lag by one tick — the
/// pre-integration sweep observed the *previous* tick's integrated
/// root. Derived-state consumers in
/// [`JeodSet::DerivedState`](crate::JeodSet::DerivedState) would then
/// observe stale child state.
///
/// Distinct fn from the pre-integration sibling so Bevy's
/// `SystemTypeSet` treats the two registrations as independent system
/// instances; the body delegates to the same logic to keep the two
/// passes byte-for-byte equivalent.
// JEOD_INV: DB.13 — kinematic state propagation routed through structural frames
// JEOD_INV: DB.17 — only the root integrates; non-root state is derived each step,
//   and the post-integration sweep is what makes "each step" mean "after the step
//   actually finished" rather than "before the step started".
#[allow(clippy::type_complexity)]
pub fn propagate_state_from_root_post_integration_system(
    mass_q: Query<(Entity, &MassPropertiesC)>,
    parents_q: Query<(Entity, &MassChildOf)>,
    kinematic_q: Query<Entity, With<KinematicChildC>>,
    names_q: Query<&Name>,
    struct_q: Query<&StructuralTransformC>,
    state_qs: ParamSet<(
        Query<(&RotationalStateC, &TranslationalStateC)>,
        Query<(&mut RotationalStateC, &mut TranslationalStateC), With<KinematicChildC>>,
    )>,
) {
    propagate_state_from_root_system(mass_q, parents_q, kinematic_q, names_q, struct_q, state_qs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{
        DynamicsConfigC, ExternalForceC, ExternalTorqueC, FrameDerivativesC, TotalForceC,
    };
    use crate::mass_tree::composite_mass_system;
    use crate::wrench::wrench_aggregation_system;
    use glam::{DMat3, DVec3};
    use jeod_sim::{MassProperties, RotationalState};

    fn add_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app
    }

    /// Run the realistic per-tick sequence: composite recomputation,
    /// then kinematic propagation, then wrench aggregation. Mirrors
    /// the plugin's schedule (without dragging in `FixedUpdate` /
    /// integration) so tests can observe the propagated state and
    /// the wrench system's defense-in-depth guard side by side.
    fn run_pipeline(app: &mut App) {
        app.add_systems(
            Update,
            (
                composite_mass_system,
                propagate_state_from_root_system.after(composite_mass_system),
                wrench_aggregation_system.after(propagate_state_from_root_system),
            ),
        );
        app.update();
    }

    fn jeod_rot_z(angle: f64) -> DMat3 {
        jeod_sim::JeodQuat::left_quat_from_eigen_rotation(angle, DVec3::Z)
            .left_quat_to_transformation()
    }

    /// Two-body chain (root + 1 kinematic child) with non-identity
    /// attach rotation and a non-trivial parent attitude/omega.
    /// The child's RotationalStateC and TranslationalStateC must
    /// equal the analytical answer of "parent state composed with
    /// link" after one App::update().
    #[test]
    fn rotated_attach_child_tracks_parent_through_link() {
        let mut app = add_test_app();

        // Parent attitude: 60° about Z. Parent ang_vel along Z at
        // 0.001 rad/s.
        let parent_q = jeod_sim::JeodQuat::left_quat_from_eigen_rotation(
            std::f64::consts::FRAC_PI_3,
            DVec3::Z,
        );
        let parent_omega = DVec3::new(0.0, 0.0, 1e-3);
        let parent_pos = DVec3::new(7e6, 0.0, 0.0);
        let parent_vel = DVec3::new(0.0, 7500.0, 0.0);

        let parent_rot = RotationalState {
            quaternion: parent_q,
            ang_vel_body: parent_omega,
        };
        let parent_trans = jeod_sim::TranslationalState {
            position: parent_pos,
            velocity: parent_vel,
        };

        let parent = app
            .world_mut()
            .spawn((
                Name::new("rotated_parent"),
                MassPropertiesC::from(MassProperties::new(10.0)),
                RotationalStateC::from_untyped(parent_rot),
                TranslationalStateC::from_untyped(parent_trans),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // Link: 30° about Z, offset (1, 0, 0).
        let t_pc = jeod_rot_z(std::f64::consts::PI / 6.0);
        let offset = DVec3::new(1.0, 0.0, 0.0);
        let child = app
            .world_mut()
            .spawn((
                Name::new("kinematic_child"),
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::with_rotation(parent, offset, t_pc),
                // Stale state — propagation must overwrite both.
                RotationalStateC::default(),
                TranslationalStateC::default(),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // First tick: wrench_aggregation_system inserts
        // KinematicChildC; propagation then writes the child's state
        // on the second tick. Run twice so we observe the steady-
        // state behaviour.
        run_pipeline(&mut app);
        app.update();

        let child_rot = app.world().get::<RotationalStateC>(child).unwrap();
        let child_trans = app.world().get::<TranslationalStateC>(child).unwrap();

        // Expected: T_inertial_body_child = T_pc · T_inertial_body_parent
        let parent_t_ib = parent_q.left_quat_to_transformation();
        let expected_child_t = t_pc * parent_t_ib;
        let actual_child_t = child_rot
            .0
            .q_inertial_body
            .as_witness()
            .left_quat_to_transformation();
        let mat_diff = (actual_child_t.x_axis - expected_child_t.x_axis).length()
            + (actual_child_t.y_axis - expected_child_t.y_axis).length()
            + (actual_child_t.z_axis - expected_child_t.z_axis).length();
        assert!(
            mat_diff < 1e-10,
            "child T_inertial_body mismatch: expected {expected_child_t:?}, got {actual_child_t:?}"
        );

        // Expected position: parent_composite_pos + r_offset_inertial,
        // where r_offset_inertial = T_inertial_struct_parent^T · pcm_to_ccm
        // and pcm_to_ccm = link_offset + 0 (child composite CoM at child
        // struct origin) − parent_composite_in_pstr.
        // Parent composite CoM (mass 10 + child mass 5 at offset 1,0,0)
        // = (10·0 + 5·1)/15 = 1/3 along x in parent struct.
        let parent_composite_in_pstr = DVec3::new(1.0 / 3.0, 0.0, 0.0);
        let pcm_to_ccm = offset - parent_composite_in_pstr;
        // Parent has identity struct→body ⇒ T_inertial_struct = T_inertial_body.
        let expected_offset_inertial = parent_t_ib.transpose() * pcm_to_ccm;
        let expected_child_pos = parent_pos + expected_offset_inertial;
        let pos_err = (child_trans.0.position.raw_si() - expected_child_pos).length();
        assert!(
            pos_err < 1e-9,
            "child position mismatch: expected {expected_child_pos:?}, got {:?}",
            child_trans.0.position.raw_si()
        );

        // Expected velocity: parent_vel + omega_inertial × r_offset_inertial.
        // omega_inertial = parent_t_ib^T · parent_omega.
        let omega_inertial = parent_t_ib.transpose() * parent_omega;
        let expected_child_vel = parent_vel + omega_inertial.cross(expected_offset_inertial);
        let vel_err = (child_trans.0.velocity.raw_si() - expected_child_vel).length();
        assert!(
            vel_err < 1e-9,
            "child velocity mismatch: expected {expected_child_vel:?}, got {:?}",
            child_trans.0.velocity.raw_si()
        );
    }

    /// Three-body kinematic chain root → mid → leaf, every link
    /// rotated 30° about Z, every body co-located at its struct
    /// origin (zero offsets) so the composite-CoM math collapses
    /// trivially: every body's composite CoM in its own struct
    /// frame is the origin.
    ///
    /// Pins two contracts:
    /// - The leaf's attitude accumulates both link rotations:
    ///   `T_inertial_body_leaf = T_pc · T_pc · T_inertial_body_root`.
    /// - Without any structural offset, the leaf's inertial position
    ///   stays at the root's position — the rotation chain doesn't
    ///   add positional drift when there's no arm.
    #[test]
    fn three_body_chain_propagates_through_middle_to_leaf() {
        let mut app = add_test_app();

        let parent_pos = DVec3::new(7e6, 0.0, 0.0);
        let parent_vel = DVec3::ZERO;
        let parent_trans = jeod_sim::TranslationalState {
            position: parent_pos,
            velocity: parent_vel,
        };

        let root = app
            .world_mut()
            .spawn((
                Name::new("root"),
                MassPropertiesC::from(MassProperties::new(10.0)),
                RotationalStateC::default(),
                TranslationalStateC::from_untyped(parent_trans),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        let t_pc = jeod_rot_z(std::f64::consts::PI / 6.0);

        // Co-located attaches (zero offset): every body sits on its
        // parent's struct origin, so the composite-CoM math at every
        // level lands on the origin and the only thing the
        // propagation system must compose is the rotation chain.
        let zero = DVec3::ZERO;

        let mid = app
            .world_mut()
            .spawn((
                Name::new("mid"),
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::with_rotation(root, zero, t_pc),
                RotationalStateC::default(),
                TranslationalStateC::default(),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        let leaf = app
            .world_mut()
            .spawn((
                Name::new("leaf"),
                MassPropertiesC::from(MassProperties::new(2.0)),
                MassChildOf::with_rotation(mid, zero, t_pc),
                RotationalStateC::default(),
                TranslationalStateC::default(),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // Run two ticks: the first installs KinematicChildC markers,
        // the second propagates and writes the steady-state outputs.
        run_pipeline(&mut app);
        app.update();

        let leaf_rot = app.world().get::<RotationalStateC>(leaf).unwrap();
        let leaf_trans = app.world().get::<TranslationalStateC>(leaf).unwrap();

        // T_leaf = T_pc · T_pc (60° about Z).
        let expected_leaf_t = t_pc * t_pc;
        let actual_leaf_t = leaf_rot
            .0
            .q_inertial_body
            .as_witness()
            .left_quat_to_transformation();
        let mat_diff = (actual_leaf_t.x_axis - expected_leaf_t.x_axis).length()
            + (actual_leaf_t.y_axis - expected_leaf_t.y_axis).length()
            + (actual_leaf_t.z_axis - expected_leaf_t.z_axis).length();
        assert!(
            mat_diff < 1e-10,
            "leaf T_inertial_body mismatch: expected {expected_leaf_t:?}, got {actual_leaf_t:?}"
        );

        // Zero offsets at every link + zero offsets between core and
        // composite CoMs (each body's composite CoM is at its struct
        // origin) ⇒ leaf composite-body position equals root
        // composite-body position.
        let pos_err = (leaf_trans.0.position.raw_si() - parent_pos).length();
        assert!(
            pos_err < 1e-9,
            "leaf position should equal root position with zero offsets: \
             expected {parent_pos:?}, got {:?}",
            leaf_trans.0.position.raw_si()
        );

        // Mid is between them — same expectation.
        let mid_pos = app
            .world()
            .get::<TranslationalStateC>(mid)
            .unwrap()
            .0
            .position
            .raw_si();
        let mid_pos_err = (mid_pos - parent_pos).length();
        assert!(
            mid_pos_err < 1e-9,
            "mid position should equal root position with zero offsets: got {mid_pos:?}"
        );
    }

    /// Failure-mode regression: the wrench-build path's fail-loud
    /// guard rejects a child with a rotated `t_parent_child` link
    /// when the child lacks a `RotationalStateC`. With
    /// `propagate_state_from_root_system` running before the wrench
    /// build, the child's attitude is filled in (derived from
    /// parent attitude composed with the attach rotation), so the
    /// guard sees a valid rotation and the chain succeeds.
    ///
    /// Mirrors the negative test
    /// `child_with_attach_rotation_and_no_rotational_state_panics`
    /// in `src/wrench.rs` — exact same chain shape, *but* the child
    /// now carries a default `RotationalStateC` (so propagation
    /// has somewhere to write the derived attitude) and the
    /// expected outcome is success, not panic. The propagation
    /// system's job is to ensure the wrench guard never fires in
    /// supported configurations; this test pins that contract.
    ///
    /// Without `propagate_state_from_root_system`, the child's
    /// stale (default-identity) RotationalStateC would be wrong
    /// (child should rotate with its parent + the attach link)
    /// and downstream consumers would silently propagate the wrong
    /// attitude. With the propagation system, the child's attitude
    /// is overwritten each step to `t_parent_child · parent_T` —
    /// the JEOD-faithful answer.
    #[test]
    fn rotated_chain_with_propagation_succeeds() {
        let mut app = add_test_app();

        let parent_q = jeod_sim::JeodQuat::left_quat_from_eigen_rotation(
            std::f64::consts::FRAC_PI_4,
            DVec3::Z,
        );
        let parent_rot = RotationalState {
            quaternion: parent_q,
            ang_vel_body: DVec3::ZERO,
        };

        let parent = app
            .world_mut()
            .spawn((
                Name::new("rotated_parent"),
                MassPropertiesC::from(MassProperties::new(10.0)),
                RotationalStateC::from_untyped(parent_rot),
                TranslationalStateC::default(),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // 90° about Z: same shape as the negative test in src/wrench.rs.
        let t_pc = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        // Child carries a default RotationalStateC (the propagation
        // system writes through it). Without propagation the wrench
        // guard would still panic the moment it observed a rotated
        // chain with mismatched per-member attitudes; with
        // propagation, by the time the wrench guard runs the child
        // attitude has been overwritten to the correct composed
        // value and the chain is consistent.
        let child = app
            .world_mut()
            .spawn((
                Name::new("kinematic_child"),
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::with_rotation(parent, DVec3::new(1.0, 0.0, 0.0), t_pc),
                RotationalStateC::default(),
                TranslationalStateC::default(),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // Two ticks for steady-state.
        run_pipeline(&mut app);
        app.update();

        // Child attitude must equal t_pc · parent_T.
        let parent_t_ib = parent_q.left_quat_to_transformation();
        let expected = t_pc * parent_t_ib;
        let child_t = app
            .world()
            .get::<RotationalStateC>(child)
            .unwrap()
            .0
            .q_inertial_body
            .as_witness()
            .left_quat_to_transformation();
        let diff = (child_t.x_axis - expected.x_axis).length()
            + (child_t.y_axis - expected.y_axis).length()
            + (child_t.z_axis - expected.z_axis).length();
        assert!(
            diff < 1e-10,
            "propagated child attitude must equal t_pc · parent_T; got diff {diff}"
        );
    }
}
