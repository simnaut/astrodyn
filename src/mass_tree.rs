//! Bevy-side ECS mass-tree adapter.
//!
//! This module is the Bevy half of issue [#271]: it exposes the
//! [`MassChildOf`] /
//! [`MassPointRef`](crate::MassPointRef) relations through the
//! [`jeod_sim::MassStorage`] trait so the same composition kernel
//! (parallel-axis / Steiner) drives both the runner's arena
//! [`jeod_sim::MassTree`] and the Bevy ECS world.
//!
//! Per the Three-Layer Architecture rule the *physics* lives in
//! `jeod_dynamics::mass_storage` and is re-exported via `jeod_sim`;
//! this module is the thin ECS-glue boundary — it adapts `Query`
//! handles to the trait surface and runs a per-step
//! [`composite_mass_system`] that walks `MassChildOf` bottom-up,
//! recomputes composites, and writes them back into
//! [`MassPropertiesC`].
//!
//! The previous arena-via-resource path —
//! [`MassTreeR`](crate::MassTreeR) plus
//! [`MassBodyIdC`](crate::MassBodyIdC) plus
//! [`AttachEvent`](crate::AttachEvent) /
//! [`DetachEvent`](crate::DetachEvent) — is preserved alongside this
//! module as a compatibility surface for in-flight mission code that
//! depends on it; new mission code should prefer the ECS-native
//! `MassChildOf` path.
//!
//! Out of scope for this PR (each is a separate sub-issue under
//! [#270] meta):
//!
//! - **#272** composite-rigid-body propagation + wrench aggregation,
//! - **#273** momentum conservation across attach (port of
//!   `combine_states_at_attach`),
//! - **#274** GJ / ABM4 integrator state reset on attach.
//!
//! [#270]: https://github.com/simnaut/bevy_jeod/issues/270
//! [#271]: https://github.com/simnaut/bevy_jeod/issues/271

use std::collections::HashMap;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use jeod_sim::{
    recompute_composites_via_storage, MassNodeView, MassPointState, MassProperties, MassStorage,
};

use crate::components::{MassChildOf, MassPropertiesC};

/// Internal cache: snapshot of an entity's *core* mass properties.
///
/// `composite_mass_system` writes the post-composition (composite)
/// values back into [`MassPropertiesC`], which then double-serves as
/// the input on the next tick. Without a separate snapshot of the
/// *core* (pre-composition) values, the next tick's kernel would
/// read stale composite-as-core, producing wrong results after the
/// topology changes (most visibly: a parent that just lost a child
/// would still be carrying the heavier composite as its "core" and
/// re-attaching a fresh child would compound the mass error).
///
/// Mirrors JEOD's separation between `MassBody::core_properties` and
/// `MassBody::composite_properties` — the arena keeps both as
/// distinct fields; the ECS keeps the composite in
/// [`MassPropertiesC`] (the field every existing consumer reads) and
/// stashes the core here.
///
/// **Cache freshness.** The cache is *re-seeded* every tick from any
/// `MassPropertiesC` that has been externally changed since the last
/// run — the system uses Bevy's `Changed<MassPropertiesC>` filter to
/// detect mission-code edits (fuel burn, staging, inertia overrides)
/// and refreshes the cache against the new core. The system's own
/// composite write-back uses [`bevy::ecs::change_detection::DetectChangesMut::bypass_change_detection`]
/// so it does not falsely re-trigger as an external change next
/// tick. The original write-once seed (PR #283 round 1) silently
/// dropped mid-sim mass edits; this revision matches the
/// `mass_update_system` contract that runtime mass changes are
/// reflected on the next step.
///
/// **Internal cache — mission code MUST NOT read or write this.**
/// [`composite_mass_system`] manages it. The struct is publicly
/// visible only because Bevy's system-param signatures require the
/// filter / data types in `Without<…>` clauses to be `pub` when the
/// system itself is `pub`; the type is hidden from rustdoc to keep
/// it off the public surface.
#[doc(hidden)] // allowed: pub-but-hidden; see doc comment above and #271
#[derive(Component, Debug, Clone, Copy)]
pub struct CoreMassPropertiesC(pub MassProperties);

/// Read view into Bevy's mass-tree relations, suitable for driving
/// the storage-agnostic composition kernel.
///
/// Built from a `Query<&MassChildOf>` (parent links) and a
/// `Query<(Entity, &MassPropertiesC)>` (per-entity core mass / inertia
/// / CoM). The view materialises children and roots once at
/// construction so the trait impl is `O(1)` per `parent` /
/// `children` / `roots` call inside the kernel walk.
///
/// Intentionally **not** a `SystemParam` itself — composition is
/// driven by the [`composite_mass_system`] (a regular system that
/// owns the queries) so adapter code that wants to invoke the kernel
/// from a `World::run_system` or one-shot context can construct the
/// view directly from queries it already holds.
///
/// The view is fully owned: it eagerly copies core mass properties
/// and entity names out of the queries at construction so it has no
/// lifetime tied to the `Query` borrows.
pub struct MassTreeView {
    parents: Vec<(Entity, Entity)>,
    /// Cached per-entity core view (mass, structure-point, name buffer).
    /// `name` owns the formatted entity-debug string the kernel uses
    /// for diagnostic panic messages.
    nodes: Vec<MassNodeRecord>,
    /// Map: query entity → index into `nodes`.
    index: HashMap<Entity, usize>,
    children_by_parent: HashMap<Entity, Vec<Entity>>,
    roots: Vec<Entity>,
}

struct MassNodeRecord {
    core: jeod_sim::MassProperties,
    structure_point: MassPointState,
    name: String,
}

impl MassTreeView {
    /// Build the view from the two queries that
    /// [`composite_mass_system`] holds.
    ///
    /// `mass_q` is the canonical "every entity that has core mass
    /// properties" set; `parents_q` is the subset that also has a
    /// [`MassChildOf`] back-link. Roots are entities in `mass_q` that
    /// don't appear as children in `parents_q`.
    /// Build the view from the read queries, treating the live
    /// `MassPropertiesC` value as the core. Suitable for one-shot
    /// composition (no prior tick has overwritten the components
    /// with composite values) and for downstream systems that want
    /// to drive [`recompute_composites_via_storage`] with their own
    /// query layout.
    pub fn from_queries<M, P>(
        mass_q: &Query<(Entity, &MassPropertiesC), M>,
        parents_q: &Query<(Entity, &MassChildOf), P>,
        names_q: &Query<&Name>,
    ) -> Self
    where
        M: bevy::ecs::query::QueryFilter,
        P: bevy::ecs::query::QueryFilter,
    {
        // Pre-compute the set of mass-bearing entities so we can
        // fail loudly on a `MassChildOf` whose parent is missing
        // from `mass_q` (PR #283 review thread PRRT_kwDORtae6c5_KBwP).
        let mass_set: HashMap<Entity, ()> = mass_q.iter().map(|(e, _)| (e, ())).collect();

        let mut edge_data: HashMap<Entity, MassChildOf> = HashMap::new();
        let mut parents: Vec<(Entity, Entity)> = Vec::new();
        let mut children_by_parent: HashMap<Entity, Vec<Entity>> = HashMap::new();
        for (child, edge) in parents_q.iter() {
            assert!(
                mass_set.contains_key(&edge.parent),
                "MassChildOf edge {child:?} -> {parent:?}: parent has no MassPropertiesC. \
                 Either add MassPropertiesC to the parent or remove the MassChildOf \
                 component from the child.",
                parent = edge.parent
            );
            edge_data.insert(child, *edge);
            parents.push((child, edge.parent));
            children_by_parent
                .entry(edge.parent)
                .or_default()
                .push(child);
        }

        let mut nodes: Vec<MassNodeRecord> = Vec::new();
        let mut index: HashMap<Entity, usize> = HashMap::new();
        let mut roots: Vec<Entity> = Vec::new();

        for (entity, mass) in mass_q.iter() {
            let untyped = mass.0.to_untyped();
            let structure_point = match edge_data.get(&entity) {
                Some(edge) => MassPointState {
                    position: edge.offset,
                    t_parent_this: edge.t_parent_child,
                },
                None => MassPointState::default(),
            };
            let name = match names_q.get(entity) {
                Ok(n) => n.as_str().to_owned(),
                Err(_) => format!("{entity:?}"),
            };
            let idx = nodes.len();
            nodes.push(MassNodeRecord {
                core: untyped,
                structure_point,
                name,
            });
            index.insert(entity, idx);
            if !edge_data.contains_key(&entity) {
                roots.push(entity);
            }
        }

        Self {
            parents,
            nodes,
            index,
            children_by_parent,
            roots,
        }
    }

    /// Number of registered nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when no entities have [`MassPropertiesC`].
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl MassStorage for MassTreeView {
    type Id = Entity;

    fn parent(&self, id: Self::Id) -> Option<Self::Id> {
        self.parents.iter().find(|(c, _)| *c == id).map(|(_, p)| *p)
    }

    fn node(&self, id: Self::Id) -> MassNodeView<'_> {
        let idx = *self.index.get(&id).unwrap_or_else(|| {
            panic!(
                "MassTreeView::node({id:?}) — entity has no MassPropertiesC. \
                 Add MassPropertiesC before participating in the mass tree."
            )
        });
        let rec = &self.nodes[idx];
        MassNodeView {
            core: rec.core,
            structure_point: rec.structure_point,
            name: rec.name.as_str(),
        }
    }

    fn children(&self, id: Self::Id) -> Vec<Self::Id> {
        self.children_by_parent
            .get(&id)
            .cloned()
            .unwrap_or_default()
    }

    fn roots(&self) -> Vec<Self::Id> {
        self.roots.clone()
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Composite mass system
// ---------------------------------------------------------------------------

/// SystemParam bundling the queries needed to build a [`MassTreeView`].
///
/// Mission code that wants to call the kernel outside of
/// [`composite_mass_system`] (e.g. a one-shot system that runs
/// composition just once after a manual attach) can take this
/// directly. Since `MassTreeView` is built from a read-borrow of
/// `MassPropertiesC` and the system writes back through a separate
/// `&mut` query, callers using both must split via Bevy's `ParamSet`
/// (the canonical pattern, mirroring what
/// [`composite_mass_system`] does internally).
#[derive(SystemParam)]
pub struct MassTreeQueries<'w, 's> {
    /// Per-entity core mass / inertia / CoM (read view).
    pub mass: Query<'w, 's, (Entity, &'static MassPropertiesC)>,
    /// `MassChildOf(parent)` parent links. Only a subset of `mass` —
    /// roots don't carry one.
    pub parents: Query<'w, 's, (Entity, &'static MassChildOf)>,
    /// Optional human-readable entity names for diagnostic messages.
    pub names: Query<'w, 's, &'static Name>,
}

/// Recompute every mass-tree composite from `MassChildOf` parent
/// links and write the results back into [`MassPropertiesC`].
///
/// Walks each tree post-order via the
/// [`jeod_sim::MassStorage`]-driven kernel (see
/// `jeod_dynamics::mass_storage::recompute_composites_via_storage`),
/// applying the parallel-axis / Steiner theorem at every internal
/// node. Atomic / leaf nodes get their composite set equal to their
/// core (matches the arena's `MassTree::recompute_composites`
/// behaviour). Every mass-bearing node — root, internal, atomic
/// leaf — also gets a fresh `inverse_inertia` whenever `mass > 0`;
/// the Bevy pipeline's rotational dynamics, gravity-gradient
/// torque, and SRP / aero torques integrate every
/// `DynamicsConfigC`-bearing entity using its own
/// `MassPropertiesC.inverse_inertia`, not just the integration
/// root, so per-node inversion is mandatory. The root's value is
/// bit-equivalent to JEOD's second invert at
/// `mass_update.cc:116-125`; non-root nodes are the natural
/// extension.
///
/// **Three-layer rule.** This system is pure ECS glue: it queries
/// components, builds a [`MassTreeView`], delegates to the shared
/// `jeod_sim::recompute_composites_via_storage` kernel, and writes
/// the result into `MassPropertiesC`. No composition math runs
/// inside this function.
///
/// **Core / composite separation.** [`MassPropertiesC`] is the
/// composite the rest of the pipeline reads. The system caches the
/// pre-composition core values into a hidden
/// [`CoreMassPropertiesC`] component (managed by this system, never
/// touched by mission code) so a parent that detaches its last
/// child correctly reverts to its own core. Without the cache the
/// previous tick's composite would shadow the original core forever.
/// JEOD makes this distinction explicit via
/// `MassBody::core_properties` vs `MassBody::composite_properties`;
/// the ECS keeps the composite in `MassPropertiesC` so existing
/// consumers (gravity, force collection, integration) need no
/// changes.
#[allow(clippy::type_complexity)]
pub fn composite_mass_system(
    mut commands: Commands,
    parents: Query<(Entity, &MassChildOf)>,
    names: Query<&Name>,
    cores_q: Query<(Entity, &CoreMassPropertiesC)>,
    mut props: ParamSet<(
        // p0: entities whose MassPropertiesC was changed this tick
        //      (or just spawned). Used to refresh CoreMassPropertiesC
        //      so mid-sim mass edits (fuel burn, staging) are picked
        //      up. The system's own composite write-back uses
        //      `bypass_change_detection` so it does not re-trigger
        //      this filter on the next tick.
        Query<(Entity, &'static MassPropertiesC), Changed<MassPropertiesC>>,
        // p1: write-back of the composite results.
        Query<&'static mut MassPropertiesC>,
    )>,
) {
    // Fast path (PR #283 review thread PRRT_kwDORtae6c5_KBwG): if
    // no entity has any `MassChildOf` edge, every body is its own
    // composite — composite == core for each — so we can skip the
    // entire build-view/kernel round-trip and the `O(N²)` HashMap
    // ceremony around it. We still have to:
    //
    //   (a) refresh `CoreMassPropertiesC` for newly-spawned or
    //       externally-edited bodies, so a later attach reads the
    //       live core (this is `O(changed)`);
    //   (b) revert any entity whose `MassPropertiesC` still carries
    //       a stale composite from a *previous* tick — the
    //       just-detached parent case. Without (b) a parent that
    //       loses its last child would carry the heavier composite
    //       forever. This walks `cores_q` once but only writes
    //       through `bypass_change_detection` when the cached core
    //       and the live composite actually differ, so true never-
    //       attached single-body scenarios pay no per-tick write.
    //
    // The full read-write loop on every entity that the previous
    // implementation paid is gone — kernel, view, and `MassChildOf`
    // edge bookkeeping are skipped entirely.
    if parents.is_empty() {
        // (a) freshly-spawned / externally-edited cores.
        {
            let changed = props.p0();
            for (entity, props_ref) in &changed {
                commands
                    .entity(entity)
                    .insert(CoreMassPropertiesC(props_ref.0.to_untyped()));
            }
        }
        // (b) revert stale composites where cache disagrees with
        //     live MassPropertiesC.
        let mut writes = props.p1();
        for (entity, core) in &cores_q {
            if let Ok(mut live) = writes.get_mut(entity) {
                let live_untyped = live.0.to_untyped();
                if live_untyped.mass != core.0.mass
                    || live_untyped.position != core.0.position
                    || live_untyped.inertia != core.0.inertia
                {
                    *live.bypass_change_detection() = MassPropertiesC::from(core.0);
                }
            }
        }
        return;
    }

    // Step 1: build the core lookup. Start from the cached
    // CoreMassPropertiesC snapshots, then *override* with any
    // MassPropertiesC that has been externally changed since last
    // tick — this is what rescues mid-sim mass edits from the
    // previous "write-once" cache. Newly-spawned entities (without
    // a cache yet) match `Changed` on their first sight, so they
    // also land in the override pass.
    let mut cores: HashMap<Entity, MassProperties> = HashMap::new();
    for (entity, core) in &cores_q {
        cores.insert(entity, core.0);
    }
    let mut to_seed: Vec<(Entity, MassProperties)> = Vec::new();
    {
        let changed = props.p0();
        for (entity, props_ref) in &changed {
            let core = props_ref.0.to_untyped();
            cores.insert(entity, core);
            to_seed.push((entity, core));
        }
    }

    // Step 2: build the view + run the kernel against the live
    // cores. `MassChildOf` and the core map together carry every
    // input the kernel needs; the view is fully owned so it has no
    // active borrow on the queries.
    let view = build_view_from_cores(&cores, &parents, &names);
    let outputs = recompute_composites_via_storage(&view);

    // Step 3: persist cache updates into Bevy (deferred via Commands
    // so we don't touch the world while a query is borrowed). This
    // covers both first-time seed (entity newly-spawned) and refresh
    // (mission code edited MassPropertiesC since last tick).
    for (entity, core) in to_seed {
        commands.entity(entity).insert(CoreMassPropertiesC(core));
    }

    // Step 4: write composites back into MassPropertiesC, using
    // `bypass_change_detection` so our own writes do not re-trigger
    // `Changed<MassPropertiesC>` on the next tick (which would
    // overwrite the just-correct CoreMassPropertiesC with the
    // composite — exactly the bug the cache is meant to prevent).
    //
    // `MassPropertiesC::from` is the canonical insertion-time bridge
    // (defined in `src/components.rs`, mirroring every other typed
    // Bevy component), so going through it keeps this module free
    // of bypass constructors.
    let mut writes = props.p1();
    for (entity, out) in outputs {
        if let Ok(mut p) = writes.get_mut(entity) {
            *p.bypass_change_detection() = MassPropertiesC::from(out.composite);
        }
    }
}

/// Build a [`MassTreeView`] from a pre-collected core map and the
/// parents query. Internal helper used by [`composite_mass_system`]
/// to avoid the Bevy `ParamSet` shape that a "(read, write) of
/// `MassPropertiesC`" + "(read) `CoreMassPropertiesC`" combo would
/// otherwise require.
fn build_view_from_cores(
    cores: &HashMap<Entity, MassProperties>,
    parents_q: &Query<(Entity, &MassChildOf)>,
    names_q: &Query<&Name>,
) -> MassTreeView {
    let mut edge_data: HashMap<Entity, MassChildOf> = HashMap::new();
    let mut parents: Vec<(Entity, Entity)> = Vec::new();
    let mut children_by_parent: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (child, edge) in parents_q.iter() {
        // Fail-loud (PR #283 review thread PRRT_kwDORtae6c5_KBwP):
        // a `MassChildOf` whose target entity has no
        // `MassPropertiesC` is broken topology — silently skipping
        // the edge would orphan the child and leave stale composites
        // upstream. Per CLAUDE.md "Fail Loudly", panic with a
        // diagnostic that names both ends of the broken edge.
        assert!(
            cores.contains_key(&edge.parent),
            "MassChildOf edge {child:?} -> {parent:?}: parent has no MassPropertiesC. \
             Either add MassPropertiesC to the parent or remove the MassChildOf \
             component from the child.",
            parent = edge.parent
        );
        edge_data.insert(child, *edge);
        parents.push((child, edge.parent));
        children_by_parent
            .entry(edge.parent)
            .or_default()
            .push(child);
    }

    let mut nodes: Vec<MassNodeRecord> = Vec::new();
    let mut index: HashMap<Entity, usize> = HashMap::new();
    let mut roots: Vec<Entity> = Vec::new();

    for (&entity, &core) in cores {
        let structure_point = match edge_data.get(&entity) {
            Some(edge) => MassPointState {
                position: edge.offset,
                t_parent_this: edge.t_parent_child,
            },
            None => MassPointState::default(),
        };
        let name = match names_q.get(entity) {
            Ok(n) => n.as_str().to_owned(),
            Err(_) => format!("{entity:?}"),
        };
        let idx = nodes.len();
        nodes.push(MassNodeRecord {
            core,
            structure_point,
            name,
        });
        index.insert(entity, idx);
        if !edge_data.contains_key(&entity) {
            roots.push(entity);
        }
    }

    MassTreeView {
        parents,
        nodes,
        index,
        children_by_parent,
        roots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, DVec3};
    use jeod_sim::MassProperties;

    fn add_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app
    }

    #[test]
    fn single_root_leaves_props_unchanged() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let core = MassProperties::new(10.0);
        let e = app.world_mut().spawn(MassPropertiesC::from(core)).id();
        app.update();

        let world = app.world();
        let stored = world.get::<MassPropertiesC>(e).unwrap().0.to_untyped();
        assert!((stored.mass - core.mass).abs() < 1e-12);
        assert_eq!(stored.position, core.position);
        // Single root with no MassChildOf — fast-path returns
        // without re-writing.
    }

    #[test]
    fn parent_composite_matches_arena_after_attach() {
        // Build the same parent + child topology in both Bevy
        // (MassChildOf) and the arena MassTree, run composition on
        // each, assert the parent composite matches.
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        // Parent at origin, mass 10; child mass 5 attached at
        // structural offset [3, 0, 0].
        let parent_core = MassProperties::with_inertia(
            10.0,
            DMat3::from_diagonal(DVec3::new(50.0, 60.0, 70.0)),
            DVec3::ZERO,
        );
        let child_core = MassProperties::new(5.0);
        let offset = DVec3::new(3.0, 0.0, 0.0);

        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(parent_core))
            .id();
        app.world_mut().spawn((
            MassPropertiesC::from(child_core),
            MassChildOf::new(parent, offset),
        ));

        app.update();

        let stored = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped();

        // Reference: same topology in the arena.
        let mut tree = jeod_sim::MassTree::new();
        let parent_id = tree.add_root("parent".into(), parent_core);
        let child_id = tree.add_body("child".into(), child_core);
        tree.attach(child_id, parent_id, offset, DMat3::IDENTITY);
        let arena = tree.get(parent_id).composite_properties;

        assert!(
            (stored.mass - arena.mass).abs() < 1e-12,
            "Bevy {} vs arena {}",
            stored.mass,
            arena.mass
        );
        let dpos = (stored.position - arena.position).length();
        assert!(dpos < 1e-12, "position diff {dpos:.3e}");
        // Inertia parity: this is the load-bearing assertion since
        // parallel-axis transformation depends on offset routing.
        for (col_a, col_b) in [
            (stored.inertia.x_axis, arena.inertia.x_axis),
            (stored.inertia.y_axis, arena.inertia.y_axis),
            (stored.inertia.z_axis, arena.inertia.z_axis),
        ] {
            let d = (col_a - col_b).length();
            assert!(d < 1e-10, "inertia col diff {d:.3e}");
        }
    }

    #[test]
    fn no_mass_children_fast_path_no_panic() {
        // Empty world: composite_mass_system must not panic.
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);
        app.update();
    }

    #[test]
    fn mid_sim_core_mass_edit_picked_up_on_next_tick() {
        // PR #283 review thread PRRT_kwDORtae6c5_KAGZ / _KBwJ
        // (write-once cache regression):
        //
        // After tick 1, an entity has its CoreMassPropertiesC
        // seeded. Mission code edits MassPropertiesC mid-sim (fuel
        // burn / staging). The next tick must pick up the new core
        // and propagate it through to the parent's composite — the
        // previous "seeded once and frozen" cache silently dropped
        // mid-sim edits.
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let parent_core = MassProperties::new(10.0);
        let child_core = MassProperties::new(5.0);
        let offset = DVec3::new(2.0, 0.0, 0.0);
        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(parent_core))
            .id();
        let child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(child_core),
                MassChildOf::new(parent, offset),
            ))
            .id();

        // Tick 1: seed cache. Composite parent mass = 15.
        app.update();
        let m1 = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!((m1 - 15.0).abs() < 1e-12, "tick1 parent mass {m1}");

        // Mission edits the child's MassPropertiesC mid-sim
        // (mass 5 -> mass 8). Bevy marks Changed<MassPropertiesC>
        // for the child, so composite_mass_system refreshes the
        // cache and recomposes.
        {
            let mut props = app.world_mut().get_mut::<MassPropertiesC>(child).unwrap();
            *props = MassPropertiesC::from(MassProperties::new(8.0));
        }
        app.update();
        let m2 = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!(
            (m2 - 18.0).abs() < 1e-12,
            "mid-sim edit not picked up: parent mass {m2} (expected 18)"
        );
    }

    #[test]
    #[should_panic(expected = "MassChildOf edge")]
    fn missing_parent_fails_loudly() {
        // PR #283 review thread PRRT_kwDORtae6c5_KBwP: a
        // `MassChildOf` whose parent has no MassPropertiesC must
        // panic, not silently treat the child as a root.
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        // Spawn a "parent" entity *without* MassPropertiesC.
        let bad_parent = app.world_mut().spawn_empty().id();
        app.world_mut().spawn((
            MassPropertiesC::from(MassProperties::new(5.0)),
            MassChildOf::new(bad_parent, DVec3::ZERO),
        ));

        // composite_mass_system must panic with a "MassChildOf
        // edge ..." diagnostic.
        app.update();
    }

    #[test]
    fn fast_path_skips_kernel_when_no_edges() {
        // PR #283 review thread PRRT_kwDORtae6c5_KBwG: with no
        // `MassChildOf` edges in the world, composite_mass_system
        // must not touch MassPropertiesC at all (every body is its
        // own composite). Bevy's change-detection ticks would
        // otherwise mark the components as changed every frame.
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let core = MassProperties::new(10.0);
        let e = app.world_mut().spawn(MassPropertiesC::from(core)).id();
        // Tick once to seed CoreMassPropertiesC and clear the
        // initial Changed flag on MassPropertiesC.
        app.update();

        // Stash the change-tick of MassPropertiesC.
        let before_tick = app
            .world()
            .entity(e)
            .get_change_ticks::<MassPropertiesC>()
            .unwrap()
            .changed;

        // Two more ticks with no edges and no external edits.
        app.update();
        app.update();

        let after_tick = app
            .world()
            .entity(e)
            .get_change_ticks::<MassPropertiesC>()
            .unwrap()
            .changed;
        assert_eq!(
            before_tick, after_tick,
            "fast path should not touch MassPropertiesC: change tick advanced from {before_tick:?} to {after_tick:?}"
        );
    }
}
