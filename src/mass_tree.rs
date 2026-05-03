//! Bevy-side ECS mass-tree adapter.
//!
//! This module is the Bevy half of issue [#271]: it exposes the
//! [`MassChildOf`](crate::MassChildOf) /
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
//! [`MassPropertiesC`](crate::MassPropertiesC).
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
    recompute_composites_via_storage, MassNodeView, MassPointState, MassProperties,
    MassPropertiesTyped, MassStorage, SelfRef,
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
/// **Internal cache — mission code MUST NOT read or write this.**
/// [`composite_mass_system`] manages it: it is inserted on first
/// composition for every entity carrying [`MassPropertiesC`] and
/// reused on subsequent ticks so the kernel always reads the
/// pre-composition core. The struct is publicly visible only because
/// Bevy's system-param signatures require the filter / data types
/// in `Without<…>` clauses to be `pub` when the system itself is
/// `pub`; we mark it `#[doc(hidden)]` to keep it out of the rustdoc
/// public surface.
#[doc(hidden)]
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
        let mut edge_data: HashMap<Entity, MassChildOf> = HashMap::new();
        let mut parents: Vec<(Entity, Entity)> = Vec::new();
        let mut children_by_parent: HashMap<Entity, Vec<Entity>> = HashMap::new();
        for (child, edge) in parents_q.iter() {
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
/// behaviour). Roots additionally get `inverse_inertia` populated
/// (JEOD only inverts at the integration root —
/// `mass_update.cc:116-125`).
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
        // p0: read-only view of MassPropertiesC, used to seed
        //      CoreMassPropertiesC for entities that don't yet have
        //      one (without conflicting with the &mut write below).
        Query<(Entity, &'static MassPropertiesC), Without<CoreMassPropertiesC>>,
        // p1: write-back of the composite results.
        Query<&'static mut MassPropertiesC>,
    )>,
) {
    // Step 1: build the core lookup. Entities seen before have a
    // CoreMassPropertiesC snapshot; entities new to the world (no
    // snapshot yet) get their current MassPropertiesC value treated
    // as core.
    let mut cores: HashMap<Entity, MassProperties> = HashMap::new();
    for (entity, core) in &cores_q {
        cores.insert(entity, core.0);
    }
    let mut newly_seeded: Vec<(Entity, MassProperties)> = Vec::new();
    {
        let fresh_q = props.p0();
        for (entity, props_ref) in &fresh_q {
            let core = props_ref.0.to_untyped();
            cores.insert(entity, core);
            newly_seeded.push((entity, core));
        }
    }

    // Step 2: build the view + run the kernel against the cached
    // cores. `MassChildOf` and the core map together carry every
    // input the kernel needs; the view is fully owned so it has no
    // active borrow on the queries.
    let view = build_view_from_cores(&cores, &parents, &names);
    let outputs = recompute_composites_via_storage(&view);

    // Step 3: persist newly-seeded core snapshots into Bevy
    // (deferred via Commands so we don't touch the world while a
    // query is borrowed).
    for (entity, core) in newly_seeded {
        commands.entity(entity).insert(CoreMassPropertiesC(core));
    }

    // Step 4: write composites back into MassPropertiesC.
    let mut writes = props.p1();
    for (entity, out) in outputs {
        if let Ok(mut p) = writes.get_mut(entity) {
            *p = MassPropertiesC(MassPropertiesTyped::<SelfRef>::from_untyped_unchecked(
                &out.composite,
            ));
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
}
