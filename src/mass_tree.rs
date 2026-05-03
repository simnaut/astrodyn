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
    /// `child -> parent` lookup. Built once at construction so
    /// [`MassStorage::parent`] is `O(1)` per call (PR #283 review
    /// thread PRRT_kwDORtae6c5_KQUU). The kernel calls `parent`
    /// inside its post-order walk; before this map existed the
    /// impl was linearly scanning a `Vec<(child, parent)>`, breaking
    /// the `O(n)` complexity the view's docstring advertises
    /// (degraded the whole walk to `O(n²)`). The arena's
    /// `MassTree::parent` achieves the same cost via slab indexing;
    /// this is the ECS-side equivalent.
    parent_by_child: HashMap<Entity, Entity>,
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
    ///
    /// Treats the live `MassPropertiesC` value as the **core**.
    /// **Important (PR #283 review threads `PRRT_kwDORtae6c5_KHnh`
    /// and `PRRT_kwDORtae6c5_KZvV`): this is correct only when no
    /// prior [`composite_mass_system`] tick has overwritten
    /// `MassPropertiesC` with composite values** — i.e. one-shot
    /// startup composition, downstream systems that build their own
    /// view *before* the system has run, or scenarios where the
    /// caller knows every entity is a leaf (so live == composite ==
    /// core trivially).
    ///
    /// **For mission code that runs the kernel after the schedule
    /// has composed at least once, use [`MassTreeQueries::build_view`]
    /// (or [`MassTreeQueries::recompute_composites`]) instead** —
    /// those route through the [`CoreMassPropertiesC`] cache and
    /// give every non-leaf entity its true core, avoiding the
    /// silent double-count that this constructor would produce on a
    /// post-system world.
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
        // from `mass_q` (PR #283 review thread PRRT_kwDORtae6c5_KBwP)
        // or whose carrier child is missing `MassPropertiesC` (PR
        // #283 review threads PRRT_kwDORtae6c5_KiLC and
        // PRRT_kwDORtae6c5_KiLP). Both ends of every edge must
        // declare their core mass properties or the kernel walk
        // would silently drop the broken edge.
        let mass_set: HashMap<Entity, ()> = mass_q.iter().map(|(e, _)| (e, ())).collect();

        let mut edge_data: HashMap<Entity, MassChildOf> = HashMap::new();
        let mut parent_by_child: HashMap<Entity, Entity> = HashMap::new();
        let mut children_by_parent: HashMap<Entity, Vec<Entity>> = HashMap::new();
        for (child, edge) in parents_q.iter() {
            assert!(
                mass_set.contains_key(&child),
                "entity {child:?} carries MassChildOf({parent:?}) but has no \
                 MassPropertiesC: every body in the mass tree must declare its \
                 core mass properties — add MassPropertiesC or remove the \
                 MassChildOf relation",
                parent = edge.parent
            );
            assert!(
                mass_set.contains_key(&edge.parent),
                "MassChildOf edge {child:?} -> {parent:?}: parent has no MassPropertiesC. \
                 Either add MassPropertiesC to the parent or remove the MassChildOf \
                 component from the child.",
                parent = edge.parent
            );
            edge_data.insert(child, *edge);
            parent_by_child.insert(child, edge.parent);
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
            parent_by_child,
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
        // O(1) HashMap lookup against the `child -> parent` map
        // built at construction time. Matches the asymptotic claim
        // in the [`MassTreeView`] docstring (PR #283 review thread
        // PRRT_kwDORtae6c5_KQUU). The arena's `MassTree::parent`
        // achieves the same cost via slab indexing; this is the
        // ECS-side equivalent.
        self.parent_by_child.get(&id).copied()
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

    fn children(&self, id: Self::Id) -> &[Self::Id] {
        // Borrow the pre-built `children_by_parent` slab directly so
        // the kernel iterates without allocating per node (PR #283
        // review thread `PRRT_kwDORtae6c5_KZvX`). Falls back to an
        // empty slice for nodes with no children — both branches
        // produce a `&[Entity]`, no `Vec` allocation either way.
        self.children_by_parent
            .get(&id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
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

/// SystemParam bundling the queries needed to build a [`MassTreeView`]
/// or read core / composite mass properties for any mass-tree entity.
///
/// Mission code that wants to call the kernel outside of
/// [`composite_mass_system`] (e.g. a one-shot system that runs
/// composition just once after a manual attach) can take this
/// directly. Since `MassTreeView` is built from a read-borrow of
/// `MassPropertiesC` and the system writes back through a separate
/// `&mut` query, callers using both must split via Bevy's `ParamSet`
/// (the canonical pattern, mirroring what
/// [`composite_mass_system`] does internally).
///
/// **Core vs composite (PR #283 review thread PRRT_kwDORtae6c5_KHnh).**
/// Once [`composite_mass_system`] has run on an entity with at least
/// one [`MassChildOf`] child, that entity's [`MassPropertiesC`]
/// holds the **composite** ("the assembly's mass"), not the core
/// ("MY mass alone"). Mission code that wants the core value must
/// read it explicitly via [`Self::core_mass`], which prefers the
/// hidden [`CoreMassPropertiesC`] cache (seeded by the system on
/// first sight and refreshed on every mid-sim core edit) and falls
/// back to [`MassPropertiesC`] for entities the system hasn't
/// touched yet (newly-spawned roots, entities without the cache,
/// pre-system one-shot recomposition). Mission code that wants the
/// composite reads [`Self::composite_mass`] (or `MassPropertiesC`
/// directly).
///
/// **Mid-tick freshness (PR #283 review threads
/// PRRT_kwDORtae6c5_KiLU and PRRT_kwDORtae6c5_KiLY).** Both
/// [`Self::core_mass`] and [`Self::build_view`] compare the entity's
/// `MassPropertiesC` change tick against the cache's change tick.
/// When mission code edits `MassPropertiesC` *after* the most recent
/// [`composite_mass_system`] run (e.g. mid-tick fuel burn / staging
/// followed by an out-of-schedule
/// [`Self::recompute_composites`]), the read prefers the live value
/// over the now-stale cache — same-tick mass edits are visible
/// without waiting for the next scheduled system run. The system's
/// own write-back uses
/// [`bevy::ecs::change_detection::DetectChangesMut::bypass_change_detection`]
/// so it does not falsely advance the live `MassPropertiesC` change
/// tick past the cache.
#[derive(SystemParam)]
pub struct MassTreeQueries<'w, 's> {
    /// Per-entity live mass / inertia / CoM, read as [`Ref`] so the
    /// freshness comparison in [`Self::core_mass`] /
    /// [`Self::build_view`] (PR #283 review threads
    /// PRRT_kwDORtae6c5_KiLU and PRRT_kwDORtae6c5_KiLY) can compare
    /// the live component's `last_changed()` tick against the
    /// `CoreMassPropertiesC` cache. Once [`composite_mass_system`]
    /// has run, this holds the **composite** for non-leaf entities.
    /// Use [`Self::core_mass`] to read the per-entity core
    /// (pre-Steiner) values. Use [`Self::composite_mass`] when the
    /// composite is what's needed.
    pub mass: Query<'w, 's, (Entity, Ref<'static, MassPropertiesC>)>,
    /// `MassChildOf(parent)` parent links. Only a subset of `mass` —
    /// roots don't carry one.
    pub parents: Query<'w, 's, (Entity, &'static MassChildOf)>,
    /// Optional human-readable entity names for diagnostic messages.
    pub names: Query<'w, 's, &'static Name>,
    /// Read-only handle to the hidden [`CoreMassPropertiesC`] cache
    /// managed by [`composite_mass_system`]. Backs
    /// [`Self::core_mass`]. Mission code never reads this directly —
    /// the cache type itself is rustdoc-hidden. This field exists
    /// only so `core_mass` can route the read through the
    /// `SystemParam`'s scheduler-managed access set.
    ///
    /// Read as [`Ref`] so the staleness comparison in
    /// [`Self::core_mass`] (PR #283 review thread
    /// PRRT_kwDORtae6c5_KiLU) can read the cache's `last_changed()`
    /// tick.
    pub(crate) cores: Query<'w, 's, Ref<'static, CoreMassPropertiesC>>,
    /// Current / previous system tick, for the wraparound-safe
    /// `Tick::is_newer_than(last_run, this_run)` comparison used by
    /// the staleness gates in [`Self::core_mass`] /
    /// [`Self::build_view`] (PR #283 review threads
    /// PRRT_kwDORtae6c5_KiLU and PRRT_kwDORtae6c5_KiLY). A naive
    /// `last_changed().get() > …` comparison would mis-order across
    /// the rare tick wraparound that Bevy clamps to ensure
    /// determinism; the helper accounts for it.
    pub(crate) ticks: bevy::ecs::system::SystemChangeTick,
}

impl MassTreeQueries<'_, '_> {
    /// Return this entity's **core** (pre-Steiner) mass properties —
    /// what the entity contributes by itself, ignoring any attached
    /// children.
    ///
    /// Prefers the hidden [`CoreMassPropertiesC`] cache (seeded by
    /// [`composite_mass_system`] from the entity's [`MassPropertiesC`]
    /// on first sight and refreshed whenever mission code edits the
    /// live `MassPropertiesC`). Falls back to the entity's live
    /// [`MassPropertiesC`] when no cache entry exists yet — that
    /// path is correct for newly-spawned entities (the system hasn't
    /// run yet, so live == core trivially) and for one-shot use
    /// before any composition has occurred.
    ///
    /// **Mid-tick freshness (PR #283 review thread
    /// PRRT_kwDORtae6c5_KiLU).** When mission code edits
    /// `MassPropertiesC` mid-tick (after the most recent
    /// [`composite_mass_system`] run), the cache is stale until the
    /// next scheduled run. This method falls through to the live
    /// `MassPropertiesC` whenever its `last_changed()` tick is
    /// strictly newer than the cache's, so same-tick edits on
    /// **leaf** entities are visible immediately. Note that for
    /// non-leaf entities the live `MassPropertiesC` is the
    /// *composite*, not the core, so a mid-tick edit to a non-leaf's
    /// `MassPropertiesC` is ambiguous — mission code that wants
    /// pre-Steiner edits on a non-leaf must call
    /// [`Self::recompute_composites`] (or wait for the next
    /// scheduled system run) to refresh the cache.
    ///
    /// Returns `None` for an entity that carries neither component,
    /// matching `Query::get`'s semantics. Callers should treat that
    /// as "not a mass-tree entity" rather than zero mass.
    ///
    /// PR #283 review thread PRRT_kwDORtae6c5_KHnh — closes the gap
    /// where mission code had no read path to a non-leaf entity's
    /// core, since [`MassPropertiesC`] post-system is the composite.
    pub fn core_mass(&self, entity: Entity) -> Option<jeod_sim::MassProperties> {
        // Mid-tick freshness gate: if the live `MassPropertiesC`
        // has been changed *after* the cache was last refreshed,
        // treat the cache as stale and prefer the live value (PR
        // #283 review thread PRRT_kwDORtae6c5_KiLU). Otherwise the
        // cache is the source of truth — for non-leaf entities the
        // cache holds the per-entity core while live holds the
        // composite, so reading live unconditionally would
        // silently swap composite for core.
        if let Ok(core) = self.cores.get(entity) {
            let live_newer = self.mass.get(entity).ok().is_some_and(|(_, m)| {
                m.last_changed()
                    .is_newer_than(core.last_changed(), self.ticks.this_run())
            });
            if !live_newer {
                return Some(core.0);
            }
        }
        self.mass.get(entity).ok().map(|(_, m)| m.0.to_untyped())
    }

    /// Return this entity's **composite** mass properties — the
    /// post-Steiner assembly value that the rest of the pipeline
    /// reads (gravity, force collection, integration). Equivalent
    /// to reading [`MassPropertiesC`] directly; provided as a named
    /// counterpart to [`Self::core_mass`] so call sites read
    /// symmetrically.
    pub fn composite_mass(&self, entity: Entity) -> Option<jeod_sim::MassProperties> {
        self.mass.get(entity).ok().map(|(_, m)| m.0.to_untyped())
    }

    /// Build a cache-backed [`MassTreeView`] using
    /// [`CoreMassPropertiesC`] for entities the system has already
    /// composed and the live [`MassPropertiesC`] for entities it
    /// hasn't touched yet (newly-spawned, pre-system one-shot use,
    /// leaves where live == core trivially).
    ///
    /// **Use this — not [`MassTreeView::from_queries`] — for any
    /// external recompute** that runs after [`composite_mass_system`]
    /// has executed at least once. The bare `from_queries` reads
    /// the live `MassPropertiesC` as the per-entity core, which
    /// silently double-counts previously-attached children for any
    /// non-leaf entity (PR #283 review thread `PRRT_kwDORtae6c5_KZvV`).
    /// `from_queries` remains correct for the strictly-pre-system
    /// (or strictly-leaf) cases where no composite write-back has
    /// happened yet, but mission code calling the kernel from a
    /// one-shot system after the schedule has run **must** route
    /// through this method.
    ///
    /// **Mid-tick freshness (PR #283 review thread
    /// PRRT_kwDORtae6c5_KiLY).** Same per-entity tick comparison as
    /// [`Self::core_mass`]: when the live `MassPropertiesC` has been
    /// edited *after* the cache was last refreshed, the per-entity
    /// core in the constructed view is the live value rather than
    /// the stale cache, so an out-of-schedule
    /// [`Self::recompute_composites`] called immediately after a
    /// mission-code edit reflects that edit.
    ///
    /// The returned view is fully owned (snapshots core properties
    /// and names at construction); callers can then drive
    /// [`jeod_sim::recompute_composites_via_storage`] without
    /// holding any borrow against this `SystemParam`.
    pub fn build_view(&self) -> MassTreeView {
        // Per-entity core map: cache first (covers every entity the
        // system has touched, including non-leaves whose live
        // `MassPropertiesC` is now the composite), live
        // `MassPropertiesC` second (covers leaves and freshly-spawned
        // entities the system hasn't seen yet — correct because for
        // those live == core). When the live `MassPropertiesC` was
        // edited *after* the cache was last refreshed, prefer the
        // live value so a mid-tick edit drives the next
        // recomposition (PR #283 review thread
        // PRRT_kwDORtae6c5_KiLY).
        let mut cores: HashMap<Entity, MassProperties> = HashMap::new();
        let this_run = self.ticks.this_run();
        for (entity, mass) in self.mass.iter() {
            let core = match self.cores.get(entity) {
                Ok(c) => {
                    if mass
                        .last_changed()
                        .is_newer_than(c.last_changed(), this_run)
                    {
                        mass.0.to_untyped()
                    } else {
                        c.0
                    }
                }
                Err(_) => mass.0.to_untyped(),
            };
            cores.insert(entity, core);
        }
        build_view_from_cores(&cores, &self.parents, &self.names)
    }

    /// Run the storage-agnostic composition kernel against a
    /// cache-backed view of the current world.
    ///
    /// Convenience wrapper around [`Self::build_view`] +
    /// [`jeod_sim::recompute_composites_via_storage`]. Returns the
    /// per-entity post-order outputs (core / composite / parent-frame
    /// shifted) for callers that want to inspect the kernel result
    /// without going through the system's write-back. Mission code
    /// that just wants composite mass should prefer reading
    /// [`MassPropertiesC`] directly after [`composite_mass_system`]
    /// has run; this method exists for one-shot, dry-run, and
    /// integration-test use cases (PR #283 review thread
    /// `PRRT_kwDORtae6c5_KZvV`).
    pub fn recompute_composites(&self) -> Vec<(Entity, jeod_sim::MassNodeOutputs)> {
        let view = self.build_view();
        jeod_sim::recompute_composites_via_storage(&view)
    }
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
    // PR #283 review thread `PRRT_kwDORtae6c5_KZvT` — gate the
    // expensive walk on whether topology or any core mass has
    // changed since the last system run. `Changed<MassChildOf>`
    // covers `Added` (Bevy treats a freshly-inserted component as
    // changed); `RemovedComponents<MassChildOf>` is the only signal
    // for a detach. `Changed<MassPropertiesC>` is what
    // `props.p0()` already filters on. With all three empty the
    // composite outputs would be byte-identical to last tick's, so
    // we early-return.
    changed_parents: Query<(), Changed<MassChildOf>>,
    mut removed_parents: RemovedComponents<MassChildOf>,
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
    // Change-detection gate (PR #283 review thread
    // `PRRT_kwDORtae6c5_KZvT`): when the world contains edges but
    // neither topology nor any core mass has changed since the last
    // system run, the composite outputs would be byte-identical to
    // last tick's. Skip the kernel walk in that case — a static
    // articulated vehicle (rover with permanently-attached arms,
    // pre-spawned constellation, etc.) pays only the gate cost per
    // tick.
    //
    // `Changed<MassChildOf>` covers added / mutated edges; Bevy
    // treats `Added` as `Changed`. `RemovedComponents<MassChildOf>`
    // is the only signal for a detach (no `Removed` query filter
    // exists in Bevy). `Changed<MassPropertiesC>` (via
    // `props.p0()`) covers mid-sim core mass edits and freshly
    // spawned entities (their first sight matches `Changed`).
    //
    // The cursor on `RemovedComponents` must be drained whether or
    // not we walk — leaving events unread causes the next tick to
    // see them again, falsely re-triggering the walk after the
    // detach has already been processed. We drain by `count()`
    // here (consumes while answering "any unread?").
    //
    // We don't gate the `parents.is_empty()` fast path on this:
    // when the world has zero edges, the fast path's case (b)
    // (just-detached parent revert) is the *only* path that
    // restores stale composites to their core. Skipping it would
    // leave the parent reading as the previous-tick composite
    // forever. The fast path is cheap anyway (linear scan over
    // `cores_q`).
    let any_topology_changed = !changed_parents.is_empty() || removed_parents.read().count() > 0;
    let any_mass_changed = !props.p0().is_empty();

    // Fast path (PR #283 review thread PRRT_kwDORtae6c5_KBwG): if
    // no entity has any `MassChildOf` edge, every body is its own
    // composite — composite == core for each — so we can skip the
    // entire build-view/kernel round-trip and the `O(N²)` HashMap
    // ceremony around it.
    //
    // Two distinct sub-cases live in this branch and they pull in
    // opposite directions, so we have to handle them precisely (PR
    // #283 review thread PRRT_kwDORtae6c5_KQUM):
    //
    //   (a) Mission code edited `MassPropertiesC` mid-tick on a
    //       standalone body (fuel burn / staging / inertia override).
    //       The live `MassPropertiesC` *is* the new source of truth
    //       and must NOT be touched by this system; we only refresh
    //       the `CoreMassPropertiesC` cache so a later attach reads
    //       the new core. These entities show up in the
    //       `Changed<MassPropertiesC>` filter (`props.p0()`).
    //   (b) A previously-attached parent had its last child detached
    //       this tick; its `MassPropertiesC` still carries the stale
    //       *composite* from the previous tick. We must revert it
    //       from `CoreMassPropertiesC` so the body once again reads
    //       as its own core. These entities are NOT in
    //       `Changed<MassPropertiesC>` (mission code didn't touch
    //       them); the cache disagrees with live precisely because
    //       last tick's kernel wrote a composite there.
    //
    // The previous implementation conflated the two: it iterated
    // `cores_q` (which holds the *previous-tick* cache snapshot, not
    // the just-edited core, because the cache refresh in (a) goes
    // through `Commands` and is deferred) and wrote the stale cache
    // back for any entity where `live != core`. For case (a) that
    // rolled the mission edit straight back, breaking the documented
    // "fuel burn / staging visible on the next step" contract.
    //
    // Fix: collect the just-edited entities into a `HashSet` first,
    // then in the revert pass skip any entity in that set — those
    // are case (a) and are pure pass-through. The remaining cores_q
    // entities are case (b) and get the legitimate revert.
    if parents.is_empty() {
        // (a) freshly-spawned / externally-edited cores: refresh the
        //     cache, do not touch `MassPropertiesC` (live IS the
        //     source of truth on these).
        let mut just_edited: std::collections::HashSet<Entity> = std::collections::HashSet::new();
        {
            let changed = props.p0();
            for (entity, props_ref) in &changed {
                commands
                    .entity(entity)
                    .insert(CoreMassPropertiesC(props_ref.0.to_untyped()));
                just_edited.insert(entity);
            }
        }
        // (b) revert stale composites where cache disagrees with
        //     live `MassPropertiesC` — but only for entities that
        //     mission code did NOT just edit. The just-edited set
        //     covers case (a) and must be left alone.
        let mut writes = props.p1();
        for (entity, core) in &cores_q {
            if just_edited.contains(&entity) {
                continue;
            }
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

    // Slow-path gate: with edges present, the kernel walk is the
    // expensive bit. Skip it when neither topology nor any core
    // mass has changed since last run. PR #283 review thread
    // `PRRT_kwDORtae6c5_KZvT`.
    //
    // We can't gate the fast path on this — the fast path itself
    // also handles the just-detached-parent revert, which has no
    // `Changed<MassPropertiesC>` signal (mission code didn't edit
    // anything; the staleness comes from last tick's kernel write
    // colliding with this tick's missing edge).
    if !any_topology_changed && !any_mass_changed {
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
    let mut parent_by_child: HashMap<Entity, Entity> = HashMap::new();
    let mut children_by_parent: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (child, edge) in parents_q.iter() {
        // Fail-loud (PR #283 review thread PRRT_kwDORtae6c5_KBwP):
        // a `MassChildOf` whose target entity has no
        // `MassPropertiesC` is broken topology — silently skipping
        // the edge would orphan the child and leave stale composites
        // upstream. Per CLAUDE.md "Fail Loudly", panic with a
        // diagnostic that names both ends of the broken edge.
        //
        // Same fail-loud guard applies symmetrically to the carrier
        // (PR #283 review thread PRRT_kwDORtae6c5_KiLC): the
        // `MassChildOf` docs require the child to also have
        // `MassPropertiesC`. Without this assert, a child without
        // core mass properties would be silently dropped here
        // (the per-entity loop below only walks `cores`), leaving
        // the parent's composite incomplete.
        assert!(
            cores.contains_key(&child),
            "entity {child:?} carries MassChildOf({parent:?}) but has no \
             MassPropertiesC: every body in the mass tree must declare its \
             core mass properties — add MassPropertiesC or remove the \
             MassChildOf relation",
            parent = edge.parent
        );
        assert!(
            cores.contains_key(&edge.parent),
            "MassChildOf edge {child:?} -> {parent:?}: parent has no MassPropertiesC. \
             Either add MassPropertiesC to the parent or remove the MassChildOf \
             component from the child.",
            parent = edge.parent
        );
        edge_data.insert(child, *edge);
        parent_by_child.insert(child, edge.parent);
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
        parent_by_child,
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
    fn fast_path_preserves_mid_tick_edit_on_standalone_body() {
        // PR #283 review thread PRRT_kwDORtae6c5_KQUM: when the
        // world has zero `MassChildOf` edges, the fast path must
        // not roll back a real mass edit on a standalone body. The
        // earlier implementation iterated `cores_q` (which holds
        // the *previous-tick* cache snapshot, since the deferred
        // `Commands::insert` for the new cache hasn't applied yet)
        // and wrote the stale cache back for any entity where
        // `live != core`, undoing the mission edit until a later
        // tick repaired the cache.
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        // Tick 1 seeds CoreMassPropertiesC with the original 10.0
        // mass and clears the initial `Changed<MassPropertiesC>`
        // flag.
        let core = MassProperties::new(10.0);
        let e = app.world_mut().spawn(MassPropertiesC::from(core)).id();
        app.update();

        // Mission code edits the body's mass mid-sim (10 -> 42),
        // matching the documented "fuel burn / staging visible on
        // the next step" contract.
        {
            let mut props = app.world_mut().get_mut::<MassPropertiesC>(e).unwrap();
            *props = MassPropertiesC::from(MassProperties::new(42.0));
        }

        // Tick 2 must run the fast path AND preserve the new mass.
        app.update();

        let after = app
            .world()
            .get::<MassPropertiesC>(e)
            .unwrap()
            .0
            .to_untyped();
        assert!(
            (after.mass - 42.0).abs() < 1e-12,
            "fast-path rolled back mid-tick edit: mass {} (expected 42)",
            after.mass
        );

        // The cache must also have been refreshed so a future
        // attach reads the new core (rather than the stale 10.0).
        let cache = app
            .world()
            .get::<CoreMassPropertiesC>(e)
            .expect("core cache present after tick");
        assert!(
            (cache.0.mass - 42.0).abs() < 1e-12,
            "core cache not refreshed: mass {} (expected 42)",
            cache.0.mass
        );
    }

    #[test]
    fn fast_path_reverts_just_detached_parent() {
        // Sibling guard for the test above: when the last child
        // is detached, the parent's `MassPropertiesC` must revert
        // from the stale composite back to its core. This is the
        // legitimate case (b) in the fast path that we must NOT
        // accidentally suppress while fixing case (a).
        use bevy::ecs::system::RunSystemOnce;

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

        // Tick 1 composes the tree: parent's `MassPropertiesC`
        // becomes the composite (mass 15).
        app.update();
        let composite_mass = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!(
            (composite_mass - 15.0).abs() < 1e-12,
            "tick1 parent composite mass {composite_mass}"
        );

        // Detach the child by removing the `MassChildOf` edge.
        // The parent is *not* edited — its `MassPropertiesC` still
        // holds the stale composite from tick 1.
        app.world_mut().entity_mut(child).remove::<MassChildOf>();
        // Bevy's `Changed` tick honours the previous frame; clear
        // any leftover state by running an empty system once so
        // the next `app.update()` sees `parents.is_empty()` true.
        app.world_mut()
            .run_system_once(|| ())
            .expect("noop system runs");

        // Tick 2: fast path must revert the parent to its core
        // (mass 10).
        app.update();
        let reverted = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped();
        assert!(
            (reverted.mass - 10.0).abs() < 1e-12,
            "just-detached parent not reverted: mass {} (expected 10)",
            reverted.mass
        );
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

    #[test]
    fn core_cache_stable_across_ticks_with_mass_update_system_present() {
        // PR #283 review thread `PRRT_kwDORtae6c5_K0dm` (round-8):
        //
        // Production scheduling runs `mass_update_system` immediately
        // before `composite_mass_system` (see `lib.rs` system order).
        // `mass_update_system` previously called
        // `mass.recompute_derived()` unconditionally on every entity,
        // which triggers `DerefMut` on `Mut<MassPropertiesC>` and
        // therefore marks `Changed<MassPropertiesC>` for every entity
        // every tick — even when the underlying value is unchanged.
        //
        // That false positive corrupted `composite_mass_system`'s
        // `Changed<MassPropertiesC>`-driven cache refresh: on tick 2,
        // every parent's `MassPropertiesC` is the *composite* that
        // tick 1's kernel wrote (via `bypass_change_detection`), and
        // the refresh pass would reseed `CoreMassPropertiesC` from
        // that composite. The kernel would then fold the children
        // into the already-composited core, producing a runaway
        // composite mass that double-counts children every tick.
        //
        // This test pins the fix: when no mission code edits any
        // `MassPropertiesC`, the core cache and the resulting
        // composite must be byte-identical across ticks.
        let mut app = add_test_app();
        // Schedule the two systems in their production order.
        app.add_systems(
            Update,
            (
                crate::systems::mass_update_system,
                composite_mass_system.after(crate::systems::mass_update_system),
            ),
        );

        // Three-body chain: grandparent (10 kg) ← parent (5 kg) ←
        // child (2 kg). Use offsets that exercise parallel-axis
        // routing through both edges so the composite is
        // unambiguously different from any individual core.
        let grandparent_core = MassProperties::new(10.0);
        let parent_core = MassProperties::new(5.0);
        let child_core = MassProperties::new(2.0);

        let grandparent = app
            .world_mut()
            .spawn(MassPropertiesC::from(grandparent_core))
            .id();
        let parent = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(parent_core),
                MassChildOf::new(grandparent, DVec3::new(2.0, 0.0, 0.0)),
            ))
            .id();
        let child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(child_core),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
            ))
            .id();

        // Tick 1: seed the cache, fold children into composites.
        app.update();

        // Snapshot the post-tick-1 state.
        let gp_core_t1 = app
            .world()
            .get::<CoreMassPropertiesC>(grandparent)
            .expect("grandparent cache seeded after tick 1")
            .0;
        let p_core_t1 = app
            .world()
            .get::<CoreMassPropertiesC>(parent)
            .expect("parent cache seeded after tick 1")
            .0;
        let c_core_t1 = app
            .world()
            .get::<CoreMassPropertiesC>(child)
            .expect("child cache seeded after tick 1")
            .0;
        let gp_composite_t1 = app
            .world()
            .get::<MassPropertiesC>(grandparent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        let p_composite_t1 = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;

        // Sanity: composites match cores plus children.
        assert!(
            (gp_composite_t1 - 17.0).abs() < 1e-12,
            "tick1 grandparent composite mass {gp_composite_t1} (expected 17)"
        );
        assert!(
            (p_composite_t1 - 7.0).abs() < 1e-12,
            "tick1 parent composite mass {p_composite_t1} (expected 7)"
        );
        // Cache holds the *cores*, not the composites.
        assert!(
            (gp_core_t1.mass - 10.0).abs() < 1e-12,
            "tick1 grandparent core cache {} (expected 10)",
            gp_core_t1.mass
        );
        assert!(
            (p_core_t1.mass - 5.0).abs() < 1e-12,
            "tick1 parent core cache {} (expected 5)",
            p_core_t1.mass
        );

        // Tick 2: no mission edits. With the bug,
        // `mass_update_system` falsely marks every
        // `MassPropertiesC` as Changed; the cache refresh in
        // `composite_mass_system` then reseeds the cores from the
        // tick-1 composites, and the kernel double-counts children.
        app.update();

        let gp_core_t2 = app
            .world()
            .get::<CoreMassPropertiesC>(grandparent)
            .unwrap()
            .0;
        let p_core_t2 = app.world().get::<CoreMassPropertiesC>(parent).unwrap().0;
        let c_core_t2 = app.world().get::<CoreMassPropertiesC>(child).unwrap().0;
        let gp_composite_t2 = app
            .world()
            .get::<MassPropertiesC>(grandparent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        let p_composite_t2 = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;

        // Cores must be byte-identical to tick 1 — no mission code
        // touched them.
        assert!(
            (gp_core_t2.mass - gp_core_t1.mass).abs() < 1e-12,
            "grandparent core cache drifted across ticks: {} -> {}",
            gp_core_t1.mass,
            gp_core_t2.mass
        );
        assert!(
            (p_core_t2.mass - p_core_t1.mass).abs() < 1e-12,
            "parent core cache drifted across ticks: {} -> {}",
            p_core_t1.mass,
            p_core_t2.mass
        );
        assert!(
            (c_core_t2.mass - c_core_t1.mass).abs() < 1e-12,
            "child core cache drifted across ticks: {} -> {}",
            c_core_t1.mass,
            c_core_t2.mass
        );
        // Composites must also be stable — a corrupted core would
        // make the composite grow each tick.
        assert!(
            (gp_composite_t2 - gp_composite_t1).abs() < 1e-12,
            "grandparent composite mass drifted across ticks: {gp_composite_t1} -> {gp_composite_t2}"
        );
        assert!(
            (p_composite_t2 - p_composite_t1).abs() < 1e-12,
            "parent composite mass drifted across ticks: {p_composite_t1} -> {p_composite_t2}"
        );
    }

    #[test]
    fn mass_tree_view_parent_lookup_matches_storage() {
        // PR #283 review thread PRRT_kwDORtae6c5_KQUU: the
        // `MassStorage::parent` impl on `MassTreeView` must do an
        // `O(1)` HashMap lookup against the pre-built
        // `parent_by_child` map and return the same answer the old
        // linear-scan impl returned. We can't directly observe the
        // asymptotic cost from a unit test, but we can pin the
        // semantics so a future regression that breaks lookup
        // correctness fires here.
        use bevy::ecs::system::RunSystemOnce;

        let mut app = add_test_app();

        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(10.0)))
            .id();
        let child_a = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(2.0)),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
            ))
            .id();
        let child_b = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(3.0)),
                MassChildOf::new(parent, DVec3::new(0.0, 1.0, 0.0)),
            ))
            .id();
        let lone_root = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(7.0)))
            .id();

        // Build the view via the public from_queries entry point.
        // This exercises the construction-time HashMap population.
        let probe = move |mass_q: Query<(Entity, &MassPropertiesC)>,
                          parents_q: Query<(Entity, &MassChildOf)>,
                          names_q: Query<&Name>| {
            let view = MassTreeView::from_queries(&mass_q, &parents_q, &names_q);
            assert_eq!(view.parent(child_a), Some(parent));
            assert_eq!(view.parent(child_b), Some(parent));
            assert_eq!(view.parent(parent), None);
            assert_eq!(view.parent(lone_root), None);
        };

        app.world_mut()
            .run_system_once(probe)
            .expect("probe system runs");
    }

    /// PR #283 review thread PRRT_kwDORtae6c5_KHnh — `MassTreeQueries`
    /// exposes a `core_mass(entity)` read path so mission code can
    /// recover the per-entity core (pre-Steiner) properties even
    /// after [`composite_mass_system`] has overwritten
    /// `MassPropertiesC` with the composite. `composite_mass(entity)`
    /// returns the post-system composite (== reading
    /// `MassPropertiesC` directly), provided as a named symmetric
    /// counterpart so call sites read clearly.
    #[test]
    fn mass_tree_queries_core_vs_composite_split() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

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
        let child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(child_core),
                MassChildOf::new(parent, offset),
            ))
            .id();

        // Run one tick of composition. Now `MassPropertiesC[parent]`
        // is the *composite* (mass 15); `CoreMassPropertiesC[parent]`
        // is the seeded core (mass 10).
        app.update();

        // One-shot system that pulls out core_mass / composite_mass
        // for both entities and stashes them in a resource.
        #[derive(Resource, Default)]
        struct Probe {
            parent_core_mass: f64,
            parent_composite_mass: f64,
            child_core_mass: f64,
            child_composite_mass: f64,
        }
        app.insert_resource(Probe::default());

        #[derive(Resource)]
        struct Targets {
            parent: Entity,
            child: Entity,
        }
        app.insert_resource(Targets { parent, child });

        fn probe(queries: MassTreeQueries, mut out: ResMut<Probe>, targets: Res<Targets>) {
            out.parent_core_mass = queries
                .core_mass(targets.parent)
                .expect("parent has core mass")
                .mass;
            out.parent_composite_mass = queries
                .composite_mass(targets.parent)
                .expect("parent has composite mass")
                .mass;
            out.child_core_mass = queries
                .core_mass(targets.child)
                .expect("child has core mass")
                .mass;
            out.child_composite_mass = queries
                .composite_mass(targets.child)
                .expect("child has composite mass")
                .mass;
        }

        // Run the probe via run_system so it picks up the cache
        // values written in the previous Update tick.
        let probe_id = app.world_mut().register_system(probe);
        app.world_mut()
            .run_system(probe_id)
            .expect("probe system runs");

        let p = app.world().resource::<Probe>();
        // Parent: core 10, composite 15 (10 + 5).
        assert!(
            (p.parent_core_mass - 10.0).abs() < 1e-12,
            "parent core mass: {} (expected 10)",
            p.parent_core_mass
        );
        assert!(
            (p.parent_composite_mass - 15.0).abs() < 1e-12,
            "parent composite mass: {} (expected 15)",
            p.parent_composite_mass
        );
        // Child (atomic leaf): core == composite == 5.
        assert!(
            (p.child_core_mass - 5.0).abs() < 1e-12,
            "child core mass: {} (expected 5)",
            p.child_core_mass
        );
        assert!(
            (p.child_composite_mass - 5.0).abs() < 1e-12,
            "child composite mass: {} (expected 5)",
            p.child_composite_mass
        );
    }

    /// PR #283 review thread `PRRT_kwDORtae6c5_KZvT` — gate the
    /// kernel walk on `Changed<MassChildOf>` /
    /// `RemovedComponents<MassChildOf>` / `Changed<MassPropertiesC>`.
    /// Build a static 3-body chain, tick once to compose, then plant
    /// a sentinel in the parent's `MassPropertiesC` (with
    /// `bypass_change_detection` so the change-tick is unaffected).
    /// On the next tick with no topology or mass edits, the system
    /// must skip the walk, leaving the sentinel intact. If the gate
    /// regresses (system walks every tick once any edge exists), the
    /// kernel will overwrite the sentinel with the correct composite
    /// and the test fails.
    #[test]
    fn gate_skips_walk_on_static_chain() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        // 3-body chain: a → b → c, all mass 1.
        let a = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(1.0)))
            .id();
        let b = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(1.0)),
                MassChildOf::new(a, DVec3::new(1.0, 0.0, 0.0)),
            ))
            .id();
        let _c = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(1.0)),
                MassChildOf::new(b, DVec3::new(1.0, 0.0, 0.0)),
            ))
            .id();

        // Tick 1: compose. After this `a`'s `MassPropertiesC` is the
        // composite (mass 3).
        app.update();
        let composite_after_tick1 = app
            .world()
            .get::<MassPropertiesC>(a)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!(
            (composite_after_tick1 - 3.0).abs() < 1e-12,
            "tick1 composite mass {composite_after_tick1}"
        );

        // Plant a sentinel in `a`'s `MassPropertiesC` *with*
        // `bypass_change_detection` so we don't trigger
        // `Changed<MassPropertiesC>`. If the walk runs on tick 2,
        // it will overwrite the sentinel; if the gate skips, the
        // sentinel survives.
        let sentinel = MassProperties::new(999.0);
        {
            let mut e = app.world_mut().entity_mut(a);
            let mut props = e
                .get_mut::<MassPropertiesC>()
                .expect("entity has MassPropertiesC");
            *props.bypass_change_detection() = MassPropertiesC::from(sentinel);
        }

        // Tick 2: no topology change, no mass change. Gate must
        // skip the walk; sentinel must survive.
        app.update();
        let after_tick2 = app
            .world()
            .get::<MassPropertiesC>(a)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!(
            (after_tick2 - 999.0).abs() < 1e-12,
            "gate did not skip walk: sentinel was overwritten ({after_tick2}, expected 999)"
        );

        // Tick 3: a real mass edit on `c`. Gate must NOT skip;
        // composite must update.
        let c_entity = {
            let mut q = app
                .world_mut()
                .query_filtered::<Entity, With<MassChildOf>>();
            // Find `c`: the child whose parent is `b`.
            let mut found: Option<Entity> = None;
            // Collect entities into a Vec first so the borrow on
            // world is released before we call get on world.
            let candidates: Vec<Entity> = q.iter(app.world()).collect();
            for cand in candidates {
                let edge = app.world().get::<MassChildOf>(cand).unwrap();
                if edge.parent == b {
                    found = Some(cand);
                    break;
                }
            }
            found.expect("found c")
        };
        {
            let mut props = app
                .world_mut()
                .get_mut::<MassPropertiesC>(c_entity)
                .unwrap();
            *props = MassPropertiesC::from(MassProperties::new(10.0));
        }
        // Also restore `a`'s `MassPropertiesC` to the previous
        // composite so the walk has a sane starting point —
        // otherwise the kernel would propagate the sentinel as `a`'s
        // own core. (This mirrors what would have happened if we
        // hadn't planted the sentinel.)
        {
            let mut e = app.world_mut().entity_mut(a);
            let mut props = e.get_mut::<MassPropertiesC>().unwrap();
            *props.bypass_change_detection() = MassPropertiesC::from(MassProperties::new(3.0));
        }
        app.update();
        let after_tick3 = app
            .world()
            .get::<MassPropertiesC>(a)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        // Tick 3 composite: a (1) + b (1) + c (10) = 12.
        assert!(
            (after_tick3 - 12.0).abs() < 1e-12,
            "post-edit composite mass: {} (expected 12)",
            after_tick3
        );
    }

    /// PR #283 review thread `PRRT_kwDORtae6c5_KZvT` — sibling guard:
    /// when an edge is *detached* but no `Changed<MassPropertiesC>`
    /// fires, the gate must not skip — `RemovedComponents<MassChildOf>`
    /// is the only signal for that case.
    #[test]
    fn gate_does_not_skip_on_detach() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(10.0)))
            .id();
        let child_a = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(2.0)),
                MassChildOf::new(parent, DVec3::new(1.0, 0.0, 0.0)),
            ))
            .id();
        let _child_b = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(3.0)),
                MassChildOf::new(parent, DVec3::new(0.0, 1.0, 0.0)),
            ))
            .id();

        // Tick 1: compose → parent composite mass 15.
        app.update();
        let composite_tick1 = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!(
            (composite_tick1 - 15.0).abs() < 1e-12,
            "tick1 composite mass {composite_tick1}"
        );

        // Detach `child_a` only — `child_b` remains. No
        // `Changed<MassPropertiesC>` fires; the only signal is
        // `RemovedComponents<MassChildOf>` plus the topology change
        // already implicit in `child_b` still being attached.
        // Mission code does NOT touch the parent's `MassPropertiesC`.
        app.world_mut().entity_mut(child_a).remove::<MassChildOf>();

        // Tick 2: kernel must run and recompute the parent's
        // composite as parent (10) + child_b (3) = 13.
        app.update();
        let composite_tick2 = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!(
            (composite_tick2 - 13.0).abs() < 1e-12,
            "gate skipped a real detach: parent mass {} (expected 13)",
            composite_tick2
        );
    }

    /// PR #283 review thread `PRRT_kwDORtae6c5_KZvV` —
    /// [`MassTreeQueries::build_view`] / [`MassTreeQueries::recompute_composites`]
    /// give external callers a cache-backed `MassTreeView` so an
    /// out-of-schedule recompute (mission one-shot system, debug
    /// inspector, integration test) does **not** silently
    /// double-count children by re-reading the post-system
    /// composite as the per-entity core.
    ///
    /// Regression scenario:
    /// - Tick 1 composes parent + child, parent's `MassPropertiesC`
    ///   becomes the composite (mass 15).
    /// - An external one-shot system calls
    ///   `MassTreeView::from_queries` (the bare entry point) and
    ///   re-runs the kernel. The bare path reads live
    ///   `MassPropertiesC` as core for the parent (mass 15) and
    ///   then *re-adds* the child (mass 5), producing a fictitious
    ///   composite mass of 20.
    /// - The cache-backed `MassTreeQueries::recompute_composites`
    ///   reads the parent's true core (10) and produces the
    ///   correct composite (15).
    #[test]
    fn build_view_uses_core_cache_to_avoid_double_count() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let parent_core = MassProperties::new(10.0);
        let child_core = MassProperties::new(5.0);
        let offset = DVec3::new(2.0, 0.0, 0.0);

        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(parent_core))
            .id();
        let _child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(child_core),
                MassChildOf::new(parent, offset),
            ))
            .id();

        // Tick 1: parent's `MassPropertiesC` becomes composite (15).
        app.update();
        let composite_after_tick1 = app
            .world()
            .get::<MassPropertiesC>(parent)
            .unwrap()
            .0
            .to_untyped()
            .mass;
        assert!(
            (composite_after_tick1 - 15.0).abs() < 1e-12,
            "tick1 composite mass {composite_after_tick1}"
        );

        #[derive(Resource, Default)]
        struct Probe {
            parent_recomposed_mass: f64,
        }
        app.insert_resource(Probe::default());

        #[derive(Resource)]
        struct Target(Entity);
        app.insert_resource(Target(parent));

        fn external_recompute(
            queries: MassTreeQueries,
            mut out: ResMut<Probe>,
            target: Res<Target>,
        ) {
            // External caller drives the kernel via the public
            // `recompute_composites` helper. Must NOT double-count
            // the child.
            let outputs = queries.recompute_composites();
            let parent_out = outputs
                .iter()
                .find(|(e, _)| *e == target.0)
                .expect("parent in kernel outputs")
                .1;
            out.parent_recomposed_mass = parent_out.composite.mass;
        }

        let id = app.world_mut().register_system(external_recompute);
        app.world_mut()
            .run_system(id)
            .expect("external recompute runs");

        let probe = app.world().resource::<Probe>();
        assert!(
            (probe.parent_recomposed_mass - 15.0).abs() < 1e-12,
            "external recompute mass: {} (expected 15, double-count would give 20)",
            probe.parent_recomposed_mass
        );
    }

    /// PR #283 review thread `PRRT_kwDORtae6c5_KiLC` —
    /// `composite_mass_system` (slow path) must panic when an
    /// entity carries a `MassChildOf` parent link but no
    /// `MassPropertiesC`. Silently dropping the broken edge would
    /// leave the parent's composite missing the (zero, but
    /// schema-violating) child contribution.
    #[test]
    #[should_panic(expected = "carries MassChildOf")]
    fn child_without_mass_properties_panics_in_system() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(10.0)))
            .id();
        // Spawn a child with `MassChildOf` but no `MassPropertiesC`.
        // The system must reject this on the first tick.
        let _child = app.world_mut().spawn(MassChildOf::at_origin(parent)).id();

        app.update();
    }

    /// PR #283 review thread `PRRT_kwDORtae6c5_KiLP` — same
    /// fail-loud guard, but exercised through the public
    /// [`MassTreeView::from_queries`] entry point.
    #[test]
    #[should_panic(expected = "carries MassChildOf")]
    fn child_without_mass_properties_panics_in_from_queries() {
        let mut app = add_test_app();

        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(10.0)))
            .id();
        let _child = app.world_mut().spawn(MassChildOf::at_origin(parent)).id();

        fn build(
            mass_q: Query<(Entity, &MassPropertiesC)>,
            parents_q: Query<(Entity, &MassChildOf)>,
            names_q: Query<&Name>,
        ) {
            let _ = MassTreeView::from_queries(&mass_q, &parents_q, &names_q);
        }
        let id = app.world_mut().register_system(build);
        // run_system propagates panics from inside the registered
        // system back to the caller; #[should_panic] catches it.
        app.world_mut().run_system(id).expect("system runs");
    }

    /// PR #283 review thread `PRRT_kwDORtae6c5_KiLU` — a mission-code
    /// mid-tick edit of `MassPropertiesC` on a leaf entity must be
    /// reflected by `MassTreeQueries::core_mass` immediately, not
    /// deferred to the next `composite_mass_system` run.
    #[test]
    fn core_mass_reflects_midtick_edit_on_leaf() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let leaf = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(10.0)))
            .id();
        // Run once so the cache is seeded.
        app.update();

        // Mid-tick edit: simulate fuel burn / staging by overwriting
        // `MassPropertiesC` between system runs.
        {
            let mut p = app.world_mut().get_mut::<MassPropertiesC>(leaf).unwrap();
            *p = MassPropertiesC::from(MassProperties::new(7.0));
        }

        #[derive(Resource, Default)]
        struct Probe {
            core_mass: f64,
        }
        app.insert_resource(Probe::default());
        #[derive(Resource)]
        struct Target(Entity);
        app.insert_resource(Target(leaf));

        fn read_core(queries: MassTreeQueries, mut out: ResMut<Probe>, t: Res<Target>) {
            out.core_mass = queries.core_mass(t.0).expect("leaf has core mass").mass;
        }
        let id = app.world_mut().register_system(read_core);
        app.world_mut().run_system(id).expect("read_core runs");

        let p = app.world().resource::<Probe>();
        assert!(
            (p.core_mass - 7.0).abs() < 1e-12,
            "stale cache returned: {} (expected 7 after mid-tick edit)",
            p.core_mass
        );
    }

    /// PR #283 review thread `PRRT_kwDORtae6c5_KiLY` —
    /// `MassTreeQueries::build_view` must use the live
    /// `MassPropertiesC` (not the stale cache) for any entity whose
    /// live value was edited *after* the cache was last refreshed.
    /// Without this, `recompute_composites` re-runs against the
    /// pre-edit core and the next composite ignores the edit.
    #[test]
    fn build_view_reflects_midtick_edit_on_leaf() {
        let mut app = add_test_app();
        app.add_systems(Update, composite_mass_system);

        let parent = app
            .world_mut()
            .spawn(MassPropertiesC::from(MassProperties::new(10.0)))
            .id();
        let _child = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::new(parent, DVec3::new(2.0, 0.0, 0.0)),
            ))
            .id();
        // Tick 1: parent composite becomes 15, cache holds core 10.
        app.update();

        // Mid-tick edit: mission code burns fuel on the parent —
        // the contract says the caller writes the parent's new core
        // into `MassPropertiesC`. Note the cache still holds 10
        // until the next system run; an out-of-schedule
        // `recompute_composites` should still see the new core 8.
        {
            let mut p = app.world_mut().get_mut::<MassPropertiesC>(parent).unwrap();
            *p = MassPropertiesC::from(MassProperties::new(8.0));
        }

        #[derive(Resource, Default)]
        struct Probe {
            parent_composite_mass: f64,
        }
        app.insert_resource(Probe::default());
        #[derive(Resource)]
        struct Target(Entity);
        app.insert_resource(Target(parent));

        fn external_recompute(queries: MassTreeQueries, mut out: ResMut<Probe>, t: Res<Target>) {
            let outputs = queries.recompute_composites();
            out.parent_composite_mass = outputs
                .iter()
                .find(|(e, _)| *e == t.0)
                .expect("parent in outputs")
                .1
                .composite
                .mass;
        }
        let id = app.world_mut().register_system(external_recompute);
        app.world_mut()
            .run_system(id)
            .expect("external_recompute runs");

        let p = app.world().resource::<Probe>();
        // Expected: parent core 8 + child 5 = 13. Stale-cache bug
        // would re-use core 10 and give 15.
        assert!(
            (p.parent_composite_mass - 13.0).abs() < 1e-12,
            "build_view used stale cache: {} (expected 13 after parent core edit 10 -> 8)",
            p.parent_composite_mass
        );
    }
}
