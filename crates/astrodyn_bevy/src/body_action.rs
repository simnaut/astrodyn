// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Bevy adapter for [`astrodyn::BodyAction`]: queue body actions
//! against an entity at startup *or mid-sim*, then have them applied
//! by [`body_action_system`] each tick before the rest of the
//! pipeline runs.
//!
//! Mirrors JEOD's `DynManager::add_body_action` /
//! `DynManager::remove_body_action` (see
//! `models/dynamics/dyn_manager/src/dyn_manager.cc:168` and `:205`)
//! and `DynManager::perform_actions` (in
//! `models/dynamics/dyn_manager/src/perform_actions.cc`).
//!
//! # Surface
//!
//! Two equivalent ways to queue an action:
//!
//! - Bevy `Message`s: write a [`BodyActionEvent::Add`] /
//!   [`BodyActionEvent::Remove`] from any system (e.g. via
//!   [`BodyActionEvent::add`] / [`BodyActionEvent::remove`]). The
//!   plugin already registers the unified message. The `*Event`
//!   suffix matches `AttachEvent` / `DetachEvent`, the other
//!   `#[derive(Message)]` types in this crate.
//! - The [`BodyActionCommandsExt`] trait on `Commands`:
//!   `commands.add_body_action(entity, action, name)` /
//!   `commands.remove_body_action(name)`. Both methods schedule a
//!   one-shot Bevy command that pushes the corresponding message to
//!   the world's pending event buffer.
//!
//! Either path lands the request in the [`BodyActionsR`] queue that
//! the per-tick [`body_action_system`] drains. When two actions
//! target the same sub-state on the same entity within one tick, the
//! later-queued one wins (JEOD semantics: each `apply` overwrites the
//! prior state).
//!
//! ## Same-tick observation: writer vs `Commands` timing
//!
//! The two surfaces differ in *when* the message becomes visible to
//! [`body_action_intake_system`] within the same `app.update()`:
//!
//! - A direct `MessageWriter<BodyActionEvent>` write (or the
//!   [`add_body_action_via`] helper) is **immediate**: the message is
//!   visible to any later-running `MessageReader<BodyActionEvent>` in
//!   the same schedule pass.
//! - [`BodyActionCommandsExt`] uses `Commands::queue`, which defers
//!   the actual message write until the next `ApplyDeferred`
//!   boundary. With no explicit ordering, that boundary lands at the
//!   end of the schedule, so an action queued via
//!   `commands.add_body_action(...)` is **not** observed until the
//!   *next* tick's intake.
//!
//! For same-tick application, callers must either:
//!
//! - use `MessageWriter<BodyActionEvent>` directly (or the
//!   [`add_body_action_via`] helper), or
//! - order the writing system `.before(body_action_intake_system)`
//!   so Bevy auto-inserts an `ApplyDeferred` between writer and
//!   intake before [`body_action_intake_system`] runs.
//!
//! The default `commands.add_body_action(...)` path (no explicit
//! `.before(body_action_intake_system)`) yields next-tick observation,
//! which is the right choice for init-time wiring (where startup
//! actions land before the first FixedUpdate intake) and for any
//! mid-sim queue where one tick of latency is acceptable.
//!
//! # Lifecycle and naming
//!
//! - Adding an action with a `name` registers it for later removal.
//!   Adding two actions with the same name is allowed; both fire in
//!   FIFO order if neither is removed first. Removing a name removes
//!   *every* still-pending action with that name. This is a strict
//!   generalization of JEOD's `remove_body_action` (see
//!   `dyn_manager.cc:211`), which removes only the first match — we
//!   drop all matching pending actions for symmetry with
//!   `add_body_action`.
//! - Adding an action without a name (`name = None`) makes it
//!   anonymous: it cannot be removed by name; it always fires once
//!   when ready and is dropped.
//! - `BodyActionEvent::Remove { name: "" }` is a no-op, matching
//!   JEOD's empty-string short-circuit (`dyn_manager.cc:207-209`).
//!   Any pending action whose name is `Some("")` survives the empty-
//!   name remove.
//! - An action that is added then removed before
//!   [`body_action_system`] runs in the same tick is never applied.
//!   This is the "remove-then-re-add" idiom that JEOD's
//!   `SIM_removable_body_action`'s `mass.py` exercises.
//!
//! # Scheduling
//!
//! [`body_action_system`] is wired by [`crate::AstrodynPlugin`] to run in
//! the `FixedUpdate` schedule between [`crate::AstrodynSet::TimeUpdate`]
//! and [`crate::AstrodynSet::EphemerisUpdate`]. That ordering matches
//! JEOD: actions resolve before ephemeris / gravity / integration
//! consume the new state.
//!
//! Both [`body_action_intake_system`] and [`body_action_system`] are
//! pinned to the `FixedUpdate` schedule only — they are deliberately
//! NOT registered in `Startup`. Each registration site gets its own
//! `Local<MessageCursor<BodyActionEvent>>`, so a dual-schedule wiring
//! would let an anonymous fire-once `BodyActionEvent::Add` apply
//! twice within one `app.update()` call (once via Startup's cursor,
//! again via FixedUpdate's, with Bevy's double-buffered `Messages`
//! keeping the write live across the buffer swap in `First`). Init-
//! time messages still land before any pipeline consumer reads the
//! body's mutable state: the FixedUpdate intake on the first tick
//! observes them and applies before
//! [`crate::AstrodynSet::EphemerisUpdate`].
//!
//! Both [`body_action_system`] and [`crate::mass_update_system`] live
//! in that same TimeUpdate→EphemerisUpdate gap, so the plugin pins
//! `body_action_system` `.before(mass_update_system)` explicitly.
//! That makes a queued [`astrodyn::BodyAction::InitMass`] land its
//! mass replacement (with the `dirty` flag set by this system after
//! the assignment) *before* the same-tick recompute walks the dirty
//! flag — so the inverse-mass / inverse-inertia caches are refreshed
//! before any consumer in `EphemerisUpdate` / `Environment` /
//! `Interaction` reads them.

use std::any::TypeId;
use std::collections::HashSet;
use std::marker::PhantomData;

use astrodyn::{BodyAction, Planet};
use bevy::ecs::message::MessageWriter;
use bevy::prelude::*;

use crate::components::{
    Abm4StateC, DynamicsConfigC, GaussJacksonStateC, MassPropertiesC, RotationalStateC,
    TranslationalStateC,
};

/// Set of planet `TypeId`s for which a per-planet body-action pipeline
/// (`BodyActionsR<P>` queue + `body_action_intake_system::<P>` +
/// `body_action_system::<P>`) has been registered with the Bevy `App`.
///
/// `crate::AstrodynPlugin::build` inserts `TypeId::of::<astrodyn::Earth>()`;
/// `crate::register_planet_systems::<P>` inserts `TypeId::of::<P>()`
/// when wiring an additional planet pipeline. The
/// [`BodyActionEvent::Add`] writer surfaces — both
/// [`BodyActionCommandsExt::add_body_action_for::<P>`] and the
/// post-intake [`body_action_unregistered_planet_fence_system`] —
/// consult this resource to refuse a planet-tagged add whose intake
/// pipeline isn't wired. Without the guard, an `add_for::<Mars>`
/// against a body in an `App` that never called
/// `register_planet_systems::<Mars>` would land in the unified
/// `Messages<BodyActionEvent>` buffer, be skipped by every existing
/// per-planet intake (TypeId mismatch), and silently age out of the
/// double-buffer with no observable effect — a Fail-Loudly violation.
#[derive(Resource, Debug, Default)]
pub struct RegisteredPlanetsR {
    pub(crate) planets: HashSet<TypeId>,
}

impl RegisteredPlanetsR {
    /// Record that planet `P`'s body-action pipeline has been wired.
    /// Called by `register_planet_systems::<P>` (and by
    /// `AstrodynPlugin::build` for Earth).
    #[inline]
    pub fn register<P: Planet>(&mut self) {
        self.planets.insert(TypeId::of::<P>());
    }

    /// `true` iff a per-planet body-action pipeline for `P` has been
    /// registered.
    #[inline]
    pub fn contains<P: Planet>(&self) -> bool {
        self.planets.contains(&TypeId::of::<P>())
    }

    /// `true` iff the supplied `TypeId` matches a registered planet.
    #[inline]
    pub fn contains_type_id(&self, type_id: TypeId) -> bool {
        self.planets.contains(&type_id)
    }
}

/// One pending body action awaiting execution.
///
/// Carried by [`BodyActionsR`] and constructed by the plugin's
/// message-draining system. Mission code does not interact with this
/// type directly; either send a [`BodyActionEvent`] or call
/// [`BodyActionCommandsExt::add_body_action`].
#[derive(Debug, Clone)]
pub(crate) struct PendingBodyAction {
    /// Subject entity the action will mutate.
    pub(crate) entity: Entity,
    /// The action itself.
    pub(crate) action: BodyAction,
    /// Optional name used by [`BodyActionEvent::Remove`] to find
    /// this pending action before it fires. JEOD's
    /// `BodyAction::action_name` is also optional; mission code that
    /// never needs to remove an action mid-flight can leave it
    /// `None`.
    pub(crate) name: Option<String>,
}

/// Bevy `Message`: one body-action lifecycle event — either queue a
/// new action or cancel pending actions by name.
///
/// A single message type carries both the `Add` and `Remove`
/// variants so the per-tick intake walk processes them in *arrival
/// order* (Bevy's per-buffer `MessageId` is monotonic within one
/// message type, so an interleaved `add → remove → add` sequence is
/// observed as written). Two parallel `Message` types would lose
/// that ordering — independent `MessageId` sequences cannot be
/// merged across types.
///
/// # Planet routing
///
/// Each `Add` carries a `planet` [`TypeId`] tag identifying which
/// planet's [`BodyActionsR<P>`] queue should receive the pending
/// action. The Earth-shorthand constructor [`Self::add`] hard-codes
/// `P = astrodyn::Earth`; multi-planet missions targeting another
/// planet pipeline call [`Self::add_for::<P>`] (or the matching
/// [`BodyActionCommandsExt::add_body_action_for::<P>`] on `Commands`).
/// `Remove` is planet-agnostic — every per-planet intake system
/// observes the cancel on the same tick and drops any matching
/// pending entry from its own queue, mirroring JEOD's
/// `DynManager::remove_body_action` which walks the single global
/// `body_actions` list with no planet filter.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use astrodyn_bevy::body_action::BodyActionEvent;
/// use astrodyn::{BodyAction, MassProperties};
///
/// fn queue_mass_change(
///     vehicle: Entity,
///     mut writer: bevy::ecs::message::MessageWriter<BodyActionEvent>,
/// ) {
///     writer.write(BodyActionEvent::add(
///         vehicle,
///         BodyAction::InitMass {
///             mass: MassProperties::new(100_000.0),
///         },
///         Some("vehicle.mass_init"),
///     ));
/// }
/// ```
// `BodyAction` carries the full parameter set for any of the init
// variants (orbital elements, NED state, mass tensors). Boxing the
// `Add` variant just to balance the enum size is wasteful for a
// message that fires at most a handful of times per scenario; the
// `Remove` variant is small but rare too. The size delta is not
// load-bearing for this allocation pattern.
#[allow(clippy::large_enum_variant)]
#[derive(Message, Debug, Clone)]
pub enum BodyActionEvent {
    /// Mirror of JEOD `DynManager::add_body_action(BodyAction&)`:
    /// queue an action to be applied as soon as `is_ready()` returns
    /// true on the next intake-then-apply pass.
    Add {
        /// Subject entity.
        entity: Entity,
        /// Action carrying its parameters.
        action: BodyAction,
        /// Optional name; required for later removal.
        name: Option<String>,
        /// `TypeId` of the planet whose [`BodyActionsR<P>`] queue
        /// should receive this pending entry. Filled by the
        /// per-planet constructors ([`Self::add`] for Earth,
        /// [`Self::add_for::<P>`] for an explicit planet); per-planet
        /// intake systems read this tag to claim only the entries
        /// they own.
        planet: TypeId,
    },
    /// Mirror of `DynManager::remove_body_action(const std::string&)`:
    /// drop *every* still-pending action with this name. Matches
    /// JEOD's linear-scan-by-name behaviour (`dyn_manager.cc:211`),
    /// generalised so two unresolved adds with the same name both
    /// drop in one remove. The cancel fans out across every
    /// per-planet [`BodyActionsR<P>`] queue (the per-planet intake
    /// systems each observe it independently), so a name-based
    /// remove from Earth-orbit code reaches a Mars-tagged add on the
    /// same name even when the calling system holds no `<P>`
    /// witness.
    Remove {
        /// Name to cancel.
        name: String,
    },
}

impl BodyActionEvent {
    /// Construct a [`BodyActionEvent::Add`] tagged for the
    /// `astrodyn::Earth` queue. Earth-only missions (the dominant
    /// case in this codebase) keep the existing call shape; non-Earth
    /// missions use [`Self::add_for::<P>`] to pin a different
    /// planet's [`BodyActionsR<P>`] queue.
    #[inline]
    pub fn add(entity: Entity, action: BodyAction, name: Option<&str>) -> Self {
        Self::add_for::<astrodyn::Earth>(entity, action, name)
    }

    /// Construct a [`BodyActionEvent::Add`] tagged for the planet
    /// `P`'s [`BodyActionsR<P>`] queue. The `<P>` witness routes the
    /// pending entry to the per-planet intake system registered by
    /// [`crate::register_planet_systems::<P>`] (or by
    /// [`crate::AstrodynPlugin`] for `P = astrodyn::Earth`).
    ///
    /// A translational `BodyAction` requires the subject entity to
    /// carry a matching `TranslationalStateC<P>` slot — a queue
    /// instance for `<Mars>` only matches entities the planet-`<Mars>`
    /// system pipeline integrates. Mass-only / rotation-only actions
    /// still pin `<P>` for routing (the apply system mutates the
    /// planet-agnostic `MassPropertiesC` / `RotationalStateC` after
    /// dequeuing) but the choice of `<P>` should match the body's
    /// integration planet so a single planet's apply pass covers all
    /// of that body's pending actions.
    #[inline]
    pub fn add_for<P: Planet>(entity: Entity, action: BodyAction, name: Option<&str>) -> Self {
        BodyActionEvent::Add {
            entity,
            action,
            name: name.map(|n| n.to_string()),
            planet: TypeId::of::<P>(),
        }
    }

    /// Construct a [`BodyActionEvent::Remove`].
    #[inline]
    pub fn remove(name: &str) -> Self {
        BodyActionEvent::Remove {
            name: name.to_string(),
        }
    }
}

/// Bevy `Resource` holding all pending body actions for planet `P`,
/// in insertion order.
///
/// One queue exists per planet `<P>` registered with the plugin:
/// [`crate::AstrodynPlugin`] inserts `BodyActionsR<astrodyn::Earth>`, and
/// [`crate::register_planet_systems::<P>`] inserts the matching
/// `BodyActionsR<P>` for any additional planet. Mission code does
/// not need to touch this resource directly — send a
/// [`BodyActionEvent`] (with [`BodyActionEvent::add`] for Earth or
/// [`BodyActionEvent::add_for::<P>`] for another planet) or use
/// [`BodyActionCommandsExt`] instead.
///
/// JEOD analog: `DynManager::body_actions` (a `std::vector<BodyAction
/// *>` walked once per `perform_actions` pass). JEOD has a single
/// global list because every JEOD `DynBody` integrates against the
/// same singleton dyn-manager-owned planet; the Bevy adapter
/// partitions by `<P>` so a Mars-orbit chief and an Earth-orbit
/// deputy in the same `World` route through disjoint queues.
#[derive(Resource, Debug)]
pub struct BodyActionsR<P: Planet> {
    /// FIFO queue of pending actions.
    pub(crate) pending: Vec<PendingBodyAction>,
    /// Phantom carries the planet witness so `BodyActionsR<Earth>`
    /// and `BodyActionsR<Mars>` are distinct resource types and
    /// Bevy's resource registry partitions them by `<P>`.
    _planet: PhantomData<fn() -> P>,
}

impl<P: Planet> Default for BodyActionsR<P> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            _planet: PhantomData,
        }
    }
}

/// Drains the [`BodyActionEvent`]s tagged for planet `P` into the
/// matching [`BodyActionsR<P>`] queue.
///
/// Each per-planet intake instantiation gets its own
/// `Local<MessageCursor<BodyActionEvent>>`, so two planet pipelines
/// reading the shared message buffer never miss or double-consume a
/// message. `Add` entries whose `planet` `TypeId` does not match
/// `<P>` are skipped — the matching planet's intake system claims
/// them. `Remove` is fan-out: every per-planet intake observes the
/// cancel and drops any matching pending entry from its own queue,
/// so a name-based remove from any code path reaches every
/// `BodyActionsR<P>` on the same tick (mirrors JEOD's single global
/// `body_actions` walk in `DynManager::remove_body_action`).
///
/// Runs strictly before [`body_action_system`] each tick so that an
/// `add → remove → add` sequence within one tick collapses to a
/// single queued action (the same idiom JEOD's `mass.py` from
/// `SIM_removable_body_action` exercises at init time). Bevy
/// preserves message arrival order within one message type, so the
/// unified [`BodyActionEvent`] enum's `MessageReader` walks the
/// add/remove operations in the order their `MessageWriter`s wrote
/// them.
pub fn body_action_intake_system<P: Planet>(
    mut messages: bevy::ecs::message::MessageReader<BodyActionEvent>,
    mut queue: ResMut<BodyActionsR<P>>,
) {
    let this_planet = TypeId::of::<P>();
    for msg in messages.read() {
        match msg {
            BodyActionEvent::Add {
                entity,
                action,
                name,
                planet,
            } => {
                if *planet != this_planet {
                    // Different planet's queue owns this entry.
                    continue;
                }
                queue.pending.push(PendingBodyAction {
                    entity: *entity,
                    action: action.clone(),
                    name: name.clone(),
                });
            }
            BodyActionEvent::Remove { name } => {
                // JEOD_INV: BA.10 — remove pending actions by `action_name`. JEOD's
                // `DynManager::remove_body_action` (`dyn_manager.cc:207-209`) returns
                // immediately when the supplied name is empty so a stray
                // `remove_body_action("")` cannot wipe every pending action whose
                // `action_name` happens to be empty (anonymous JEOD actions register
                // with a default-constructed `std::string`). The Bevy adapter
                // preserves that no-op: `remove("")` does nothing. We further
                // drop *every* still-pending entry whose name matches a
                // non-empty `name`, a strict generalisation of JEOD's
                // first-match-and-erase loop (covered by
                // `tests::remove_drops_all_pending_with_matching_name`).
                // The cancel is planet-agnostic: every per-planet
                // intake walks its own `BodyActionsR<P>`.pending and
                // drops matching entries, so a remove from a system
                // that holds no `<P>` witness still reaches a
                // Mars-tagged add on the same name.
                if name.is_empty() {
                    continue;
                }
                queue
                    .pending
                    .retain(|act| act.name.as_deref() != Some(name.as_str()));
            }
        }
    }
}

/// Fail-loud fence that catches `BodyActionEvent::Add` messages
/// whose `planet` `TypeId` does not match any registered per-planet
/// body-action pipeline.
///
/// The per-planet [`body_action_intake_system::<P>`] each filter their
/// own `Local<MessageCursor<BodyActionEvent>>` and silently `continue`
/// past `Add` entries whose `planet` field doesn't match `<P>`. With
/// no fence, an `add_for::<Mars>` issued in an `App` that never called
/// [`crate::register_planet_systems::<astrodyn::Mars>`] would be
/// skipped by every existing intake and age out of the message
/// double-buffer with no observable effect — a Fail-Loudly violation
/// (the action targets a body, but the pipeline that should mutate
/// that body's `<Mars>` storage was never wired).
///
/// This fence reads the same shared message buffer through its own
/// `Local<MessageCursor>` and panics on any `Add` whose `planet`
/// `TypeId` is not in [`RegisteredPlanetsR`]. Pinned to run after
/// every per-planet intake (so the well-formed adds have already
/// been claimed) but before any `body_action_system::<P>` apply pass
/// (so the panic precedes the silent state mutation that the
/// downstream apply systems would *not* perform on the unregistered
/// planet).
///
/// Pairs with the call-site assertion in
/// [`BodyActionCommandsExt::add_body_action_for::<P>`]: the
/// `Commands` path catches the misconfiguration at queue-flush time,
/// the fence catches the direct-`MessageWriter` path at intake time.
/// Both diagnostics name the unregistered planet via
/// [`std::any::type_name`] and point to
/// [`crate::register_planet_systems`] as the fix.
pub fn body_action_unregistered_planet_fence_system(
    mut messages: bevy::ecs::message::MessageReader<BodyActionEvent>,
    registered: Res<RegisteredPlanetsR>,
) {
    for msg in messages.read() {
        if let BodyActionEvent::Add {
            entity,
            name,
            planet,
            ..
        } = msg
        {
            assert!(
                registered.contains_type_id(*planet),
                "BodyActionEvent::Add for planet TypeId {planet:?} against entity \
                 {entity:?} (action_name={name:?}) but no per-planet body-action \
                 pipeline is registered for that planet — the unified \
                 `Messages<BodyActionEvent>` buffer holds the entry, but no \
                 `body_action_intake_system::<P>` will claim it (every existing \
                 intake will skip it on `TypeId` mismatch) and the message will \
                 age out of the double-buffer with no observable effect. \
                 Fix: call `astrodyn_bevy::register_planet_systems::<P>(&mut app)` \
                 during `App` setup for the planet whose body this action targets, \
                 before writing `BodyActionEvent::add_for::<P>` (or use \
                 `BodyActionCommandsExt::add_body_action_for::<P>`, which performs \
                 the same registration check at the call site). `AstrodynPlugin::build` \
                 pre-registers `astrodyn::Earth`; additional planets must be \
                 registered explicitly.",
            );
        }
    }
}

/// Apply every ready pending action in [`BodyActionsR`] to its
/// subject entity, removing applied actions from the queue.
///
/// Mirrors JEOD `DynManager::perform_actions`
/// (`models/dynamics/dyn_manager/src/perform_actions.cc:41`):
///
/// ```text
/// for action in body_actions:
///     if action.is_ready():
///         action.apply()
///         body_actions.erase(action)
/// ```
///
/// # Per-action mutation site
///
/// - `BodyAction::InitMass` → replaces [`MassPropertiesC`]'s inner
///   `MassProperties`, then sets `dirty = true` on the replacement.
///   `MassProperties::new` / `::with_inertia` themselves leave
///   `dirty = false` (they precompute `inverse_mass` /
///   `inverse_inertia` from the supplied scalars), so the explicit
///   flip here is what tells the same-tick
///   [`crate::mass_update_system`] to walk the entry — its `dirty`
///   guard makes the recompute a no-op for un-flipped entries, so
///   marking dirty after every `InitMass` is the safe default that
///   also covers callers passing a hand-built `MassProperties` with
///   an out-of-sync `inverse_mass`. Also resets multi-step
///   integrator history (Gauss–Jackson / ABM4) on the entity (IG.37):
///   force/mass = acceleration, so a mid-sim mass change makes the
///   predictor / corrector history (which records derivatives from
///   the prior mass) inconsistent with the new dynamics whenever any
///   non-gravitational force is present.
/// - `BodyAction::InitTrans` /
///   `BodyAction::InitTransOrbital` /
///   `BodyAction::InitTransLvlh` /
///   `BodyAction::InitTransNed` →
///   replaces [`TranslationalStateC`]. Resets multi-step integrator
///   history (Gauss–Jackson / ABM4) on the entity (IG.37).
/// - `BodyAction::InitRot` → replaces [`RotationalStateC`]. Resets
///   multi-step integrator history on the entity (IG.37).
///
/// # Failure modes
///
/// Panics if the subject entity does not carry the component the
/// action targets. This matches the "Fail Loudly" rule: an action
/// applied to a wrong-type entity is a misconfiguration, not a
/// silently-skipped operation.
#[allow(clippy::type_complexity)]
pub fn body_action_system<P: Planet>(
    mut queue: ResMut<BodyActionsR<P>>,
    mut bodies: Query<
        (
            Option<&mut TranslationalStateC<P>>,
            Option<&mut RotationalStateC>,
            Option<&mut MassPropertiesC>,
            Option<&mut GaussJacksonStateC>,
            Option<&mut Abm4StateC>,
        ),
        // JEOD_INV: BA.01 — subject must be a DynBody-equivalent entity. `DynamicsConfigC`
        // is required on every dynamic body; gating the query on it both narrows
        // the match to body-like entities (the prior `Option<...>`-only filter
        // matched every entity in the world) and yields a `QueryDoesNotMatch`
        // error from `get_mut` when a caller targets a non-body entity, which
        // surfaces the misconfiguration with the correct diagnostic.
        With<DynamicsConfigC>,
    >,
) {
    let mut idx = 0;
    while idx < queue.pending.len() {
        let action_ref = &queue.pending[idx];
        // JEOD_INV: BA.09 — `is_ready` consulted before `apply` on every pass; not-ready actions
        // stay in the queue (mirror of `perform_actions.cc:45-62`).
        if !action_ref.action.is_ready() {
            idx += 1;
            continue;
        }
        // Take the action by removing it from the queue *before*
        // applying it; an action whose `apply_*` panics still has the
        // entry stripped, so a recovered World won't replay the
        // bad action on the next tick.
        let action = queue.pending.remove(idx);
        let (mut trans, mut rot, mut mass, mut gj, mut abm) = bodies
            .get_mut(action.entity)
            .unwrap_or_else(|err| {
                panic!(
                    "BodyAction subject entity {:?} (action_name={:?}) is not a recognised vehicle entity \
                     (despawned, never spawned, or missing DynamicsConfigC — every dynamic body carries DynamicsConfigC). \
                     Spawn the entity with the dynamic-body Components before queuing a BodyAction. (bevy query error: {err:?})",
                    action.entity, action.name,
                )
            });
        // Track whether translational / rotational state were mutated
        // so we can reset multi-step integrator history afterwards
        // (mirrors the IG.37 attach/detach reset path). `mass_mutated`
        // is tracked alongside so the integrator reset also fires on
        // an `InitMass` action: force/mass = acceleration, so the
        // predictor / corrector history recorded under the prior mass
        // is inconsistent with the new dynamics whenever any
        // non-gravitational force is present.
        let mut state_mutated = false;
        let mut mass_mutated = false;
        if let Some(state) = action.action.apply_translational() {
            let comp = trans
                .as_deref_mut()
                .unwrap_or_else(|| {
                    panic!(
                        "BodyAction targets translational state on entity {entity:?} (action_name={name:?}) \
                         but the entity has no `TranslationalStateC<{planet}>` slot, and this apply pass \
                         only mutates `<{planet}>`-tagged storage (the action was routed to \
                         `BodyActionsR<{planet}>` by its planet tag in `BodyActionEvent::Add::planet`). \
                         Two fixes: \
                         (a) if the body's integration planet really is `{planet}`, spawn it with \
                         `TranslationalStateC::<{planet}>` (`VehicleConfig::spawn_bevy::<{planet}>` is \
                         the canonical entry point); \
                         (b) if the body integrates against another planet `Q`, queue the action via \
                         `BodyActionEvent::add_for::<Q>` (or `BodyActionCommandsExt::add_body_action_for::<Q>`) \
                         so it lands in `BodyActionsR<Q>` and the matching `body_action_system::<Q>` \
                         pass mutates `TranslationalStateC<Q>` instead. Adding a wrong-planet \
                         translational slot to the entity is not a valid workaround — it would \
                         silently land the action in the wrong planet's storage.",
                        entity = action.entity, name = action.name, planet = std::any::type_name::<P>(),
                    )
                });
            // Action-fire boundary — `BodyAction::apply_translational`
            // returns the ECS-agnostic `TranslationalState` (#397).
            // allowed: typed↔raw kernel boundary at action-fire time.
            comp.0 = crate::typed_bridge::trans_raw_to_planet::<P>(&state);
            state_mutated = true;
        }
        if let Some(state) = action.action.apply_rotational() {
            let comp = rot
                .as_deref_mut()
                .unwrap_or_else(|| {
                    panic!(
                        "BodyAction targets rotational state on entity {:?} (action_name={:?}) but the entity has no RotationalStateC. \
                         Add `RotationalStateC::default()` to the entity before queuing this action.",
                        action.entity, action.name,
                    )
                });
            // Same action-fire boundary as the translational branch above (#397).
            // allowed: typed↔raw kernel boundary at action-fire time.
            comp.0 = crate::typed_bridge::rot_raw_to_self_ref(&state);
            state_mutated = true;
        }
        if let Some(props) = action.action.apply_mass() {
            let comp = mass
                .as_deref_mut()
                .unwrap_or_else(|| {
                    panic!(
                        "BodyAction targets mass properties on entity {:?} (action_name={:?}) but the entity has no MassPropertiesC. \
                         Add `MassPropertiesC::from(MassPropertiesTyped::<SelfRef>::new(420_000.0.kg()))` (or `with_inertia(...)`) to the entity before queuing this action.",
                        action.entity, action.name,
                    )
                });
            // Replace the typed wrapper, then mark the new value
            // `dirty` so `mass_update_system` recomputes the inverse
            // caches. `MassProperties::new` / `with_inertia` set
            // `dirty = false` (they precompute `inverse_mass` /
            // `inverse_inertia` from the supplied scalars), so without
            // this flip the per-tick recompute path is skipped — fine
            // for those constructors, but a downstream caller that
            // hands us a hand-built `MassProperties` with an
            // out-of-sync `inverse_mass` would be missed. Marking
            // `dirty` here is the safe action-fire contract: the
            // recompute is a `dirty`-guarded no-op when nothing
            // changed.
            // Action-fire boundary — `MassProperties` is the ECS-agnostic
            // untyped form (#397).
            // allowed: typed↔raw kernel boundary at action-fire time.
            comp.0 = crate::typed_bridge::mass_raw_to_self_ref(&props);
            comp.0.dirty = true;
            mass_mutated = true;
        }
        if state_mutated || mass_mutated {
            // JEOD_INV: IG.37 — multi-step integrator history must be reset on
            // any mid-sim state or mass change. JEOD's `dyn_body_init_*`
            // actions overwrite a body's translational / rotational state
            // (or its mass) mid-run; per JEOD's attach/detach analog,
            // leaving Gauss–Jackson / ABM4 predictor history pointing at
            // the prior dynamics corrupts the next integrate. A mid-sim
            // mass change matters for the same structural reason: the
            // history records `accel = force / mass` samples under the old
            // mass, and any non-gravitational force (drag, SRP, thrust)
            // makes those samples inconsistent with the new acceleration
            // the predictor will compute on the next step. The reset is a
            // no-op for single-step integrators (`gj` / `abm` will be
            // `None` on RK4 entities), so this branch is free for the
            // common path.
            astrodyn::reset_integrators(
                gj.as_deref_mut().map(|c| c.0.inner_mut()),
                abm.as_deref_mut().map(|c| c.0.inner_mut()),
            );
        }
        // Do not advance idx: the queue shifted left by one when we
        // removed the applied action.
    }
}

/// `Commands` extension for queueing body actions without writing to
/// a `MessageWriter` directly.
///
/// JEOD analog: the bare `dynamics.dyn_manager.add_body_action(...)`
/// call in JEOD `Modified_data/*.py`. Mission code that doesn't want
/// to thread a `MessageWriter<BodyActionEvent>` through every
/// system can drop into `Commands` instead — both paths land in the
/// same [`BodyActionsR`] queue.
///
/// # Timing
///
/// Both methods route through `Commands::queue`, so the underlying
/// `BodyActionEvent` write is deferred until the next `ApplyDeferred`
/// boundary — typically the end of the current schedule. As a
/// result, actions queued via this trait are **not observed by
/// [`body_action_intake_system`] until the next tick** unless the
/// writing system is explicitly ordered
/// `.before(body_action_intake_system)` (which causes Bevy to insert
/// an `ApplyDeferred` between them, flushing the queued command
/// before intake runs).
///
/// For same-tick observation without ordering ceremony, prefer a
/// direct `MessageWriter<BodyActionEvent>` (or the
/// [`add_body_action_via`] helper): writer-based sends are immediate.
pub trait BodyActionCommandsExt {
    /// Queue a [`BodyAction`] against `entity` for the
    /// `astrodyn::Earth` queue. Equivalent to sending a
    /// `BodyActionEvent::Add` constructed via [`BodyActionEvent::add`].
    /// Non-Earth missions use [`Self::add_body_action_for::<P>`].
    ///
    /// # Timing
    ///
    /// The implementation uses `Commands::queue`, so the
    /// `BodyActionEvent::Add` write happens at `ApplyDeferred` time —
    /// not at the call site. With no extra ordering, the message is
    /// invisible to [`body_action_intake_system`] until the next
    /// tick. Callers that need the action to apply on the *current*
    /// tick must either:
    ///
    /// - write to `MessageWriter<BodyActionEvent>` directly (e.g. via
    ///   the [`add_body_action_via`] helper), which is immediate, or
    /// - order their writing system
    ///   `.before(body_action_intake_system)` so Bevy auto-inserts an
    ///   `ApplyDeferred` boundary that flushes the queued command
    ///   before intake walks the message buffer.
    fn add_body_action(&mut self, entity: Entity, action: BodyAction, name: Option<&str>);

    /// Queue a [`BodyAction`] against `entity` for planet `P`'s
    /// [`BodyActionsR<P>`] queue. Equivalent to sending a
    /// `BodyActionEvent::Add` constructed via
    /// [`BodyActionEvent::add_for::<P>`]. Same `Commands::queue`
    /// timing as [`add_body_action`](Self::add_body_action).
    fn add_body_action_for<P: Planet>(
        &mut self,
        entity: Entity,
        action: BodyAction,
        name: Option<&str>,
    );

    /// Cancel every pending body action whose name matches `name`.
    /// Equivalent to sending a [`BodyActionEvent::Remove`]. The
    /// cancel is planet-agnostic — every per-planet
    /// [`BodyActionsR<P>`] queue observes it on the same tick.
    ///
    /// Same `Commands::queue` deferral applies as
    /// [`add_body_action`](Self::add_body_action): the
    /// `BodyActionEvent::Remove` becomes visible at the next
    /// `ApplyDeferred` boundary, so a same-tick remove requires
    /// either a direct `MessageWriter` write or
    /// `.before(body_action_intake_system)` ordering on the calling
    /// system.
    fn remove_body_action(&mut self, name: &str);
}

impl<'w, 's> BodyActionCommandsExt for Commands<'w, 's> {
    fn add_body_action(&mut self, entity: Entity, action: BodyAction, name: Option<&str>) {
        self.add_body_action_for::<astrodyn::Earth>(entity, action, name);
    }

    fn add_body_action_for<P: Planet>(
        &mut self,
        entity: Entity,
        action: BodyAction,
        name: Option<&str>,
    ) {
        let name = name.map(|n| n.to_string());
        self.queue(move |world: &mut World| {
            // Fail loudly when planet `P`'s pipeline isn't registered:
            // the matching `body_action_intake_system::<P>` never runs
            // and the message would otherwise age out of the
            // double-buffer with no observable effect. The diagnostic
            // names the unregistered planet, the subject entity, and
            // the API call to wire it up.
            if let Some(registered) = world.get_resource::<RegisteredPlanetsR>() {
                assert!(
                    registered.contains::<P>(),
                    "BodyAction queued for planet `{planet}` against entity {entity:?} \
                     (action_name={name:?}) but no per-planet body-action pipeline is \
                     registered for that planet. The unified `Messages<BodyActionEvent>` \
                     buffer would never be drained for this `<P>` and the action would \
                     silently age out of the double-buffer. \
                     Fix: call `astrodyn_bevy::register_planet_systems::<{planet}>(&mut app)` \
                     during `App` setup (after `app.add_plugins(AstrodynPlugin)`), before \
                     queuing actions for `{planet}`-integrated bodies. \
                     `AstrodynPlugin::build` pre-registers `astrodyn::Earth`; additional \
                     planets must be registered explicitly.",
                    planet = std::any::type_name::<P>(),
                );
            }
            let mut writer = world.resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>();
            writer.write(BodyActionEvent::Add {
                entity,
                action,
                name,
                planet: TypeId::of::<P>(),
            });
        });
    }

    fn remove_body_action(&mut self, name: &str) {
        let name = name.to_string();
        self.queue(move |world: &mut World| {
            let mut writer = world.resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>();
            writer.write(BodyActionEvent::Remove { name });
        });
    }
}

/// Convenience: explicit `MessageWriter` overload of
/// [`BodyActionCommandsExt::add_body_action`] for systems that already
/// hold a writer (faster; avoids an indirect `Commands::queue`
/// closure).
///
/// Equivalent to `writer.write(BodyActionEvent::add(entity, action,
/// name))`. Provided here so the call site reads as
/// `add_body_action_via(...)` and matches the JEOD vocabulary even
/// when going through a writer.
///
/// Unlike [`BodyActionCommandsExt::add_body_action`], this writes the
/// message synchronously: the result is visible to any later-running
/// `MessageReader<BodyActionEvent>` (including
/// [`body_action_intake_system`]) within the same schedule pass — no
/// `ApplyDeferred` boundary is required.
#[inline]
pub fn add_body_action_via(
    writer: &mut MessageWriter<BodyActionEvent>,
    entity: Entity,
    action: BodyAction,
    name: Option<&str>,
) {
    writer.write(BodyActionEvent::add(entity, action, name));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{
        DynamicsConfigC, MassPropertiesC, RotationalStateC, TranslationalStateC,
    };
    use astrodyn::{
        DynamicsConfig, JeodQuat, MassProperties, OrbitalElementSet, OrbitalElements,
        RotationalState,
    };
    use glam::DVec3;

    fn build_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_message::<BodyActionEvent>();
        // The unit tests in this module exercise the Earth-tagged
        // queue path. Per-planet `BodyActionsR<P>` registration in
        // production lives in `AstrodynPlugin::build` (Earth) and
        // `register_planet_systems::<P>` (other planets); here we
        // wire the Earth instantiation manually to keep the unit
        // tests free of the full plugin scaffolding.
        app.init_resource::<BodyActionsR<astrodyn::Earth>>();
        app.add_systems(
            Update,
            (
                body_action_intake_system::<astrodyn::Earth>,
                body_action_system::<astrodyn::Earth>,
            )
                .chain(),
        );
        app
    }

    fn spawn_vehicle(app: &mut App) -> Entity {
        app.world_mut()
            .spawn((
                TranslationalStateC::<astrodyn::Earth>::default(),
                RotationalStateC::default(),
                MassPropertiesC::from(crate::typed_bridge::mass_raw_to_self_ref(
                    &(MassProperties::new(400_000.0)),
                )),
                // `body_action_system` filters by `With<DynamicsConfigC>`;
                // a real vehicle entity always carries this Component.
                DynamicsConfigC(DynamicsConfig {
                    translational_dynamics: true,
                    rotational_dynamics: true,
                    three_dof: false,
                }),
            ))
            .id()
    }

    fn write_msg(app: &mut App, msg: BodyActionEvent) {
        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
            .write(msg);
    }

    #[test]
    fn add_then_remove_before_apply_skips_action() {
        // Mirrors JEOD `mass.py`:
        //   add(mass_init with mass=400000)
        //   remove("vehicle.mass_init")
        //   add(mass_init with mass=100000)
        // Final mass should be 100000.
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);

        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(400_000.0),
                },
                Some("vehicle.mass_init"),
            ),
        );
        write_msg(&mut app, BodyActionEvent::remove("vehicle.mass_init"));
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(100_000.0),
                },
                Some("vehicle.mass_init"),
            ),
        );
        app.update();

        let final_mass = crate::typed_bridge::mass_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<MassPropertiesC>()
                .expect("mass props present")
                .0,
        )
        .mass;
        assert_eq!(final_mass, 100_000.0);
    }

    #[test]
    fn rot_init_writes_rotational_state() {
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);
        let q = JeodQuat::identity();
        let omega = DVec3::new(0.0, 0.0, 0.01);
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitRot {
                    quaternion: q,
                    ang_vel_body: omega,
                },
                None,
            ),
        );
        app.update();
        let state: RotationalState = crate::typed_bridge::rot_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<RotationalStateC>()
                .expect("rot state present")
                .0,
        );
        assert_eq!(state.quaternion, q);
        assert_eq!(state.ang_vel_body, omega);
    }

    #[test]
    fn trans_orbital_init_writes_translational_state() {
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);
        const MU: f64 = 3.986_004_415e14;
        let mut elements = OrbitalElements::default();
        elements.semi_major_axis = 7.0e6;
        elements.e_mag = 0.001;
        elements.inclination = 51.6_f64.to_radians();
        elements.long_asc_node = 0.1;
        elements.arg_periapsis = 0.2;
        elements.true_anom = 0.3;

        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitTransOrbital {
                    set: OrbitalElementSet::SmaEccIncAscnodeArgperTanom,
                    elements,
                    time_periapsis: 0.0,
                    mu: MU,
                },
                None,
            ),
        );
        app.update();
        let trans = crate::typed_bridge::trans_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<TranslationalStateC<astrodyn::Earth>>()
                .expect("trans state present")
                .0,
        );
        assert!(trans.position.length() > 1.0e6);
        assert!(trans.velocity.length() > 1.0);
    }

    #[test]
    fn commands_extension_add_then_remove() {
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);

        // Use the Commands extension to queue + remove + queue.
        fn queue_actions(In(entity): In<Entity>, mut commands: Commands) {
            commands.add_body_action(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(400_000.0),
                },
                Some("vehicle.mass_init"),
            );
            commands.remove_body_action("vehicle.mass_init");
            commands.add_body_action(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(100_000.0),
                },
                Some("vehicle.mass_init"),
            );
        }

        app.world_mut()
            .run_system_cached_with(queue_actions, entity)
            .expect("run_system_cached_with");
        app.update();

        let final_mass = crate::typed_bridge::mass_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<MassPropertiesC>()
                .expect("mass props present")
                .0,
        )
        .mass;
        assert_eq!(final_mass, 100_000.0);
    }

    #[test]
    fn anonymous_action_cannot_be_removed_by_name() {
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(123.0),
                },
                None,
            ),
        );
        write_msg(&mut app, BodyActionEvent::remove("anything"));
        app.update();

        let final_mass = crate::typed_bridge::mass_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<MassPropertiesC>()
                .expect("mass props present")
                .0,
        )
        .mass;
        assert_eq!(final_mass, 123.0);
    }

    #[test]
    fn two_writes_with_same_name_apply_in_order() {
        // Two `add`s sharing a name with no intervening `remove`
        // both fire in FIFO order. JEOD's
        // `DynManager::add_body_action` checks for *pointer*
        // duplicates, not name duplicates, so two distinct actions
        // sharing a name are legal. Last-write-wins on the mutated
        // component.
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(11.0),
                },
                Some("dup"),
            ),
        );
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(22.0),
                },
                Some("dup"),
            ),
        );
        app.update();
        let final_mass = crate::typed_bridge::mass_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<MassPropertiesC>()
                .expect("mass props present")
                .0,
        )
        .mass;
        assert_eq!(final_mass, 22.0);
    }

    #[test]
    fn remove_drops_all_pending_with_matching_name() {
        // `remove` drops *every* pending action with the name —
        // mirrors JEOD's linear-scan-by-name (`dyn_manager.cc:211`)
        // generalised so two adds with the same name both clear in
        // a single remove.
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(11.0),
                },
                Some("dup"),
            ),
        );
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(22.0),
                },
                Some("dup"),
            ),
        );
        write_msg(&mut app, BodyActionEvent::remove("dup"));
        app.update();
        // Neither add fired: the entity still has its original
        // 400 000 kg from `spawn_vehicle`.
        let final_mass = crate::typed_bridge::mass_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<MassPropertiesC>()
                .expect("mass props present")
                .0,
        )
        .mass;
        assert_eq!(final_mass, 400_000.0);
    }

    #[test]
    fn empty_name_remove_is_noop() {
        // JEOD `dyn_manager.cc:207-209`: `remove_body_action("")`
        // returns immediately. The Bevy adapter must keep that
        // contract — otherwise a stray `Remove { name: "" }` would
        // wipe every anonymous pending action whose name happens to
        // be the empty string. This test queues two named adds, then
        // sends a `Remove { name: "" }`; both adds must still fire.
        let mut app = build_app();
        let entity = spawn_vehicle(&mut app);
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(11.0),
                },
                // Name is `""` — an explicitly empty (not `None`) name
                // is the case JEOD's empty-string short-circuit
                // protects against.
                Some(""),
            ),
        );
        write_msg(
            &mut app,
            BodyActionEvent::add(
                entity,
                BodyAction::InitMass {
                    mass: MassProperties::new(22.0),
                },
                Some(""),
            ),
        );
        write_msg(&mut app, BodyActionEvent::remove(""));
        app.update();
        // Both adds fired in FIFO order, last-write-wins on the mass.
        // If the empty-name `remove` had iterated `retain` it would
        // have cleared both pending entries and the mass would still
        // be the spawn-time 400 000.
        let final_mass = crate::typed_bridge::mass_typed_to_raw(
            &app.world()
                .entity(entity)
                .get::<MassPropertiesC>()
                .expect("mass props present")
                .0,
        )
        .mass;
        assert_eq!(final_mass, 22.0);
    }
}
