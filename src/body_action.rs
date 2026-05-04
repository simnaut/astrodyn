//! Bevy adapter for [`jeod_sim::BodyAction`]: queue body actions
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
//! # Lifecycle and naming
//!
//! - Adding an action with a `name` registers it for later removal.
//!   Adding two actions with the same name is allowed; both fire in
//!   FIFO order if neither is removed first. Removing a name removes
//!   *every* still-pending action with that name, mirroring JEOD's
//!   `remove_body_action` linear-scan-by-name (see
//!   `dyn_manager.cc:211`).
//! - Adding an action without a name (`name = None`) makes it
//!   anonymous: it cannot be removed by name; it always fires once
//!   when ready and is dropped.
//! - `BodyActionEvent::Remove { name: "" }` is a no-op, matching
//!   JEOD's empty-string short-circuit (`dyn_manager.cc:207-209`).
//!   Any pending action whose name is `Some("")` survives the empty-
//!   name remove.
//! - An action that is added then removed before
//!   [`body_action_system`] runs in the same tick is never applied.
//!   This is the "remove-then-readd" idiom that JEOD's
//!   `SIM_removable_body_action`'s `mass.py` exercises.
//!
//! # Scheduling
//!
//! [`body_action_system`] is wired by [`crate::JeodPlugin`] to run in
//! the `FixedUpdate` schedule between [`crate::JeodSet::TimeUpdate`]
//! and [`crate::JeodSet::EphemerisUpdate`]. That ordering matches
//! JEOD: actions resolve before ephemeris / gravity / integration
//! consume the new state.
//!
//! Both [`body_action_system`] and [`crate::mass_update_system`] live
//! in that same TimeUpdate→EphemerisUpdate gap, so the plugin pins
//! `body_action_system` `.before(mass_update_system)` explicitly.
//! That makes a queued [`jeod_sim::BodyAction::InitMass`] land its
//! mass replacement (with the `dirty` flag set by this system after
//! the assignment) *before* the same-tick recompute walks the dirty
//! flag — so the inverse-mass / inverse-inertia caches are refreshed
//! before any consumer in `EphemerisUpdate` / `Environment` /
//! `Interaction` reads them.

use bevy::ecs::message::MessageWriter;
use bevy::prelude::*;
use jeod_sim::BodyAction;

use crate::components::{
    Abm4StateC, DynamicsConfigC, GaussJacksonStateC, MassPropertiesC, RotationalStateC,
    TranslationalStateC,
};

/// One pending body action awaiting execution.
///
/// Carried by [`BodyActionsR`] and constructed by the plugin's
/// message-draining system. Mission code does not interact with this
/// type directly; either send a [`BodyActionEvent`] or call
/// [`BodyActionCommandsExt::add_body_action`].
#[derive(Debug, Clone)]
pub struct PendingBodyAction {
    /// Subject entity the action will mutate.
    pub entity: Entity,
    /// The action itself.
    pub action: BodyAction,
    /// Optional name used by [`BodyActionEvent::Remove`] to find
    /// this pending action before it fires. JEOD's
    /// `BodyAction::action_name` is also optional; mission code that
    /// never needs to remove an action mid-flight can leave it
    /// `None`.
    pub name: Option<String>,
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
/// Convenience constructors [`Self::add`] / [`Self::remove`] keep
/// call sites short. The Bevy adapter further provides
/// [`BodyActionCommandsExt`] on `Commands` for the same vocabulary.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::body_action::BodyActionEvent;
/// use jeod_sim::{BodyAction, MassProperties};
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
    },
    /// Mirror of `DynManager::remove_body_action(const std::string&)`:
    /// drop *every* still-pending action with this name. Matches
    /// JEOD's linear-scan-by-name behaviour (`dyn_manager.cc:211`),
    /// generalised so two unresolved adds with the same name both
    /// drop in one remove.
    Remove {
        /// Name to cancel.
        name: String,
    },
}

impl BodyActionEvent {
    /// Construct a [`BodyActionEvent::Add`].
    #[inline]
    pub fn add(entity: Entity, action: BodyAction, name: Option<&str>) -> Self {
        BodyActionEvent::Add {
            entity,
            action,
            name: name.map(|n| n.to_string()),
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

/// Bevy `Resource` holding all pending body actions, in insertion
/// order.
///
/// Inserted by [`crate::JeodPlugin`]. Mission code does not need to
/// touch this resource directly — send a [`BodyActionEvent`] or
/// use [`BodyActionCommandsExt`] instead.
///
/// JEOD analog: `DynManager::body_actions` (a `std::vector<BodyAction
/// *>` walked once per `perform_actions` pass).
#[derive(Resource, Default, Debug)]
pub struct BodyActionsR {
    /// FIFO queue of pending actions.
    pub pending: Vec<PendingBodyAction>,
}

/// Drains [`BodyActionEvent`]s into [`BodyActionsR`].
///
/// Runs strictly before [`body_action_system`] each tick so that an
/// `add → remove → add` sequence within one tick collapses to a
/// single queued action (the same idiom JEOD's `mass.py` from
/// `SIM_removable_body_action` exercises at init time). Bevy
/// preserves message arrival order within one message type, so the
/// unified [`BodyActionEvent`] enum's `MessageReader` walks the
/// add/remove operations in the order their `MessageWriter`s wrote
/// them.
pub fn body_action_intake_system(
    mut messages: bevy::ecs::message::MessageReader<BodyActionEvent>,
    mut queue: ResMut<BodyActionsR>,
) {
    for msg in messages.read() {
        match msg {
            BodyActionEvent::Add {
                entity,
                action,
                name,
            } => {
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

/// Apply every ready [`PendingBodyAction`] to its subject entity,
/// removing applied actions from the queue.
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
///   an out-of-sync `inverse_mass`.
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
pub fn body_action_system(
    mut queue: ResMut<BodyActionsR>,
    mut bodies: Query<
        (
            Option<&mut TranslationalStateC>,
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
        // (mirrors the IG.37 attach/detach reset path).
        let mut state_mutated = false;
        if let Some(state) = action.action.apply_translational() {
            let comp = trans
                .as_deref_mut()
                .unwrap_or_else(|| {
                    panic!(
                        "BodyAction targets translational state on entity {:?} (action_name={:?}) but the entity has no TranslationalStateC. \
                         Add `TranslationalStateC::default()` to the entity (or spawn via `VehicleConfig::spawn_bevy`) before queuing this action.",
                        action.entity, action.name,
                    )
                });
            // allowed: action-fire boundary — `BodyAction::apply_translational` returns the
            // ECS-agnostic `TranslationalState` (the kernels in `jeod_dynamics::body_init`
            // are untyped). Lifting back to the typed `<PlanetInertial<SelfPlanet>>` storage
            // is a one-time relabel at the action-fire boundary, identical in shape to the
            // `VehicleConfig::spawn_bevy` initial-state lift in `lib.rs`. Not a per-step
            // bypass.
            comp.0 = jeod_sim::TranslationalStateTyped::from_untyped_unchecked(&state);
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
            // allowed: same action-fire boundary as the translational branch above.
            // `RotationalState` is the ECS-agnostic untyped form; the typed
            // `<SelfRef>` re-tag here is a one-time witness reinstatement at the
            // mutation site, not a per-step lift.
            comp.0 = jeod_sim::RotationalStateTyped::from_untyped_unchecked(&state);
            state_mutated = true;
        }
        if let Some(props) = action.action.apply_mass() {
            let comp = mass
                .as_deref_mut()
                .unwrap_or_else(|| {
                    panic!(
                        "BodyAction targets mass properties on entity {:?} (action_name={:?}) but the entity has no MassPropertiesC. \
                         Add `MassPropertiesC::from(MassProperties::new(...))` to the entity before queuing this action.",
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
            // allowed: action-fire boundary — `MassProperties` is the ECS-agnostic
            // untyped form returned by `BodyAction::apply_mass`. The
            // `MassPropertiesTyped<SelfRef>` re-tag is a one-time relabel matching
            // the `MassPropertiesC::from(MassProperties)` pattern used at spawn
            // time. Not a per-step bypass.
            comp.0 = jeod_sim::MassPropertiesTyped::from_untyped_unchecked(&props);
            comp.0.dirty = true;
        }
        if state_mutated {
            // JEOD_INV: IG.37 — multi-step integrator history must be reset on
            // any mid-sim state change. JEOD's `dyn_body_init_*` actions
            // overwrite a body's translational / rotational state mid-run,
            // and (per JEOD's attach/detach analog) leaving Gauss–Jackson /
            // ABM4 predictor history pointing at the prior state corrupts
            // the next integrate. The reset is a no-op for single-step
            // integrators (`gj` / `abm` will be `None` on RK4 entities),
            // so this branch is free for the common path.
            jeod_sim::reset_integrators(
                gj.as_deref_mut().map(|c| &mut c.0),
                abm.as_deref_mut().map(|c| &mut c.0),
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
pub trait BodyActionCommandsExt {
    /// Queue a [`BodyAction`] against `entity`. Equivalent to
    /// sending a [`BodyActionEvent::Add`].
    fn add_body_action(&mut self, entity: Entity, action: BodyAction, name: Option<&str>);

    /// Cancel every pending body action whose name matches `name`.
    /// Equivalent to sending a [`BodyActionEvent::Remove`].
    fn remove_body_action(&mut self, name: &str);
}

impl<'w, 's> BodyActionCommandsExt for Commands<'w, 's> {
    fn add_body_action(&mut self, entity: Entity, action: BodyAction, name: Option<&str>) {
        let name = name.map(|n| n.to_string());
        self.queue(move |world: &mut World| {
            let mut writer = world.resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>();
            writer.write(BodyActionEvent::Add {
                entity,
                action,
                name,
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
    use glam::DVec3;
    use jeod_sim::{
        DynamicsConfig, JeodQuat, MassProperties, OrbitalElementSet, OrbitalElements,
        RotationalState,
    };

    fn build_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_message::<BodyActionEvent>();
        app.init_resource::<BodyActionsR>();
        app.add_systems(
            Update,
            (body_action_intake_system, body_action_system).chain(),
        );
        app
    }

    fn spawn_vehicle(app: &mut App) -> Entity {
        app.world_mut()
            .spawn((
                TranslationalStateC::default(),
                RotationalStateC::default(),
                MassPropertiesC::from(MassProperties::new(400_000.0)),
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

        let final_mass = app
            .world()
            .entity(entity)
            .get::<MassPropertiesC>()
            .expect("mass props present")
            .0
            .to_untyped()
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
        let state: RotationalState = app
            .world()
            .entity(entity)
            .get::<RotationalStateC>()
            .expect("rot state present")
            .0
            .to_untyped();
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
        let trans = app
            .world()
            .entity(entity)
            .get::<TranslationalStateC>()
            .expect("trans state present")
            .0
            .to_untyped();
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

        let final_mass = app
            .world()
            .entity(entity)
            .get::<MassPropertiesC>()
            .expect("mass props present")
            .0
            .to_untyped()
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

        let final_mass = app
            .world()
            .entity(entity)
            .get::<MassPropertiesC>()
            .expect("mass props present")
            .0
            .to_untyped()
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
        let final_mass = app
            .world()
            .entity(entity)
            .get::<MassPropertiesC>()
            .expect("mass props present")
            .0
            .to_untyped()
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
        let final_mass = app
            .world()
            .entity(entity)
            .get::<MassPropertiesC>()
            .expect("mass props present")
            .0
            .to_untyped()
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
        let final_mass = app
            .world()
            .entity(entity)
            .get::<MassPropertiesC>()
            .expect("mass props present")
            .0
            .to_untyped()
            .mass;
        assert_eq!(final_mass, 22.0);
    }
}
