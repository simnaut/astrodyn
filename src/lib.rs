#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod body_action;
pub mod bundles;
pub mod components;
pub mod frame_attach_system;
pub mod frame_param;
pub mod kinematic_propagation;
pub mod mass_tree;
pub mod prelude;
pub mod recipes;
pub mod sets;
pub mod source_mutator;
pub mod systems;
pub mod validation;
pub mod wrench;

pub use body_action::{
    add_body_action_via, body_action_intake_system, body_action_system,
    body_action_unregistered_planet_fence_system, BodyActionCommandsExt, BodyActionEvent,
    BodyActionsR, RegisteredPlanetsR,
};
pub use bundles::*;
pub use components::*;
pub use frame_attach_system::{
    frame_attach_system, propagate_frame_attached_state_post_integration_system,
    propagate_frame_attached_state_system,
};
pub use kinematic_propagation::{
    propagate_state_from_root_post_integration_system, propagate_state_from_root_system,
};
pub use mass_tree::{composite_mass_system, MassTreeQueries, MassTreeView};
pub use sets::*;
pub use source_mutator::{SourceMutator, SourceReader};
pub use systems::*;
pub use wrench::wrench_aggregation_system;

use bevy::prelude::*;

// Re-export jeod_sim types that form the public atmosphere API.
pub use jeod_sim::atmosphere::{AtmosphereConfig, AtmosphereModel};

/// Bevy resource wrapping `SimulationTime`.
// JEOD_INV: TM.07 — JEOD uses -1.0 sentinel; we call recompute_derived() at construction instead
#[derive(Resource, Debug, Deref, DerefMut)]
pub struct SimulationTimeR(pub jeod_sim::SimulationTime);

impl Default for SimulationTimeR {
    fn default() -> Self {
        Self(jeod_sim::SimulationTime::at_j2000(
            jeod_sim::default_leap_second_table(),
        ))
    }
}

/// Optional Bevy resource for polar motion (xp, yp) in radians.
///
/// When inserted, the `planet_fixed_rotation_system` includes polar motion
/// in the RNP composition: W(xp,yp) × R(GAST) × N × P.
/// When absent, polar motion is omitted (equivalent to `enable_polar=false`).
///
/// For time-varying polar motion, update this resource each step from
/// IERS EOP data.
#[derive(Resource, Debug, Clone, Copy)]
pub struct PolarMotionR {
    /// Polar motion x_p in radians.
    pub xp: f64,
    /// Polar motion y_p in radians.
    pub yp: f64,
}

/// Bevy resource wrapping [`AtmosphereConfig`] with an entity reference for
/// the planet whose rotation matrix is used for geodetic conversion.
#[derive(Resource, Debug, Clone)]
pub struct AtmosphereModelR {
    /// ECS-agnostic atmosphere configuration (model, radii, wind).
    pub config: AtmosphereConfig,
    /// Entity of the planet whose `PlanetFixedRotationC` is used.
    /// `None` means no rotation (position assumed planet-fixed).
    pub planet_entity: Option<Entity>,
}

/// Bevy resource wrapping [`jeod_sim::Ephemeris`] for DE4xx ephemeris access.
///
/// When inserted, `planet_fixed_rotation_system` can use `MoonDE421` rotation
/// and `ephemeris_update_system` can update source positions from DE421/DE440.
#[derive(Resource, Deref, DerefMut)]
pub struct EphemerisR(pub jeod_sim::Ephemeris);

/// Bevy resource wrapping `MassTree` for multi-body vehicles.
///
/// Shared by all entities that have [`components::MassBodyIdC`].
/// The `staging_system` processes [`components::AttachEvent`] and
/// [`components::DetachEvent`] to modify the tree and sync
/// composite mass properties back to affected entities.
///
/// This resource is not inserted automatically by [`JeodPlugin`]. Applications
/// that use staging must insert `MassTreeR` before sending
/// [`components::AttachEvent`] or
/// [`components::DetachEvent`]. If the resource is absent, staging
/// events are silently drained.
#[derive(Resource, Deref, DerefMut)]
pub struct MassTreeR(pub jeod_sim::MassTree);

/// Bevy resource holding the [`Entity`] of the root frame entity in
/// the ECS-native frame hierarchy. The root frame entity carries
/// [`components::FrameTransC`] / [`components::FrameRotC`] /
/// [`components::FrameAngVelC`] at identity and is the parent of every
/// source's inertial frame entity (which is the parent of every body
/// or pfix frame entity, and so on). Spawned by [`JeodPlugin::build`]
/// before any source/body registration so the registration systems can
/// `ChildOf`-link their frame entities to it. Mission code reads
/// cross-frame state via [`crate::frame_param::RelativeFrameState`]
/// and [`crate::frame_param::FrameOrigin`].
#[derive(Resource, Debug, Clone, Copy, Deref, DerefMut)]
pub struct RootFrameEntityR(pub Entity);

/// Unified JEOD plugin — registers all pipeline systems and schedule sets.
///
/// The seven [`JeodSet`] pipeline stages run in Bevy's `FixedUpdate`
/// schedule, which acts as a single JEOD-style integration group: every
/// body matched by the integrating systems advances together at the
/// schedule's shared `dt`. (Auxiliary registration systems —
/// `register_source_frames_system` / `register_body_frames_system` — also
/// run in `Startup` and `PreUpdate` to catch late-spawned entities; they
/// no-op for already-registered ones.) Multi-stage integrators (RK4, etc.)
/// loop internally inside [`JeodSet::Integration`] — they do *not*
/// trigger multiple schedule passes. See the [`sets`] module docs for
/// the full mapping and the recipe for scenarios that need separate
/// integration groups.
pub struct JeodPlugin;

impl Plugin for JeodPlugin {
    fn build(&self, app: &mut App) {
        // ── Schedule set ordering ──
        // JEOD_INV: DM.04 — init order: time -> ephemeris -> environment -> interaction -> forces -> integration -> derived
        // JEOD_INV: DM.13 — ephemeris updated before gravity (EphemerisUpdate before Environment)
        app.configure_sets(
            FixedUpdate,
            (
                JeodSet::TimeUpdate,
                JeodSet::EphemerisUpdate.after(JeodSet::TimeUpdate),
                JeodSet::Environment.after(JeodSet::EphemerisUpdate),
                JeodSet::Interaction.after(JeodSet::Environment),
                JeodSet::ForceCollection.after(JeodSet::Interaction),
                JeodSet::Integration.after(JeodSet::ForceCollection),
                JeodSet::DerivedState.after(JeodSet::Integration),
            ),
        );

        // ── Resources ──
        app.init_resource::<SimulationTimeR>();

        // ── ECS-native root frame entity ──
        // Spawn the root frame entity. Source / body registration
        // `ChildOf`-links its frame entities under this one; the ECS
        // hierarchy is the single source of truth for all frame-tree
        // state.
        //
        // Only spawn when the caller hasn't pre-installed
        // `RootFrameEntityR`. A mission (or a second `JeodPlugin::build`
        // call — `Plugin::build` is not idempotent on its own) cannot
        // silently leak the previously-spawned root entity and
        // re-parent future frame entities under a different root than
        // the existing ones. When pre-installed we validate that the
        // referenced entity still exists and carries the required
        // frame components / `InertialFrameMarker` so source / body
        // registration's `ChildOf`-links and the typed
        // `<RootInertial>` assumptions hold. Per the "Fail Loudly"
        // rule a stale or wrong-kind pre-installed entity panics with
        // a diagnostic that names the broken assumption and tells the
        // caller how to fix it.
        if !app.world().contains_resource::<RootFrameEntityR>() {
            let root_frame_entity = app
                .world_mut()
                .spawn((
                    Name::new("root.frame"),
                    components::InertialFrameMarker,
                    components::FrameTransC::default(),
                    components::FrameRotC::default(),
                    components::FrameAngVelC::default(),
                ))
                .id();
            app.insert_resource(RootFrameEntityR(root_frame_entity));
        } else {
            let root_frame_entity = app.world().resource::<RootFrameEntityR>().0;
            assert!(
                app.world().get_entity(root_frame_entity).is_ok(),
                "JeodPlugin: pre-installed RootFrameEntityR ({root_frame_entity:?}) \
                 references an entity that no longer exists in the world. Source / \
                 body registration will `ChildOf`-link new frame entities under \
                 this dangling reference and panic later. Insert the resource only \
                 after spawning the root frame entity in the same `App`, or remove \
                 the pre-installation and let JeodPlugin own root-frame creation.",
            );
            assert!(
                app.world()
                    .entity(root_frame_entity)
                    .contains::<components::InertialFrameMarker>(),
                "JeodPlugin: pre-installed RootFrameEntityR ({root_frame_entity:?}) \
                 is missing `InertialFrameMarker`. The plugin assumes the root \
                 frame is inertial — source / body registration tags new children \
                 with `InertialFrameMarker` and the typed Bevy components \
                 (`Position<RootInertial>`, \
                 `TranslationalStateC<P>` storing `<PlanetInertial<P>>`) \
                 are all phantom-tagged for an inertial root. Add \
                 `InertialFrameMarker` to the entity, or let JeodPlugin spawn the \
                 root frame.",
            );
            assert!(
                app.world()
                    .entity(root_frame_entity)
                    .contains::<components::FrameTransC>()
                    && app
                        .world()
                        .entity(root_frame_entity)
                        .contains::<components::FrameRotC>()
                    && app
                        .world()
                        .entity(root_frame_entity)
                        .contains::<components::FrameAngVelC>(),
                "JeodPlugin: pre-installed RootFrameEntityR ({root_frame_entity:?}) \
                 is missing one or more of the required frame components \
                 (`FrameTransC`, `FrameRotC`, `FrameAngVelC`). Frame-tree \
                 consumers read these directly from the root entity. Insert all \
                 three (each with `Default::default()` for an inertial root), or \
                 let JeodPlugin spawn the root frame.",
            );
        }

        // ── Events ──
        // The Bevy adapter registers the canonical runtime-resolved
        // `AttachEvent<SelfRef, SelfRef>`. Mission code that mints
        // concrete vehicle phantoms (e.g. `define_vehicle!(Iss)` /
        // `define_vehicle!(Soyuz)`) and wants typed
        // `AttachEvent<Iss, Soyuz>` pumping must register the matching
        // `add_message::<AttachEvent<Iss, Soyuz>>()` itself; the
        // canonical `staging_system` reads the `<SelfRef, SelfRef>`
        // instantiation only.
        app.add_message::<AttachEvent<jeod_sim::SelfRef, jeod_sim::SelfRef>>();
        app.add_message::<DetachEvent>();
        app.add_message::<FrameAttachEvent>();
        app.add_message::<FrameDetachEvent>();
        // Body-action lifecycle: callers add / remove
        // body-action requests through this single message type or
        // through `BodyActionCommandsExt` on `Commands`. The
        // per-planet intake system drains the matching tagged adds
        // into `BodyActionsR<P>`; the per-planet apply system walks
        // that resource and mutates ready actions' subjects. Earth's
        // queue is registered here for single-planet missions;
        // additional planets call `register_planet_systems::<P>`.
        app.add_message::<body_action::BodyActionEvent>();
        app.init_resource::<body_action::BodyActionsR<jeod_sim::Earth>>();
        // Track which planets have a per-planet body-action pipeline
        // registered. Earth is wired below by `JeodPlugin::build`;
        // additional planets call `register_planet_systems::<P>`,
        // which inserts `TypeId::of::<P>()` into this resource. The
        // `BodyActionEvent::Add` writer surfaces consult this set to
        // refuse a planet-tagged add whose intake pipeline isn't
        // wired, panicking with a diagnostic that names the
        // unregistered planet rather than letting the message age out
        // of the double-buffer silently (Fail-Loudly).
        app.init_resource::<body_action::RegisteredPlanetsR>();
        app.world_mut()
            .resource_mut::<body_action::RegisteredPlanetsR>()
            .register::<jeod_sim::Earth>();

        // ── Systems ──
        // Source-frame registration runs at Startup to spawn the ECS
        // frame entity for every spawned source, and again before each
        // FixedUpdate's EphemerisUpdate to catch late-spawned sources.
        // The latter filters by `Without<FrameEntityC>` so
        // already-registered sources are skipped — registering is
        // one-time per source. Body-frame registration follows so
        // bodies can resolve `IntegSourceC(Some(source_entity))`
        // against an already-registered source.
        //
        // Registration is wired into three schedules so it catches every
        // spawn surface:
        //   - Startup: initial spawns before any tick.
        //   - PreUpdate: catches entities spawned during the previous
        //     frame's `Update` / `PostUpdate`. They are registered before
        //     the *next* frame's `Update` runs. Same-frame spawn-and-
        //     mutate inside one `Update` (spawn + `SourceMutator` call in
        //     consecutive systems of the same frame) is *not* supported
        //     by this scheduling, since `Update` runs after `PreUpdate`;
        //     callers needing that pattern must add a manual
        //     registration call in `Update` with explicit ordering.
        //   - Before `JeodSet::EphemerisUpdate` (FixedUpdate): catches
        //     entities spawned between fixed ticks before they hit the
        //     ephemeris / rotation / integration pipeline.
        // Each pass is a no-op for already-registered entities (the
        // `Without<FrameEntityC>` / `Without<PfixFrameEntityC>` filters
        // make repeated runs cost a single query iteration).
        // `register_pfix_frames_system` covers a rare but real case:
        // a source spawned without `PlanetFixedRotationC` that gains
        // it after the initial registration. The main
        // `register_source_frames_system` filters by
        // `Without<FrameEntityC>` so it can't observe that mutation;
        // the dedicated pfix pass uses `Without<PfixFrameEntityC>` +
        // `With<PlanetFixedRotationC>` instead.
        app.add_systems(
            Startup,
            (
                systems::register_source_frames_system::<jeod_sim::Earth>,
                systems::register_pfix_frames_system::<jeod_sim::Earth>
                    .after(systems::register_source_frames_system::<jeod_sim::Earth>),
                systems::register_body_frames_system::<jeod_sim::Earth>
                    .after(systems::register_pfix_frames_system::<jeod_sim::Earth>),
                // Maintain `MassPointRef` ↔ `MassPropertiesC` invariant
                // for bodies that gain or lose mass after the one-time
                // body-frame registration pass.
                systems::sync_body_mass_point_ref_system
                    .after(systems::register_body_frames_system::<jeod_sim::Earth>),
                // Body-action systems are intentionally NOT registered
                // in `Startup`. Bevy gives every system instance an
                // independent `Local<MessageCursor<BodyActionEvent>>`
                // (one per registration site), so registering the same
                // intake function in both `Startup` and `FixedUpdate`
                // would let messages written before the first
                // `app.update()` be observed once by each cursor — an
                // anonymous fire-once `BodyActionEvent::Add` would
                // therefore apply twice (once at the end of `Startup`,
                // again on the first `FixedUpdate` tick after Bevy's
                // double-buffer aging keeps the message live). Pinning
                // the only intake / apply registration to `FixedUpdate`
                // makes the cursor singular and the action-fire count
                // exact. Init-time messages still land before any
                // pipeline consumer reads the body's mutable state:
                // Bevy's double-buffered `Messages` keeps Startup-era
                // writes alive across the buffer swap that happens in
                // `First`, so the FixedUpdate intake on the first tick
                // observes them and applies before
                // `JeodSet::EphemerisUpdate`.
            ),
        );
        // Reject configs where a single frame entity carries multiple
        // kinematic-spec components: the four driver systems use
        // `Without<...>` filters to advertise pairwise-disjoint
        // queries to the scheduler, so an entity with two specs would
        // be silently dropped from every driver and propagate stale
        // `FrameRotC` / `FrameAngVelC` instead of panicking. The
        // fail-loud rule forbids that silent path.
        //
        // Two layers of enforcement:
        //
        // 1. **Component `on_insert` hooks** —
        //    `register_joint_kinematics_exclusivity_hooks` installs a
        //    Bevy lifecycle hook on each of the four spec components
        //    that fires the moment an insertion lands a second spec
        //    on an entity, regardless of which schedule the insert
        //    happened in (`Startup`, `Update`, `FixedUpdate`, an
        //    observer, …). This is the primary guard: it catches
        //    runtime spawns / inserts that the `PostStartup`
        //    validator never sees.
        //
        // 2. **`PostStartup` validator** — defense in depth, kept so
        //    a startup-time misconfiguration panics with a single
        //    aggregated message that lists every offending entity at
        //    once (the per-insert hooks panic on the *first* offender
        //    they observe, which is the right shape for runtime but
        //    less helpful when several stacked-spec entities are
        //    declared together at startup). Wired into `PostStartup`
        //    (not `Startup`) so it observes commands flushed at the
        //    end of `Startup`.
        systems::register_joint_kinematics_exclusivity_hooks(app);
        app.add_systems(PostStartup, systems::validate_joint_kinematics_exclusivity);
        app.add_systems(
            PreUpdate,
            (
                systems::register_source_frames_system::<jeod_sim::Earth>,
                systems::register_pfix_frames_system::<jeod_sim::Earth>
                    .after(systems::register_source_frames_system::<jeod_sim::Earth>),
                systems::register_body_frames_system::<jeod_sim::Earth>
                    .after(systems::register_pfix_frames_system::<jeod_sim::Earth>),
                systems::sync_body_mass_point_ref_system
                    .after(systems::register_body_frames_system::<jeod_sim::Earth>),
            ),
        );
        // ECS frame-entity cleanup on owner despawn: the registration
        // sites in `register_source_frames_system` /
        // `register_body_frames_system` (and the pfix branch of the
        // former / `register_pfix_frames_system`) spawn frame entities
        // that must be despawned when the owning source / body / pfix
        // entity is gone. The retired-pfix observer takes care of the
        // orphan stashed by a `RotationModel::None` toggle that wasn't
        // followed by a re-toggle before despawn.
        app.add_observer(systems::on_retired_pfix_frame_entity_despawn);
        app.add_observer(systems::on_frame_entity_despawn);
        app.add_observer(systems::on_source_pfix_frame_entity_despawn);
        // Split into two add_systems calls to stay within Bevy's tuple size limit.
        app.add_systems(
            FixedUpdate,
            (
                // Time advance
                systems::time_advance_system.in_set(JeodSet::TimeUpdate),
                // Catch dynamically-spawned sources before they hit
                // `planet_fixed_rotation_system` / `ephemeris_update_system`.
                systems::register_source_frames_system::<jeod_sim::Earth>
                    .before(JeodSet::EphemerisUpdate),
                // Late-attached `PlanetFixedRotationC` → pfix child node
                // (see `register_pfix_frames_system` doc).
                systems::register_pfix_frames_system::<jeod_sim::Earth>
                    .after(systems::register_source_frames_system::<jeod_sim::Earth>)
                    .before(JeodSet::EphemerisUpdate),
                // Catch dynamically-spawned bodies (after source registration so
                // any IntegSourceC reference resolves to a registered source).
                systems::register_body_frames_system::<jeod_sim::Earth>
                    .after(systems::register_pfix_frames_system::<jeod_sim::Earth>)
                    .before(JeodSet::EphemerisUpdate),
                // Late-acquired / late-lost `MassPropertiesC` →
                // insert / remove `MassPointRef` for bodies that have
                // already passed through `register_body_frames_system`.
                systems::sync_body_mass_point_ref_system
                    .after(systems::register_body_frames_system::<jeod_sim::Earth>)
                    .before(JeodSet::EphemerisUpdate),
                // Validation runs *after* registration but before any
                // pipeline consumer touches the new components. The
                // frame-switch / non-root checks walk `Query<&ChildOf>`
                // on the body's `FrameEntityC` and read the source's
                // `FrameEntityC` (both populated by the `register_*`
                // systems above). Pinning validation to
                // `before(JeodSet::TimeUpdate)` would panic with "not
                // a registered gravity source" on the first tick after
                // a between-tick spawn, even though the same
                // `FixedUpdate` would have registered the entity a few
                // systems later. Slotting validation after the
                // registration trio (and still before
                // `JeodSet::EphemerisUpdate`, where the gravity /
                // ephemeris / pfix consumers live) preserves the
                // "validate before consumers" intent without racing
                // the frame-tree wiring.
                validation::validate_jeod_invariants::<jeod_sim::Earth>
                    .after(systems::register_body_frames_system::<jeod_sim::Earth>)
                    .before(JeodSet::EphemerisUpdate),
                // After ephemeris_update_system writes new source
                // position / velocity, mirror the values into the
                // source's frame entity so frame-tree consumers
                // (`RelativeFrameState`, `FrameOrigin`,
                // frame-switch evaluation) see the latest state.
                systems::sync_source_to_frame_system::<jeod_sim::Earth>
                    .in_set(JeodSet::EphemerisUpdate)
                    .after(systems::ephemeris_update_system::<jeod_sim::Earth>)
                    .after(systems::planet_fixed_rotation_system::<jeod_sim::Earth>),
                // Planet-fixed rotation (RNP)
                systems::planet_fixed_rotation_system::<jeod_sim::Earth>
                    .in_set(JeodSet::EphemerisUpdate),
                // Ephemeris position updates (DE4xx)
                systems::ephemeris_update_system::<jeod_sim::Earth>
                    .in_set(JeodSet::EphemerisUpdate),
                // Tidal ΔC20 (must run after planet-fixed rotation)
                systems::tidal_update_system::<jeod_sim::Earth>
                    .in_set(JeodSet::EphemerisUpdate)
                    .after(systems::planet_fixed_rotation_system::<jeod_sim::Earth>),
                // Mass update: recompute inverse_mass/inverse_inertia each step.
                systems::mass_update_system
                    .after(JeodSet::TimeUpdate)
                    .before(JeodSet::EphemerisUpdate),
                // Mass-tree composite recomputation: walks
                // `MassChildOf` edges bottom-up via the
                // `jeod_sim::MassStorage` trait and writes composite
                // mass / inertia / CoM back into `MassPropertiesC`.
                // Runs after `mass_update_system` so the per-entity
                // inverse caches are fresh, and before
                // `JeodSet::EphemerisUpdate` so downstream gravity /
                // interaction / integration systems see the
                // composite. Fast-paths to a no-op when no entity
                // carries `MassChildOf`.
                mass_tree::composite_mass_system
                    .after(systems::mass_update_system)
                    .before(JeodSet::EphemerisUpdate),
                // Gravity pre-computation
                systems::gravity_computation_system::<jeod_sim::Earth>.in_set(JeodSet::Environment),
                // Atmosphere evaluation
                systems::atmosphere_update_system::<jeod_sim::Earth>.in_set(JeodSet::Environment),
                // Interactions
                // Mass tree staging (attach/detach) — runs before interactions
                // so mass changes affect the current step's forces and integration.
                systems::staging_system::<jeod_sim::Earth>
                    .after(JeodSet::Environment)
                    .before(JeodSet::Interaction),
                // Detached-subtree ballistic propagation: advance every
                // entity carrying `DetachedSubtreeStateC` by `dt` under
                // free-flight kinematics (no force, no torque). Runs in
                // parallel with the integration of attached bodies —
                // detached subtrees are not part of any integrator's
                // wrench-aggregation walk anymore. Mirrors
                // `jeod_runner::Simulation::step_detached_subtrees`.
                //
                // Detached entities still carry `FrameEntityC`, so
                // `sync_body_to_frame_system` and `frame_switch_system`
                // would otherwise read the body's pre-step
                // `TranslationalStateC` and write it into the body's
                // frame entity before `step_detached_system` overwrites
                // it — leaving the frame tree desynced for one tick.
                // Pin `step_detached_system` before both so the synced
                // frame entity reflects the post-step body state.
                systems::step_detached_system::<jeod_sim::Earth>
                    .in_set(JeodSet::Integration)
                    .before(systems::sync_body_to_frame_system::<jeod_sim::Earth>)
                    .before(systems::frame_switch_system::<jeod_sim::Earth>),
                systems::aero_drag_system::<jeod_sim::Earth>.in_set(JeodSet::Interaction),
                systems::gravity_torque_system.in_set(JeodSet::Interaction),
                systems::flat_plate_srp_system::<jeod_sim::Earth>.in_set(JeodSet::Interaction),
                systems::cannonball_srp_system::<jeod_sim::Earth>.in_set(JeodSet::Interaction),
            ),
        );
        app.add_systems(
            FixedUpdate,
            (
                // Kinematically prescribed joint frames. Same
                // EphemerisUpdate stage as planet-fixed rotation: both
                // write FrameRotC / FrameAngVelC on frame entities and
                // must run before any frame-tree consumer in
                // Environment / Interaction / Integration /
                // DerivedState. Lives in this second add_systems call
                // (rather than alongside planet_fixed_rotation_system)
                // only to stay within Bevy's 20-tuple `IntoSystem`
                // limit; set membership controls ordering, not which
                // `add_systems` call carries the system.
                systems::joint_kinematics_system.in_set(JeodSet::EphemerisUpdate),
                // Sibling kinematic-joint drivers for the richer specs
                // (sinusoidal, closure, multi-DOF). All four write
                // `FrameRotC` / `FrameAngVelC` and run inside the same
                // `EphemerisUpdate` set; pairwise disjointness on
                // those mutable accesses is enforced three ways so
                // the scheduler can dispatch them in parallel without
                // a borrow conflict and so a misconfigured entity
                // can't silently drop out of the pipeline:
                //
                // * **Per-system `Without<...>` filters** — each
                //   driver excludes the other three spec components,
                //   so the queries are structurally disjoint at the
                //   `Query` level. This is the signal Bevy needs to
                //   parallelize the four systems on the same set.
                // * **Component `on_insert` hooks** —
                //   `register_joint_kinematics_exclusivity_hooks`
                //   panics at insertion time when a stacked-spec
                //   entity is created or mutated, including spawns
                //   that happen long after `Startup` (FixedUpdate,
                //   Update, observers, …).
                // * **PostStartup validation** —
                //   `validate_joint_kinematics_exclusivity` walks
                //   every frame entity once at `PostStartup` and
                //   reports *every* offender at startup in one
                //   aggregated message; defense in depth alongside
                //   the hooks.
                //
                // Spec components are semantic alternatives, not
                // stackable: a joint is *either* constant-rate, *or*
                // sinusoidal, *or* closure, *or* multi-DOF.
                systems::sinusoidal_joint_kinematics_system.in_set(JeodSet::EphemerisUpdate),
                systems::closure_joint_kinematics_system.in_set(JeodSet::EphemerisUpdate),
                systems::multi_dof_joint_kinematics_system.in_set(JeodSet::EphemerisUpdate),
                // Frame-attached body attach/detach event processing
                // (port of `Simulation::attach_to_frame` /
                // `detach_from_frame`). Pinned between
                // `JeodSet::EphemerisUpdate` and `JeodSet::Environment`
                // so the propagation pass below sees freshly-processed
                // events on the same tick they were dispatched. Runner
                // counterpart: events are applied before stage 3a in
                // `Simulation::step_internal`.
                frame_attach_system::frame_attach_system::<jeod_sim::Earth>
                    .after(JeodSet::EphemerisUpdate)
                    .before(JeodSet::Environment),
                // Frame-attached body kinematic propagation. Derives
                // every `FrameAttachedC` body's `TranslationalStateC`
                // / `RotationalStateC` from its parent frame entity's
                // current state composed with the captured offset.
                // Pinned before `JeodSet::Environment` so gravity /
                // atmosphere see the freshly-derived parent-frame
                // composition rather than a one-tick-stale body state,
                // and before `JeodSet::Interaction` so drag / SRP /
                // gravity-torque also read the post-composition state.
                // Mirrors stage 3a of the runner's
                // `Simulation::step_internal` in
                // `crates/jeod_runner/src/simulation/step/mod.rs`.
                // Runs after `frame_attach_system` so freshly-attached
                // bodies pick up the parent-frame composition the same
                // tick they were attached.
                frame_attach_system::propagate_frame_attached_state_system::<jeod_sim::Earth>
                    .after(frame_attach_system::frame_attach_system::<jeod_sim::Earth>)
                    .before(JeodSet::Environment),
                // Body-action lifecycle: drain `BodyActionEvent`
                // (`Add` / `Remove` variants) into `BodyActionsR`,
                // then apply every ready action. Runs before
                // `JeodSet::EphemerisUpdate` so a mid-tick mass /
                // state replacement is visible to gravity, atmosphere,
                // integration, and derived state in the same tick.
                // Runs after `sync_body_mass_point_ref_system` (in the
                // first `add_systems` call) so a freshly-spawned body
                // has a `MassPointRef` before its `MassPropertiesC` is
                // mutated by an `InitMass` action.
                //
                // `sync_body_mass_point_ref_system` mutates the world
                // via `Commands::insert` / `Commands::remove`, so the
                // `MassPointRef` (or its absence) only becomes
                // visible after a flush. With Bevy's
                // `auto_insert_apply_deferred` (the default), the
                // explicit `.after(sync_body_mass_point_ref_system)`
                // ordering causes the scheduler to auto-insert an
                // `ApplyDeferred` between the two systems, so this
                // intake pass observes the up-to-date
                // `MassPointRef`. The same is true for the
                // `register_body_frames_system` chain that
                // `sync_body_mass_point_ref_system` itself depends on
                // — every `Commands`-using ancestor in the chain
                // gets an auto-flush before the next ordered system
                // runs.
                //
                // Lives in this second `add_systems` to stay within
                // Bevy's 20-tuple `IntoSystem` limit.
                body_action::body_action_intake_system::<jeod_sim::Earth>
                    .after(JeodSet::TimeUpdate)
                    .after(systems::sync_body_mass_point_ref_system)
                    .before(systems::mass_update_system)
                    .before(JeodSet::EphemerisUpdate),
                // Fail-loud fence on unregistered-planet adds. Runs
                // after the Earth intake (so the well-formed Earth
                // adds have already been claimed) and before any
                // `body_action_system::<P>` apply pass; reads the
                // shared `Messages<BodyActionEvent>` buffer through
                // its own `Local<MessageCursor>` and panics on any
                // `Add` whose `planet` `TypeId` is not registered in
                // `RegisteredPlanetsR`. Wired here so single-planet
                // (Earth-only) missions still get the guard with no
                // explicit `register_planet_systems::<Earth>` call.
                body_action::body_action_unregistered_planet_fence_system
                    .after(JeodSet::TimeUpdate)
                    .after(body_action::body_action_intake_system::<jeod_sim::Earth>)
                    .before(body_action::body_action_system::<jeod_sim::Earth>)
                    .before(systems::mass_update_system)
                    .before(JeodSet::EphemerisUpdate),
                // Strictly ordered before `mass_update_system` so a
                // queued `BodyAction::InitMass` lands its `dirty=true`
                // mass replacement *before* the per-tick recompute walks
                // the dirty flag — the recompute then runs against the
                // newly applied mass on the same tick. The first
                // `add_systems` call places `mass_update_system`
                // `.after(JeodSet::TimeUpdate).before(JeodSet::EphemerisUpdate)`,
                // i.e. in the same TimeUpdate→EphemerisUpdate gap as
                // `body_action_system` — without this explicit ordering
                // Bevy is free to schedule the two in either order and
                // a same-tick mass propagation would be a coin flip.
                body_action::body_action_system::<jeod_sim::Earth>
                    .after(JeodSet::TimeUpdate)
                    .after(body_action::body_action_intake_system::<jeod_sim::Earth>)
                    .after(body_action::body_action_unregistered_planet_fence_system)
                    .before(systems::mass_update_system)
                    .before(JeodSet::EphemerisUpdate),
                // Pre-integration kinematic state propagation: walks
                // MassChildOf chains pre-order from each root and
                // overwrites every kinematic child's `RotationalStateC`
                // / `TranslationalStateC` with the parent's state
                // composed with the link's `t_parent_child` rotation
                // and offset. Mirrors JEOD
                // `DynBody::propagate_state_from_structure` and the
                // runner's stage 3b. Pinned before
                // `JeodSet::Environment` so kinematic children inherit
                // the freshly-derived root state before gravity /
                // atmosphere read body state, and after the
                // frame-attached propagation so a frame-attached
                // mass-tree root has its parent-frame-derived state
                // available before the kinematic walk reads it (a
                // body that is both a frame-attach target and a
                // mass-tree root would otherwise hand its kinematic
                // descendants a stale pre-frame-attach root state).
                // Fast-path no-op when no entity carries `MassChildOf`.
                kinematic_propagation::propagate_state_from_root_system::<jeod_sim::Earth>
                    .after(
                        frame_attach_system::propagate_frame_attached_state_system::<jeod_sim::Earth>,
                    )
                    .before(JeodSet::Environment),
                // Force collection and integration
                systems::force_collection_system.in_set(JeodSet::ForceCollection),
                // Composite-rigid-body wrench aggregation: walk
                // MassChildOf chains leaves → root and accumulate
                // each child's force/torque (and parallel-axis cross
                // term) into the root's TotalForceC. Non-root children
                // are zeroed so the existing integration_system does
                // not double-count them. Reads the kinematic-propagated
                // child states written by the pre-Environment
                // `propagate_state_from_root_system` pass, so the
                // per-entity `T_inertial_struct` is correct for every
                // chain member. Fast-path no-op when no entity carries
                // MassChildOf.
                wrench::wrench_aggregation_system
                    .in_set(JeodSet::ForceCollection)
                    .after(systems::force_collection_system),
                systems::integration_system::<jeod_sim::Earth>.in_set(JeodSet::Integration),
                // After integration, sync the body's typed state into
                // its frame entity's `FrameTransC` so frame-switch
                // evaluation and downstream `RelativeFrameState` /
                // `FrameOrigin` queries see current distances.
                systems::sync_body_to_frame_system::<jeod_sim::Earth>
                    .in_set(JeodSet::Integration)
                    .after(systems::integration_system::<jeod_sim::Earth>),
                // Evaluate distance-based frame switches and reparent
                // the body's frame entity in the ECS hierarchy on
                // trigger.
                systems::frame_switch_system::<jeod_sim::Earth>
                    .in_set(JeodSet::Integration)
                    .after(systems::sync_body_to_frame_system::<jeod_sim::Earth>),
                // Post-integration frame-attached body propagation.
                // Symmetric to the pre-integration sweep above: the
                // integrator just produced fresh source-body state and
                // `frame_switch_system` may have rewritten frame-tree
                // state mid-step, so re-derive every `FrameAttachedC`
                // body's `TranslationalStateC` / `RotationalStateC` so
                // `JeodSet::DerivedState` consumers
                // (`orbital_elements_system`, `geodetic_system`,
                // `lvlh_system`, `solar_beta_system`,
                // `earth_lighting_system`) observe the parent
                // reference frame's *current* state rather than a
                // one-tick-stale composition.
                //
                // Mirrors stage 8c of the runner's
                // `Simulation::step_internal` in
                // `crates/jeod_runner/src/simulation/step/mod.rs`.
                // Must run *before* the post-integration kinematic
                // walk below: a frame-attached body that is also a
                // mass-tree root would otherwise hand its kinematic
                // descendants a stale pre-frame-attach root state —
                // same constraint as the pre-integration ordering,
                // applied to the post-integration sweep.
                frame_attach_system::propagate_frame_attached_state_post_integration_system::<
                    jeod_sim::Earth,
                >
                    .in_set(JeodSet::Integration)
                    .after(systems::frame_switch_system::<jeod_sim::Earth>),
                // Post-integration kinematic state propagation
                // (root → leaves). The integrator just produced fresh
                // root-body state; rerun the kinematic walk so every
                // non-root child's `RotationalStateC` /
                // `TranslationalStateC` reflects the same-tick parent
                // state. Mirrors JEOD's
                // `DynBody::propagate_state_from_structure` invocation
                // at the end of every integration cycle and the
                // runner's stage 8d post-integration sweep.
                kinematic_propagation::propagate_state_from_root_post_integration_system::<
                    jeod_sim::Earth,
                >
                    .in_set(JeodSet::Integration)
                    .after(
                        frame_attach_system::propagate_frame_attached_state_post_integration_system::<jeod_sim::Earth>,
                    ),
            ),
        );
        // Third `add_systems` call: derived-state systems live here only
        // to stay within Bevy's 20-tuple `IntoSystem` limit on the
        // second block. Set membership pins them to `JeodSet::DerivedState`
        // regardless of which `add_systems` call carries them.
        app.add_systems(
            FixedUpdate,
            (
                // Derived states
                systems::orbital_elements_system::<jeod_sim::Earth>.in_set(JeodSet::DerivedState),
                systems::euler_angles_system.in_set(JeodSet::DerivedState),
                systems::lvlh_system::<jeod_sim::Earth>.in_set(JeodSet::DerivedState),
                systems::geodetic_system::<jeod_sim::Earth>.in_set(JeodSet::DerivedState),
                systems::solar_beta_system::<jeod_sim::Earth>.in_set(JeodSet::DerivedState),
                systems::earth_lighting_system::<jeod_sim::Earth>.in_set(JeodSet::DerivedState),
            ),
        );
    }
}

/// Register the planet-generic system instantiations needed for a
/// downstream multi-planet mission.
///
/// `JeodPlugin::build` registers every planet-generic system with
/// `<jeod_sim::Earth>` so that single-planet Earth missions work out of
/// the box without any extra registration call. A mission that
/// integrates bodies in multiple planet-inertial frames (e.g. a
/// Mars-orbit chief + Earth-orbit deputy) calls this helper once per
/// *additional* planet:
///
/// ```ignore
/// use bevy::prelude::*;
/// use bevy_jeod::{JeodPlugin, register_planet_systems};
/// use jeod_sim::Mars;
///
/// let mut app = App::new();
/// app.add_plugins(JeodPlugin);            // registers Earth instantiations
/// register_planet_systems::<Mars>(&mut app); // adds Mars instantiations
/// ```
///
/// Each instantiation only matches entities whose Planet-flavored
/// components carry the same `<P>` tag — `register_body_frames_system::<Earth>`
/// only registers bodies with `TranslationalStateC<Earth>`,
/// `register_body_frames_system::<Mars>` only the Mars-tagged ones,
/// etc. The two registrations therefore cover disjoint entity sets
/// and run in parallel without conflict.
///
/// Schedule ordering and set membership mirror the Earth registrations
/// in `JeodPlugin::build` — this helper is the structural single
/// source of truth for the per-planet system set.
pub fn register_planet_systems<P: jeod_sim::Planet>(app: &mut App) {
    // Per-planet body-action queue: the unified `BodyActionEvent`
    // message buffer fans out to one `BodyActionsR<P>` per registered
    // planet. The matching `body_action_intake_system::<P>` claims
    // only the entries whose `Add::planet` `TypeId` matches `P`;
    // remaining `Add`s land in another planet's queue, and `Remove`s
    // fan out across every queue (a name-based cancel from any code
    // path reaches a Mars-tagged add even when the calling system
    // holds no `<P>` witness). `init_resource` is idempotent — a
    // mission re-registering Earth via `register_planet_systems::<Earth>`
    // (already wired by `JeodPlugin::build`) does not double-init the
    // queue.
    app.init_resource::<body_action::BodyActionsR<P>>();
    // Track this planet in the registry consulted by the
    // `BodyActionEvent::Add` writer surfaces (the
    // `BodyActionCommandsExt::add_body_action_for::<P>` call-site
    // assertion and the per-tick `body_action_unregistered_planet_fence_system`).
    // Without this insertion, `add_for::<P>` would reach the apply
    // pipeline but the fence would treat `P` as unregistered and
    // panic — which is correct for missions that forgot
    // `register_planet_systems::<P>`, but wrong for missions that
    // *did* call it. `RegisteredPlanetsR::register` is idempotent
    // (it's a `HashSet::insert`), so a redundant call is harmless.
    app.world_mut()
        .resource_mut::<body_action::RegisteredPlanetsR>()
        .register::<P>();
    app.add_systems(
        Startup,
        (
            systems::register_source_frames_system::<P>,
            systems::register_pfix_frames_system::<P>
                .after(systems::register_source_frames_system::<P>),
            systems::register_body_frames_system::<P>
                .after(systems::register_pfix_frames_system::<P>),
        ),
    );
    app.add_systems(
        PreUpdate,
        (
            systems::register_source_frames_system::<P>,
            systems::register_pfix_frames_system::<P>
                .after(systems::register_source_frames_system::<P>),
            systems::register_body_frames_system::<P>
                .after(systems::register_pfix_frames_system::<P>),
        ),
    );
    app.add_systems(
        FixedUpdate,
        (
            systems::register_source_frames_system::<P>.before(JeodSet::EphemerisUpdate),
            systems::register_pfix_frames_system::<P>
                .after(systems::register_source_frames_system::<P>)
                .before(JeodSet::EphemerisUpdate),
            systems::register_body_frames_system::<P>
                .after(systems::register_pfix_frames_system::<P>)
                .before(JeodSet::EphemerisUpdate),
            // Per-planet validator instantiation. Mirrors the schedule
            // slot of the Earth registration in `JeodPlugin::build`:
            // after `register_body_frames_system::<P>` so the body's
            // `FrameEntityC` parent chain is wired (the frame-switch
            // and non-root-integ checks walk it), before
            // `JeodSet::EphemerisUpdate` so consumers of validated
            // state see the post-validation gravity-control
            // auto-corrections.
            validation::validate_jeod_invariants::<P>
                .after(systems::register_body_frames_system::<P>)
                .before(JeodSet::EphemerisUpdate),
            systems::sync_source_to_frame_system::<P>
                .in_set(JeodSet::EphemerisUpdate)
                .after(systems::ephemeris_update_system::<P>)
                .after(systems::planet_fixed_rotation_system::<P>),
            systems::planet_fixed_rotation_system::<P>.in_set(JeodSet::EphemerisUpdate),
            systems::ephemeris_update_system::<P>.in_set(JeodSet::EphemerisUpdate),
            systems::tidal_update_system::<P>
                .in_set(JeodSet::EphemerisUpdate)
                .after(systems::planet_fixed_rotation_system::<P>),
            systems::gravity_computation_system::<P>.in_set(JeodSet::Environment),
            systems::atmosphere_update_system::<P>.in_set(JeodSet::Environment),
            systems::staging_system::<P>
                .after(JeodSet::Environment)
                .before(JeodSet::Interaction),
            systems::step_detached_system::<P>
                .in_set(JeodSet::Integration)
                .before(systems::sync_body_to_frame_system::<P>)
                .before(systems::frame_switch_system::<P>),
            systems::aero_drag_system::<P>.in_set(JeodSet::Interaction),
            systems::flat_plate_srp_system::<P>.in_set(JeodSet::Interaction),
            systems::cannonball_srp_system::<P>.in_set(JeodSet::Interaction),
        ),
    );
    app.add_systems(
        FixedUpdate,
        (
            // Per-planet body-action intake + apply. Mirrors the
            // Earth registration in `JeodPlugin::build`: intake runs
            // after `JeodSet::TimeUpdate` and after the mass-point-ref
            // sync, before `mass_update_system` and
            // `JeodSet::EphemerisUpdate`; apply chains after intake
            // with the same surrounding ordering. The two systems
            // partition by `<P>` (the intake claims only `Add`
            // messages whose `planet` `TypeId` matches `P`; the apply
            // walks `BodyActionsR<P>`), so multiple planet
            // pipelines never fight for the same pending entry.
            body_action::body_action_intake_system::<P>
                .after(JeodSet::TimeUpdate)
                .after(systems::sync_body_mass_point_ref_system)
                .before(body_action::body_action_unregistered_planet_fence_system)
                .before(systems::mass_update_system)
                .before(JeodSet::EphemerisUpdate),
            body_action::body_action_system::<P>
                .after(JeodSet::TimeUpdate)
                .after(body_action::body_action_intake_system::<P>)
                .after(body_action::body_action_unregistered_planet_fence_system)
                .before(systems::mass_update_system)
                .before(JeodSet::EphemerisUpdate),
            frame_attach_system::frame_attach_system::<P>
                .after(JeodSet::EphemerisUpdate)
                .before(JeodSet::Environment),
            frame_attach_system::propagate_frame_attached_state_system::<P>
                .after(frame_attach_system::frame_attach_system::<P>)
                .before(JeodSet::Environment),
            kinematic_propagation::propagate_state_from_root_system::<P>
                .after(frame_attach_system::propagate_frame_attached_state_system::<P>)
                .before(JeodSet::Environment),
            systems::integration_system::<P>.in_set(JeodSet::Integration),
            systems::sync_body_to_frame_system::<P>
                .in_set(JeodSet::Integration)
                .after(systems::integration_system::<P>),
            systems::frame_switch_system::<P>
                .in_set(JeodSet::Integration)
                .after(systems::sync_body_to_frame_system::<P>),
            frame_attach_system::propagate_frame_attached_state_post_integration_system::<P>
                .in_set(JeodSet::Integration)
                .after(systems::frame_switch_system::<P>),
            kinematic_propagation::propagate_state_from_root_post_integration_system::<P>
                .in_set(JeodSet::Integration)
                .after(
                    frame_attach_system::propagate_frame_attached_state_post_integration_system::<P>,
                ),
            systems::orbital_elements_system::<P>.in_set(JeodSet::DerivedState),
            systems::lvlh_system::<P>.in_set(JeodSet::DerivedState),
            systems::geodetic_system::<P>.in_set(JeodSet::DerivedState),
            systems::solar_beta_system::<P>.in_set(JeodSet::DerivedState),
            systems::earth_lighting_system::<P>.in_set(JeodSet::DerivedState),
        ),
    );
}

// ── Bevy spawn helpers for the typestate VehicleBuilder ──

/// Bevy-side terminal for [`jeod_sim::VehicleBuilder`].
///
/// `VehicleBuilder<Ready>::build()` returns a [`jeod_sim::VehicleConfig`]
/// that the standalone `jeod_runner::Simulation` consumes via
/// `SimulationBuilder::add_body`. This trait provides the parallel
/// terminal for Bevy: given a runtime mapping from gravity-source indices
/// (the `usize`-indexed [`GravityControl`](jeod_sim::GravityControl)s in
/// the built config) to ECS [`Entity`]s, it spawns the vehicle entity
/// with all the required JEOD components attached.
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::{PlanetBundle, VehicleConfigBevyExt};
/// use jeod_sim::recipes::{constants, orbital_elements, vehicle};
/// use jeod_sim::{GravityControl, VehicleBuilder, EARTH};
///
/// let mut app = App::new();
/// app.add_systems(Startup, |mut commands: Commands| {
///     let earth = commands.spawn(PlanetBundle::<jeod_sim::Earth>::point_mass("Earth", &EARTH)).id();
///     let cfg = VehicleBuilder::new()
///         .from_orbital_elements(orbital_elements::iss(), constants::mu_ggm05c())
///         .three_dof_point_mass(vehicle::iss_mass())
///         .rk4()
///         .gravity(GravityControl::new_spherical(0_usize, false))
///         .build();
///     cfg.spawn_bevy::<jeod_sim::Earth>(&mut commands, &[earth]);
/// });
/// app.update();
/// ```
pub trait VehicleConfigBevyExt {
    /// Spawn a Bevy entity carrying the core components implied by this
    /// vehicle configuration, with the translational-state slot tagged
    /// for planet `P`.
    ///
    /// Currently inserts: translational state (as
    /// `TranslationalStateC<P>`), optional rotational state, optional
    /// mass properties, dynamics config, gravity controls, integrator
    /// type, structural transform, optional external force / torque,
    /// and (when `compute_gravity_gradient`) a default gravity torque
    /// component. `source_entities` resolves each `usize` index in
    /// `gravity_controls` to the corresponding ECS [`Entity`].
    ///
    /// Also wires `integ_source` (translated to
    /// [`components::IntegSourceC`] when `Some`) and `frame_switches`
    /// (translated to [`components::FrameSwitchesC`] when non-empty),
    /// retagging each `usize` source index to the matching ECS
    /// [`Entity`] from `source_entities`.
    ///
    /// Not yet wired (callers must insert these manually): drag, SRP
    /// (flat-plate / cannonball), shadow body, derived-state requests
    /// (orbital elements, Euler, LVLH, geodetic, solar beta, earth
    /// lighting). These are tracked for future expansion of
    /// `spawn_bevy`.
    ///
    /// # Planet selection
    ///
    /// `<P>` selects the planet whose [`PlanetInertial`](jeod_sim::PlanetInertial)
    /// frame the body integrates in. A single-planet Earth mission
    /// pins `cfg.spawn_bevy::<jeod_sim::Earth>(...)`; a Mars-orbit
    /// constellation pins `cfg.spawn_bevy::<jeod_sim::Mars>(...)` and
    /// must also have called
    /// [`register_planet_systems::<jeod_sim::Mars>`] so the matching
    /// per-planet system pipeline is wired. The inserted
    /// `TranslationalStateC<P>` carries the same untyped
    /// [`jeod_sim::TranslationalState`] (the configuration-time
    /// builder is planet-agnostic on the data side); the witness for
    /// `<P>` lives in the call-site turbofish, not in the builder.
    /// Queued translational `BodyAction`s for this body must be
    /// routed to the `<P>` queue via
    /// [`crate::body_action::BodyActionEvent::add_for::<P>`] (or the
    /// matching
    /// [`crate::body_action::BodyActionCommandsExt::add_body_action_for::<P>`])
    /// so the `body_action_system::<P>` apply pass mutates the same
    /// `TranslationalStateC<P>` slot this helper inserts.
    ///
    /// # Panics
    ///
    /// Panics if any of the following `usize` source indices is out of
    /// bounds in `source_entities`:
    ///
    /// - any `GravityControl::source_name` in `gravity_controls.controls`
    /// - the `integ_source` value (when `Some`)
    /// - any `FrameSwitchConfig::target_source` in `frame_switches`
    ///
    /// All three panics share the same diagnostic shape, telling the
    /// caller to spawn all gravity sources before invoking `spawn_bevy`.
    ///
    /// Returns the spawned vehicle entity ID.
    fn spawn_bevy<P: jeod_sim::Planet>(
        self,
        commands: &mut Commands,
        source_entities: &[Entity],
    ) -> Entity;
}

/// Resolve a `usize` source index against the caller-supplied entity
/// table, panicking with a descriptive error when the index is out of
/// bounds. Centralizes the error message so every site in
/// [`VehicleConfigBevyExt::spawn_bevy`] that translates a source index
/// produces the same actionable diagnostic.
fn resolve_source_entity(source_entities: &[Entity], idx: usize, what: &str) -> Entity {
    *source_entities.get(idx).unwrap_or_else(|| {
        panic!(
            "spawn_bevy: {what} references source index {idx} but only {len} source \
             entities were provided. Spawn all gravity sources before calling spawn_bevy.",
            what = what,
            idx = idx,
            len = source_entities.len()
        )
    })
}

impl VehicleConfigBevyExt for jeod_sim::VehicleConfig {
    fn spawn_bevy<P: jeod_sim::Planet>(
        self,
        commands: &mut Commands,
        source_entities: &[Entity],
    ) -> Entity {
        // Translate `GravityControls<usize>` to `GravityControls<Entity>` by
        // retagging the source identifier on each control via the
        // `GravityControl::retag_source` helper. The field list lives in
        // exactly one place (`jeod_gravity::gravity_controls`), so adding a
        // new field there does not require touching this site.
        let entity_controls = jeod_sim::GravityControls::<Entity> {
            controls: self
                .gravity_controls
                .controls
                .into_iter()
                .map(|c| {
                    c.retag_source(|idx| {
                        resolve_source_entity(source_entities, idx, "GravityControl")
                    })
                })
                .collect(),
        };

        let dynamics_config = jeod_sim::DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: self.rot.is_some(),
            three_dof: self.rot.is_none(),
        };

        let mut entity = commands.spawn((
            components::TranslationalStateC::<P>::from(self.trans),
            components::DynamicsConfigC(dynamics_config),
            components::GravityControlsC(entity_controls),
            components::IntegratorTypeC(self.integrator),
            components::StructuralTransformC(jeod_sim::FrameTransform::from_matrix(
                self.t_struct_body,
            )),
        ));
        if let Some(rot) = self.rot {
            entity.insert(components::RotationalStateC::from(rot));
        }
        if let Some(mass) = self.mass {
            entity.insert(components::MassPropertiesC::from(mass));
        }
        if self.external_force != glam::DVec3::ZERO {
            // `VehicleConfig.external_force` is still an untyped
            // `DVec3` field on the `jeod_sim` runtime fluent builder
            // API. The Bevy `ExternalForceC` is typed (`Force<RootInertial>`),
            // so this is a one-time insertion-time lift — not a per-step
            // bypass. Migrating `VehicleConfig` itself to typed external
            // fields is a deeper refactor inside `jeod_sim`; out of
            // scope for this Bevy-adapter boundary.
            let f = jeod_sim::Force::<jeod_sim::RootInertial>::from_raw_si(self.external_force); // allowed: insertion-time boundary — VehicleConfig still exposes external_force as untyped DVec3, lifted once into Force<RootInertial> at spawn time (not a per-step bypass).
            entity.insert(components::ExternalForceC(f));
        }
        if self.external_torque != glam::DVec3::ZERO {
            let t = jeod_sim::Torque::<jeod_sim::BodyFrame<jeod_sim::SelfRef>>::from_raw_si(
                self.external_torque,
            ); // allowed: same insertion-time boundary as `external_force` above — VehicleConfig still untyped.
            entity.insert(components::ExternalTorqueC(t));
        }
        if self.compute_gravity_gradient {
            entity.insert(components::GravityTorqueC::default());
        }
        // Non-root integration: translate the `usize` source index to
        // the matching ECS Entity so `register_body_frames_system` can
        // parent the body's frame entity under that source's frame
        // entity. `IntegSourceC(None)` is the implicit default (root),
        // so we only insert when the builder set a non-default integ
        // source.
        if let Some(idx) = self.integ_source {
            let src = resolve_source_entity(source_entities, idx, "integ_source");
            entity.insert(components::IntegSourceC(Some(src)));
        }
        // Frame switches: translate each `FrameSwitchConfig<usize>` to
        // `FrameSwitchConfig<Entity>` by retagging `target_source`. The
        // bevy adapter's `frame_switch_system` reads
        // `FrameSwitchConfig<Entity>` directly. Skip the insertion when
        // the builder didn't configure any switches.
        if !self.frame_switches.is_empty() {
            let entity_switches: Vec<jeod_sim::FrameSwitchConfig<Entity>> = self
                .frame_switches
                .into_iter()
                .map(|sw| jeod_sim::FrameSwitchConfig::<Entity> {
                    target_source: resolve_source_entity(
                        source_entities,
                        sw.target_source,
                        "FrameSwitchConfig::target_source",
                    ),
                    switch_sense: sw.switch_sense,
                    switch_distance: sw.switch_distance,
                    active: sw.active,
                })
                .collect();
            entity.insert(components::FrameSwitchesC(entity_switches));
        }
        entity.id()
    }
}
