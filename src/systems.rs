//! Bevy `Systems` that delegate per-body work to `jeod_sim` per-body
//! orchestration functions. Each system queries the relevant components,
//! calls into `jeod_sim`, and writes the result back. No physics
//! algorithms live here.
//!
//! Frame-tree state lives entirely on Bevy entities: every source /
//! body has a [`crate::components::FrameEntityC`] handle pointing at
//! its frame entity, which carries
//! [`crate::components::FrameTransC`] / [`crate::components::FrameRotC`] /
//! [`crate::components::FrameAngVelC`]. Cross-frame queries flow
//! through [`crate::frame_param::RelativeFrameState`] and
//! [`crate::frame_param::FrameOrigin`].

use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{
    Acceleration, AngularAcceleration, BodyFrame, Force, Position, RootInertial, SelfPlanet,
    SelfRef, Torque, Velocity,
};

use crate::components::*;
use crate::frame_param::{FrameOrigin, RelativeFrameState};
use crate::AtmosphereModelR;
use crate::SimulationTimeR;

// ── Frame-tree source registration ──

/// Auto-register every gravity-source entity (carrying [`GravitySourceC`])
/// by spawning its frame entity as a child of the root frame entity and
/// attaching [`FrameEntityC`] back to the source. The frame entity's
/// [`FrameTransC`] is initialized from [`SourceInertialPositionC`] and
/// (when present) [`SourceInertialVelocityC`].
///
/// A [`PfixFrameEntityC`] is additionally inserted iff the source also
/// carries [`PlanetFixedRotationC`] — that's the indicator
/// `planet_fixed_rotation_system` filters on; without it the source
/// never rotates and a pfix frame would be a permanent identity. When
/// `PlanetFixedRotationC` is present and `RotationModelC` is omitted,
/// the same `EarthRNP` default applies as in
/// `planet_fixed_rotation_system`.
///
/// This is the Bevy analog of `jeod_runner::Simulation::add_source` —
/// it makes the source state observable to gravity / integration via
/// [`crate::frame_param::FrameOrigin`] and to mission code via
/// [`crate::frame_param::RelativeFrameState`].
///
/// **Divergence from jeod_runner**: every source becomes a child of
/// the root frame, including the central body. `jeod_runner` renames
/// the root frame to `<central>.inertial` and reuses it. The Bevy
/// adapter keeps a generic root and treats all sources uniformly so
/// the registration order doesn't matter and so adding a body in a
/// non-Earth-central simulation doesn't require special-casing
/// "central" sources. Frame-switch parity lives at the orchestration
/// layer, where this divergence is invisible.
#[allow(clippy::type_complexity)]
pub fn register_source_frames_system(
    mut commands: Commands,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sources: Query<
        (
            Entity,
            Option<&Name>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
            Option<&RotationModelC>,
            Option<&PlanetFixedRotationC>,
        ),
        (With<GravitySourceC>, Without<FrameEntityC>),
    >,
) {
    for (entity, name, pos, vel, rotation_model, pfix_rot) in &sources {
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("source{:?}", entity));
        // Initialize the source frame entity's FrameTransC from the
        // entity's current typed state. Reading both Position and
        // (optional) Velocity lets sources that already carry a
        // non-zero `SourceInertialVelocityC` start with the right
        // velocity in the tree; sources without the velocity component
        // get zero, matching their ECS state.
        let init_pos = pos.0.raw_si();
        let init_vel = vel.map_or(glam::DVec3::ZERO, |v| v.0.raw_si());

        // Spawn the source's frame entity parented under the root
        // frame entity. The frame entity's FrameTransC / FrameRotC /
        // FrameAngVelC IS the per-frame state — there is no parallel
        // arena to keep in sync.
        let source_frame_entity = commands
            .spawn((
                Name::new(format!("{label}.frame.inertial")),
                InertialFrameMarker,
                FrameTransC {
                    position: init_pos,
                    velocity: init_vel,
                },
                FrameRotC::default(),
                FrameAngVelC::default(),
                ChildOf(root_frame_entity.0),
            ))
            .id();
        commands
            .entity(entity)
            .insert(FrameEntityC(source_frame_entity));

        // Create a pfix child frame only if this source actually
        // rotates. The presence of `PlanetFixedRotationC` is the
        // indicator — `planet_fixed_rotation_system` queries
        // `&mut PlanetFixedRotationC`, so an entity without it never
        // rotates, and a pfix frame would be a permanent identity.
        // Plain point-mass sources spawned without
        // `PlanetFixedRotationC` get no pfix frame, matching
        // `jeod_runner` for the same case. When rotation IS present
        // and `RotationModelC` is omitted, the EarthRNP default
        // applies — same default as `planet_fixed_rotation_system`.
        if pfix_rot.is_some() {
            let default_model = jeod_sim::RotationModel::EarthRNP;
            let model_value = rotation_model.map_or(default_model, |m| m.0);
            if !matches!(model_value, jeod_sim::RotationModel::None) {
                let pfix_frame_entity = commands
                    .spawn((
                        Name::new(format!("{label}.frame.pfix")),
                        PlanetFixedFrameMarker,
                        FrameTransC::default(),
                        FrameRotC::default(),
                        FrameAngVelC::default(),
                        ChildOf(source_frame_entity),
                    ))
                    .id();
                commands
                    .entity(entity)
                    .insert(PfixFrameEntityC(pfix_frame_entity));
            }
        }
    }
}

/// Register a [`PfixFrameEntityC`] for sources that were registered
/// without [`PlanetFixedRotationC`] and acquired it later, or for
/// sources whose [`RotationModelC`] just toggled back from
/// [`jeod_sim::RotationModel::None`] to a rotating model.
/// [`register_source_frames_system`] filters by `Without<FrameEntityC>`,
/// so it cannot pick up an entity that gained `PlanetFixedRotationC`
/// after its initial registration.
///
/// Same registration semantics as [`register_source_frames_system`]'s
/// pfix branch: gated on [`PlanetFixedRotationC`], `EarthRNP` default
/// when [`RotationModelC`] is absent, no frame entity when the rotation
/// model is explicitly [`jeod_sim::RotationModel::None`].
///
/// **Reuse path**: when an entity carries a [`RetiredPfixFrameEntityC`]
/// (the planet just toggled back from `RotationModel::None` to a
/// rotating model), this system reuses the stashed pfix frame entity
/// instead of spawning a fresh one. Its `Name` is restored to the
/// canonical `<label>.frame.pfix` and its `FrameTransC` /
/// `FrameRotC` / `FrameAngVelC` are reset to identity. This bounds
/// the world's pfix-frame entity count at one per source regardless
/// of toggle-cycle count.
#[allow(clippy::type_complexity)]
pub fn register_pfix_frames_system(
    mut commands: Commands,
    sources: Query<
        (
            Entity,
            Option<&Name>,
            // The source's own frame entity: the spawned pfix frame
            // entity ChildOf-links under it. Required for registration
            // — `register_source_frames_system` always inserts it.
            &FrameEntityC,
            Option<&RotationModelC>,
            // ECS-entity retirement marker so we reuse instead of leak
            // on toggle cycles.
            Option<&RetiredPfixFrameEntityC>,
        ),
        (
            With<GravitySourceC>,
            With<PlanetFixedRotationC>,
            Without<PfixFrameEntityC>,
        ),
    >,
    mut frame_trans: Query<&mut FrameTransC>,
    mut frame_rots: Query<&mut FrameRotC>,
    mut frame_ang_vels: Query<&mut FrameAngVelC>,
) {
    for (entity, name, source_frame_entity, rotation_model, retired_entity) in &sources {
        let default_model = jeod_sim::RotationModel::EarthRNP;
        let model_value = rotation_model.map_or(default_model, |m| m.0);
        if matches!(model_value, jeod_sim::RotationModel::None) {
            continue;
        }
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("source{:?}", entity));

        let pfix_frame_entity = if let Some(retired_e) = retired_entity {
            // Reuse: restore canonical name (via Commands so we
            // don't need a `Query<&mut Name, With<PlanetFixedFrameMarker>>`
            // that would conflict with the outer query's
            // `Option<&Name>` access at runtime) and reset typed
            // state to identity. The orphan's
            // `ChildOf(source_frame_entity.0)` edge was preserved
            // across the toggle cycle, so the hierarchy is already
            // correct.
            commands
                .entity(retired_e.0)
                .insert(Name::new(format!("{label}.frame.pfix")));
            // Fail loud if the retired pfix frame entity has lost any
            // of its FrameTransC / FrameRotC / FrameAngVelC components
            // (or has been despawned out from under us). Silently
            // skipping these resets would let stale rotation, angular
            // velocity, or translation state leak into the reused
            // entity. The retirement path in
            // `planet_fixed_rotation_system` (the only producer of
            // `RetiredPfixFrameEntityC`) guarantees the entity stays
            // alive with all three components attached, so an `Err`
            // here means the entity was despawned or stripped
            // externally — which is a misconfiguration, not a
            // recoverable state.
            let mut t = frame_trans.get_mut(retired_e.0).unwrap_or_else(|err| {
                panic!(
                    "register_pfix_frames_system: source {entity:?} \
                     carries RetiredPfixFrameEntityC({:?}) but that \
                     entity has no FrameTransC ({err:?}). The retired \
                     pfix frame entity must be alive with FrameTransC / \
                     FrameRotC / FrameAngVelC intact (set up by \
                     planet_fixed_rotation_system on retirement). Do not \
                     despawn or strip components from a retired pfix \
                     frame entity while its source still carries the \
                     marker.",
                    retired_e.0
                )
            });
            *t = FrameTransC::default();
            let mut r = frame_rots.get_mut(retired_e.0).unwrap_or_else(|err| {
                panic!(
                    "register_pfix_frames_system: source {entity:?} \
                     carries RetiredPfixFrameEntityC({:?}) but that \
                     entity has no FrameRotC ({err:?}). The retired \
                     pfix frame entity must be alive with FrameTransC / \
                     FrameRotC / FrameAngVelC intact (set up by \
                     planet_fixed_rotation_system on retirement). Do not \
                     despawn or strip components from a retired pfix \
                     frame entity while its source still carries the \
                     marker.",
                    retired_e.0
                )
            });
            *r = FrameRotC::default();
            let mut av = frame_ang_vels.get_mut(retired_e.0).unwrap_or_else(|err| {
                panic!(
                    "register_pfix_frames_system: source {entity:?} \
                     carries RetiredPfixFrameEntityC({:?}) but that \
                     entity has no FrameAngVelC ({err:?}). The retired \
                     pfix frame entity must be alive with FrameTransC / \
                     FrameRotC / FrameAngVelC intact (set up by \
                     planet_fixed_rotation_system on retirement). Do not \
                     despawn or strip components from a retired pfix \
                     frame entity while its source still carries the \
                     marker.",
                    retired_e.0
                )
            });
            *av = FrameAngVelC::default();
            commands.entity(entity).remove::<RetiredPfixFrameEntityC>();
            retired_e.0
        } else {
            commands
                .spawn((
                    Name::new(format!("{label}.frame.pfix")),
                    PlanetFixedFrameMarker,
                    FrameTransC::default(),
                    FrameRotC::default(),
                    FrameAngVelC::default(),
                    ChildOf(source_frame_entity.0),
                ))
                .id()
        };
        commands
            .entity(entity)
            .insert(PfixFrameEntityC(pfix_frame_entity));
    }
}

/// Sync each gravity source's typed state from the ECS components
/// (`SourceInertialPositionC` + optional `SourceInertialVelocityC`) into
/// its frame entity's [`FrameTransC`] each step. Mirrors
/// `jeod_runner::Simulation::update_ephemeris`'s post-DE4xx writeback —
/// required so frame-tree consumers
/// ([`crate::frame_param::RelativeFrameState`],
/// [`crate::frame_param::FrameOrigin`], frame-switch evaluation,
/// per-stage source interpolation in [`integration_system`]) see the
/// current source state rather than the registration-time snapshot.
///
/// Velocity source-of-truth precedence:
///
/// 1. [`SourceInertialVelocityC`] when present — the explicit
///    per-source velocity component.
/// 2. Otherwise [`TranslationalStateC`]'s velocity —
///    `ephemeris_update_system` populates it for ephemeris-driven
///    sources that don't carry the standalone velocity component
///    (Sun / Moon entities used by SRP / earth-lighting are typically
///    spawned this way via `SunBundle` / `MoonBundle`).
/// 3. Otherwise leave the frame entity's velocity unchanged.
///
/// Runs in `JeodSet::EphemerisUpdate` after `ephemeris_update_system`
/// (which writes the ECS components from DE4xx) so the frame-entity
/// sync sees the latest values.
#[allow(clippy::type_complexity)]
pub fn sync_source_to_frame_system(
    sources: Query<(
        &FrameEntityC,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TranslationalStateC>,
    )>,
    mut frame_states: Query<&mut FrameTransC>,
) {
    for (fe, pos, vel, trans) in &sources {
        let position = pos.0.raw_si();
        let velocity = vel
            .map(|v| v.0.raw_si())
            .or_else(|| trans.map(|t| t.0.velocity.raw_si()));

        // Write to the source's frame entity. The referenced entity
        // must exist and carry FrameTransC — `register_source_frames_system`
        // spawns it with `FrameTransC` populated from the source's
        // initial state, and the despawn observers tear it down in
        // lockstep with the source. Fail loud if `FrameEntityC` points
        // at a stale / missing entity instead of silently dropping the
        // sync.
        let mut frame_trans = frame_states.get_mut(fe.0).unwrap_or_else(|err| {
            panic!(
                "sync_source_to_frame_system: source has \
                 FrameEntityC({:?}) but that entity has no FrameTransC \
                 ({err:?}). The source's frame entity must be alive \
                 with FrameTransC attached (spawned by PlanetBundle / \
                 register_*_frames_system). Either remove the stale \
                 FrameEntityC marker before despawning the frame \
                 entity, or ensure the frame entity stays alive for \
                 as long as the source carries the handle.",
                fe.0
            )
        });
        frame_trans.position = position;
        if let Some(v) = velocity {
            frame_trans.velocity = v;
        }
    }
}

/// Auto-register every vehicle entity (carrying [`TranslationalStateC`])
/// by spawning the body's frame entity with
/// `ChildOf(integ_frame_entity)` and attaching [`FrameEntityC`] to the
/// body. The body's integration frame is determined by:
///
/// 1. `IntegSourceC(Some(source_entity))` — child of that source's
///    frame entity (panics if the source isn't yet registered).
/// 2. Otherwise — child of the root inertial frame entity
///    ([`crate::RootFrameEntityR`]).
///
/// The body's initial state is read from [`TranslationalStateC`] and
/// written into the new frame entity's [`FrameTransC`] so the
/// hierarchy is consistent from the first step. The integration frame
/// is then queryable via `Query<&ChildOf>` on the body's frame entity
/// (no explicit integration-frame handle component).
///
/// Runs at `Startup` and again before `JeodSet::EphemerisUpdate` to
/// catch dynamically-spawned bodies. Filters by
/// `Without<FrameEntityC>` so the registration is one-time per body.
#[allow(clippy::type_complexity)]
pub fn register_body_frames_system(
    mut commands: Commands,
    // The ECS-side root frame entity, used as the body's frame
    // parent when no IntegSourceC is supplied.
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sources: Query<&FrameEntityC, With<GravitySourceC>>,
    bodies: Query<
        (
            Entity,
            Option<&Name>,
            &TranslationalStateC,
            Option<&IntegSourceC>,
            // Wire the frame-side `MassPointRef` back-pointer at
            // body-frame registration time for any entity that also
            // carries `MassPropertiesC` (i.e. participates in the
            // mass tree). In the current Bevy adapter the body /
            // mass / frame ECS entity is one and the same, so the
            // back-pointer resolves to `MassPointRef(self)`. The
            // component is skipped for kinematic-only bodies (no
            // `MassPropertiesC`), matching the "absent for
            // kinematic-only attaches" contract on the type.
            Has<MassPropertiesC>,
        ),
        (
            With<TranslationalStateC>,
            With<DynamicsConfigC>,
            Without<FrameEntityC>,
        ),
    >,
) {
    for (entity, name, trans, integ_source, has_mass) in &bodies {
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("body{:?}", entity));

        // Resolve the integration frame entity. Default: root inertial.
        let integ_frame_entity = match integ_source.and_then(|c| c.0) {
            Some(source_entity) => {
                sources
                    .get(source_entity)
                    .map(|fe| fe.0)
                    .unwrap_or_else(|err| {
                        panic!(
                            "register_body_frames_system: body {entity:?} has \
                         IntegSourceC pointing at {source_entity:?}, but that \
                         entity is not a registered gravity source (missing \
                         FrameEntityC + GravitySourceC). Spawn the source via \
                         PlanetBundle before the body, or remove IntegSourceC. \
                         Underlying error: {err:?}"
                        )
                    })
            }
            None => root_frame_entity.0,
        };

        // The body frame entity carries the body's current state
        // relative to its integ frame. For root-integrated bodies
        // this is the absolute inertial state; for non-root bodies
        // the body's TranslationalStateC is interpreted as already in
        // integ-frame coordinates (mission code is responsible for
        // supplying state in the integ-frame).
        let init_pos = trans.0.position.raw_si();
        let init_vel = trans.0.velocity.raw_si();

        // Tag the integ frame entity with `IntegrationFrameMarker`
        // (idempotent insert via Commands). The body's integration
        // frame is queried via `Query<&ChildOf>` on the body's frame
        // entity by gravity / integration / frame-switch consumers —
        // the body frame entity's parent *is* the integration frame.
        commands
            .entity(integ_frame_entity)
            .insert(IntegrationFrameMarker);
        let body_frame_entity = commands
            .spawn((
                Name::new(format!("{label}.frame.body")),
                BodyFrameMarker,
                FrameTransC {
                    position: init_pos,
                    velocity: init_vel,
                },
                FrameRotC::default(),
                FrameAngVelC::default(),
                ChildOf(integ_frame_entity),
            ))
            .id();
        let mut entity_cmds = commands.entity(entity);
        entity_cmds.insert(FrameEntityC(body_frame_entity));
        // Wire the frame-side `MassPointRef` back-pointer for any
        // entity that also carries `MassPropertiesC` (i.e.
        // participates in the mass tree). In the current Bevy adapter
        // the body / mass / frame ECS entity is one and the same, so
        // the back-pointer resolves to `MassPointRef(self)`.
        if has_mass {
            entity_cmds.insert(MassPointRef(entity));
        }
    }
}

/// Maintain the `MassPointRef` ↔ `MassPropertiesC` invariant on bodies
/// that have already passed through [`register_body_frames_system`].
///
/// `register_body_frames_system` is filtered by `Without<FrameEntityC>`
/// so it sees each body exactly once. That makes the
/// `Has<MassPropertiesC>`-driven `MassPointRef` insertion only correct
/// at the body's first sight — a body that starts kinematic-only and
/// later acquires `MassPropertiesC` would never receive the
/// back-pointer, and a body that loses `MassPropertiesC` after first
/// registration would keep a stale one.
///
/// This system handles the post-registration transitions:
///
/// - **Acquired mass**: a registered body (carrying `FrameEntityC` +
///   `DynamicsConfigC`) with `MassPropertiesC` but no `MassPointRef`
///   gets one inserted (the back-pointer resolves to the body's own
///   entity, mirroring the "body / mass / frame ECS entity is one
///   and the same" invariant the initial registration uses).
/// - **Lost mass**: a registered body with `MassPointRef` whose
///   `MassPropertiesC` has been removed gets the stale `MassPointRef`
///   removed (the "absent for kinematic-only attaches" contract on
///   the type — keeping a stale back-pointer would lie about whether
///   the frame still participates in the mass tree).
///
/// Runs in the same scheduling slots as
/// [`register_body_frames_system`] (Startup, PreUpdate, FixedUpdate
/// before `JeodSet::EphemerisUpdate`) so the invariant is restored
/// before any consumer (gravity, force collection, integration) reads
/// the back-pointer this tick.
///
/// The query filter combines `With<FrameEntityC>` (the post-PR4
/// "registered" gate, which sources also carry) with
/// `With<DynamicsConfigC>` (which sources don't carry) to restrict
/// the iteration to bodies. Brand-new bodies that
/// `register_body_frames_system` will register this same tick are
/// excluded by the filter — `Commands` are deferred until the next
/// system flush, so those bodies don't yet carry `FrameEntityC` when
/// this system runs.
#[allow(clippy::type_complexity)]
pub fn sync_body_mass_point_ref_system(
    mut commands: Commands,
    acquired: Query<
        Entity,
        (
            With<FrameEntityC>,
            With<DynamicsConfigC>,
            With<MassPropertiesC>,
            Without<MassPointRef>,
        ),
    >,
    lost: Query<
        Entity,
        (
            With<FrameEntityC>,
            With<DynamicsConfigC>,
            With<MassPointRef>,
            Without<MassPropertiesC>,
        ),
    >,
) {
    for entity in &acquired {
        commands.entity(entity).insert(MassPointRef(entity));
    }
    for entity in &lost {
        commands.entity(entity).remove::<MassPointRef>();
    }
}

// ── Frame-tree despawn cleanup ──
//
// Frame entities are owned by the source / body / pfix entity that
// references them via `FrameEntityC` / `PfixFrameEntityC` /
// `RetiredPfixFrameEntityC`. When the owner despawns, the referenced
// frame entity needs to despawn alongside so the world's entity count
// stays bounded and future re-spawns of the same `Name` aren't
// shadowed by an orphan.
//
// We use [`Despawn`] (not [`Remove`]) so component-only removals
// — notably `planet_fixed_rotation_system`'s toggle-to-`None` path
// that does `commands.entity(e).remove::<PfixFrameEntityC>()` — don't
// double-fire and tear down the entity that the retirement path is
// stashing for reuse. Each observer cleans up only its own handle;
// ordering across the per-component `Despawn` triggers is therefore
// irrelevant.
//
// Out of scope: a body whose [`IntegSourceC`]'s source entity is
// despawned remains alive but integrates against a now-despawned
// frame. Mission code is responsible for despawning dependent bodies.

/// On entity despawn, despawn the orphan pfix *frame entity* stashed
/// in [`RetiredPfixFrameEntityC`] (left over from a
/// `RotationModel::None` toggle that wasn't followed by a re-toggle
/// before despawn). The orphan ECS entity has no other owner — it
/// was kept alive specifically so the next `None → rotating` retoggle
/// could reuse it — so without this observer it would leak when the
/// owning source despawns.
///
/// `try_despawn` (not `despawn`) because the retired pfix entity's
/// `ChildOf` parent is the source frame entity, which is despawned
/// recursively by [`on_frame_entity_despawn`] when the source
/// despawns; the retired pfix may already be gone by the time this
/// observer's command flushes.
pub fn on_retired_pfix_frame_entity_despawn(
    trigger: On<Despawn, RetiredPfixFrameEntityC>,
    sources: Query<&RetiredPfixFrameEntityC>,
    mut commands: Commands,
) {
    if let Ok(retired) = sources.get(trigger.entity) {
        commands.entity(retired.0).try_despawn();
    }
}

/// On entity despawn, despawn the *frame entity* the source / body
/// entity carries in [`FrameEntityC`]. Without this observer,
/// despawning a source or body would leave its dual-write frame
/// entity (and the pfix child it parents, when present) alive
/// indefinitely under the root frame entity, growing the entity
/// count over time and potentially shadowing future re-spawns of
/// the same `Name`.
///
/// Fires for *any* entity that carries [`FrameEntityC`], i.e. both
/// source entities (registered by [`register_source_frames_system`])
/// and body entities (registered by [`register_body_frames_system`]).
/// The cleanup logic is identical for the two cases — the despawning
/// entity hands us its frame-entity handle and we tear down the
/// referenced frame entity — so the observer is named for the
/// component it watches, not for either of the owner kinds (a
/// previous name `on_source_frame_entity_despawn` misled readers
/// into thinking the observer only handled sources).
///
/// `try_despawn` (not `despawn`) because Bevy's `ChildOf` /
/// `Children` relationship already triggers recursive despawn on the
/// frame entity's children — the pfix child of a source frame, the
/// body frame entity if a body shares the integration frame entity
/// — so a sibling observer ([`on_source_pfix_frame_entity_despawn`])
/// firing on the same entity-despawn event may find its target
/// already queued for despawn. `try_despawn` silently no-ops in that
/// case.
///
/// Pairs with [`on_source_pfix_frame_entity_despawn`] (covers the
/// source's pfix child) and
/// [`on_retired_pfix_frame_entity_despawn`] (covers a stashed
/// orphan from a `RotationModel::None` toggle that wasn't followed
/// by a re-toggle before despawn) to provide complete cleanup for
/// the spawn sites in [`register_source_frames_system`],
/// [`register_pfix_frames_system`], and
/// [`register_body_frames_system`].
pub fn on_frame_entity_despawn(
    trigger: On<Despawn, FrameEntityC>,
    owners: Query<&FrameEntityC>,
    mut commands: Commands,
) {
    if let Ok(frame) = owners.get(trigger.entity) {
        commands.entity(frame.0).try_despawn();
    }
}

/// On entity despawn, despawn the pfix *frame entity* the source
/// entity carries in [`PfixFrameEntityC`]. Pair to
/// [`on_frame_entity_despawn`] for the source's pfix child.
///
/// Independent of [`on_frame_entity_despawn`] so the
/// per-component `Despawn` order doesn't matter: in the common case
/// the pfix entity is `ChildOf(source_frame_entity)` and gets
/// despawned recursively when its parent does, but this observer is
/// the safety net for any future configuration where the pfix entity
/// is parented elsewhere (or for entities that hold
/// `PfixFrameEntityC` without `FrameEntityC`). `try_despawn` silently
/// no-ops when the recursive despawn has already claimed the entity.
pub fn on_source_pfix_frame_entity_despawn(
    trigger: On<Despawn, PfixFrameEntityC>,
    owners: Query<&PfixFrameEntityC>,
    mut commands: Commands,
) {
    if let Ok(frame) = owners.get(trigger.entity) {
        commands.entity(frame.0).try_despawn();
    }
}

/// Sync each vehicle's [`TranslationalStateC`] into its frame entity's
/// [`FrameTransC`]. Required so [`frame_switch_system`] and
/// downstream [`crate::frame_param::RelativeFrameState`] /
/// [`crate::frame_param::FrameOrigin`] queries see current body state
/// when evaluating switch distances and computing cross-frame state.
///
/// Runs in `JeodSet::Integration` after `integration_system` and
/// before `frame_switch_system`.
///
/// The `With<DynamicsConfigC>` filter narrows the iteration to actual
/// dynamic bodies — gravity-source entities (planets) also carry
/// `TranslationalStateC` + `FrameEntityC` post-registration but their
/// frame entity is updated by `sync_source_to_frame_system` from the
/// source-side state instead.
pub fn sync_body_to_frame_system(
    bodies: Query<(&TranslationalStateC, &FrameEntityC), With<DynamicsConfigC>>,
    mut frame_states: Query<&mut FrameTransC>,
) {
    for (trans, frame_entity) in &bodies {
        let position = trans.0.position.raw_si();
        let velocity = trans.0.velocity.raw_si();

        // The referenced body frame entity must exist and carry
        // FrameTransC — `register_body_frames_system` spawns it with
        // `FrameTransC` populated from the body's initial state, and
        // the despawn observers tear it down in lockstep with the
        // body. Fail loud if `FrameEntityC` points at a stale /
        // missing entity instead of silently dropping the sync.
        let mut frame_trans = frame_states.get_mut(frame_entity.0).unwrap_or_else(|err| {
            panic!(
                "sync_body_to_frame_system: body has FrameEntityC({:?}) \
                 but that entity has no FrameTransC ({err:?}). The \
                 body's frame entity must be alive with FrameTransC \
                 attached (spawned by register_body_frames_system). \
                 Either remove the stale FrameEntityC marker before \
                 despawning the frame entity, or ensure the frame \
                 entity stays alive for as long as the body carries \
                 the handle.",
                frame_entity.0
            )
        });
        frame_trans.position = position;
        frame_trans.velocity = velocity;
    }
}

/// Evaluate distance-based [`FrameSwitchesC`] entries for each body.
/// On trigger, this system:
///
/// 1. Reparents the body's frame entity under the target source's
///    frame entity via
///    `commands.entity(body_frame).insert(ChildOf(target_frame))`.
/// 2. Rewrites the body's [`TranslationalStateC`] (and the body
///    frame entity's [`FrameTransC`]) in the new integration
///    frame's coordinates, computed via
///    [`crate::frame_param::RelativeFrameState`].
/// 3. Flips [`GravityControlsC`]'s `differential` flags so the new
///    central source becomes non-differential and the prior
///    central source (and any others) becomes differential.
///
/// JEOD reference: `dyn_body_frame_switch.cc:173-182`. The trigger
/// predicates and gravity-control flip mirror
/// `jeod_runner::Simulation`'s `evaluate_and_apply_frame_switch` over
/// the arena; the Bevy variant reads/writes the ECS hierarchy
/// directly via [`crate::frame_param::RelativeFrameState`] (which
/// `impl FrameStorage`s and shares the storage-agnostic
/// `compute_relative_state` algorithm in `jeod_frames` with the
/// runner's arena).
///
/// Runs in `JeodSet::Integration` after [`sync_body_to_frame_system`].
/// Bodies without [`FrameSwitchesC`] entries (or whose entries are
/// all `active = false`) are skipped.
// JEOD_INV: DB.14 — distance-based integration-frame switch reparents
// the body's frame entity under the target source's frame entity and
// rewrites translational state into the new frame's coordinates.
#[allow(clippy::type_complexity)]
pub fn frame_switch_system(
    mut commands: Commands,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sources: Query<&FrameEntityC, With<GravitySourceC>>,
    parents: Query<&ChildOf>,
    rel: RelativeFrameState,
    mut bodies: Query<(
        Entity,
        &mut TranslationalStateC,
        &FrameEntityC,
        &mut FrameSwitchesC,
        &mut GravityControlsC,
    )>,
) {
    // Build a set of registered source frame entities once per call so
    // the per-body integ-frame validation below is O(1) rather than
    // O(sources). Without this, the inner check is a linear scan over
    // every source for every body every tick (O(bodies * sources)),
    // which dominates with many bodies and/or many sources even when
    // no switch fires.
    let known_source_frames: std::collections::HashSet<Entity> =
        sources.iter().map(|fe| fe.0).collect();
    for (body_entity, mut trans, body_frame_entity, mut switches, mut gravity_controls) in
        &mut bodies
    {
        if switches.0.is_empty() {
            continue;
        }
        // The body's current integration frame is the parent of its
        // frame entity in the ECS hierarchy.
        let current_integ_frame_entity = parents
            .get(body_frame_entity.0)
            .unwrap_or_else(|err| {
                panic!(
                    "frame_switch_system: body {body_entity:?} frame entity {fe:?} \
                     has no ChildOf parent ({err:?}). The body's frame entity must \
                     be parented under its integration frame entity (set by \
                     register_body_frames_system).",
                    fe = body_frame_entity.0,
                )
            })
            .parent();
        // Validate the current integ frame entity is the root frame
        // entity or a registered source's frame entity. Anything else
        // means the registration / integ-source wiring is corrupt.
        let current_is_known = current_integ_frame_entity == root_frame_entity.0
            || known_source_frames.contains(&current_integ_frame_entity);
        assert!(
            current_is_known,
            "frame_switch_system: body {body_entity:?} frame entity \
             {fe:?} has parent {parent:?} which is neither the root \
             frame entity ({root_e:?}) nor a registered source's \
             frame entity. The integration frame entity must be one \
             of those — register the source via PlanetBundle before \
             spawning the body, or attach the body under the root.",
            fe = body_frame_entity.0,
            parent = current_integ_frame_entity,
            root_e = root_frame_entity.0,
        );

        // Find the first active switch whose predicate triggers.
        let mut trigger_idx = None;
        for (idx, sw) in switches.0.iter().enumerate() {
            if !sw.active {
                continue;
            }
            // Resolve the target source's frame entity. Fail loud if
            // the target isn't a registered gravity source — same
            // contract as `evaluate_and_apply_frame_switch`'s
            // `FrameSwitchTargetMissing` error. The query filter is
            // `With<GravitySourceC>`, so a missing match means the
            // target either isn't a gravity source at all (no
            // `GravitySourceC`) or is one but `FrameEntityC` was never
            // inserted by `register_source_frames_system` — both are
            // user misconfigurations the diagnostic must enumerate.
            let target_frame_entity =
                sources
                    .get(sw.target_source)
                    .map(|fe| fe.0)
                    .unwrap_or_else(|err| {
                        panic!(
                            "frame_switch_system: body {body_entity:?} switch evaluation failed: \
                         target source {target:?} is not a registered gravity source — \
                         it is missing GravitySourceC and/or FrameEntityC. Spawn it via \
                         PlanetBundle (which inserts both) before referencing it from a \
                         FrameSwitchConfig. Underlying error: {err:?}",
                            target = sw.target_source,
                        )
                    });
            // OnApproach: distance from body to target's frame
            // origin. OnDeparture: body's distance from its current
            // integration frame's origin (i.e. body's
            // `TranslationalStateC.position` magnitude, which equals
            // its FrameTransC position in the current integ frame).
            // Mirrors `jeod_runner::evaluate_and_apply_frame_switch`.
            let threshold_sq = sw.switch_distance * sw.switch_distance;
            let triggered = match sw.switch_sense {
                jeod_sim::SwitchSense::OnApproach => {
                    let pos_in_target = rel.position(target_frame_entity, body_frame_entity.0);
                    pos_in_target.length_squared() < threshold_sq
                }
                jeod_sim::SwitchSense::OnDeparture => {
                    trans.0.position.raw_si().length_squared() > threshold_sq
                }
            };
            if triggered {
                trigger_idx = Some(idx);
                break;
            }
        }

        let Some(idx) = trigger_idx else {
            continue;
        };

        let target_source = switches.0[idx].target_source;
        switches.0[idx].active = false;
        // Re-resolve the target frame entity; lookup proven Some above.
        let new_parent_frame_entity = sources.get(target_source).map(|fe| fe.0).expect(
            "frame_switch_system: target source resolved during evaluation \
             but failed during application — caller-side mutation between lookups",
        );

        // Compute the body's full state expressed in the new target
        // frame's coordinates *before* reparenting. The walk uses the
        // body frame entity's pre-switch `ChildOf` parent (the old
        // integ frame) so the math composes through the existing
        // hierarchy — same algorithm `evaluate_and_apply_frame_switch`
        // runs over the arena (`reparent` then read the post-reparent
        // state).
        let new_state = rel.relative_state(new_parent_frame_entity, body_frame_entity.0);

        // Reparent the body's frame entity under the target source's
        // frame entity, and write the new FrameTransC in the same
        // deferred Commands batch so a post-reparent
        // `RelativeFrameState` walk on the next system flush finds
        // the body in the new parent's coordinates. Without the
        // FrameTransC update, the stored value would still reflect
        // the old parent's frame and downstream consumers would
        // observe a discontinuity-equal-to-(new_origin - old_origin)
        // on this tick. Using `Commands::insert` (rather than a
        // `&mut FrameTransC` query) avoids a static query-conflict
        // with `RelativeFrameState`'s read-only `&FrameTransC`
        // query — the body frame entity's `FrameTransC` is rewritten
        // when the Commands buffer flushes after this system.
        commands
            .entity(body_frame_entity.0)
            .insert(ChildOf(new_parent_frame_entity))
            .insert(FrameTransC {
                position: new_state.trans.position,
                velocity: new_state.trans.velocity,
            });

        // Mirror the new state into the body's TranslationalStateC.
        // Re-wrap as the Component's `<PlanetInertial<SelfPlanet>>`
        // phantom — `new_state.trans` carries planet-inertial
        // coordinates of the *target* source's planet (this is the
        // post-switch frame) which the wildcard `SelfPlanet` tags
        // without committing to a compile-time planet identity. Same
        // boundary lift `evaluate_and_apply_frame_switch` performs.
        type PiPos = jeod_sim::Position<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>;
        type PiVel = jeod_sim::Velocity<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>;
        let pos_typed = PiPos::from_raw_si(new_state.trans.position); // allowed: frame-switch boundary lift, see comment above
        let vel_typed = PiVel::from_raw_si(new_state.trans.velocity); // allowed: same frame-switch boundary lift
        trans.0.position = pos_typed;
        trans.0.velocity = vel_typed;

        // Flip gravity controls: target source becomes
        // non-differential (central body), all others become
        // differential. Identity match by `Entity` — same convention
        // `evaluate_and_apply_frame_switch` uses.
        for ctrl in &mut gravity_controls.0.controls {
            ctrl.differential = ctrl.source_name != target_source;
        }
        // `IntegSourceC` (the config-time intent) is intentionally
        // untouched — the live truth lives in the body frame
        // entity's `ChildOf` parent.
    }
}

// ── Time ──

/// Advance every JEOD-tracked time scale by the Bevy `Time<Fixed>` delta
/// each step (TAI/UTC/UT1/TDB/TT/GMST). Runs in
/// [`JeodSet::TimeUpdate`](crate::JeodSet::TimeUpdate).
// JEOD_INV: TM.03 — time types updated in dependency order (delegates to SimulationTime::advance)
pub fn time_advance_system(mut sim_time: ResMut<SimulationTimeR>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs_f64();
    sim_time.advance(dt);
}

// ── Ephemeris / Frames ──

/// Computes the inertial-to-planet-fixed rotation matrix for each entity
/// that carries a `PlanetFixedRotationC` component.
///
/// Dispatches per-entity via `RotationModelC`:
///
/// - `EarthRNP`: IAU 2000A precession-nutation + GAST + optional polar motion
/// - `MarsIAU`: IAU pole + spin + nutation Fourier series
/// - `MoonIAU`: IAU 2009 pole + prime meridian
/// - `MoonDE421`: DE421 BPC libration (requires `EphemerisR`)
/// - `None`: skip (leaves `PlanetFixedRotationC` unchanged)
///
/// When `RotationModelC` is absent, defaults to `EarthRNP`.
///
/// Earth RNP is lazy-computed once per step and reused across all `EarthRNP`
/// entities.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn planet_fixed_rotation_system(
    mut commands: Commands,
    sim_time: Res<SimulationTimeR>,
    polar: Option<Res<crate::PolarMotionR>>,
    ephemeris: Option<Res<crate::EphemerisR>>,
    mut query: Query<(
        Entity,
        &mut PlanetFixedRotationC,
        Option<&RotationModelC>,
        Option<&PlanetOmegaC>,
        Option<&mut PlanetAngularVelocityC>,
        Option<&PfixFrameEntityC>,
    )>,
    mut frame_rots: Query<&mut FrameRotC>,
    mut frame_ang_vels: Query<&mut FrameAngVelC>,
) {
    let polar_params = polar.map(|p| (p.xp, p.yp));
    // Lazy-compute Earth RNP only if needed (most common case). Cache the
    // already-typed `FrameTransform` rather than the bare matrix so the
    // expensive `from_matrix` work (matrix→quat extraction + renormalization)
    // happens once per tick total, not once per EarthRNP entity per tick —
    // all EarthRNP entities share the same rotation each step.
    type EarthRot =
        jeod_sim::FrameTransform<jeod_sim::RootInertial, jeod_sim::PlanetFixed<SelfPlanet>>;
    let mut earth_rotation: Option<EarthRot> = Option::None;
    let mut earth_rotation_raw: Option<glam::DMat3> = Option::None;
    for (entity, mut rot, model, omega, ang_vel, pfix_frame_entity) in &mut query {
        let default_model = jeod_sim::RotationModel::EarthRNP;
        let rotation_model = model.map_or(&default_model, |m| &m.0);
        // Track whether we wrote a rotation this tick — controls
        // `PlanetAngularVelocityC` and pfix frame-entity writes.
        let rotated = !matches!(rotation_model, jeod_sim::RotationModel::None);
        // Capture the raw DMat3 too so we can sync the pfix frame
        // entity (FrameRotC + FrameAngVelC) from the same data.
        let mut raw_matrix: Option<glam::DMat3> = None;
        match rotation_model {
            jeod_sim::RotationModel::None => {}
            jeod_sim::RotationModel::EarthRNP => {
                let mat = *earth_rotation_raw.get_or_insert_with(|| {
                    jeod_sim::compute_t_parent_this_from_tjt_with_polar(
                        sim_time.gmst_seconds,
                        sim_time.tt_tjt(),
                        polar_params,
                    )
                });
                let rotation = *earth_rotation.get_or_insert_with(|| {
                    // allowed: matrix is JEOD's RNP-derived rotation; the
                    // RootInertial → PlanetFixed<SelfPlanet> phantoms match the kernel
                    // by construction
                    jeod_sim::FrameTransform::from_matrix(mat)
                });
                rot.0 = rotation;
                raw_matrix = Some(mat);
            }
            jeod_sim::RotationModel::MarsIAU => {
                let tt_s_since_j2000 =
                    (sim_time.tt_tjt() - jeod_sim::J2000_TT_TJT) * jeod_sim::SECONDS_PER_DAY;
                let mat = jeod_sim::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                // allowed: matrix from JEOD-ported IAU Mars rotation formula
                rot.0 = jeod_sim::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
            jeod_sim::RotationModel::MoonIAU => {
                let tdb_jd = sim_time.tdb_julian_date();
                let tdb_s_since_j2000 =
                    (tdb_jd - jeod_sim::J2000_TT_JD) * jeod_sim::SECONDS_PER_DAY;
                let mat = jeod_sim::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                // allowed: matrix from JEOD-ported IAU Moon rotation formula
                rot.0 = jeod_sim::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
            jeod_sim::RotationModel::MoonDE421 => {
                let eph = ephemeris.as_ref().expect(
                    "RotationModel::MoonDE421 requires the EphemerisR resource with a BPC \
                     loaded. Insert EphemerisR before stepping the simulation, or switch the \
                     body to RotationModel::MoonIAU.",
                );
                let tdb_jd = sim_time.tdb_julian_date();
                let mat = eph
                    .get_body_rotation(jeod_sim::EphemerisBody::Moon, tdb_jd)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Moon DE421 BPC rotation query failed at TDB JD {tdb_jd}: {err:?}. \
                             The loaded BPC kernel does not cover this epoch; load a kernel \
                             whose coverage includes the simulation epoch."
                        )
                    });
                // allowed: matrix from NASA SPICE BPC kernel (DE421 / Moon-PA)
                rot.0 = jeod_sim::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
        }

        // ── Planet angular velocity ──
        // JEOD `planet_rnp.cc` writes `ang_vel_this = [0, 0, planet_omega]`
        // on the pfix frame node. Mirror that on (a) the
        // `PlanetAngularVelocityC` ECS component and (b) the pfix
        // frame entity's `FrameRotC` / `FrameAngVelC` so velocity
        // composition both via the typed component and via
        // [`crate::frame_param::RelativeFrameState`] reads the
        // correct rate.
        if rotated {
            // Falling back to `0.0` for a rotating planet (`RotationModelC`
            // present but `PlanetOmegaC` absent) silently misreports the
            // pfix angular velocity as zero for manual-spawn call sites
            // that include `PlanetFixedRotationC` + `RotationModelC` but
            // not `PlanetOmegaC`. Map the rotation model to the
            // canonical `PlanetConfig::omega` when the explicit
            // override is absent.
            let default_omega = match rotation_model {
                jeod_sim::RotationModel::None => 0.0,
                jeod_sim::RotationModel::EarthRNP => jeod_sim::EARTH.omega,
                jeod_sim::RotationModel::MarsIAU => jeod_sim::MARS.omega,
                jeod_sim::RotationModel::MoonIAU | jeod_sim::RotationModel::MoonDE421 => {
                    jeod_sim::MOON.omega
                }
            };
            let omega_value = omega.map(|o| o.0).unwrap_or(default_omega);
            if let Some(mut ang_vel_c) = ang_vel {
                // Mint `AngularVelocity<PlanetFixed<SelfPlanet>>` from the
                // scalar `PlanetOmegaC`. JEOD's `planet_rnp.cc` writes
                // [0, 0, omega] in the pfix frame; this is the typed-API
                // boundary for that scalar → typed-vector lift.
                type PlanetAngVel = jeod_sim::AngularVelocity<jeod_sim::PlanetFixed<SelfPlanet>>;
                let raw = glam::DVec3::new(0.0, 0.0, omega_value);
                ang_vel_c.0 = PlanetAngVel::from_raw_si(raw); // allowed: scalar omega → typed AngularVelocity boundary
            }
            // Write the pfix frame entity's FrameRotC / FrameAngVelC.
            // When `PfixFrameEntityC` is present the referenced entity
            // must be alive with FrameRotC / FrameAngVelC intact
            // (spawned by `register_pfix_frames_system`, torn down in
            // lockstep with the marker by the despawn observers and
            // the rotation-toggle retirement path). A stale handle
            // here would silently desync the pfix-frame state from
            // the rotation matrix on `PlanetFixedRotationC`.
            if let (Some(matrix), Some(pfix_fe)) = (raw_matrix, pfix_frame_entity) {
                let mut frame_rot = frame_rots.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system: source {entity:?} has \
                         PfixFrameEntityC({:?}) but that entity has no \
                         FrameRotC ({err:?}). The pfix frame entity must be \
                         alive with FrameRotC / FrameAngVelC attached \
                         (spawned by register_pfix_frames_system). Either \
                         remove the stale PfixFrameEntityC marker before \
                         despawning the pfix frame entity, or ensure the \
                         pfix frame entity stays alive for as long as the \
                         source carries the handle.",
                        pfix_fe.0
                    )
                });
                frame_rot.q_parent_this =
                    jeod_sim::JeodQuat::left_quat_from_transformation(&matrix);
                frame_rot.t_parent_this = matrix;
                let mut frame_av = frame_ang_vels.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system: source {entity:?} has \
                         PfixFrameEntityC({:?}) but that entity has no \
                         FrameAngVelC ({err:?}). The pfix frame entity must \
                         be alive with FrameRotC / FrameAngVelC attached \
                         (spawned by register_pfix_frames_system). Either \
                         remove the stale PfixFrameEntityC marker before \
                         despawning the pfix frame entity, or ensure the \
                         pfix frame entity stays alive for as long as the \
                         source carries the handle.",
                        pfix_fe.0
                    )
                });
                frame_av.0 = glam::DVec3::new(0.0, 0.0, omega_value);
            }
        } else {
            // `RotationModel::None`: actively clear the rotation
            // matrix, angular velocity, and pfix-frame state. Without
            // this, a runtime toggle from a rotating model to `None`
            // would leave the last-tick rotation matrix on
            // `PlanetFixedRotationC`, the last-tick omega on
            // `PlanetAngularVelocityC`, and the last-tick state on
            // the pfix frame entity — so frame-tree queries would
            // still report a rotating planet-fixed frame even though
            // the source is configured as non-rotating.
            // allowed: explicit identity clear when rotation model toggles to None;
            // the RootInertial → PlanetFixed<SelfPlanet> phantoms are correct by
            // construction (same shape as the rotating-branch from_matrix sites).
            rot.0 = jeod_sim::FrameTransform::from_matrix(glam::DMat3::IDENTITY);
            if let Some(mut ang_vel_c) = ang_vel {
                type PlanetAngVel = jeod_sim::AngularVelocity<jeod_sim::PlanetFixed<SelfPlanet>>;
                ang_vel_c.0 = PlanetAngVel::from_raw_si(glam::DVec3::ZERO); // allowed: zero-omega clear → typed AngularVelocity boundary
            }
            if let Some(pfix_fe) = pfix_frame_entity {
                // Clear the pfix frame entity's state to identity so
                // any `RelativeFrameState` reader sees the source as
                // non-rotating. When `PfixFrameEntityC` is present,
                // the referenced entity must be alive with
                // FrameRotC / FrameAngVelC intact; silently skipping
                // the clear would leave the pfix rotation/omega
                // frozen at the last rotating-tick value while the
                // toggle-to-`None` removes the public component.
                let mut frame_rot = frame_rots.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system (RotationModel::None \
                         clear): source {entity:?} has PfixFrameEntityC({:?}) \
                         but that entity has no FrameRotC ({err:?}). The \
                         pfix frame entity must be alive with FrameRotC / \
                         FrameAngVelC attached (spawned by \
                         register_pfix_frames_system). Either remove the \
                         stale PfixFrameEntityC marker before despawning \
                         the pfix frame entity, or ensure the pfix frame \
                         entity stays alive for as long as the source \
                         carries the handle.",
                        pfix_fe.0
                    )
                });
                *frame_rot = FrameRotC::default();
                let mut frame_av = frame_ang_vels.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system (RotationModel::None \
                         clear): source {entity:?} has PfixFrameEntityC({:?}) \
                         but that entity has no FrameAngVelC ({err:?}). The \
                         pfix frame entity must be alive with FrameRotC / \
                         FrameAngVelC attached (spawned by \
                         register_pfix_frames_system). Either remove the \
                         stale PfixFrameEntityC marker before despawning \
                         the pfix frame entity, or ensure the pfix frame \
                         entity stays alive for as long as the source \
                         carries the handle.",
                        pfix_fe.0
                    )
                });
                frame_av.0 = glam::DVec3::ZERO;

                // Retire the pfix frame entity for reuse on the next
                // toggle back to a rotating model. The orphan entity
                // stays alive (its `ChildOf(source_frame_entity)`
                // edge is preserved so the `RetiredPfixFrameEntityC`
                // reuse path in `register_pfix_frames_system`
                // doesn't have to re-parent), its `Name` is
                // overwritten with a stable `.retired` sentinel so
                // name-based lookups won't shadow a future live
                // entity, and the source's `PfixFrameEntityC` is
                // removed and replaced with `RetiredPfixFrameEntityC`.
                // This bounds the world's pfix-frame entity count at
                // one per source regardless of toggle-cycle count.
                commands
                    .entity(pfix_fe.0)
                    .insert(Name::new(format!("pfix.retired:{:?}", pfix_fe.0)));
                commands
                    .entity(entity)
                    .remove::<PfixFrameEntityC>()
                    .insert(RetiredPfixFrameEntityC(pfix_fe.0));
            }
        }
    }
}

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
/// Scheduled in [`crate::JeodSet::EphemerisUpdate`] alongside
/// `planet_fixed_rotation_system` so the joint frame's rotation /
/// angular velocity are current before any consumer that walks the
/// frame tree (gravity, derived state, integration) reads them.
///
/// "Kinematic" means the angle is an *input*, not an integrated
/// state — there is no torque, inertia, or momentum. Joint dynamics
/// (free-swinging joints, IK, constraint-derived joint forces) are
/// out of scope; see the deferred-dynamics meta.
pub fn joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<(&JointKinematicsC, &mut FrameRotC, &mut FrameAngVelC)>,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) = jeod_sim::evaluate_joint_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// Computes tidal ΔC20 for each gravity source that has a `TidalConfigC`.
///
/// Runs after `planet_fixed_rotation_system` so the rotation matrix is current.
/// Sources without `TidalConfigC` keep their default `TidalDeltaC20C::default()`
/// (a zero-valued [`jeod_sim::Ratio`]).
pub fn tidal_update_system(
    mut query: Query<(&TidalConfigC, &PlanetFixedRotationC, &mut TidalDeltaC20C)>,
) {
    for (config, rotation, mut delta) in &mut query {
        // `TidalConfigC` already wraps `TidalConfigTyped` — the dimensional
        // lift happened once at insertion (`TidalConfigC::from_untyped`),
        // so the system reads the typed value directly with no per-tick
        // `Vec` allocation or per-body f64 → typed conversion.
        // `compute_delta_c20_typed` returns `Ratio`, matching
        // `TidalDeltaC20C`'s storage type.
        delta.0 = jeod_sim::compute_delta_c20_typed(&config.0, rotation.0.matrix_ref());
    }
}

/// Updates source positions from DE4xx ephemeris each step.
///
/// Queries entities with `EphemerisBodyC` + `SourceInertialPositionC` and
/// looks up the current position/velocity from the `EphemerisR` resource.
/// Also updates `SourceInertialVelocityC` and `TranslationalStateC` when
/// present (velocity for relativistic corrections; translational state for
/// Sun/Moon entities used by SRP, solar beta, and earth lighting systems).
///
/// Placed in `JeodSet::EphemerisUpdate`.
pub fn ephemeris_update_system(
    ephemeris: Option<Res<crate::EphemerisR>>,
    sim_time: Res<SimulationTimeR>,
    mut query: Query<(
        &EphemerisBodyC,
        &mut SourceInertialPositionC,
        Option<&mut SourceInertialVelocityC>,
        Option<&mut TranslationalStateC>,
    )>,
) {
    let Some(eph) = ephemeris else {
        return;
    };
    let tdb_jd = sim_time.tdb_julian_date();
    for (ephem_body, mut source_pos, source_vel, trans_state) in &mut query {
        // Typed sibling: returns `(Position<RootInertial>, Velocity<RootInertial>)`
        // directly, matching the typed component storage. Bit-identical to
        // the deprecated f64 path — the kernel itself extracts SI base
        // values from ANISE and re-wraps them.
        let (pos_typed, vel_typed) = eph
            .get_state_typed(ephem_body.target, ephem_body.observer, tdb_jd)
            .unwrap_or_else(|e| {
                panic!(
                    "Ephemeris lookup failed for {:?} wrt {:?} at TDB JD {tdb_jd}: {e}",
                    ephem_body.target, ephem_body.observer,
                )
            });
        source_pos.0 = pos_typed;
        if let Some(mut sv) = source_vel {
            sv.0 = vel_typed;
        }
        if let Some(mut ts) = trans_state {
            // TranslationalStateC wraps `TranslationalStateTyped<PlanetInertial<SelfPlanet>>`;
            // `pos_typed` / `vel_typed` are root-inertial-tagged by the
            // ephemeris API. Relabel via `from_raw_si` to the
            // wildcard-tagged planet-inertial frame the Component
            // stores. The numeric SI values (m, m/s) are preserved
            // exactly — only the phantom tag changes.
            type PiPos = jeod_sim::Position<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>;
            type PiVel = jeod_sim::Velocity<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>;
            ts.0.position = PiPos::from_raw_si(pos_typed.raw_si()); // allowed: ephemeris boundary, RootInertial → PlanetInertial<SelfPlanet> wildcard relabel
            ts.0.velocity = PiVel::from_raw_si(vel_typed.raw_si()); // allowed: same ephemeris boundary relabel
        }
    }
}

// ── Dynamics ──

/// Recompute derived mass quantities (`inverse_mass`, `inverse_inertia`) each step.
///
/// Port of JEOD's `(DYNAMICS, "scheduled") dyn_body.mass.update_mass_properties()`.
/// JEOD runs this every timestep so that runtime mass changes (fuel burn,
/// staging, attach/detach) are reflected in the dynamics before the next
/// derivative computation.
///
/// Placed before `JeodSet::EphemerisUpdate` so gravity and force collection
/// see current mass properties.
///
/// **Change-detection contract**: the dirty-flag check below is read through
/// `Mut::deref` (immutable access), and `recompute_derived()` is only
/// invoked — triggering `DerefMut` and marking the component as
/// `Changed` — when the entity actually needs updating. Without this
/// gate, an unconditional `mass.recompute_derived()` (whose body is a
/// `dirty`-guarded no-op) still triggers `DerefMut` on every entity
/// every tick, and `composite_mass_system`'s downstream
/// `Changed<MassPropertiesC>` filter would match every parent every
/// tick — corrupting the `CoreMassPropertiesC` cache by reseeding it
/// from the previous-tick composite. The `dirty` field is only set
/// `true` by mission code that genuinely mutates `mass`/`inertia`, so
/// it is the correct signal here.
pub fn mass_update_system(mut query: Query<&mut MassPropertiesC>) {
    for mut mass in &mut query {
        // Read `dirty` via `Mut::deref` (no `DerefMut`), so entities
        // that don't need recomputation are not falsely marked
        // `Changed`. `recompute_derived` is itself a no-op when
        // `!dirty`, so the gate preserves behavior.
        if mass.0.dirty {
            mass.recompute_derived();
        }
    }
}

/// Collects non-gravity forces and all torques into `TotalForceC`.
///
/// Delegates to [`jeod_sim::collect_and_resolve_forces`] for frame-aware
/// force/torque aggregation and frame derivative computation.
///
/// Gravity is intentionally **excluded** because the integration system
/// recomputes it at each RK4 stage for 4th-order accuracy. Non-gravity
/// forces (aero, SRP) are approximately constant over one timestep and
/// are added to the per-stage gravity inside the integrator.
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
#[allow(clippy::type_complexity)]
pub fn force_collection_system(
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; their
    // `TotalForceC` / `FrameDerivativesC` are no longer consumed by
    // any integrator. Skip them so downstream consumers don't see
    // stale aggregated forces on bodies that aren't reacting to them.
    mut query: Query<
        (
            &mut TotalForceC,
            Option<&mut FrameDerivativesC>,
            Option<&GravityAccelerationC>,
            Option<&RotationalStateC>,
            Option<&MassPropertiesC>,
            Option<&AerodynamicForceC>,
            Option<&RadiationForceC>,
            Option<&GravityTorqueC>,
            Option<&StructuralTransformC>,
            Option<&ExternalForceC>,
            Option<&ExternalTorqueC>,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
) {
    for (
        mut total,
        derivs,
        grav,
        rot_state,
        mass,
        aero,
        srp,
        grav_torque,
        struct_xform,
        ext_force,
        ext_torque,
    ) in &mut query
    {
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
        // `GravityAccelerationC` stores `Acceleration<RootInertial>`; the
        // existing `collect_and_resolve_forces` kernel takes a raw
        // `DVec3`, so drop the phantom here. The kernel's frame
        // contract (gravity in inertial) matches the component's
        // phantom by construction.
        let grav_accel = grav.map_or(DVec3::ZERO, |g| g.grav_accel.raw_si());

        // Map Bevy component references to jeod_interactions types for jeod_sim.
        let aero_ref = aero.map(|a| jeod_sim::AerodynamicForce {
            force: a.force,
            torque: a.torque,
        });
        let srp_ref = srp.map(|s| jeod_sim::RadiationForce {
            force: s.force,
            torque: s.torque,
        });
        // GravityTorqueC stores `Torque<BodyFrame<SelfRef>>`; the
        // untyped `collect_and_resolve_forces` boundary still expects a
        // raw `DVec3` in the body frame — drop the phantom at the call
        // site only.
        let gravity_torque_val = grav_torque.map(|gt| gt.0.raw_si());

        // RotationalStateC and MassPropertiesC now wrap typed siblings;
        // convert to untyped at the kernel boundary. (The kernel
        // signature still takes the untyped form. Migrating the kernel
        // signature itself is out of scope for the ECS-surface typing;
        // the win here is at the ECS surface where mission code
        // interacts.)
        let rot_untyped = rot_state.map(|r| r.0.to_untyped());
        let mass_untyped = mass.map(|m| m.0.to_untyped());

        let (collected, frame_derivs_raw) = jeod_sim::collect_and_resolve_forces(
            aero_ref.as_ref(),
            srp_ref.as_ref(),
            gravity_torque_val,
            rot_untyped.as_ref(),
            t_struct_body,
            mass_untyped.as_ref(),
            grav_accel,
        );

        // The kernel returns untyped TotalForce / FrameDerivatives;
        // re-wrap as the component's typed form. The `RootInertial` and
        // `BodyFrame<SelfRef>` phantoms match the kernel's documented
        // frame contracts (force inertial, torque body).
        total.0 =
            // allowed: typed↔untyped kernel boundary; the kernel signature in
            // jeod_sim is still untyped, so re-wrapping is the canonical
            // adapter pattern (analogous to the From<Untyped> impls in
            // src/components.rs).
            jeod_sim::TotalForceTyped::<jeod_sim::SelfRef, RootInertial>::from_untyped_unchecked(
                &collected,
            );
        let mut frame_derivs =
            // allowed: typed↔untyped kernel boundary, see TotalForceTyped comment above
            jeod_sim::FrameDerivativesTyped::<RootInertial, jeod_sim::SelfRef>::from_untyped_unchecked(
                &frame_derivs_raw,
            );

        // Apply external force/torque (set by caller between steps).
        // Matches simulation.rs:846-855 logic. ExternalForceC and
        // ExternalTorqueC carry typed phantoms; the totals are typed
        // too, so the accumulator stays in typed land throughout.
        if let Some(ef) = ext_force {
            if ef.0.raw_si() != DVec3::ZERO {
                total.0.force += ef.0;
                if let Some(mass) = mass {
                    // `Force<RootInertial> / Mass → Acceleration<RootInertial>`
                    // is the typed identity here; we go through raw_si
                    // for the scalar inverse_mass multiply (it's an
                    // untyped f64 by design — see jeod_dynamics::mass
                    // doc on why inverse_mass stays untyped).
                    let accel_contrib = ef.0.raw_si() * mass.0.inverse_mass;
                    frame_derivs.trans_accel +=
                        // allowed: scalar inverse_mass is untyped by design; rewrap.
                        Acceleration::<RootInertial>::from_raw_si(accel_contrib);
                }
            }
        }
        if let Some(et) = ext_torque {
            if et.0.raw_si() != DVec3::ZERO {
                total.0.torque += et.0;
                if let Some(mass) = mass {
                    let alpha_contrib = mass.0.inverse_inertia * et.0.raw_si();
                    frame_derivs.rot_accel +=
                        // allowed: same untyped inverse_inertia boundary as above.
                        AngularAcceleration::<BodyFrame<SelfRef>>::from_raw_si(alpha_contrib);
                }
            }
        }

        if let Some(mut derivs) = derivs {
            derivs.0 = frame_derivs;
        }
    }
}

/// Advances translational (and optionally rotational) state by one timestep.
///
/// Delegates to [`jeod_sim::integrate_body`] for 6-DOF/3-DOF routing and
/// integration stepping. Gravity is recomputed at each intermediate state
/// for proper multi-stage accuracy.
///
/// The integration method is determined by the optional `IntegratorTypeC`
/// component (RK4, RKF45, GaussJackson, Abm4). When absent, RK4 is used.
/// GaussJackson requires `GaussJacksonStateC`; ABM4 requires `Abm4StateC`.
///
/// Per-body integration-frame origins (relative to root) are queried via
/// the [`FrameOrigin`] SystemParam, which walks the ECS frame hierarchy
/// (`Query<&ChildOf>` on the body's frame entity).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn integration_system(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // The body query filter excludes two disjoint populations:
    //   * Kinematic-chain children — composite-rigid-body integration
    //     only advances the root of every `MassChildOf` chain.
    //     `wrench_aggregation_system` tags every non-root chain member
    //     with `KinematicChildC`. Without this filter, zeroing a
    //     child's `TotalForceC` would not be enough — the per-RK-stage
    //     gravity recompute below would still drift the child's
    //     translational state every step.
    //   * Detached subtrees — advanced ballistically by
    //     `step_detached_system`. Integrating them here would
    //     double-step the same entity per tick, mirroring the runner
    //     split between `Simulation::bodies` and
    //     `Simulation::detached_subtrees`.
    // See `KinematicChildC` and `DetachedSubtreeStateC` for the
    // detailed lifecycles.
    // JEOD_INV: DB.17 — kinematic children skip integration.
    // JEOD_INV: DB.21 — detached subtrees skip integration.
    mut bodies: Query<
        (
            Entity,
            &DynamicsConfigC,
            &mut TranslationalStateC,
            Option<&mut RotationalStateC>,
            Option<&MassPropertiesC>,
            &GravityControlsC,
            &mut TotalForceC,
            Option<&IntegratorTypeC>,
            Option<&mut GaussJacksonStateC>,
            Option<&mut Abm4StateC>,
            Option<&mut FlatPlateConfigC>,
            Option<&StructuralTransformC>,
            Option<&mut RadiationForceC>,
            Option<&mut FrameDerivativesC>,
            Option<&FrameEntityC>,
        ),
        (
            Without<KinematicChildC>,
            Without<crate::DetachedSubtreeStateC>,
        ),
    >,
    sources: Query<
        (
            &GravitySourceC,
            Option<&PlanetFixedRotationC>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
            Option<&TidalDeltaC20C>,
            Option<&TidalConfigC>,
            // Fallback velocity source for ephemeris-driven sources (Sun /
            // Moon via SunBundle / MoonBundle) that don't carry
            // SourceInertialVelocityC.
            Option<&TranslationalStateC>,
        ),
        // Static disjointness vs. the `bodies` query's `&mut
        // TranslationalStateC`: no integrated body is also a gravity
        // source. Without this filter Bevy can't prove the queries
        // don't alias and panics with `assert_component_access_compatibility`.
        Without<DynamicsConfigC>,
    >,
    time: Res<Time<Fixed>>,
    sim_time: Res<SimulationTimeR>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }
    // Dynamic timestep matches `jeod_runner::run_integration`'s
    // `integ_dt = sim_dt * time_scale_factor` so reversed/scaled time
    // produces consistent gravity at RK sub-stages.
    let integ_dt = dt * sim_time.0.time_scale_factor;

    // Helper closure for gravity at an intermediate state — reused by both
    // the standard and coupled dispatch branches. The integrator passes
    // raw `DVec3` per-stage states (the integrator internals are not
    // yet typed); we wrap into `Position<RootInertial>` / `Velocity<RootInertial>`
    // for the typed `*_typed` kernels and unwrap before returning.
    //
    // `integ_origin_pos` / `integ_origin_vel` are the per-body integration
    // frame's translational state (relative to root) at step start. For
    // root-integrated bodies both are zero — the original behavior. For
    // non-root bodies the integ frame may itself be moving, so each
    // RK sub-stage advances the origin linearly by `time_frac * integ_dt`,
    // matching `jeod_runner::run_integration`. Source positions are
    // similarly interpolated when the integ frame moves, so the Newtonian
    // gravity field stays consistent across stages. PPN (relativistic)
    // corrections use step-start source state — runner does the same
    // (`step/integrate.rs:199-202`).
    let eval_gravity = |entity: Entity,
                        controls: &GravityControlsC,
                        pos: DVec3,
                        vel: DVec3,
                        integ_origin_pos: DVec3,
                        integ_origin_vel: DVec3,
                        time_frac: f64|
     -> DVec3 {
        // Per-stage interpolation of the integration frame's origin and
        // each source's position, mirroring jeod_runner's pattern in
        // `step/integrate.rs:172-184`. `sub_dt` is gated on the integ
        // frame actually moving so root-integrated bodies stay
        // bit-identical to the pre-N3 path.
        let stage_dt = time_frac * integ_dt;
        let stage_origin_pos = integ_origin_pos + integ_origin_vel * stage_dt;
        let sub_dt = if integ_origin_vel != DVec3::ZERO {
            stage_dt
        } else {
            0.0
        };
        // The standard `integrate_body` (and `integrate_body_coupled`
        // for the thermal-SRP path) accept a `gravity_fn` closure
        // that receives raw `DVec3` per-stage state. These lifts are
        // inside `jeod_sim` boundary territory, not at the Bevy ECS
        // surface where the typed quantities live.
        let typed_abs_pos = Position::<RootInertial>::from_raw_si(pos + stage_origin_pos); // allowed: integrator-kernel boundary
        let typed_abs_vel = Velocity::<RootInertial>::from_raw_si(vel + integ_origin_vel); // allowed: integrator-kernel boundary
        let typed_origin = Position::<RootInertial>::from_raw_si(stage_origin_pos); // allowed: integrator-kernel boundary

        // Helper: resolve a source's effective velocity, falling back to
        // `TranslationalStateC.velocity` when the explicit
        // `SourceInertialVelocityC` component is absent. Without the
        // fallback, ephemeris-driven Sun/Moon sources (spawned via
        // SunBundle/MoonBundle, which include `TranslationalStateC`
        // but not `SourceInertialVelocityC`) get treated as stationary
        // at every RK sub-stage.
        let source_vel =
            |v: Option<&SourceInertialVelocityC>, ts: Option<&TranslationalStateC>| -> DVec3 {
                v.map(|v| v.0.raw_si())
                    .or_else(|| ts.map(|t| t.0.velocity.raw_si()))
                    .unwrap_or(DVec3::ZERO)
            };

        let typed_accel = jeod_sim::accumulate_gravity_typed(
            typed_abs_pos,
            &controls.0,
            typed_origin,
            |source_entity| match sources.get(source_entity) {
                Ok((s, r, p, v, tidal, tidal_config, ts)) => {
                    let base_pos = p.0.raw_si();
                    let stage_pos = if sub_dt != 0.0 {
                        base_pos + source_vel(v, ts) * sub_dt
                    } else {
                        base_pos
                    };
                    Some(jeod_sim::ResolvedSource {
                        source: &s.0,
                        rotation: r.map(|r| r.0.matrix_ref()),
                        position: stage_pos,
                        delta_c20: tidal.map_or(0.0, |t| t.0.value),
                        has_delta_coeffs: tidal_config.is_some(),
                    })
                }
                Err(_) => {
                    panic!(
                        "Entity {entity:?}: GravityControl references source \
                         {source_entity:?} which does not exist or lacks \
                         GravitySourceC + SourceInertialPositionC."
                    );
                }
            },
        );
        let mut accel = typed_accel.grav_accel.raw_si();

        // PPN (relativistic) corrections use step-start source positions
        // and velocities — `jeod_runner::run_integration` snapshots both
        // outside the per-stage closure (`step/integrate.rs:199-202`),
        // so per-stage interpolation here would drift from runner.
        let rel = jeod_sim::accumulate_relativistic_corrections_typed(
            typed_abs_pos,
            typed_abs_vel,
            &controls.0,
            |source_entity| {
                sources
                    .get(source_entity)
                    .ok()
                    .map(|(s, _, p, v, _, _, ts)| {
                        // Step-start values for PPN — runner does the
                        // same (snapshots `src_pos`/`src_vel` outside
                        // the per-stage closure).
                        jeod_sim::ResolvedRelativisticSource {
                            mu: s.mu,
                            position: p.0.raw_si(),
                            velocity: source_vel(v, ts),
                        }
                    })
            },
        );
        accel += rel.raw_si();

        accel
    };

    for (
        entity,
        config,
        mut state,
        mut rot_state,
        mass,
        controls,
        mut total_force,
        integrator,
        mut gj_state,
        mut abm4_state,
        mut flat_config,
        struct_xform,
        mut srp_force,
        mut frame_derivs,
        body_frame_entity,
    ) in &mut bodies
    {
        // Per-body integration-frame origin (relative to root). Computed
        // once per step — the integ frame doesn't move during a single
        // integration step, so the multi-stage RK4 sub-evaluations
        // reuse the same value.
        //
        // The body's integration frame is the parent of its frame
        // entity in the ECS hierarchy (set at registration by
        // `register_body_frames_system`). Bodies registered before
        // the frames-as-entities components landed have no
        // `FrameEntityC`; treat those as root-integrated, matching
        // the pre-migration default.
        let integ_frame_entity = body_frame_entity
            .and_then(|fe| parents.get(fe.0).ok().map(|child_of| child_of.parent()));
        let (integ_origin_pos, integ_origin_vel) = match integ_frame_entity {
            Some(integ_e) if integ_e != root_frame_entity.0 => {
                frame_origin.origin_in(root_frame_entity.0, integ_e)
            }
            _ => (DVec3::ZERO, DVec3::ZERO),
        };
        let integrator_type = integrator.map_or(jeod_sim::IntegratorType::Rk4, |c| c.0);
        if matches!(integrator_type, jeod_sim::IntegratorType::GaussJackson(..)) {
            assert!(
                gj_state.is_some(),
                "Entity {entity:?}: IntegratorTypeC is GaussJackson but \
                 GaussJacksonStateC component is missing. Create the state \
                 from the same config used in IntegratorTypeC, e.g.: \
                 GaussJacksonStateC(GaussJacksonState::new(config))"
            );
        }
        if matches!(integrator_type, jeod_sim::IntegratorType::Abm4) {
            assert!(
                abm4_state.is_some(),
                "Entity {entity:?}: IntegratorTypeC is Abm4 but \
                 Abm4StateC component is missing. Add \
                 Abm4StateC(Abm4State::new()) to the entity."
            );
        }

        // Derivative-class thermal fork: the SRP system cached step-start
        // inputs into `flat_config.stage_inputs`. Recompute SRP force +
        // temperature derivatives per RK4 stage through
        // `integrate_body_coupled`. See `jeod_runner::Simulation::step_internal`
        // for the sister implementation.
        let stage_inputs_and_order = flat_config
            .as_ref()
            .and_then(|fc| fc.stage_inputs.map(|si| (si, fc.integration_order)));
        if let Some((srp_inputs, thermal_order)) = stage_inputs_and_order {
            assert!(
                matches!(integrator_type, jeod_sim::IntegratorType::Rk4),
                "Entity {entity:?}: derivative-class ThermalIntegrationOrder \
                 requires RK4 integrator; use Scheduled or switch integrator.",
            );
            let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
            // Drop typed phantoms at the kernel boundary. `total_force`
            // accumulators are typed (`Force<RootInertial>` / `Torque<BodyFrame>`);
            // the integrator API still consumes raw `DVec3`.
            let non_grav_non_srp_force = total_force.force.raw_si();
            let constant_torque = total_force.torque.raw_si();
            let mut final_srp_inertial_force = DVec3::ZERO;
            let mut final_srp_torque = DVec3::ZERO;
            let mut k1_temp_dots: Option<Vec<f64>> = None;
            // Convert typed state to the untyped form the kernel wants.
            // After `integrate_body_coupled` mutates the untyped copies
            // we re-wrap as typed for storage.
            let mass_copy_untyped = mass.map(|m| m.0.to_untyped());
            let mut state_untyped = state.0.to_untyped();
            let mut rot_state_untyped = rot_state.as_ref().map(|r| r.0.to_untyped());
            let thermal = flat_config
                .as_mut()
                .expect("stage_inputs_and_order => flat_config present");
            jeod_sim::integrate_body_coupled(
                config,
                &mut state_untyped,
                rot_state_untyped.as_mut(),
                mass_copy_untyped.as_ref(),
                |stage_trans, stage_rot, stage_thermal, time_frac| {
                    let gravity_accel = eval_gravity(
                        entity,
                        controls,
                        stage_trans.position,
                        stage_trans.velocity,
                        integ_origin_pos,
                        integ_origin_vel,
                        time_frac,
                    );
                    let t_inertial_body = stage_rot.map_or(glam::DMat3::IDENTITY, |r| {
                        r.quaternion.left_quat_to_transformation()
                    });
                    let t_inertial_struct =
                        jeod_sim::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
                    // Per-stage flux recompute from intermediate vehicle
                    // position — matches JEOD's derivative-class
                    // `RadiationSource::calculate_flux`. Sun position is
                    // step-constant (ephemeris is scheduled-class).
                    //
                    // RF.10: `stage_trans.position` is the integrator's
                    // intermediate `DVec3` in the body's *integration*
                    // frame, which equals root inertial only when the
                    // body's frame entity is a direct child of the
                    // root frame entity. For non-root integration we
                    // shift via the per-stage origin before
                    // differencing against `srp_inputs.sun_position`
                    // (which is typed `Position<RootInertial>`). Mirrors
                    // `jeod_runner::run_integration`'s coupled SRP path
                    // (`crates/jeod_runner/src/simulation/step/integrate.rs:299-305`).
                    use jeod_sim::{Position, RootInertial};
                    let stage_dt = time_frac * integ_dt;
                    let stage_origin = if integ_origin_vel != DVec3::ZERO {
                        integ_origin_pos + integ_origin_vel * stage_dt
                    } else {
                        integ_origin_pos
                    };
                    let stage_pos_root: Position<RootInertial> =
                        // allowed: typed-API boundary — `stage_trans.position`
                        // arrives as the integrator's untyped intermediate
                        // DVec3; `stage_pos_root` is the root-inertial value
                        // after the integ-origin shift, ready for the typed
                        // `srp_inputs.sun_position` subtraction.
                        Position::<RootInertial>::from_raw_si(stage_trans.position + stage_origin);
                    let sun_to_vehicle: Position<RootInertial> =
                        stage_pos_root - srp_inputs.sun_position;
                    let sun_to_vehicle = sun_to_vehicle.raw_si();
                    let distance = sun_to_vehicle.length().max(1.0);
                    let stage_flux_inertial_hat = sun_to_vehicle / distance;
                    let stage_flux_mag = jeod_sim::solar_flux_at_distance(distance);
                    let flux_struct_hat = t_inertial_struct * stage_flux_inertial_hat;
                    let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
                        &stage_thermal.plates,
                        &stage_thermal.t_pow4_cached,
                        flux_struct_hat,
                        stage_flux_mag,
                        srp_inputs.center_grav,
                        srp_inputs.illum_factor,
                    );
                    let srp_force_inertial = t_inertial_struct.transpose() * srp_result.force;
                    final_srp_inertial_force = srp_force_inertial;
                    final_srp_torque = srp_result.torque;
                    let temp_dots = match thermal_order {
                        jeod_sim::ThermalIntegrationOrder::DerivativeRk4 => srp_result.temp_dots,
                        jeod_sim::ThermalIntegrationOrder::DerivativeFirstOrder => {
                            if time_frac == 0.0 {
                                k1_temp_dots = Some(srp_result.temp_dots.clone());
                                srp_result.temp_dots
                            } else {
                                k1_temp_dots
                                    .as_ref()
                                    .expect("stage 1 runs before stages 2-4")
                                    .clone()
                            }
                        }
                        jeod_sim::ThermalIntegrationOrder::Scheduled => {
                            unreachable!("Scheduled bodies do not enter the coupled path")
                        }
                    };
                    // `srp_result.torque` is structural-frame per
                    // `FlatPlateSrpResult` docs; `constant_torque` is
                    // body-frame (from `collect_and_resolve_forces`).
                    // Rotate to body frame before summing so the coupled
                    // integrator's rotational dynamics are correct when
                    // `t_struct_body` != IDENTITY.
                    let srp_torque_body = t_struct_body * srp_result.torque;
                    jeod_sim::CoupledStageEval {
                        gravity_accel,
                        non_grav_force: non_grav_non_srp_force + srp_force_inertial,
                        torque: constant_torque + srp_torque_body,
                        temp_dots,
                    }
                },
                &mut thermal.0,
                dt,
                sim_time.0.time_scale_factor,
            );

            // Re-wrap kernel-mutated untyped state back into typed
            // components. The frame phantoms are unchanged (the typed
            // storage's `<PlanetInertial<SelfPlanet>>` /
            // `<BodyFrame<SelfRef>>` are the same frames the kernel
            // was operating in — the kernel computes everything in
            // the body's integration frame, which the Component tags
            // as planet-inertial via the `SelfPlanet` wildcard).
            type PiTrans =
                jeod_sim::TranslationalStateTyped<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>;
            state.0 = PiTrans::from_untyped_unchecked(&state_untyped); // allowed: typed↔untyped kernel boundary (integrate_body_coupled signature is untyped); analogous to From<Untyped> impls.
            if let (Some(rs), Some(ru)) = (rot_state.as_mut(), rot_state_untyped) {
                // allowed: same typed↔untyped kernel boundary as above.
                rs.0 = jeod_sim::RotationalStateTyped::<SelfRef>::from_untyped_unchecked(&ru);
            }

            // Write representative `RadiationForceC` from stage 4 so
            // `VehicleOutput`-equivalent observers still see the SRP force.
            if let Some(ref mut srp_force) = srp_force {
                srp_force.force = final_srp_inertial_force;
                srp_force.torque = final_srp_torque;
            }

            // Backfill `TotalForceC` and `FrameDerivativesC` with the
            // final-stage SRP contribution so downstream observers see
            // SRP-inclusive values, matching the Scheduled-mode invariant
            // that `TotalForceC` / `FrameDerivativesC` reflect every
            // applied force / resulting acceleration. In derivative modes
            // this is a "representative stage" (stage 4) snapshot, same
            // as `RadiationForceC` above.
            // allowed: SRP kernel returns DVec3; re-wrap into the typed
            // accumulators (`Force<RootInertial>` / `Torque<BodyFrame<SelfRef>>`).
            total_force.force += Force::<RootInertial>::from_raw_si(final_srp_inertial_force);
            let final_srp_torque_body = t_struct_body * final_srp_torque;
            // allowed: same SRP-kernel boundary.
            total_force.torque += Torque::<BodyFrame<SelfRef>>::from_raw_si(final_srp_torque_body);
            if let (Some(ref mut fd), Some(mass_p)) = (frame_derivs.as_mut(), mass_copy_untyped) {
                // allowed: typed↔untyped acceleration accumulator boundary.
                fd.trans_accel += Acceleration::<RootInertial>::from_raw_si(
                    final_srp_inertial_force * mass_p.inverse_mass,
                );
                // allowed: typed↔untyped angular-acceleration boundary.
                fd.rot_accel += AngularAcceleration::<BodyFrame<SelfRef>>::from_raw_si(
                    mass_p.inverse_inertia * final_srp_torque_body,
                );
            }
            continue;
        }

        // Standard (Scheduled or no-SRP) path. Same typed↔untyped
        // bridging as the coupled path: extract untyped at entry,
        // re-wrap typed at exit.
        let mut state_untyped = state.0.to_untyped();
        let mut rot_state_untyped = rot_state.as_ref().map(|r| r.0.to_untyped());
        let mass_untyped = mass.map(|m| m.0.to_untyped());
        jeod_sim::integrate_body(
            config,
            &mut state_untyped,
            rot_state_untyped.as_mut(),
            mass_untyped.as_ref(),
            |pos, vel, time_frac| {
                eval_gravity(
                    entity,
                    controls,
                    pos,
                    vel,
                    integ_origin_pos,
                    integ_origin_vel,
                    time_frac,
                )
            },
            total_force.force.raw_si(),
            total_force.torque.raw_si(),
            dt,
            sim_time.0.time_scale_factor,
            integrator_type,
            gj_state.as_mut().map(|g| &mut g.0),
            abm4_state.as_mut().map(|a| &mut a.0),
        );
        // Re-wrap kernel-mutated state back into typed components;
        // integrate_body signature is untyped, so re-wrapping is the
        // canonical adapter step (analogous to From<Untyped> impls).
        state.0 =
            // allowed: typed↔untyped kernel boundary; planet-inertial frame matches the body's integration frame.
            jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>::from_untyped_unchecked(&state_untyped);
        if let (Some(rs), Some(ru)) = (rot_state.as_mut(), rot_state_untyped) {
            // allowed: typed↔untyped kernel boundary
            rs.0 = jeod_sim::RotationalStateTyped::<SelfRef>::from_untyped_unchecked(&ru);
        }
    }
}

// ── Gravity ──

/// Pre-computes gravity for each dynamic body.
///
/// Gravity is precomputed here in the Environment stage but is recomputed at
/// each integrator stage by the integration system for multi-stage accuracy.
///
/// Delegates to [`jeod_sim::accumulate_gravity`] for the per-body accumulation
/// loop, providing a closure that resolves Bevy entity references.
///
/// Bodies whose frame entity is a child of a non-root integration frame
/// have their integration-frame origin (relative to root inertial)
/// added to `body.position` to recover the absolute inertial position
/// for the gravity field; the same origin is passed to
/// [`jeod_sim::accumulate_gravity_typed`] so the differential gravity
/// correction subtracts the integ frame's own acceleration toward each
/// source. The integration frame is determined from the body's
/// `FrameEntityC` parent via `Query<&ChildOf>` (no explicit
/// integration-frame handle component), and the origin is queried via
/// the [`FrameOrigin`] SystemParam — typed
/// `(Position<RootInertial>, Velocity<RootInertial>)` directly, so no
/// `from_raw_si` lift is needed at the boundary.
#[allow(clippy::type_complexity)]
pub fn gravity_computation_system(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // Filter excludes detached subtrees: only attached bodies participate
    // in gravity / force-collection / integration. Detached subtrees coast
    // ballistically (no force, no torque) via `step_detached_system`, so
    // populating `GravityAccelerationC` on them is wasted work and would
    // expose stale values to diagnostics / logging consumers. Mirrors the
    // runner's split between `Simulation::bodies` and
    // `Simulation::detached_subtrees` — gravity is only evaluated on the
    // integrated set.
    // JEOD_INV: DB.21 — detached subtrees skip gravity evaluation.
    mut bodies: Query<
        (
            Entity,
            &TranslationalStateC,
            &GravityControlsC,
            &mut GravityAccelerationC,
            Option<&FrameEntityC>,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
    sources: Query<(
        &GravitySourceC,
        Option<&PlanetFixedRotationC>,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TidalDeltaC20C>,
        Option<&TidalConfigC>,
        // Fallback velocity source for ephemeris-driven sources that
        // don't carry SourceInertialVelocityC.
        Option<&TranslationalStateC>,
    )>,
) {
    for (entity, state, controls, mut accel, body_frame) in &mut bodies {
        // `TranslationalStateC` stores typed
        // `Position<PlanetInertial<SelfPlanet>>` /
        // `Velocity<PlanetInertial<SelfPlanet>>`. For root-integrated
        // bodies the integ frame numerically equals root inertial, so
        // the raw values match what gravity wants. For non-root
        // bodies we shift to absolute root-inertial coordinates below
        // via the body frame entity's parent +
        // `FrameOrigin::origin_in_root`. The shift is the only safe
        // path from `PlanetInertial<P>` to `RootInertial` (RF.10);
        // relabel via `from_raw_si` so the compiler accepts the
        // addition with the typed root-inertial origin.
        let body_pos = Position::<RootInertial>::from_raw_si(state.position.raw_si()); // allowed: gravity shift-site, planet-inertial → root-inertial via integ-origin offset
        let body_vel = Velocity::<RootInertial>::from_raw_si(state.velocity.raw_si()); // allowed: same gravity shift-site

        // Integration-frame origin (relative to root) — zero for
        // root-integrated bodies. Shared helper documents the cases
        // (no `FrameEntityC` legacy entity → zero; integ frame is the
        // root frame → zero; otherwise walk via `FrameOrigin`).
        let (integ_origin, integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let abs_pos = body_pos + integ_origin;

        let typed_accel = jeod_sim::accumulate_gravity_typed(
            abs_pos,
            &controls.0,
            integ_origin,
            |source_entity| match sources.get(source_entity) {
                Ok((source, rot, pos, _, tidal, tidal_config, _)) => {
                    Some(jeod_sim::ResolvedSource {
                        source: &source.0,
                        rotation: rot.map(|r| r.0.matrix_ref()),
                        position: pos.0.raw_si(),
                        delta_c20: tidal.map_or(0.0, |t| t.0.value),
                        // JEOD gates on n_deltacoeffs > 0 (tidal config
                        // present), not on whether ΔC20 component exists.
                        has_delta_coeffs: tidal_config.is_some(),
                    })
                }
                Err(_) => {
                    panic!(
                        "Entity {entity:?}: GravityControl references source \
                         {source_entity:?} which does not exist or lacks \
                         GravitySourceC + SourceInertialPositionC."
                    );
                }
            },
        );
        accel.0 = typed_accel;

        // Apply relativistic (post-Newtonian PPN) corrections after Newtonian
        // gravity, matching Simulation::step() stage 4b ordering. PPN depends
        // on |r_body - r_source| and v_body in inertial frame, so for
        // non-root-integrated bodies we lift `body_pos`/`body_vel` from
        // integ-frame coords into absolute inertial coords first.
        // Reuse the typed origin computed above: (integ_origin,
        // integ_origin_vel) are already Position/Velocity<Inertial>.
        let abs_body_pos = body_pos + integ_origin;
        let abs_body_vel = body_vel + integ_origin_vel;
        let rel_accel = jeod_sim::accumulate_relativistic_corrections_typed(
            abs_body_pos,
            abs_body_vel,
            &controls.0,
            |source_entity| {
                sources
                    .get(source_entity)
                    .ok()
                    .map(|(s, _, p, v, _, _, ts)| {
                        // Fall back to TranslationalStateC.velocity when
                        // SourceInertialVelocityC is absent — same precedence
                        // as `sync_source_to_frame_system`.
                        let velocity = v
                            .map(|v| v.0.raw_si())
                            .or_else(|| ts.map(|t| t.0.velocity.raw_si()))
                            .unwrap_or(DVec3::ZERO);
                        jeod_sim::ResolvedRelativisticSource {
                            mu: s.mu,
                            position: p.0.raw_si(),
                            velocity,
                        }
                    })
            },
        );
        accel.grav_accel += rel_accel;
    }
}

// ── Atmosphere ──

// JEOD_INV: AT.01 — active flag gates computation (no AtmosphericStateC component = no computation)
// JEOD_INV: AT.02 — atmosphere model pointer non-null for update (AtmosphereModelR resource checked)
/// Update atmospheric state for entities that have `AtmosphericStateC`.
///
/// Delegates to [`jeod_sim::evaluate_atmosphere`] for the per-body evaluation
/// pipeline (planet-fixed rotation, geodetic conversion, model dispatch, wind).
pub fn atmosphere_update_system(
    atmos_model: Option<Res<AtmosphereModelR>>,
    sim_time: Option<Res<SimulationTimeR>>,
    planet_query: Query<&PlanetFixedRotationC>,
    mut query: Query<(&TranslationalStateC, &mut AtmosphericStateC)>,
) {
    // JEOD_INV: AT.02 — early return if no atmosphere model resource
    let Some(model) = atmos_model else {
        return;
    };

    // JEOD_INV: AT.03 — planet-fixed position required for geodetic altitude
    let t_inertial_pfix = if let Some(entity) = model.planet_entity {
        let Ok(r) = planet_query.get(entity) else {
            panic!(
                "AtmosphereModelR.planet_entity is set ({entity:?}) but entity has no \
                 PlanetFixedRotationC. In JEOD, the planet-fixed frame is always \
                 available for atmosphere computation. Add PlanetFixedRotationC to \
                 the planet entity or set planet_entity to None for spherical fallback."
            );
        };
        Some(*r.0.matrix_ref())
    } else {
        None
    };

    let tai_tjt = sim_time.as_ref().map(|t| t.tai_tjt);
    for (state, mut atmos) in &mut query {
        // MET atmosphere requires time for seasonal variation. Check only when
        // entities with AtmosphericStateC actually exist (avoids panic when MET
        // is configured but no bodies need atmosphere yet).
        if tai_tjt.is_none() {
            if let jeod_sim::AtmosphereModel::Met(_) = &model.config.model {
                panic!(
                    "MET atmosphere requires SimulationTimeR resource for TJT. \
                     Ensure JeodPlugin is added (it provides SimulationTimeR)."
                );
            }
        }
        **atmos = jeod_sim::evaluate_atmosphere(
            &model.config,
            state.position.raw_si(),
            t_inertial_pfix.as_ref(),
            tai_tjt,
        );
    }
}

// ── Interactions ──

/// Compute aerodynamic drag for entities with all required components.
///
/// Placed in `JeodSet::Interaction`.
// JEOD_INV: IN.03 — AerodynamicDrag.active gates computation (structural: no DragConfigC -> no drag)
#[allow(clippy::type_complexity)]
pub fn aero_drag_system(
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; skip
    // drag so `AerodynamicForceC` doesn't hold stale values that no
    // integrator consumes (the runner's split between `bodies` and
    // `detached_subtrees` only evaluates drag on the integrated set).
    mut query: Query<
        (
            &DragConfigC,
            &AtmosphericStateC,
            &TranslationalStateC,
            &RotationalStateC,
            Option<&StructuralTransformC>,
            &mut AerodynamicForceC,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
) {
    for (drag_config, atmos, state, rot, struct_xform, mut aero_force) in &mut query {
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());

        // `DragConfigC` and `TranslationalStateC` both store typed values;
        // the system reads them directly. The result carries
        // `StructuralFrame<SelfRef>` phantoms, which the structural-frame
        // `AerodynamicForceC` unwraps via `.raw_si()` for storage (the
        // structural-frame Component still uses raw DVec3; that's a
        // remaining typed-storage boundary).
        let rot_untyped = rot.0.to_untyped();
        // Bevy adapter stores body velocity as
        // `Velocity<PlanetInertial<SelfPlanet>>`. Drag's typed sibling
        // is parameterized over a concrete `P`, so the call site does
        // a wildcard → `PlanetInertial<Earth>` phantom relabel (no
        // integ-origin shift — drag stays in planet-inertial
        // throughout). Bit-identical and asserts the Earth-orbit
        // assumption that the body's planet is Earth.
        use jeod_sim::{Earth, PlanetInertial, Velocity};
        // allowed: wildcard `<SelfPlanet>` → concrete `<Earth>` relabel
        // for the typed sibling; bit-identical (no arithmetic).
        let drag_velocity = Velocity::<PlanetInertial<Earth>>::from_raw_si(state.velocity.raw_si());
        let result = jeod_sim::compute_drag_typed::<Earth, SelfRef>(
            &drag_config.0,
            atmos,
            drag_velocity,
            Some(&rot_untyped),
            t_struct_body,
        );

        aero_force.force = result.force.raw_si();
        aero_force.torque = result.torque.raw_si();
    }
}

/// Compute gravity gradient torque.
///
/// Placed in `JeodSet::Interaction`.
// JEOD_INV: IN.01 — GravityTorque.subject_body required (structural: query requires all components)
// JEOD_INV: IN.02 — GravityTorque.active gates computation (structural: no GravityTorqueC -> no torque)
pub fn gravity_torque_system(
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; their
    // gravity gradient torque is no longer consumed by any integrator
    // and would otherwise hold stale values. Skip them.
    mut query: Query<
        (
            &GravityAccelerationC,
            &RotationalStateC,
            &MassPropertiesC,
            &mut GravityTorqueC,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
) {
    for (grav, rot, mass, mut torque) in &mut query {
        // MassPropertiesC stores `InertiaTensor<BodyFrame<SelfRef>>`
        // directly; read it without lifting. Same for the rotational
        // state — it's already typed.
        let rot_untyped = rot.0.to_untyped();
        torque.0 = jeod_sim::compute_gravity_torque_typed::<SelfRef>(
            &grav.grav_grad,
            &rot_untyped,
            mass.0.inertia,
        );
    }
}

/// Compute illumination factor from all shadow-casting bodies.
fn compute_illum_factor(
    vehicle_pos: DVec3,
    sun_pos: DVec3,
    shadow_bodies: &Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
) -> f64 {
    let mut illum = 1.0_f64;
    for (body_state, shadow) in shadow_bodies.iter() {
        let factor = jeod_sim::compute_shadow_fraction(
            vehicle_pos,
            sun_pos,
            body_state.position.raw_si(),
            shadow.radius,
            jeod_sim::SOLAR_RADIUS,
        );
        illum = illum.min(factor);
    }
    illum
}

// ── Derived States ──

/// Compute orbital elements for entities with `OrbitalElementsConfigC`.
///
/// Placed in `JeodSet::DerivedState`.
pub fn orbital_elements_system(
    mut query: Query<(
        &TranslationalStateC,
        &OrbitalElementsConfigC,
        &mut OrbitalElementsC,
    )>,
    sources: Query<&GravitySourceC>,
) {
    for (state, config, mut elements) in &mut query {
        let Ok(source) = sources.get(config.gravity_source) else {
            elements.0 = Default::default();
            continue;
        };
        // The Bevy `OrbitalElementsC` component is parameterized by
        // `SelfPlanet` (per-entity planet identity is dynamic, keyed by
        // `config.gravity_source`). Drive the planet-erased
        // `compute_orbital_elements` so the result is already
        // `<SelfPlanet>`-tagged — no relabel step needed, and the
        // previous `<Earth>` → relabel path through
        // `compute_orbital_elements_typed::<Earth>` is no longer
        // available because `OrbitalElements::relabel` is restricted
        // to a `<SelfPlanet>` receiver to prevent silent cross-planet
        // retagging.
        match jeod_sim::compute_orbital_elements(
            source.mu,
            state.position.raw_si(),
            state.velocity.raw_si(),
        ) {
            Ok(oe) => elements.0 = oe,
            Err(_) => elements.0 = Default::default(),
        }
    }
}

/// Compute Euler angles for entities with `EulerAnglesConfigC`.
///
/// Placed in `JeodSet::DerivedState`.
pub fn euler_angles_system(
    mut query: Query<(
        Option<&RotationalStateC>,
        &EulerAnglesConfigC,
        &mut EulerAnglesC,
    )>,
) {
    for (rot_opt, config, mut angles) in &mut query {
        if let Some(rot) = rot_opt {
            // The "_typed" function takes untyped input but returns
            // typed `[Angle; 3]` (the typed-output naming convention
            // documented in jeod_sim::derived). Convert at the call.
            let rot_untyped = rot.0.to_untyped();
            angles.0 = jeod_sim::compute_body_euler_angles_typed(&rot_untyped, config.sequence);
        } else {
            angles.0 = Default::default();
        }
    }
}

/// Compute LVLH frame for entities with `LvlhFrameC`.
///
/// Presence of `LvlhFrameC` alone enables computation (no separate config needed).
///
/// Placed in `JeodSet::DerivedState`.
pub fn lvlh_system(mut query: Query<(&TranslationalStateC, &mut LvlhFrameC)>) {
    for (state, mut lvlh) in &mut query {
        // Typed throughout — `TranslationalStateC` carries
        // `PlanetInertial<SelfPlanet>` on the Bevy adapter. LVLH stays
        // in planet-inertial (no integ-origin shift), but the typed
        // sibling is parameterized over a concrete `P`, so the call
        // site does a wildcard → `PlanetInertial<Earth>` phantom
        // relabel. Bit-identical and asserts the Earth-orbit
        // assumption.
        use jeod_sim::{Earth, PlanetInertial};
        // allowed: wildcard `<SelfPlanet>` → concrete `<Earth>` relabel
        // for the typed sibling; bit-identical (no arithmetic).
        let pos = jeod_sim::Position::<PlanetInertial<Earth>>::from_raw_si(state.position.raw_si());
        // allowed: same relabel as `pos` above.
        let vel = jeod_sim::Velocity::<PlanetInertial<Earth>>::from_raw_si(state.velocity.raw_si());
        lvlh.0 = jeod_sim::compute_body_lvlh_frame_typed::<Earth>(pos, vel);
    }
}

/// Compute geodetic state for entities with `GeodeticConfigC`.
///
/// Placed in `JeodSet::DerivedState`.
pub fn geodetic_system(
    mut query: Query<(&TranslationalStateC, &GeodeticConfigC, &mut GeodeticStateC)>,
    planets: Query<(&PlanetFixedRotationC, &PlanetC)>,
) {
    for (state, config, mut geodetic) in &mut query {
        let Ok((rot, planet)) = planets.get(config.planet) else {
            geodetic.0 = Default::default();
            continue;
        };
        // Position is already typed `Position<PlanetInertial<SelfPlanet>>`;
        // geodetic stays in planet-inertial (no integ-origin shift),
        // and the typed sibling is parameterized over a concrete `P`,
        // so the call site does a wildcard → `PlanetInertial<Earth>`
        // phantom relabel. The ellipsoid-radii lift on the next call
        // is the typed-units boundary on planet shape (a config-time
        // conversion, not a per-step bypass).
        use jeod_sim::F64Ext;
        use jeod_sim::{Earth, PlanetInertial};
        // allowed: wildcard `<SelfPlanet>` → concrete `<Earth>` relabel
        // for the typed sibling; bit-identical (no arithmetic).
        let pos = jeod_sim::Position::<PlanetInertial<Earth>>::from_raw_si(state.position.raw_si());
        geodetic.0 = jeod_sim::compute_body_geodetic_typed::<Earth>(
            pos,
            rot.0.matrix_ref(),
            planet.r_eq.m(),
            planet.r_pol.m(),
        );
    }
}

/// Compute the typed root-inertial origin offset of `body_frame`'s
/// integration frame — the RF.10 shift that lifts a body's
/// `PlanetInertial<SelfPlanet>` state into absolute `RootInertial`
/// coordinates. Returns `(zero, zero)` when:
///
/// - the body has no [`FrameEntityC`] (legacy entities registered
///   before the frames-as-entities components landed are treated as
///   root-integrated), or
/// - the body's frame entity's parent is the root frame.
///
/// In both cases the integ-origin shift is identically zero, so
/// relabeling the body state to `RootInertial` is a no-op
/// numerically. For non-root-integrated bodies the shift is the
/// translational state of the integration frame relative to root,
/// supplied by the [`FrameOrigin`] SystemParam.
///
/// Mirrors the `body_integ_origins` helper that
/// `jeod_runner::Simulation::step_internal` builds before each shift
/// site (gravity, integration, derived states); same algorithm,
/// ECS-backed storage.
fn body_integ_origin_in_root(
    body_frame: Option<&FrameEntityC>,
    parents: &Query<&ChildOf>,
    root_frame_entity: Entity,
    frame_origin: &FrameOrigin,
) -> (Position<RootInertial>, Velocity<RootInertial>) {
    let integ_frame_entity =
        body_frame.and_then(|fe| parents.get(fe.0).ok().map(|child_of| child_of.parent()));
    match integ_frame_entity {
        Some(integ_e) if integ_e != root_frame_entity => {
            frame_origin.origin_in_root(root_frame_entity, integ_e)
        }
        _ => (
            Position::<RootInertial>::zero(),
            Velocity::<RootInertial>::zero(),
        ),
    }
}

/// Compute solar beta angle for entities with `SolarBetaC`.
///
/// Requires a `SunMarker` entity to exist in the world.
///
/// Placed in `JeodSet::DerivedState`.
#[allow(clippy::type_complexity)]
pub fn solar_beta_system(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    mut query: Query<
        (&TranslationalStateC, Option<&FrameEntityC>, &mut SolarBetaC),
        Without<SunMarker>,
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // No SunMarker present: clear stale solar beta values
            for (_, _, mut beta) in &mut query {
                beta.0 = Default::default();
            }
            return;
        }
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found in solar_beta_system. \
                 JEOD assumes exactly one Sun body; ensure exactly one SunMarker entity exists."
            );
        }
    };
    for (state, body_frame, mut beta) in &mut query {
        // Solar beta is a root-inertial-shift consumer (RF.10): the
        // kernel mixes the body state with the Sun position in
        // absolute root-inertial coordinates. For non-root-integrated
        // bodies the body's `<PlanetInertial<SelfPlanet>>` storage is
        // integ-frame-relative, not absolute root-inertial — passing
        // it raw to the root-inertial kernel would compute solar beta
        // off by the inter-source separation distance. Lift to
        // absolute root-inertial via the integ-origin shift, then
        // call the typed kernel. `Angle.value` reads radians (the SI
        // base unit), so the f64 `SolarBetaC` storage is bit-identical
        // for root-integrated bodies (where the shift is zero).
        let (integ_origin, integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let body_pos_rel = Position::<RootInertial>::from_raw_si(state.position.raw_si()); // allowed: integ-origin shift adds origin offset on the next line; relabel is a phantom-tag attachment matching the runner's `body.trans.to_inertial(&o)` boundary.
        let body_vel_rel = Velocity::<RootInertial>::from_raw_si(state.velocity.raw_si()); // allowed: same boundary as `body_pos_rel`.
        let body_pos = body_pos_rel + integ_origin;
        let body_vel = body_vel_rel + integ_origin_vel;
        // Sun is registered through `SunBundle` and integrates in the
        // root frame, so its `<PlanetInertial<SelfPlanet>>` storage is
        // numerically root-inertial; the relabel here is the boundary
        // step that pins the framing convention at the consumer call
        // site rather than asserting it once at registration.
        let sun_pos = Position::<RootInertial>::from_raw_si(sun_state.position.raw_si()); // allowed: Sun is root-integrated by SunBundle construction (its frame entity's parent is the root frame, integ origin = zero); relabel is the consumer-boundary step.
        beta.0 = jeod_sim::compute_body_solar_beta_typed(body_pos, body_vel, sun_pos).value;
    }
}

/// Compute earth lighting (eclipse/albedo) for entities with `EarthLightingConfigC`.
///
/// Requires `SunMarker` and `MoonMarker` entities in the world.
///
/// Placed in `JeodSet::DerivedState`.
#[allow(clippy::type_complexity)]
pub fn earth_lighting_system(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    mut query: Query<
        (
            &TranslationalStateC,
            Option<&FrameEntityC>,
            &EarthLightingConfigC,
            &mut EarthLightingStateC,
        ),
        (Without<SunMarker>, Without<MoonMarker>),
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
    moon_query: Query<&TranslationalStateC, With<MoonMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // No SunMarker present: clear stale earth lighting values
            for (_, _, _, mut lighting) in &mut query {
                lighting.0 = Default::default();
            }
            return;
        }
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found in earth_lighting_system. \
                 JEOD assumes exactly one Sun body."
            );
        }
    };
    let moon_state = match moon_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // No MoonMarker present: clear stale earth lighting values
            for (_, _, _, mut lighting) in &mut query {
                lighting.0 = Default::default();
            }
            return;
        }
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with MoonMarker found in earth_lighting_system. \
                 JEOD assumes exactly one Moon body."
            );
        }
    };
    for (state, body_frame, config, mut lighting) in &mut query {
        // Earth lighting is a root-inertial-shift consumer (RF.10):
        // the kernel mixes the body position with the Sun and Moon
        // positions, all expected in absolute root-inertial
        // coordinates. For non-root-integrated bodies the body's
        // `<PlanetInertial<SelfPlanet>>` storage is integ-frame-
        // relative; lift it to absolute root-inertial via the integ-
        // origin shift before passing to the kernel. Sun and Moon
        // are root-integrated by the SunBundle / MoonBundle
        // construction (their frame entities are children of the
        // root frame), so their positions need no shift.
        let (integ_origin, _integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let body_pos_root = state.position.raw_si() + integ_origin.raw_si();
        lighting.0 = jeod_sim::compute_earth_lighting(
            body_pos_root,
            sun_state.position.raw_si(),
            moon_state.position.raw_si(),
            config.sun_radius,
            config.earth_radius,
            config.moon_radius,
        );
    }
}

/// Compute flat-plate SRP with thermal emission and shadow detection.
///
// JEOD_INV: IN.06 — RadiationPressure.active gates computation (structural: no FlatPlateConfigC → no SRP)
// JEOD_INV: IN.09 — RadiationSource planet must exist (SunMarker required; panics on multiple)
/// For entities with `FlatPlateConfigC`. Handles:
/// - Solar flux at vehicle distance
/// - Conical shadow from `ShadowBodyC` entities
/// - Per-plate absorption, diffuse/specular reflection, thermal emission
/// - Temperature integration (forward Euler)
/// - Force is rotated from structural to inertial by this system before writing `RadiationForceC`
///
/// Kinematic children of a `MassChildOf` chain (entities carrying
/// [`KinematicChildC`]) are excluded from this system. Until the
/// kinematic-propagation system (design-doc Section 15.3
/// `propagate_state_from_root_system`) lands, a kinematic child's
/// own `TranslationalStateC` / `RotationalStateC` are not advanced
/// in lock-step with the chain root — they stay frozen at whatever
/// the world had when the chain was assembled. Reading those stale
/// states to compute solar pressure here would silently produce SRP
/// for a position the body is no longer at. Excluding kinematic
/// children entirely (rather than feeding them stale state) is the
/// fail-loud-but-conservative choice: kinematic-child appendages get
/// no SRP this PR, and the follow-up that introduces propagated
/// child state will route SRP through the live composite-derived
/// values.
///
/// Placed in `JeodSet::Interaction`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn flat_plate_srp_system(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // Filter excludes both kinematic-chain children (their
    // `TranslationalStateC` / `RotationalStateC` stay frozen until
    // the kinematic-propagation system lands; computing SRP from
    // stale state would produce solar pressure at the wrong
    // location) and detached subtrees (they coast ballistically;
    // `RadiationForceC` and the per-stage thermal cache stay
    // zeroed because no integrator consumes their forces).
    // JEOD_INV: DB.21 — detached subtrees skip SRP.
    mut query: Query<
        (
            &mut FlatPlateConfigC,
            &TranslationalStateC,
            Option<&RotationalStateC>,
            Option<&MassPropertiesC>,
            Option<&StructuralTransformC>,
            Option<&FrameEntityC>,
            &mut RadiationForceC,
        ),
        (
            Without<SunMarker>,
            Without<CannonballSrpC>,
            Without<crate::DetachedSubtreeStateC>,
            Without<KinematicChildC>,
        ),
    >,
    // Cleanup query for kinematic children: drop any prior-tick
    // `RadiationForceC` / `stage_inputs` left over from when the
    // entity was last in the main query (i.e. before it became a
    // chain member). Without this clear, `force_collection_system`
    // would still accumulate the stale SRP into the child's
    // `TotalForceC`, and `wrench_aggregation_system` would shift
    // that stale wrench up to the parent — silently producing SRP
    // for a position the body is no longer at.
    mut kinematic_cleanup: Query<
        (&mut FlatPlateConfigC, &mut RadiationForceC),
        (
            With<KinematicChildC>,
            Without<SunMarker>,
            Without<CannonballSrpC>,
        ),
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
    time: Res<Time<Fixed>>,
) {
    // Drop stale state for any kinematic-child SRP body. Runs first
    // so a transition from non-kinematic → kinematic this tick
    // never carries a leftover SRP force into the wrench-aggregation
    // walk.
    for (mut flat_config, mut srp_force) in &mut kinematic_cleanup {
        flat_config.stage_inputs = None;
        srp_force.force = DVec3::ZERO;
        srp_force.torque = DVec3::ZERO;
    }

    let sun_state = match sun_query.single() {
        Ok(s) => Some(s),
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => None,
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found. In JEOD, RadiationPressure \
                 has exactly one RadiationSource (value member). Ensure exactly one \
                 Sun entity exists."
            );
        }
    };

    let dt = time.delta_secs_f64();

    for (mut flat_config, state, rot, mass, struct_xform, body_frame, mut srp_force) in &mut query {
        // Clear per-step SRP state unconditionally (before the Sun check)
        // so derivative-mode entities don't retain stale `stage_inputs` or
        // force/torque if the Sun entity is removed between steps — which
        // would otherwise incorrectly drive the coupled RK4 path. Mirrors
        // the unconditional clearing in `jeod_runner::Simulation`.
        flat_config.stage_inputs = None;
        srp_force.force = DVec3::ZERO;
        srp_force.torque = DVec3::ZERO;

        let Some(sun_state) = sun_state else {
            continue;
        };

        // SRP is a root-inertial-shift consumer (RF.10): `sun_to_vehicle`
        // and the conical-shadow geometry both mix the body position
        // with the Sun / shadow-body positions, which are tagged
        // `<RootInertial>` (they integrate in root). For non-root-
        // integrated bodies the body's `<PlanetInertial<SelfPlanet>>`
        // storage is integ-frame-relative, so passing it raw to the
        // SRP / shadow kernels would compute `sun_to_vehicle` off by
        // the Earth–planet separation distance — wrong flux direction
        // and wrong illumination factor. Lift the body position to
        // absolute root-inertial via the integ-origin shift before
        // mixing. Both the scheduled-class and derivative-class
        // branches read `pos_raw` for `sun_to_vehicle`, distance, and
        // `compute_illum_factor`, so the shift applies to both — only
        // the temperature integration cadence differs between them.
        let (integ_origin, _integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let pos_raw = state.position.raw_si() + integ_origin.raw_si();
        // Sun is registered through `SunBundle` and integrates in the
        // root frame, so its `<PlanetInertial<SelfPlanet>>` storage is
        // numerically root-inertial; no integ-origin shift needed for
        // the Sun position.
        let sun_pos_raw = sun_state.position.raw_si();

        let sun_to_vehicle = pos_raw - sun_pos_raw;
        let distance = sun_to_vehicle.length();
        if distance < 1.0 {
            // Too close to the Sun to compute flux: force/torque/
            // stage_inputs already zeroed above.
            continue;
        }
        let flux_inertial_hat = sun_to_vehicle / distance;
        let flux_mag = jeod_sim::solar_flux_at_distance(distance);

        // Shadow fraction (step-constant; matches JEOD's scheduled-class
        // shadow evaluation across all three integration orders).
        let illum_factor = compute_illum_factor(pos_raw, sun_pos_raw, &shadow_bodies);
        let center_grav = mass.map_or(DVec3::ZERO, |m| m.0.center_of_mass.raw_si());

        match flat_config.integration_order {
            jeod_sim::ThermalIntegrationOrder::Scheduled => {
                // Scheduled-class (SIM_3_ORBIT): SRP force + Euler T once
                // per step. Force fed to the orbital integrator is
                // step-constant.
                let t_inertial_body = rot.map_or(glam::DMat3::IDENTITY, |r| {
                    r.0.q_inertial_body
                        .as_witness()
                        .left_quat_to_transformation()
                });
                let t_struct_body =
                    struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
                let t_inertial_struct =
                    jeod_sim::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
                let flux_struct_hat = t_inertial_struct * flux_inertial_hat;

                let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
                    &flat_config.plates,
                    &flat_config.t_pow4_cached,
                    flux_struct_hat,
                    flux_mag,
                    center_grav,
                    illum_factor,
                );

                let force_inertial = t_inertial_struct.transpose() * srp_result.force;
                srp_force.force = force_inertial;
                srp_force.torque = srp_result.torque;

                // Integrate plate temperatures (forward Euler) — shared with
                // `Simulation` runner via `FlatPlateState::integrate_temperatures`.
                if dt > 0.0 {
                    flat_config.integrate_temperatures(&srp_result.temp_dots, dt);
                }
            }
            jeod_sim::ThermalIntegrationOrder::DerivativeFirstOrder
            | jeod_sim::ThermalIntegrationOrder::DerivativeRk4 => {
                // Derivative-class: SRP force (and optionally T) recomputed
                // per RK4 stage by the integration system. Cache the
                // step-start inputs on the plate state here; `RadiationForceC`
                // stays at the zero cleared above — the integration system
                // writes a representative final-stage value.
                // `sun_state.position` is now stored as the wildcard
                // `<PlanetInertial<SelfPlanet>>`; the SRP derivative
                // closure expects a root-inertial Sun position (RF.10
                // shift-site). Relabel at the boundary —
                // bit-identical numerics; the Sun's ephemeris-driven
                // inertial position numerically coincides with the
                // root frame's representation.
                type RootPos = jeod_sim::Position<jeod_sim::RootInertial>;
                let sun_pos_root = RootPos::from_raw_si(sun_state.position.raw_si()); // allowed: SRP shift-site, Sun position relabeled to RootInertial for kernel
                flat_config.stage_inputs = Some(jeod_sim::FlatPlateStageInputs {
                    sun_position: sun_pos_root,
                    illum_factor,
                    center_grav,
                });
            }
        }
    }
}

/// Compute cannonball SRP using JEOD's `RadiationDefaultSurface` formula.
///
/// Force = (flux/c) * cx_area * [1 + albedo*diffuse*(4/9)] * flux_hat * illum_factor.
///
/// For entities with `CannonballSrpC`. Requires `SunMarker` entity in the world.
/// Optional shadow detection via `ShadowBodyC` entities.
/// Writes force to `RadiationForceC` (torque is always zero for cannonball).
///
/// Placed in `JeodSet::Interaction`.
#[allow(clippy::type_complexity)]
pub fn cannonball_srp_system(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; skip
    // cannonball SRP so `RadiationForceC` doesn't hold stale values
    // that no integrator consumes.
    mut query: Query<
        (
            &CannonballSrpC,
            &TranslationalStateC,
            Option<&FrameEntityC>,
            &mut RadiationForceC,
        ),
        (
            Without<SunMarker>,
            Without<FlatPlateConfigC>,
            Without<crate::DetachedSubtreeStateC>,
        ),
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => return,
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found. \
                 Ensure exactly one Sun entity exists."
            );
        }
    };

    for (config, state, body_frame, mut srp_force) in &mut query {
        // Cannonball SRP is a root-inertial-shift consumer (RF.10):
        // the kernel mixes the body position with the Sun position
        // (expected root-inertial). Lift the body's
        // `<PlanetInertial<SelfPlanet>>` storage to absolute root-
        // inertial via the integ-origin shift before mixing — same
        // boundary discipline as the flat-plate / solar-beta sites.
        let (integ_origin, _integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let pos_raw = state.position.raw_si() + integ_origin.raw_si();
        let sun_pos_raw = sun_state.position.raw_si();
        let illum_factor = compute_illum_factor(pos_raw, sun_pos_raw, &shadow_bodies);

        srp_force.force = jeod_sim::compute_cannonball_srp(
            pos_raw,
            sun_pos_raw,
            config.cx_area,
            config.albedo,
            config.diffuse,
            illum_factor,
        );
        srp_force.torque = DVec3::ZERO;
    }
}

/// Process mass-tree attach/detach messages and sync composite properties.
///
/// Runs before interactions so that mass changes from staging are
/// reflected in the current step's interaction forces, force collection,
/// and integration.
///
/// On `AttachEvent` this system:
///
/// 1. snapshots both bodies' pre-attach composite-body inertial state
///    (`TranslationalStateC` + `RotationalStateC`) and pre-attach
///    composite mass properties,
/// 2. mutates the [`crate::MassTreeR`] arena (which recomputes composite
///    mass properties for every affected node),
/// 3. runs [`jeod_sim::stage_attach_combine`] (the
///    momentum-conservation port of JEOD's `combine_states_at_attach`,
///    `models/dynamics/dyn_body/src/dyn_body_attach.cc`) to derive the
///    merged composite-body inertial state — preserves linear momentum
///    about the integration-frame origin and angular momentum about
///    the new combined CoM,
/// 4. writes the merged state back into the parent entity's
///    [`crate::TranslationalStateC`] / [`crate::RotationalStateC`],
/// 5. removes [`crate::DetachedSubtreeStateC`] from the child entity if
///    it was previously detached (the captured ballistic state is now
///    consumed by the combine).
///
/// On `DetachEvent` this system:
///
/// 1. captures the about-to-be-detached subtree's instantaneous
///    composite-body inertial state via
///    [`jeod_sim::stage_detach_capture`],
/// 2. mutates the arena (which recomputes the former parent's composite
///    mass to reflect the lost subtree),
/// 3. inserts [`crate::DetachedSubtreeStateC`] on the detached entity
///    so [`step_detached_system`] can advance the subtree ballistically
///    each tick.
///
/// Both branches end with the IG.37 mark + reset for any body whose
/// composite mass changed — multi-step integrators (GJ, ABM4) must
/// drop their predictor history on topology change.
///
/// Note: [`crate::MassTreeR`] must be present as a resource for attach/detach
/// messages to have any effect.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::DetachEvent;
///
/// // A user-defined system that emits a DetachEvent for a known booster
/// // entity (e.g. one cached in a Resource).
/// #[derive(Resource)]
/// struct Booster(Entity);
///
/// fn detach_booster(
///     booster: Res<Booster>,
///     mut detach_messages: bevy::ecs::message::MessageWriter<DetachEvent>,
/// ) {
///     detach_messages.write(DetachEvent { child: booster.0 });
/// }
///
/// let mut app = App::new();
/// app.add_message::<DetachEvent>();
/// app.add_systems(Update, detach_booster);
/// ```
#[allow(clippy::type_complexity)]
pub fn staging_system(
    mut commands: Commands,
    tree: Option<ResMut<crate::MassTreeR>>,
    mut attach_events: bevy::ecs::message::MessageReader<crate::AttachEvent>,
    mut detach_events: bevy::ecs::message::MessageReader<crate::DetachEvent>,
    mut bodies: Query<(
        Entity,
        &crate::MassBodyIdC,
        &mut MassPropertiesC,
        Option<&mut TranslationalStateC>,
        Option<&mut RotationalStateC>,
    )>,
    detached_q: Query<Entity, With<crate::DetachedSubtreeStateC>>,
    mut integrators: Query<(
        &crate::MassBodyIdC,
        Option<&mut GaussJacksonStateC>,
        Option<&mut Abm4StateC>,
    )>,
) {
    // No mass tree resource → drain events and return.
    let Some(mut tree) = tree else {
        attach_events.clear();
        detach_events.clear();
        return;
    };

    // The set of mass-tree node ids whose composite mass changes due
    // to the events processed below — i.e. whose multi-step integrator
    // state must be marked topology-dirty (Site A) and later reset
    // (Site B). We accumulate it INLINE with each event-handler branch
    // so the dirty-marking is structurally bound to the topology
    // mutation call site, then mark in one query pass, then reset in
    // a separate observation pass. Splitting Site A and Site B is the
    // structural fix for IG.37 fail-loud (see JEOD_invariants.md): a
    // future code path that adds a new event branch and forgets the
    // reset pass will leave the dirty flag set, so the next
    // `integrate()` panics with the IG.37 diagnostic rather than
    // silently propagating stale predictor history.
    let mut affected_ids: Vec<jeod_sim::MassBodyId> = Vec::new();

    // Per-attach work item: captures the pre-attach snapshot needed by
    // `combine_states_at_attach` plus the post-mutation parent entity
    // we'll write the merged composite-body state into. Built before
    // the topology mutation so the snapshot is independent of the
    // tree's post-attach state.
    struct AttachWork {
        parent_entity: Entity,
        child_entity: Entity,
        parent_id: jeod_sim::MassBodyId,
        // Pre-attach snapshot for the kernel.
        parent_position: glam::DVec3,
        parent_velocity: glam::DVec3,
        parent_quaternion: jeod_sim::JeodQuat,
        parent_ang_vel_body: glam::DVec3,
        parent_mass: jeod_sim::MassProperties,
        orig_parent_cm_struct: glam::DVec3,
        parent_t_inertial_struct: glam::DMat3,
        child_position: glam::DVec3,
        child_velocity: glam::DVec3,
        child_quaternion: jeod_sim::JeodQuat,
        child_ang_vel_body: glam::DVec3,
        child_mass: jeod_sim::MassProperties,
        // Was the child carrying a `DetachedSubtreeStateC` immediately
        // before this attach? If so the entry is consumed and removed.
        child_was_detached: bool,
        // Was the parent carrying a `DetachedSubtreeStateC` immediately
        // before this attach? If so the parent is still a free-flying
        // tree root post-attach (no integrated ancestor); its tracked
        // ballistic state must be replaced with the merged composite-
        // body state so `step_detached_system` continues advancing the
        // correct value next tick (rather than overwriting the merged
        // state with the stale pre-attach `DetachedSubtreeStateC`).
        parent_was_detached: bool,
    }

    let mut attach_work: Vec<AttachWork> = Vec::new();
    // Per-detach work: captured pre-detach composite-body state to be
    // attached to the detached entity as `DetachedSubtreeStateC` once
    // the topology mutation is done.
    let mut detach_work: Vec<(Entity, jeod_sim::DetachedSubtreeState)> = Vec::new();

    for evt in attach_events.read() {
        // Look up child + parent. Fail-loud per CLAUDE.md if either
        // entity is not a mass-tree body.
        let (_, child_body_id, child_mass_c, child_trans, child_rot) =
            bodies.get(evt.child).unwrap_or_else(|_| {
                panic!(
                    "AttachEvent.child = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC. Spawn the body via the mass-tree API before attaching.",
                    evt.child
                )
            });
        let child_id = child_body_id.0;
        let child_mass: jeod_sim::MassProperties = child_mass_c.0.to_untyped();
        let (child_position, child_velocity) = child_trans
            .as_ref()
            .map(|t| (t.0.position.raw_si(), t.0.velocity.raw_si()))
            .unwrap_or((glam::DVec3::ZERO, glam::DVec3::ZERO));
        let (child_quaternion, child_ang_vel_body) = child_rot
            .as_ref()
            .map(|r| {
                let untyped = r.0.to_untyped();
                (untyped.quaternion, untyped.ang_vel_body)
            })
            .unwrap_or((jeod_sim::JeodQuat::identity(), glam::DVec3::ZERO));

        let (_, parent_body_id, parent_mass_c, parent_trans, parent_rot) =
            bodies.get(evt.parent).unwrap_or_else(|_| {
                panic!(
                    "AttachEvent.parent = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC. Spawn the parent via the mass-tree API before attaching.",
                    evt.parent
                )
            });
        let parent_id = parent_body_id.0;
        let parent_mass: jeod_sim::MassProperties = parent_mass_c.0.to_untyped();
        let (parent_position, parent_velocity) = parent_trans
            .as_ref()
            .map(|t| (t.0.position.raw_si(), t.0.velocity.raw_si()))
            .unwrap_or((glam::DVec3::ZERO, glam::DVec3::ZERO));
        let (parent_quaternion, parent_ang_vel_body) = parent_rot
            .as_ref()
            .map(|r| {
                let untyped = r.0.to_untyped();
                (untyped.quaternion, untyped.ang_vel_body)
            })
            .unwrap_or((jeod_sim::JeodQuat::identity(), glam::DVec3::ZERO));

        // T_inertial_to_struct = T_struct_to_body^T · T_inertial_to_body
        // Per JEOD `dyn_body_collect.cc:219-221` and
        // `jeod_dynamics::compute_t_inertial_struct` — the kernel needs
        // this to rotate the structure-frame CoM-shift vector
        // (`combined.position - orig_parent_cm_struct`) into the
        // inertial frame for the parent's post-attach position.
        let parent_t_struct_to_body = parent_mass.t_parent_this;
        let parent_t_inertial_to_body = parent_quaternion.left_quat_to_transformation();
        let parent_t_inertial_struct = jeod_sim::compute_t_inertial_struct(
            &parent_t_struct_to_body,
            &parent_t_inertial_to_body,
        );

        // The bodies whose composite mass changes are the child plus
        // every ancestor of the new parent in the pre-attach tree
        // (`MassTree::recompute_composites` walks the entire forest
        // post-order, so any ancestor of the new parent is touched).
        // Capture the chain BEFORE mutating the tree.
        affected_ids.push(child_id);
        affected_ids.extend(tree.ancestors_inclusive(parent_id));

        let child_was_detached = detached_q.contains(evt.child);
        let parent_was_detached = detached_q.contains(evt.parent);

        attach_work.push(AttachWork {
            parent_entity: evt.parent,
            child_entity: evt.child,
            parent_id,
            parent_position,
            parent_velocity,
            parent_quaternion,
            parent_ang_vel_body,
            parent_mass,
            orig_parent_cm_struct: parent_mass.position,
            parent_t_inertial_struct,
            child_position,
            child_velocity,
            child_quaternion,
            child_ang_vel_body,
            child_mass,
            child_was_detached,
            parent_was_detached,
        });

        tree.attach(child_id, parent_id, evt.offset, evt.t_parent_child);
    }

    // Per-detach post-mutation work: tree_root entity whose
    // `TranslationalStateC` / `RotationalStateC` (and possibly
    // `DetachedSubtreeStateC`) must be shifted by the inertial-frame
    // composite-CoM delta after the topology change, since the parent's
    // composite-CoM moves within its own struct frame when the subtree
    // leaves. Mirrors the runner's `detach_subtree` parent-side update.
    struct ParentShift {
        tree_root_entity: Entity,
        parent_pre_position: glam::DVec3,
        parent_pre_velocity: glam::DVec3,
        parent_pre_quat: jeod_sim::JeodQuat,
        parent_pre_ang_vel_body: glam::DVec3,
        parent_pre_composite_props: jeod_sim::MassProperties,
        parent_was_detached: bool,
    }
    let mut parent_shifts: Vec<(jeod_sim::MassBodyId, ParentShift)> = Vec::new();

    // Build a one-shot id → entity map by scanning the bodies query.
    // The detach handler needs to look up the tree root's entity from
    // its `MassBodyId` so it can read the parent's composite-body
    // inertial state — runner's `detach_subtree` indexes
    // `self.bodies` directly; ECS-side we reconstruct the mapping.
    let id_to_entity: std::collections::HashMap<jeod_sim::MassBodyId, Entity> = bodies
        .iter()
        .map(|(e, body_id, _, _, _)| (body_id.0, e))
        .collect();

    for evt in detach_events.read() {
        let (_, child_body_id, _, _, _) = bodies.get(evt.child).unwrap_or_else(|_| {
            panic!(
                "DetachEvent.child = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC.",
                evt.child
            )
        });
        let child_id = child_body_id.0;

        // Walk up to the current tree root. The runner's
        // `detach_subtree` does this same walk; the parent's composite-
        // body inertial state lives at the root (only the integrated /
        // free-flying root carries the merged composite — attached
        // children's `TranslationalStateC` is stale post-attach, since
        // post-attach state is propagated down from the root by
        // `propagate_state_from_root_system` rather than re-merged at
        // each child).
        let mut tree_root_id = child_id;
        while let Some(p) = tree.parent(tree_root_id) {
            tree_root_id = p;
        }
        if tree_root_id == child_id {
            // Detaching a body that has no parent in the mass tree is
            // a misconfiguration: the rigid-body subtree is already
            // free-flying with respect to every other tree, so there
            // is no parent composite to derive child state from.
            panic!(
                "DetachEvent.child = {:?} (mass id {:?}) has no parent in the mass tree — \
                 detaching a tree root is a no-op in JEOD and indicates a stale event \
                 (e.g. firing DetachEvent twice without a re-AttachEvent in between).",
                evt.child, child_id,
            );
        }

        let tree_root_entity = *id_to_entity.get(&tree_root_id).unwrap_or_else(|| {
            panic!(
                "DetachEvent.child = {:?}: tree root {:?} has no entity in the bodies query — \
                 every mass-tree node must be spawned with `MassBodyIdC` before any \
                 attach/detach event references it.",
                evt.child, tree_root_id,
            )
        });

        // Pre-mutation snapshot of the parent's composite-body inertial
        // state (read from the root entity, which is the only place
        // post-attach that carries the merged composite — attached
        // children's `TranslationalStateC` is stale, populated by
        // root-down propagation rather than re-merged in place).
        // Keeping these as raw f64 fields (not borrowing the query)
        // avoids holding a borrow across the `bodies.iter()` /
        // `bodies.get_mut` calls below.
        //
        // `parent_pre_composite_props` is read from the legacy
        // `MassTreeR` arena rather than the entity's
        // `MassPropertiesC` because the ECS-tree fast path in
        // `composite_mass_system` reverts `MassPropertiesC` to its
        // `CoreMassPropertiesC` cache for any entity that has no
        // `MassChildOf` edge, and the arena attach/detach path
        // exercised here never adds those edges. Without this
        // arena-read, by the time the detach handler runs (in the
        // same tick, after `composite_mass_system`),
        // `parent_mass_c.0.to_untyped()` would yield the just-reverted
        // *core* mass instead of the live post-attach composite — and
        // the CoM-shift formula below would key off
        // `composite_properties.position == core.position` (typically
        // zero), corrupting the parent's post-detach inertial position.
        // The arena tree is the same source of truth the runner reads
        // in `Simulation::detach_subtree`, so this also keeps the two
        // adapters bit-identical for the parent-side post-detach
        // CoM-shift. Mirrors `jeod_runner::Simulation::detach_subtree`'s
        // `tree.get(tree_root_id).composite_properties` access.
        //
        // JEOD_INV: MA.23 — composite-property reads at detach must
        // see the live (pre-detach) composite, not a downstream
        // cache; the `MassTree` arena is the canonical store.
        let (
            parent_pre_position,
            parent_pre_velocity,
            parent_pre_quat,
            parent_pre_ang_vel_body,
            parent_pre_composite_props,
        ) = {
            let (_, _, _, parent_trans, parent_rot) = bodies
                .get(tree_root_entity)
                .expect("id_to_entity points at a valid mass body");
            let position = parent_trans
                .as_ref()
                .map(|t| t.0.position.raw_si())
                .unwrap_or(glam::DVec3::ZERO);
            let velocity = parent_trans
                .as_ref()
                .map(|t| t.0.velocity.raw_si())
                .unwrap_or(glam::DVec3::ZERO);
            let (q, w) = parent_rot
                .as_ref()
                .map(|r| {
                    let u = r.0.to_untyped();
                    (u.quaternion, u.ang_vel_body)
                })
                .unwrap_or((jeod_sim::JeodQuat::identity(), glam::DVec3::ZERO));
            let composite = tree.get(tree_root_id).composite_properties;
            (position, velocity, q, w, composite)
        };

        // Walk root → subtree applying `propagate_forward` at each
        // level using the mass-tree's `composite_wrt_pstr` offsets.
        // This is the JEOD-faithful derivation of the subtree's
        // instantaneous composite-body inertial state at the detach
        // instant — i.e. the rigid-body composition of the parent's
        // composite-body state plus the subtree's offset within the
        // composite. Runner does the same in `detach_subtree`.
        let mut chain: Vec<jeod_sim::MassBodyId> = Vec::new();
        let mut walker = child_id;
        while walker != tree_root_id {
            chain.push(walker);
            walker = tree
                .parent(walker)
                .expect("chain walk hit a parentless intermediate before reaching tree root");
        }
        chain.reverse();

        let parent_composite_state = jeod_sim::RefFrameState {
            trans: jeod_sim::RefFrameTrans {
                position: parent_pre_position,
                velocity: parent_pre_velocity,
            },
            rot: jeod_sim::RefFrameRot {
                q_parent_this: parent_pre_quat,
                t_parent_this: parent_pre_quat.left_quat_to_transformation(),
                ang_vel_this: parent_pre_ang_vel_body,
            },
        };
        let mut current_state = parent_composite_state;
        let mut current_node_id = tree_root_id;
        for next_id in &chain {
            let next_node = tree.get(*next_id);
            let current_node = tree.get(current_node_id);
            // Body-aware step (matches runner's detach walk):
            //   offset_in_current_body = T_current_struct_to_body
            //                          · (next.composite_wrt_pstr.position
            //                             − current.composite_properties.position)
            //   T_current_body_to_next_body = T_next_struct_to_body
            //                               · next.structure_point.t_parent_this
            //                               · T_current_body_to_struct
            let t_current_struct_to_body = current_node.composite_properties.t_parent_this;
            let t_next_struct_to_body = next_node.composite_properties.t_parent_this;
            let offset_struct =
                next_node.composite_wrt_pstr.position - current_node.composite_properties.position;
            let offset_in_current_body = t_current_struct_to_body * offset_struct;
            let t_current_body_to_next_body = t_next_struct_to_body
                * next_node.structure_point.t_parent_this
                * t_current_struct_to_body.transpose();
            let rel = jeod_sim::MassPointState {
                position: offset_in_current_body,
                t_parent_this: t_current_body_to_next_body,
            };
            current_state = jeod_sim::propagate_forward(&current_state, &rel);
            current_node_id = *next_id;
        }
        let subtree_state = current_state;

        let captured = jeod_sim::stage_detach_capture(
            subtree_state.trans.position,
            subtree_state.trans.velocity,
            subtree_state.rot.q_parent_this,
            subtree_state.rot.ang_vel_this,
        );
        detach_work.push((evt.child, captured));

        // Stash the parent-side post-mutation update for later (after
        // tree.detach + composite mass sync). The CoM-shift uses the
        // pre/post composite properties — the post is read after
        // mutation so we record only the pre-state here.
        let parent_was_detached_root = detached_q.contains(tree_root_entity);
        parent_shifts.push((
            tree_root_id,
            ParentShift {
                tree_root_entity,
                parent_pre_position,
                parent_pre_velocity,
                parent_pre_quat,
                parent_pre_ang_vel_body,
                parent_pre_composite_props,
                parent_was_detached: parent_was_detached_root,
            },
        ));

        // Bodies whose composite changes: the (about-to-be-detached)
        // child plus the former parent's full ancestor chain. Capture
        // BEFORE mutating the tree.
        affected_ids.push(child_id);
        affected_ids.extend(tree.ancestors_inclusive(tree_root_id));
        tree.detach(child_id);
    }

    if affected_ids.is_empty() && attach_work.is_empty() && detach_work.is_empty() {
        return;
    }
    affected_ids.sort_unstable();
    affected_ids.dedup();

    // Sync composite mass properties for all affected nodes.
    //
    // These writes go through `bypass_change_detection` because the value being
    // written is the *composite* (post-Steiner) mass, not a core-mass
    // edit by mission code. The `composite_mass_system` ECS path uses
    // `Changed<MassPropertiesC>` to detect mid-sim core edits (fuel
    // burn, propellant offload) and refresh its hidden
    // [`crate::mass_tree::CoreMassPropertiesC`] cache. If the legacy
    // arena `staging_system` write tripped that filter, the next tick
    // the ECS path would seed `CoreMassPropertiesC` from a *composite*
    // value — corrupting the core cache so every subsequent
    // recomposition would Steiner-shift the already-composed mass on
    // top of itself. Bypassing change detection here keeps the two
    // composition paths (legacy arena via `MassBodyIdC`/`AttachEvent`
    // and ECS-native via `MassChildOf`) safe to coexist on the same
    // entity during the migration window. The `MassPropertiesC` value
    // is still updated; only the change-detection signal is silenced.
    for (_, body_id, mut mass, _, _) in &mut bodies {
        if affected_ids.binary_search(&body_id.0).is_ok() {
            *mass.bypass_change_detection() =
                MassPropertiesC::from(tree.get(body_id.0).composite_properties);
        }
    }

    // Run the JEOD momentum-conservation combine for every staged
    // attach. This must happen *after* the composite-mass sync above
    // so the merged mass we feed the kernel matches the parent's
    // post-attach `MassPropertiesC` (which is what subsequent
    // gravity / force-collection / integration reads in the same tick).
    //
    // JEOD_INV: DB.13 — state propagation across attached subtrees: only the
    // root carries the integrated composite-body state; child sub-trees ride
    // it via the MassChildOf / mass-tree composition (not yet propagated
    // through derived frames; see #198 frame-attached body integration).
    // JEOD_INV: DB.14 — integration-frame switch on attach: the combined
    // body integrates in the parent's frame; here we update the parent's
    // composite_body state; frame-side switching belongs to #280.
    // JEOD_INV: DB.21 — only unattached bodies integrate: after attach the
    // detached-subtree-state is removed from the child so it stops drifting
    // ballistically; the integrated body's state is the merged composite.
    for work in &attach_work {
        let combined_mass = tree.get(work.parent_id).composite_properties;
        let merged = jeod_sim::stage_attach_combine(jeod_sim::StageAttachInputs {
            parent_position: work.parent_position,
            parent_velocity: work.parent_velocity,
            parent_quaternion: work.parent_quaternion,
            parent_ang_vel_body: work.parent_ang_vel_body,
            parent_mass: work.parent_mass,
            orig_parent_cm_struct: work.orig_parent_cm_struct,
            parent_t_inertial_struct: work.parent_t_inertial_struct,
            child_position: work.child_position,
            child_velocity: work.child_velocity,
            child_quaternion: work.child_quaternion,
            child_ang_vel_body: work.child_ang_vel_body,
            child_mass: work.child_mass,
            combined_mass,
        });

        if let Ok((_, _, _, mut trans, mut rot)) = bodies.get_mut(work.parent_entity) {
            if let Some(ref mut t) = trans {
                t.0 =
                    // allowed: stage_attach_combine kernel boundary; the
                    // kernel returns untyped DVec3 by design, so re-wrapping
                    // as TranslationalStateTyped<PlanetInertial<SelfPlanet>>
                    // is the same typed↔untyped pattern as the
                    // From<TranslationalState> impl on TranslationalStateC.
                    jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>::from_untyped_unchecked(
                        &jeod_sim::TranslationalState {
                            position: merged.position,
                            velocity: merged.velocity,
                        },
                    );
            }
            if let Some(ref mut r) = rot {
                // allowed: stage_attach_combine kernel boundary; same
                // typed↔untyped re-wrap pattern as the translational case
                // above. The output quaternion is the parent's pre-attach
                // unit-norm quaternion (per `combine_states_at_attach`'s
                // "merged body inherits parent attitude"), so the
                // NormalizedQuat witness in from_untyped_unchecked is
                // satisfied.
                r.0 = jeod_sim::RotationalStateTyped::<jeod_sim::SelfRef>::from_untyped_unchecked(
                    &jeod_sim::RotationalState {
                        quaternion: merged.quaternion,
                        ang_vel_body: merged.ang_vel_body,
                    },
                );
            }
        }

        if work.child_was_detached {
            // Re-attach consumes the captured ballistic state — the
            // child is no longer free-flying.
            commands
                .entity(work.child_entity)
                .remove::<crate::DetachedSubtreeStateC>();
        }

        if work.parent_was_detached {
            // The merged composite is still a free-flying tree root
            // (the parent had no integrated ancestor to graft onto).
            // Replace the parent's stale pre-attach `DetachedSubtreeStateC`
            // with the merged composite-body inertial state so
            // `step_detached_system` continues advancing the right
            // value next tick rather than overwriting the merged state
            // with the captured pre-attach snapshot.
            //
            // JEOD_INV: DB.21 — detached subtrees keep advancing
            // ballistically post-attach; the merged composite simply
            // becomes the new "free-flying root" state.
            let updated = jeod_sim::DetachedSubtreeState {
                composite_position: merged.position,
                composite_velocity: merged.velocity,
                composite_attitude: jeod_sim::DetachedSubtreeState::attitude_from_raw_jeod_quat(
                    merged.quaternion,
                ),
                composite_ang_vel_body: merged.ang_vel_body,
            };
            commands
                .entity(work.parent_entity)
                .insert(crate::DetachedSubtreeStateC(updated));
        }
    }

    // Apply detach captures: insert `DetachedSubtreeStateC` on each
    // detached child so `step_detached_system` advances it ballistically
    // each tick.
    for (entity, captured) in detach_work {
        commands
            .entity(entity)
            .insert(crate::DetachedSubtreeStateC(captured));
    }

    // Parent-side post-detach composite-CoM shift: when a subtree is
    // removed from a tree, the parent's composite-CoM moves within its
    // own struct frame. The parent's rigid-body structure point hasn't
    // moved in inertial space, but the composite-body inertial state
    // (which is what `TranslationalStateC` stores after the
    // composite-body refactor) must shift by the corresponding
    // kinematic offset to track the new (smaller) composite. Mirrors
    // `jeod_runner::Simulation::detach_subtree`'s integrated-body /
    // detached-parent branches; both produce the same inertial
    // CoM-delta formula.
    //
    // JEOD_INV: DB.13 — composite-body propagation on topology change.
    for (tree_root_id, shift) in parent_shifts {
        let parent_post_composite_props = tree.get(tree_root_id).composite_properties;
        let cm_delta_struct =
            parent_post_composite_props.position - shift.parent_pre_composite_props.position;
        // composite_properties.t_parent_this is struct→body. Compose
        // with the body's inertial-to-body to map struct → inertial.
        let t_struct_to_body = shift.parent_pre_composite_props.t_parent_this;
        let cm_delta_body = t_struct_to_body * cm_delta_struct;
        let t_inertial_to_body = shift.parent_pre_quat.left_quat_to_transformation();
        let cm_delta_inertial = t_inertial_to_body.transpose() * cm_delta_body;
        // Velocity offset from rigid-body rotation: ω × Δr in body
        // frame, then rotated to inertial.
        let omega_body = shift.parent_pre_ang_vel_body;
        let dvel_inertial = t_inertial_to_body.transpose() * omega_body.cross(cm_delta_body);

        let new_position = shift.parent_pre_position + cm_delta_inertial;
        let new_velocity = shift.parent_pre_velocity + dvel_inertial;

        if let Ok((_, _, _, Some(mut t), _)) = bodies.get_mut(shift.tree_root_entity) {
            t.0 =
                // allowed: detach-handler kernel boundary; same
                // typed↔untyped re-wrap pattern as the attach branch
                // above. The CoM-shift is a pure kinematic update —
                // it does not introduce a new frame, so wrapping as
                // `PlanetInertial<SelfPlanet>` is the same convention
                // as the pre-detach value.
                jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>::from_untyped_unchecked(
                    &jeod_sim::TranslationalState {
                        position: new_position,
                        velocity: new_velocity,
                    },
                );
        }

        if shift.parent_was_detached {
            // The parent is itself a detached free-flying root — keep
            // its `DetachedSubtreeStateC` in lock-step with the shifted
            // `TranslationalStateC` so the next `step_detached_system`
            // tick advances from the post-detach composite state.
            // Quaternion / ang_vel are unchanged because the parent's
            // body axes don't rotate just because mass left the tree
            // (composite_properties.t_parent_this == core_properties
            // .t_parent_this throughout — see mass tree recompute).
            let updated = jeod_sim::DetachedSubtreeState {
                composite_position: new_position,
                composite_velocity: new_velocity,
                composite_attitude: jeod_sim::DetachedSubtreeState::attitude_from_raw_jeod_quat(
                    shift.parent_pre_quat,
                ),
                composite_ang_vel_body: shift.parent_pre_ang_vel_body,
            };
            commands
                .entity(shift.tree_root_entity)
                .insert(crate::DetachedSubtreeStateC(updated));
        }
    }

    // Site A: mark every affected body's integrators dirty.
    // JEOD_INV: IG.37 — kept strictly before Site B so a regression
    // that drops Site B leaves the dirty flag set and panics on next
    // integrate.
    for (body_id, mut gj_opt, mut abm_opt) in &mut integrators {
        if affected_ids.binary_search(&body_id.0).is_ok() {
            if let Some(ref mut gj) = gj_opt {
                gj.0.mark_topology_dirty();
            }
            if let Some(ref mut abm) = abm_opt {
                abm.0.mark_topology_dirty();
            }
        }
    }

    // Site B: reset integrator history. Mirrors JEOD's
    // `dyn_body_attach.cc::reset_integrators()` (lines 860, 871) and
    // `dyn_body_detach.cc:271-273`.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    for (body_id, mut gj_opt, mut abm_opt) in &mut integrators {
        if affected_ids.binary_search(&body_id.0).is_ok() {
            jeod_sim::reset_integrators(
                gj_opt.as_mut().map(|c| &mut c.0),
                abm_opt.as_mut().map(|c| &mut c.0),
            );
        }
    }
}

/// Advance every entity carrying [`crate::DetachedSubtreeStateC`] by
/// the schedule's fixed `dt` under ballistic dynamics — no force, no
/// torque. Position drifts at `composite_velocity`; attitude rotates
/// at `composite_ang_vel_body` via JEOD's left-multiply convention
/// (`q̇ = -½(ω ⊗ q)`, owned by [`jeod_sim::BodyAttitude`]).
///
/// Also synchronizes the entity's [`crate::TranslationalStateC`] /
/// [`crate::RotationalStateC`] with the advanced subtree state each
/// tick so downstream consumers (gravity-source position lookups,
/// derived-state systems, mission code) see the body's current
/// inertial state without having to special-case detached vs
/// integrated bodies. Mirrors
/// `jeod_runner::Simulation::step_detached_subtrees`.
///
/// The ballistic timestep is `dt * time_scale_factor` (matching
/// `integration_system`'s `integ_dt` and the runner's
/// `step_detached_subtrees(dt * time.time_scale_factor)`); under
/// reversed or scaled time the detached subtree advances at the same
/// rate as integrated bodies, so the two stay phase-locked.
///
/// JEOD_INV: DB.21 — only unattached bodies integrate; detached subtrees
/// drift ballistically here while the integrator targets the integrated
/// body.
pub fn step_detached_system(
    time: Res<Time<Fixed>>,
    sim_time: Res<SimulationTimeR>,
    mut detached: Query<(
        &mut crate::DetachedSubtreeStateC,
        Option<&mut TranslationalStateC>,
        Option<&mut RotationalStateC>,
    )>,
) {
    let dt = time.delta().as_secs_f64();
    if dt == 0.0 {
        return;
    }
    let integ_dt = dt * sim_time.0.time_scale_factor;
    for (mut state, trans, rot) in &mut detached {
        state.0.step_ballistic(integ_dt);
        if let Some(mut t) = trans {
            t.0 =
                // allowed: DetachedSubtreeState kernel boundary; the
                // ballistic-step result is returned as raw DVec3 fields by
                // design — re-wrapping into TranslationalStateTyped is the
                // same typed↔untyped pattern as the
                // From<TranslationalState> impl on TranslationalStateC.
                jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>::from_untyped_unchecked(
                    &jeod_sim::TranslationalState {
                        position: state.0.composite_position,
                        velocity: state.0.composite_velocity,
                    },
                );
        }
        if let Some(mut r) = rot {
            // allowed: DetachedSubtreeState kernel boundary. The advanced
            // `composite_attitude` is a `BodyAttitude<SelfRef>` whose
            // `to_jeod_quat` returns the underlying scalar-first
            // left-transformation quaternion. The wrapper guarantees
            // unit-norm post-step (that's the whole point of
            // `BodyAttitude::advance_under_body_rate`), so the
            // NormalizedQuat witness in from_untyped_unchecked is
            // satisfied.
            r.0 = jeod_sim::RotationalStateTyped::<jeod_sim::SelfRef>::from_untyped_unchecked(
                &jeod_sim::RotationalState {
                    quaternion: state.0.composite_attitude.to_jeod_quat(),
                    ang_vel_body: state.0.composite_ang_vel_body,
                },
            );
        }
    }
}
