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
    Acceleration, AngularAcceleration, BodyFrame, Force, Planet, Position, RootInertial, SelfRef,
    Torque, Velocity,
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
pub fn register_source_frames_system<P: Planet>(
    mut commands: Commands,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sources: Query<
        (
            Entity,
            Option<&Name>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
            Option<&RotationModelC>,
            Option<&PlanetFixedRotationC<P>>,
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
pub fn register_pfix_frames_system<P: Planet>(
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
            With<PlanetFixedRotationC<P>>,
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
pub fn sync_source_to_frame_system<P: Planet>(
    sources: Query<(
        &FrameEntityC,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TranslationalStateC<P>>,
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
pub fn register_body_frames_system<P: Planet>(
    mut commands: Commands,
    // The ECS-side root frame entity, used as the body's frame
    // parent when no IntegSourceC is supplied.
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sources: Query<&FrameEntityC, With<GravitySourceC>>,
    bodies: Query<
        (
            Entity,
            Option<&Name>,
            &TranslationalStateC<P>,
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
            With<TranslationalStateC<P>>,
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
pub fn sync_body_to_frame_system<P: Planet>(
    bodies: Query<(&TranslationalStateC<P>, &FrameEntityC), With<DynamicsConfigC>>,
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
pub fn frame_switch_system<P: Planet>(
    mut commands: Commands,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sources: Query<&FrameEntityC, With<GravitySourceC>>,
    parents: Query<&ChildOf>,
    rel: RelativeFrameState,
    mut bodies: Query<(
        Entity,
        &mut TranslationalStateC<P>,
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
        // Re-wrap as the Component's `PlanetInertial<P>` phantom —
        // `new_state.trans` carries planet-inertial coordinates of the
        // *target* source's planet (this is the post-switch frame)
        // which the same `<P>` parameter tags. The system instantiation
        // for `<P>` is responsible for matching the body's planet
        // identity at the call site (see `register_planet_systems`);
        // each instantiation only matches bodies with `TranslationalStateC<P>`.
        // Same boundary lift `evaluate_and_apply_frame_switch` performs.
        // allowed: frame-switch boundary lift, see comment above
        let pos_typed = jeod_sim::Position::<jeod_sim::PlanetInertial<P>>::from_raw_si(
            new_state.trans.position,
        );
        // allowed: same frame-switch boundary lift
        let vel_typed = jeod_sim::Velocity::<jeod_sim::PlanetInertial<P>>::from_raw_si(
            new_state.trans.velocity,
        );
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
pub fn planet_fixed_rotation_system<P: Planet>(
    mut commands: Commands,
    sim_time: Res<SimulationTimeR>,
    polar: Option<Res<crate::PolarMotionR>>,
    ephemeris: Option<Res<crate::EphemerisR>>,
    mut query: Query<(
        Entity,
        &mut PlanetFixedRotationC<P>,
        Option<&RotationModelC>,
        Option<&PlanetOmegaC>,
        Option<&mut PlanetAngularVelocityC<P>>,
        Option<&PfixFrameEntityC>,
    )>,
    mut frame_rots: Query<&mut FrameRotC>,
    mut frame_ang_vels: Query<&mut FrameAngVelC>,
) {
    let polar_params = polar.map(|p| (p.xp, p.yp));
    // Lazy-compute Earth RNP once per system invocation when an
    // `EarthRNP` rotation-model entity is matched. Cache the
    // already-typed `FrameTransform` rather than the bare matrix so the
    // expensive `from_matrix` work (matrix→quat extraction + renormalization)
    // happens once per tick total, not once per EarthRNP entity per tick —
    // all matched entities share the same rotation each step.
    type PlanetRot<P> = jeod_sim::FrameTransform<jeod_sim::RootInertial, jeod_sim::PlanetFixed<P>>;
    let mut earth_rotation: Option<PlanetRot<P>> = Option::None;
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
                    // RootInertial → PlanetFixed<P> phantoms match the
                    // kernel by construction (system instantiation pins P).
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
                // Mint `AngularVelocity<PlanetFixed<P>>` from the
                // scalar `PlanetOmegaC`. JEOD's `planet_rnp.cc` writes
                // [0, 0, omega] in the pfix frame; this is the typed-API
                // boundary for that scalar → typed-vector lift.
                let raw = glam::DVec3::new(0.0, 0.0, omega_value);
                let typed = jeod_sim::AngularVelocity::<jeod_sim::PlanetFixed<P>>::from_raw_si(raw); // allowed: scalar omega → typed AngularVelocity boundary
                ang_vel_c.0 = typed;
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
            // the RootInertial → PlanetFixed<P> phantoms are correct by
            // construction (same shape as the rotating-branch from_matrix sites).
            rot.0 = jeod_sim::FrameTransform::from_matrix(glam::DMat3::IDENTITY);
            if let Some(mut ang_vel_c) = ang_vel {
                // allowed: zero-omega clear → typed AngularVelocity boundary
                ang_vel_c.0 = jeod_sim::AngularVelocity::<jeod_sim::PlanetFixed<P>>::from_raw_si(
                    glam::DVec3::ZERO,
                );
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
///
/// The `Without<...>` filters on the three sibling kinematic-spec
/// components are a *parallelism signal* for Bevy's scheduler — they
/// make this query structurally disjoint from the sinusoidal /
/// closure / multi-DOF drivers so the four systems can dispatch in
/// parallel under `JeodSet::EphemerisUpdate` without a runtime borrow
/// conflict on `FrameRotC` / `FrameAngVelC`. They are *not* the
/// correctness mechanism that rejects stacked-spec entities.
///
/// Stacked-spec rejection is enforced by the per-component
/// `on_insert` hooks installed via
/// [`register_joint_kinematics_exclusivity_hooks`]: inserting a
/// second kinematic-spec component on an entity that already carries
/// one panics immediately, naming the entity and both spec
/// components, before any driver query ever runs. The PostStartup
/// [`validate_joint_kinematics_exclusivity`] pass is defense in
/// depth — it catches stacking patterns that bypass the hook order
/// (e.g., a `Bundle` whose components arrive in the same archetype
/// move, or future spec components added without registering a hook)
/// and aggregates every offender into a single startup-time panic.
#[allow(clippy::type_complexity)]
pub fn joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (&JointKinematicsC, &mut FrameRotC, &mut FrameAngVelC),
        (
            Without<SinusoidalJointKinematicsC>,
            Without<ClosureJointKinematicsC>,
            Without<MultiDofJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) = jeod_sim::evaluate_joint_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// Drives sinusoidal kinematic joint frames each tick.
///
/// Sibling of [`joint_kinematics_system`] that handles
/// [`SinusoidalJointKinematicsC`]-tagged frame entities. Reads the
/// same `tai_seconds` clock and writes the same
/// [`FrameRotC`] / [`FrameAngVelC`] storage, so downstream consumers
/// that walk the frame tree see uniform output across the kinematic
/// styles.
///
/// Scheduled in [`crate::JeodSet::EphemerisUpdate`] alongside
/// `planet_fixed_rotation_system` and `joint_kinematics_system` —
/// the joint frame's rotation / angular velocity must be current
/// before any consumer that walks the frame tree (gravity, derived
/// state, integration) reads them.
///
/// The `Without<...>` filters mirror the contract documented on
/// [`joint_kinematics_system`]: they are a parallelism signal that
/// keeps the four kinematic-spec drivers pairwise-disjoint at the
/// query level. The correctness mechanism that rejects stacked-spec
/// entities is the on_insert hooks installed by
/// [`register_joint_kinematics_exclusivity_hooks`] (panic at
/// insertion); [`validate_joint_kinematics_exclusivity`] is
/// PostStartup defense in depth.
#[allow(clippy::type_complexity)]
pub fn sinusoidal_joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (
            &SinusoidalJointKinematicsC,
            &mut FrameRotC,
            &mut FrameAngVelC,
        ),
        (
            Without<JointKinematicsC>,
            Without<ClosureJointKinematicsC>,
            Without<MultiDofJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) =
            jeod_sim::evaluate_sinusoidal_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// Drives closure (fixed-pose) kinematic joint frames each tick.
///
/// Sibling of [`joint_kinematics_system`] that handles
/// [`ClosureJointKinematicsC`]-tagged frame entities. The output is
/// constant in time, so the system writes the same `FrameRotC` /
/// `FrameAngVelC` value every step. Scheduled in
/// [`crate::JeodSet::EphemerisUpdate`] alongside
/// `joint_kinematics_system` so the closure-pinned frame's rotation
/// is materialized before any frame-tree consumer reads it.
///
/// The `Without<...>` filters mirror the contract documented on
/// [`joint_kinematics_system`]: they are a parallelism signal that
/// keeps the four kinematic-spec drivers pairwise-disjoint at the
/// query level. The correctness mechanism that rejects stacked-spec
/// entities is the on_insert hooks installed by
/// [`register_joint_kinematics_exclusivity_hooks`] (panic at
/// insertion); [`validate_joint_kinematics_exclusivity`] is
/// PostStartup defense in depth.
#[allow(clippy::type_complexity)]
pub fn closure_joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (&ClosureJointKinematicsC, &mut FrameRotC, &mut FrameAngVelC),
        (
            Without<JointKinematicsC>,
            Without<SinusoidalJointKinematicsC>,
            Without<MultiDofJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) = jeod_sim::evaluate_closure_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// Drives multi-DOF kinematic joint frames each tick.
///
/// Sibling of [`joint_kinematics_system`] that handles
/// [`MultiDofJointKinematicsC`]-tagged frame entities. Each entity
/// carries an N-stage chain (`N <= MAX_MULTI_DOF_AXES`); the kernel
/// folds the per-stage `(rotation, ang_vel)` contributions through
/// `RefFrameState::incr_right` so the output is bit-identical to a
/// chain of N single-DOF joint entities walked through the frame
/// tree. Scheduled in [`crate::JeodSet::EphemerisUpdate`] for the
/// same reason as the other joint-kinematics systems.
///
/// The `Without<...>` filters mirror the contract documented on
/// [`joint_kinematics_system`]: they are a parallelism signal that
/// keeps the four kinematic-spec drivers pairwise-disjoint at the
/// query level. The correctness mechanism that rejects stacked-spec
/// entities is the on_insert hooks installed by
/// [`register_joint_kinematics_exclusivity_hooks`] (panic at
/// insertion); [`validate_joint_kinematics_exclusivity`] is
/// PostStartup defense in depth.
#[allow(clippy::type_complexity)]
pub fn multi_dof_joint_kinematics_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<
        (&MultiDofJointKinematicsC, &mut FrameRotC, &mut FrameAngVelC),
        (
            Without<JointKinematicsC>,
            Without<SinusoidalJointKinematicsC>,
            Without<ClosureJointKinematicsC>,
        ),
    >,
) {
    let elapsed = sim_time.tai_seconds;
    for (spec, mut rot, mut ang_vel) in &mut query {
        let (q_parent_this, ang_vel_this) =
            jeod_sim::evaluate_multi_dof_kinematics(&spec.0, elapsed);
        rot.q_parent_this = q_parent_this;
        rot.t_parent_this = q_parent_this.left_quat_to_transformation();
        ang_vel.0 = ang_vel_this;
    }
}

/// PostStartup-time guard that asserts at most one of the four
/// joint-kinematic spec components is present on any single entity.
///
/// The four joint-kinematic drivers
/// ([`joint_kinematics_system`], [`sinusoidal_joint_kinematics_system`],
/// [`closure_joint_kinematics_system`], [`multi_dof_joint_kinematics_system`])
/// each carry `Without<...>` filters for the other three spec
/// components so Bevy's scheduler can dispatch them in parallel under
/// `JeodSet::EphemerisUpdate` without contending for `FrameRotC` /
/// `FrameAngVelC`. That filter discipline turns an entity that
/// accidentally carries two specs into a *silent drop* from every
/// driver — its `FrameRotC` would never be written and the joint
/// frame would advertise stale (or default-identity) state to every
/// downstream `RelativeFrameState` walk.
///
/// Per the project's fail-loud rule, that misconfiguration must
/// panic at the earliest detection point with a diagnostic that
/// names the offending entity and the specs it carries. The primary
/// guard is the per-component `on_insert` hook installed by
/// [`register_joint_kinematics_exclusivity_hooks`], which fires at
/// insertion time and catches every stacking pattern (Startup,
/// FixedUpdate, observers, …). This `PostStartup` validator is
/// defense in depth: it walks every entity that already carries at
/// least one kinematic spec once before the first `FixedUpdate` tick
/// and emits a *single* aggregated panic message that lists every
/// offending entity at once. The `on_insert` hooks panic on the
/// first stacked insertion they observe, which is right for runtime
/// but less informative when the user declares several stacked
/// entities together at startup; the aggregated startup pass keeps
/// that path actionable.
///
/// The four spec components are declarative alternatives — a joint
/// is *either* constant-rate, *or* sinusoidal, *or* a closure pose,
/// *or* a multi-DOF chain — so stacking two of them has no
/// meaningful semantics. If a future kinematic style needs to
/// compose with an existing one, that composition belongs in a new
/// dedicated spec (e.g., extend `SingleDofKinematics` and route
/// through `MultiDofJointKinematicsC`), not in two parallel
/// drivers racing for the same storage.
///
/// # Panics
/// Panics if any entity carries more than one of `JointKinematicsC`,
/// [`SinusoidalJointKinematicsC`], [`ClosureJointKinematicsC`], or
/// [`MultiDofJointKinematicsC`]. The message lists every offending
/// entity together with the specs it carries.
#[allow(clippy::type_complexity)]
pub fn validate_joint_kinematics_exclusivity(
    query: Query<
        (
            Entity,
            Has<JointKinematicsC>,
            Has<SinusoidalJointKinematicsC>,
            Has<ClosureJointKinematicsC>,
            Has<MultiDofJointKinematicsC>,
        ),
        Or<(
            With<JointKinematicsC>,
            With<SinusoidalJointKinematicsC>,
            With<ClosureJointKinematicsC>,
            With<MultiDofJointKinematicsC>,
        )>,
    >,
) {
    let mut offenders: Vec<String> = Vec::new();
    for (entity, has_const, has_sin, has_close, has_multi) in &query {
        let count = usize::from(has_const)
            + usize::from(has_sin)
            + usize::from(has_close)
            + usize::from(has_multi);
        if count > 1 {
            let mut names: Vec<&'static str> = Vec::new();
            if has_const {
                names.push("JointKinematicsC");
            }
            if has_sin {
                names.push("SinusoidalJointKinematicsC");
            }
            if has_close {
                names.push("ClosureJointKinematicsC");
            }
            if has_multi {
                names.push("MultiDofJointKinematicsC");
            }
            offenders.push(format!("{entity:?} carries [{}]", names.join(", ")));
        }
    }
    assert!(
        offenders.is_empty(),
        "Joint-kinematics spec components are mutually exclusive — each frame entity \
         must carry at most one of JointKinematicsC, SinusoidalJointKinematicsC, \
         ClosureJointKinematicsC, MultiDofJointKinematicsC. Offending entities: {}. \
         Fix: pick a single kinematic style per joint frame; for composed motions \
         use MultiDofJointKinematicsC with a chain of SingleDofKinematics stages.",
        offenders.join("; ")
    );
}

/// Format a "stacked specs" panic diagnostic for a single offending
/// entity. Centralized so the `on_insert` hooks below and any future
/// detection site share one message shape — a mission engineer reading
/// the panic always sees the same actionable instructions regardless
/// of which path tripped the check.
fn format_stacked_specs_panic(entity: Entity, names: &[&'static str]) -> String {
    format!(
        "Joint-kinematics spec components are mutually exclusive — each frame entity \
         must carry at most one of JointKinematicsC, SinusoidalJointKinematicsC, \
         ClosureJointKinematicsC, MultiDofJointKinematicsC. Offending entity: \
         {entity:?} carries [{}]. \
         Fix: pick a single kinematic style per joint frame; for composed motions \
         use MultiDofJointKinematicsC with a chain of SingleDofKinematics stages.",
        names.join(", ")
    )
}

/// Shared body of the four joint-kinematics `on_insert` hooks. Reads
/// every spec flag off the entity's post-insertion archetype, counts
/// how many distinct specs are present, and panics with
/// [`format_stacked_specs_panic`] if more than one is. `self_name`
/// is the spec component whose hook is firing — included in the
/// panic so a mission engineer reading the backtrace sees which
/// insertion attempt tripped the check.
///
/// `on_insert` runs after the bundle's components are already added
/// to the entity's archetype, so the four `contains::<...>` reads
/// on the `DeferredWorld` reflect the full post-insertion state.
fn check_stacked_specs(
    world: bevy::ecs::world::DeferredWorld<'_>,
    entity: Entity,
    self_name: &'static str,
) {
    let entity_ref = world.get_entity(entity).expect(
        "joint-kinematics on_insert hook: entity must exist when its component is inserted",
    );
    let has_const = entity_ref.contains::<JointKinematicsC>();
    let has_sin = entity_ref.contains::<SinusoidalJointKinematicsC>();
    let has_close = entity_ref.contains::<ClosureJointKinematicsC>();
    let has_multi = entity_ref.contains::<MultiDofJointKinematicsC>();
    let count = usize::from(has_const)
        + usize::from(has_sin)
        + usize::from(has_close)
        + usize::from(has_multi);
    if count <= 1 {
        return;
    }
    let mut names: Vec<&'static str> = Vec::new();
    if has_const {
        names.push("JointKinematicsC");
    }
    if has_sin {
        names.push("SinusoidalJointKinematicsC");
    }
    if has_close {
        names.push("ClosureJointKinematicsC");
    }
    if has_multi {
        names.push("MultiDofJointKinematicsC");
    }
    panic!(
        "{} (triggered while inserting {self_name})",
        format_stacked_specs_panic(entity, &names)
    );
}

// `ComponentHook` is `fn(DeferredWorld, HookContext)` — a plain
// function pointer with no captures. Each spec component therefore
// gets its own dedicated `fn` item that forwards to
// `check_stacked_specs` with a hard-coded `self_name`.

fn on_insert_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "JointKinematicsC");
}

fn on_insert_sinusoidal_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "SinusoidalJointKinematicsC");
}

fn on_insert_closure_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "ClosureJointKinematicsC");
}

fn on_insert_multi_dof_joint_kinematics_c(
    world: bevy::ecs::world::DeferredWorld<'_>,
    ctx: bevy::ecs::lifecycle::HookContext,
) {
    check_stacked_specs(world, ctx.entity, "MultiDofJointKinematicsC");
}

/// Register `on_insert` hooks on every joint-kinematics spec component
/// so any insertion that lands a second spec on an entity panics
/// immediately.
///
/// The `PostStartup` validator
/// ([`validate_joint_kinematics_exclusivity`]) is a startup-time
/// safety net: it walks the world *once* before the first
/// `FixedUpdate` tick and catches misconfigurations declared in
/// `Startup` systems. It cannot observe entities spawned after
/// `PostStartup` — for example, a `Commands::spawn(...)` issued from
/// a `FixedUpdate` user system, an `Update` system, or an event
/// handler. Without a runtime guard those late spawns slip past every
/// driver's `Without<...>` filter and silently propagate stale
/// `FrameRotC` / `FrameAngVelC`, which the project's fail-loud rule
/// forbids.
///
/// Bevy 0.18 component lifecycle hooks (`on_insert`) close that gap:
/// every kinematic-spec insertion — `spawn`, `insert`, or `replace`
/// — fires its hook before the next system observes the new
/// component, so a bad insertion panics at the insertion site rather
/// than silently propagating bad state. The hook reads the entity's
/// post-insertion archetype to count how many of the four spec
/// components are present and panics with the same diagnostic shape
/// as [`validate_joint_kinematics_exclusivity`] if more than one is.
///
/// Idempotent re-registration of the *same* spec component (insert A
/// onto an entity that already has A) does not trip the hook: the
/// count of distinct kinematic specs is unchanged. Only stacking
/// distinct specs panics.
///
/// `JeodPlugin::build` calls this once during plugin setup. Tests
/// that exercise the joint-kinematics pipeline without `JeodPlugin`
/// can call this directly to install the same guard.
pub fn register_joint_kinematics_exclusivity_hooks(app: &mut App) {
    let world = app.world_mut();
    world
        .register_component_hooks::<JointKinematicsC>()
        .on_insert(on_insert_joint_kinematics_c);
    world
        .register_component_hooks::<SinusoidalJointKinematicsC>()
        .on_insert(on_insert_sinusoidal_joint_kinematics_c);
    world
        .register_component_hooks::<ClosureJointKinematicsC>()
        .on_insert(on_insert_closure_joint_kinematics_c);
    world
        .register_component_hooks::<MultiDofJointKinematicsC>()
        .on_insert(on_insert_multi_dof_joint_kinematics_c);
}

/// Computes tidal ΔC20 for each gravity source that has a `TidalConfigC`.
///
/// Runs after `planet_fixed_rotation_system` so the rotation matrix is current.
/// Sources without `TidalConfigC` keep their default `TidalDeltaC20C::default()`
/// (a zero-valued [`jeod_sim::Ratio`]).
pub fn tidal_update_system<P: Planet>(
    mut query: Query<(&TidalConfigC, &PlanetFixedRotationC<P>, &mut TidalDeltaC20C)>,
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
/// Also updates `SourceInertialVelocityC` and `TranslationalStateC<P>` when
/// present (velocity for relativistic corrections; translational state for
/// Sun/Moon entities used by SRP, solar beta, and earth lighting systems).
///
/// Generic over `P: Planet` so the relabel from `RootInertial` → `PlanetInertial<P>`
/// matches the planet phantom on the `TranslationalStateC<P>` instance being
/// updated. Each plugin instantiation only matches sources whose
/// translational state carries the matching `<P>` tag — Sun/Moon ephemeris
/// bodies typically lack `TranslationalStateC` (so `Option<&mut ...>` is
/// `None` and the relabel is skipped) or carry a tag matching the planet
/// they orbit. See `register_planet_systems` for downstream multi-planet
/// instantiation.
///
/// Placed in `JeodSet::EphemerisUpdate`.
#[allow(clippy::type_complexity)]
pub fn ephemeris_update_system<P: Planet>(
    ephemeris: Option<Res<crate::EphemerisR>>,
    sim_time: Res<SimulationTimeR>,
    mut query: Query<(
        &EphemerisBodyC,
        &mut SourceInertialPositionC,
        Option<&mut SourceInertialVelocityC>,
        Option<&mut TranslationalStateC<P>>,
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
            // TranslationalStateC wraps `TranslationalStateTyped<PlanetInertial<P>>`;
            // `pos_typed` / `vel_typed` are root-inertial-tagged by the
            // ephemeris API. Relabel via `from_raw_si` to the
            // `<P>`-tagged planet-inertial frame the Component stores.
            // The numeric SI values (m, m/s) are preserved exactly —
            // only the phantom tag changes. The query filter guarantees
            // we only land in this branch when the matched body's
            // `<P>` matches the system instantiation, so the relabel is
            // sound.
            let pos_si = pos_typed.raw_si();
            let vel_si = vel_typed.raw_si();
            // allowed: ephemeris boundary, RootInertial → PlanetInertial<P> relabel
            ts.0.position = jeod_sim::Position::<jeod_sim::PlanetInertial<P>>::from_raw_si(pos_si);
            // allowed: same ephemeris boundary relabel
            ts.0.velocity = jeod_sim::Velocity::<jeod_sim::PlanetInertial<P>>::from_raw_si(vel_si);
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
pub fn integration_system<P: Planet>(
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
    // JEOD_INV: DB.21 — detached subtrees and frame-attached bodies
    //   skip integration. The frame-attach filter mirrors the runner's
    //   `if body.frame_attach.is_some() { continue; }` guard in
    //   `step::integrate.rs`; bodies attached to a non-body reference
    //   frame have their state derived each tick by
    //   `propagate_frame_attached_state_system` (parent frame's
    //   current state composed with the captured offset) and the
    //   integrator must not stomp the kinematic value with a
    //   force-driven update.
    mut bodies: Query<
        (
            Entity,
            &DynamicsConfigC,
            &mut TranslationalStateC<P>,
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
            Without<crate::components::FrameAttachedC>,
        ),
    >,
    sources: Query<
        (
            &GravitySourceC,
            Option<&PlanetFixedRotationC<P>>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
            Option<&TidalDeltaC20C>,
            Option<&TidalConfigC>,
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

        // Helper: resolve a source's effective velocity from the
        // typed `SourceInertialVelocityC` (which is
        // `Velocity<RootInertial>` — planet-agnostic). Sources that
        // lack this component coast at zero velocity within the step.
        //
        // `SourceInertialVelocityC` is opt-in: `PlanetBundle`,
        // `SunBundle`, and `MoonBundle` do not insert it, and
        // `ephemeris_update_system` only writes through it when it is
        // already present (it does not auto-insert from
        // `EphemerisBodyC`). Callers who want a moving source for
        // per-stage gravity interpolation or relativistic source
        // resolution must attach `SourceInertialVelocityC` explicitly.
        //
        // No `TranslationalStateC<P>` fallback is offered here. The
        // `<P>` instantiation runs gravity-computation in
        // `PlanetInertial<P>` for the body's planet, and a Sun /
        // ephemeris source's `TranslationalStateC<P>` carries that
        // body-side `<P>` tag (per `SunBundle` / `MoonBundle`'s
        // construction-time convention) — so the velocity it stores
        // is "Sun's velocity tagged as the central planet's inertial
        // frame," which has no well-defined source-motion meaning.
        // Treating the source as stationary when no
        // `SourceInertialVelocityC` is present matches
        // `sync_source_to_frame_system`'s precedence: explicit
        // velocity component first, otherwise treat as no source-
        // motion contribution to the per-step kernel.
        let source_vel = |v: Option<&SourceInertialVelocityC>| -> DVec3 {
            v.map(|v| v.0.raw_si()).unwrap_or(DVec3::ZERO)
        };

        let typed_accel = jeod_sim::accumulate_gravity_typed(
            typed_abs_pos,
            &controls.0,
            typed_origin,
            |source_entity| match sources.get(source_entity) {
                Ok((s, r, p, v, tidal, tidal_config)) => {
                    let base_pos = p.0.raw_si();
                    let stage_pos = if sub_dt != 0.0 {
                        base_pos + source_vel(v) * sub_dt
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
                sources.get(source_entity).ok().map(|(s, _, p, v, _, _)| {
                    // Step-start values for PPN — runner does the
                    // same (snapshots `src_pos`/`src_vel` outside
                    // the per-stage closure).
                    jeod_sim::ResolvedRelativisticSource {
                        mu: s.mu,
                        position: p.0.raw_si(),
                        velocity: source_vel(v),
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
                        // Drop the typed `Position<StructuralFrame<SelfRef>>`
                        // phantom into the kernel's raw-DVec3 contract; the
                        // typed field is the storage-time guard.
                        srp_inputs.center_grav.raw_si(),
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
            // storage's `<PlanetInertial<P>>` / `<BodyFrame<SelfRef>>`
            // are the same frames the kernel was operating in — the
            // kernel computes everything in the body's integration
            // frame, which the Component tags as planet-inertial with
            // the system instantiation's `<P>` parameter that matches
            // this entity by query filter).
            type PiTrans<P> = jeod_sim::TranslationalStateTyped<jeod_sim::PlanetInertial<P>>;
            state.0 = PiTrans::<P>::from_untyped_unchecked(&state_untyped); // allowed: typed↔untyped kernel boundary (integrate_body_coupled signature is untyped); analogous to From<Untyped> impls.
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
            // allowed: typed↔untyped kernel boundary; planet-inertial frame matches the body's integration frame (system instantiation's `<P>` parameter, gated by the bodies query filter).
            jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<P>>::from_untyped_unchecked(&state_untyped);
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
pub fn gravity_computation_system<P: Planet>(
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
            &TranslationalStateC<P>,
            &GravityControlsC,
            &mut GravityAccelerationC,
            Option<&FrameEntityC>,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
    sources: Query<(
        &GravitySourceC,
        Option<&PlanetFixedRotationC<P>>,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TidalDeltaC20C>,
        Option<&TidalConfigC>,
    )>,
) {
    for (entity, state, controls, mut accel, body_frame) in &mut bodies {
        // `TranslationalStateC` stores typed
        // `Position<PlanetInertial<P>>` /
        // `Velocity<PlanetInertial<P>>`. For root-integrated
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
                Ok((source, rot, pos, _, tidal, tidal_config)) => {
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
                sources.get(source_entity).ok().map(|(s, _, p, v, _, _)| {
                    // Source velocity flows through the planet-agnostic
                    // `SourceInertialVelocityC` (`Velocity<RootInertial>`).
                    // It is opt-in: `PlanetBundle`, `SunBundle`, and
                    // `MoonBundle` do not insert it, and
                    // `ephemeris_update_system` only writes through it
                    // when it is already present (no auto-insert from
                    // `EphemerisBodyC`). Sources without the component
                    // coast at zero velocity for the relativistic
                    // correction — callers who want PPN to see source
                    // motion must attach `SourceInertialVelocityC`
                    // explicitly.
                    let velocity = v.map(|v| v.0.raw_si()).unwrap_or(DVec3::ZERO);
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
pub fn atmosphere_update_system<P: Planet>(
    atmos_model: Option<Res<AtmosphereModelR>>,
    sim_time: Option<Res<SimulationTimeR>>,
    planet_query: Query<&PlanetFixedRotationC<P>>,
    mut query: Query<(&TranslationalStateC<P>, &mut AtmosphericStateC<P>)>,
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
        **atmos = jeod_sim::evaluate_atmosphere_typed::<P>(
            &model.config,
            state.position,
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
pub fn aero_drag_system<P: Planet>(
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; skip
    // drag so `AerodynamicForceC` doesn't hold stale values that no
    // integrator consumes (the runner's split between `bodies` and
    // `detached_subtrees` only evaluates drag on the integrated set).
    mut query: Query<
        (
            &DragConfigC,
            &AtmosphericStateC<P>,
            &TranslationalStateC<P>,
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
        // The body velocity and atmospheric state both carry the
        // concrete planet `<P>` at the type level (matching the
        // system instantiation's `<P>` parameter, gated by the bodies
        // query filter), so they pass straight into the typed kernel
        // without a relabel.
        let result = jeod_sim::compute_drag_typed::<P, SelfRef>(
            &drag_config.0,
            &atmos.0,
            state.velocity,
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
fn compute_illum_factor<P: Planet>(
    vehicle_pos: DVec3,
    sun_pos: DVec3,
    shadow_bodies: &Query<(&TranslationalStateC<P>, &ShadowBodyC), Without<SunMarker>>,
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
/// Generic over `P: Planet` so the result is correctly typed. The
/// `mu` value read from the configured `gravity_source` entity must
/// physically correspond to planet `P` (RF.11): for an Earth-orbit
/// instantiation `<Earth>` the `gravity_source` should point at the
/// Earth entity, not at Sun/Moon. The system instantiation's `<P>`
/// determines which bodies it processes (only those carrying
/// `OrbitalElementsC<P>`).
///
/// Placed in `JeodSet::DerivedState`.
pub fn orbital_elements_system<P: Planet>(
    mut query: Query<(
        &TranslationalStateC<P>,
        &OrbitalElementsConfigC,
        &mut OrbitalElementsC<P>,
    )>,
    sources: Query<&GravitySourceC>,
) {
    for (state, config, mut elements) in &mut query {
        let Ok(source) = sources.get(config.gravity_source) else {
            elements.0 = Default::default();
            continue;
        };
        // `OrbitalElementsC<P>` and the typed kernel result both pin
        // the planet to `P`. Mint a `GravParam<P>` from the source's
        // f64 mu at the call boundary; the caller is responsible for
        // wiring `gravity_source` to a source whose `mu` matches `P`
        // (RF.11). Misconfigurations (e.g. an Earth-orbit body whose
        // `OrbitalElementsConfigC.gravity_source` points at Sun)
        // produce numerically-wrong elements at *runtime*, not at
        // compile time — Bevy's runtime ECS link cannot enforce the
        // mu↔planet match structurally.
        let mu_p = jeod_sim::GravParam::<P>::from_si(source.mu);
        match jeod_sim::compute_orbital_elements_typed::<P>(mu_p, state.position, state.velocity) {
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
pub fn lvlh_system<P: Planet>(mut query: Query<(&TranslationalStateC<P>, &mut LvlhFrameC)>) {
    for (state, mut lvlh) in &mut query {
        // `TranslationalStateC<P>` already carries `PlanetInertial<P>`,
        // matching the typed kernel's `P` parameter directly — no
        // relabel needed. LVLH stays in planet-inertial throughout
        // (no integ-origin shift).
        lvlh.0 = jeod_sim::compute_body_lvlh_frame_typed::<P>(state.position, state.velocity);
    }
}

/// Compute geodetic state for entities with `GeodeticConfigC`.
///
/// Placed in `JeodSet::DerivedState`.
pub fn geodetic_system<P: Planet>(
    mut query: Query<(
        &TranslationalStateC<P>,
        &GeodeticConfigC,
        &mut GeodeticStateC,
    )>,
    planets: Query<(&PlanetFixedRotationC<P>, &PlanetC)>,
) {
    for (state, config, mut geodetic) in &mut query {
        let Ok((rot, planet)) = planets.get(config.planet) else {
            geodetic.0 = Default::default();
            continue;
        };
        // Position is already typed `Position<PlanetInertial<P>>` —
        // matches the typed kernel's `P` directly, no relabel needed.
        // Geodetic stays in planet-inertial throughout (no integ-origin
        // shift). The ellipsoid-radii lift below is the typed-units
        // boundary on planet shape (a config-time conversion, not a
        // per-step bypass).
        use jeod_sim::F64Ext;
        geodetic.0 = jeod_sim::compute_body_geodetic_typed::<P>(
            state.position,
            rot.0.matrix_ref(),
            planet.r_eq.m(),
            planet.r_pol.m(),
        );
    }
}

/// Compute the typed root-inertial origin offset of `body_frame`'s
/// integration frame — the RF.10 shift that lifts a body's
/// `PlanetInertial<P>` state into absolute `RootInertial`
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

/// Lazy fail-loud variant of [`body_integ_origin_in_root`] for systems
/// that take `Option<Res<RootFrameEntityR>>`: a body with a
/// `FrameEntityC` whose parent is *not* the root needs the
/// integ-origin shift, and the shift cannot be computed without the
/// root entity. Panicking here surfaces the misconfiguration at the
/// exact site where wrong physics would otherwise propagate silently
/// (per the *Fail Loudly* rule in CLAUDE.md): a non-root-integrated
/// body's `TranslationalStateC` is planet-relative, and treating
/// every integ-origin as zero would feed planet-relative coordinates
/// into a kernel that composes in root-inertial — silently producing
/// merged states off by the integration-frame's full translational
/// state (~3.8e8 m / 1 km/s for lunar bodies).
///
/// Pure root-integrated worlds (the common minimal-test shape: no
/// `JeodPlugin`, so no `FrameEntityC` on bodies) keep working — the
/// `body_frame.is_none()` branch returns zero without consulting the
/// root entity. Tests that exercise non-root-integrated bodies must
/// register `JeodPlugin` (which inserts `RootFrameEntityR` and the
/// frame-tree infrastructure) or supply an equivalent mock resource.
fn body_integ_origin_in_root_lazy(
    body_frame: Option<&FrameEntityC>,
    parents: &Query<&ChildOf>,
    root_frame_entity: Option<Entity>,
    frame_origin: &FrameOrigin,
) -> (Position<RootInertial>, Velocity<RootInertial>) {
    // Resolve the body's integ-frame entity (parent of its
    // `FrameEntityC` in the frame-tree). Two legitimate paths return
    // a zero origin without consulting the frame tree:
    //
    //   * `body_frame.is_none()` — the body has no `FrameEntityC` at
    //     all (minimal-test shape with no `JeodPlugin`); root-
    //     integrated by convention.
    //
    // A body that *does* carry `FrameEntityC` but whose frame entity
    // has no `ChildOf` parent is malformed: every frame entity must
    // be parented in the frame tree (under the root frame entity for
    // root-integrated bodies, or under a planet's inertial frame
    // entity for planet-integrated bodies). Treating that corruption
    // as "root-integrated" would silently feed planet-relative coords
    // into a kernel that composes in root-inertial — exactly the
    // failure mode the rest of the staging path rejects loudly.
    let Some(fe) = body_frame else {
        return (
            Position::<RootInertial>::zero(),
            Velocity::<RootInertial>::zero(),
        );
    };
    let integ_e = parents
        .get(fe.0)
        .map(|child_of| child_of.parent())
        .unwrap_or_else(|err| {
            panic!(
                "malformed frame tree: body's FrameEntityC ({:?}) has no ChildOf parent \
                 ({err:?}). Every body frame entity must be parented under either the root \
                 frame entity (root-integrated) or a planet's inertial frame entity \
                 (planet-integrated). Detached or freshly reparented bodies must restore \
                 the ChildOf edge before the next staging/step; treating this as \
                 root-integrated would feed planet-relative coordinates into a \
                 root-inertial kernel and silently corrupt the merged composite by the \
                 missing integ-frame's full root-inertial state. Likely cause: an attach \
                 or detach handler dropped the frame-tree reparent step.",
                fe.0,
            )
        });
    // The body has a registered frame entity. Without the root entity
    // we cannot tell whether `integ_e == root` (root-integrated, safe
    // zero shift) or `integ_e != root` (non-root, load-bearing shift).
    // Demand the resource and panic with a fix-it diagnostic if it is
    // absent — silently returning zero in the latter case would
    // corrupt the merged composite by the integration-frame's full
    // root-inertial state.
    let root = root_frame_entity.unwrap_or_else(|| {
        panic!(
            "RootFrameEntityR resource not present, but a body carries FrameEntityC \
             ({:?}) whose integ-frame parent is {integ_e:?} — the integ-origin shift \
             cannot be computed without the root frame entity. JeodPlugin must be \
             loaded for systems that lift integration-frame coordinates to \
             root-inertial (staging_system, step_detached_system). If your test \
             intentionally omits JeodPlugin, also omit FrameEntityC from the body \
             (root-integrated bodies skip this path entirely).",
            fe.0,
        )
    });
    if integ_e == root {
        (
            Position::<RootInertial>::zero(),
            Velocity::<RootInertial>::zero(),
        )
    } else {
        frame_origin.origin_in_root(root, integ_e)
    }
}

/// Compute solar beta angle for entities with `SolarBetaC`.
///
/// Requires a `SunMarker` entity to exist in the world.
///
/// Generic over `P: Planet` so the body's planet-inertial state and
/// the Sun's `TranslationalStateC<P>` (which by convention stores the
/// Sun position in the body's planet-inertial frame for the
/// single-planet pipeline) match at the type level. Multi-planet
/// instantiation registers a separate Sun-state component per planet.
///
/// Placed in `JeodSet::DerivedState`.
#[allow(clippy::type_complexity)]
pub fn solar_beta_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    mut query: Query<
        (
            &TranslationalStateC<P>,
            Option<&FrameEntityC>,
            &mut SolarBetaC,
        ),
        Without<SunMarker>,
    >,
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
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
        // bodies the body's `<PlanetInertial<P>>` storage is
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
        // root frame, so its `<PlanetInertial<P>>` storage is
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
/// Generic over `P: Planet` so the body's planet-inertial state and the
/// Sun / Moon `TranslationalStateC<P>` (which by convention store the
/// solar-system body positions in the body's planet-inertial frame for
/// the single-planet pipeline) match at the type level.
///
/// Placed in `JeodSet::DerivedState`.
#[allow(clippy::type_complexity)]
pub fn earth_lighting_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    mut query: Query<
        (
            &TranslationalStateC<P>,
            Option<&FrameEntityC>,
            &EarthLightingConfigC,
            &mut EarthLightingStateC,
        ),
        (Without<SunMarker>, Without<MoonMarker>),
    >,
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
    moon_query: Query<&TranslationalStateC<P>, With<MoonMarker>>,
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
        // `<PlanetInertial<P>>` storage is integ-frame-
        // relative; lift it to absolute root-inertial via the integ-
        // origin shift before passing to the typed kernel. Sun and
        // Moon are root-integrated by the SunBundle / MoonBundle
        // construction (their frame entities are children of the
        // root frame), so their positions need no shift — only a
        // boundary relabel from `<PlanetInertial<P>>` to
        // `<RootInertial>` to satisfy the typed entry's frame contract.
        let (integ_origin, _integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let body_pos_rel = Position::<RootInertial>::from_raw_si(state.position.raw_si()); // allowed: integ-origin shift adds origin offset on the next line; relabel matches the runner's `body.trans.to_inertial(&o)` boundary.
        let body_pos = body_pos_rel + integ_origin;
        // Sun / Moon are root-integrated by SunBundle / MoonBundle
        // (their frame entity's parent is the root frame, integ
        // origin = zero); the relabel here is the consumer-boundary
        // step that pins the framing convention at the call site.
        let sun_pos = Position::<RootInertial>::from_raw_si(sun_state.position.raw_si()); // allowed: Sun is root-integrated by SunBundle construction (its frame entity's parent is the root frame, integ origin = zero); relabel is the consumer-boundary step.
        let moon_pos = Position::<RootInertial>::from_raw_si(moon_state.position.raw_si()); // allowed: Moon is root-integrated by MoonBundle construction (its frame entity's parent is the root frame, integ origin = zero); relabel is the consumer-boundary step.
        lighting.0 = jeod_sim::compute_earth_lighting_typed(
            body_pos,
            sun_pos,
            moon_pos,
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
pub fn flat_plate_srp_system<P: Planet>(
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
            &TranslationalStateC<P>,
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
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC<P>, &ShadowBodyC), Without<SunMarker>>,
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
        // integrated bodies the body's `<PlanetInertial<P>>`
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
        // root frame, so its `<PlanetInertial<P>>` storage is
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
        // The CoM is in the vehicle's structural frame; tag the typed
        // wildcard at this boundary so `FlatPlateStageInputs.center_grav`
        // (also typed) accepts it without a raw `DVec3` mismatch. Inner
        // SRP kernels go back through `.raw_si()`.
        let center_grav_raw = mass.map_or(DVec3::ZERO, |m| m.0.center_of_mass.raw_si());
        let center_grav = jeod_sim::Vec3Ext::m_at::<jeod_sim::StructuralFrame<jeod_sim::SelfRef>>(
            center_grav_raw,
        );

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
                    // Drop typed wildcard for the kernel's raw-DVec3 contract.
                    center_grav.raw_si(),
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
                // `sun_state.position` is stored as
                // `<PlanetInertial<P>>`; the SRP derivative
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
pub fn cannonball_srp_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; skip
    // cannonball SRP so `RadiationForceC` doesn't hold stale values
    // that no integrator consumes.
    mut query: Query<
        (
            &CannonballSrpC,
            &TranslationalStateC<P>,
            Option<&FrameEntityC>,
            &mut RadiationForceC,
        ),
        (
            Without<SunMarker>,
            Without<FlatPlateConfigC>,
            Without<crate::DetachedSubtreeStateC>,
        ),
    >,
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC<P>, &ShadowBodyC), Without<SunMarker>>,
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
        // `<PlanetInertial<P>>` storage to absolute root-
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
///    the new combined CoM. When the parent and child resolve to
///    different integration-frame entities (post root-equivalence
///    fold), each body's pre-attach state is lifted to root inertial
///    via its own `IntegOrigin` before the kernel call so the
///    cross-body composition arithmetic operates on a single inertial
///    frame, and the merged composite is lowered back through the
///    parent's integ origin for the writeback,
/// 4. writes the merged state back into the parent entity's
///    [`crate::TranslationalStateC`] / [`crate::RotationalStateC`],
/// 5. for the cross-integration-frame case, reparents the child's
///    body-frame entity (and every kinematic descendant of the child
///    in the mass tree) under the parent's integ-frame entity via
///    `commands.entity(...).insert(ChildOf(...))`, mirroring JEOD's
///    `dyn_body_attach.cc::attach_establish_links` →
///    `dyn_body_integration.cc::set_integ_frame` recursion. JEOD's
///    `set_integ_frame` itself "does not update state"
///    (`dyn_body_integration.cc:85-86`) — JEOD relies on the
///    immediately-following `propagate_state()` call inside
///    `attach_update_properties` to refill descendants' parent-relative
///    storage from the merged root. Our adapter has no equivalent
///    same-call propagation: the next tick's
///    [`propagate_state_from_root_system`](crate::propagate_state_from_root_system)
///    walk runs many systems later, and the
///    `TranslationalStateC`-is-already-integ-frame-relative contract
///    (`register_body_frames_system`) would otherwise leave every
///    descendant's stored numerics inconsistent with the freshly
///    reparented frame-tree topology for the staging → propagate
///    window. Each reparented descendant's `TranslationalStateC` and
///    body-frame `FrameTransC` are therefore shifted in-place by
///    `(old_integ_origin - new_integ_origin)` (root-inertial
///    coordinates) during this same staging tick — same physical
///    pose, just relabeled into the new integration frame's
///    coordinates. `frame_switch_system` does the symmetric
///    pair (reparent + state rewrite) for its own distance-triggered
///    switches; this is the cross-integ-frame attach analogue,
/// 6. removes [`crate::DetachedSubtreeStateC`] from the child entity if
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
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn staging_system<P: Planet>(
    mut commands: Commands,
    tree: Option<ResMut<crate::MassTreeR>>,
    mut attach_events: bevy::ecs::message::MessageReader<crate::AttachEvent>,
    mut detach_events: bevy::ecs::message::MessageReader<crate::DetachEvent>,
    mut bodies: Query<(
        Entity,
        &crate::MassBodyIdC,
        &mut MassPropertiesC,
        Option<&mut TranslationalStateC<P>>,
        Option<&mut RotationalStateC>,
    )>,
    body_frames: Query<&FrameEntityC>,
    parents: Query<&ChildOf>,
    detached_q: Query<Entity, With<crate::DetachedSubtreeStateC>>,
    // Per-body component presence used by the cross-integ-frame fence
    // to tell apart three distinct "FrameEntityC absent / present"
    // populations:
    //
    //   * **Mass-only attach participant** — entity carries
    //     `MassBodyIdC` + `MassPropertiesC` but lacks at least one of
    //     `DynamicsConfigC` / `TranslationalStateC`. Registration will
    //     never visit it, so `FrameEntityC` will never be inserted.
    //     Legitimate `MassBody`-without-`DynBody` configuration; the
    //     fence has no frame node to protect for it.
    //
    //   * **Registration-race** — entity carries both eligibility
    //     components (`DynamicsConfigC` + `TranslationalStateC`, the
    //     filter for `register_body_frames_system`) but lacks
    //     `FrameEntityC`. `register_body_frames_system` has not yet run
    //     this tick (deferred `Commands` flush ordering). Letting the
    //     attach proceed would silently corrupt the frame tree on the
    //     next register pass. Per Fail Loudly this must panic.
    //
    //   * **Partially-stripped dynamic body** — entity carries
    //     `FrameEntityC` (registration ran) but at least one of
    //     `DynamicsConfigC` / `TranslationalStateC` /
    //     `RotationalStateC` has been removed since. Reading state for
    //     the kernel would silently substitute zero/identity, and the
    //     combine-back-write below conditionally writes the merged
    //     composite only if those components are present — so the
    //     merged state would be silently dropped. Per Fail Loudly the
    //     fence must surface this as well; matches JEOD's
    //     `attach_validate_child` rejecting "Child body has an
    //     incomplete state" (`dyn_body_attach.cc:131-135`).
    eligibility: Query<(
        Has<DynamicsConfigC>,
        Has<TranslationalStateC<P>>,
        Has<RotationalStateC>,
    )>,
    // Frame-state query needed by `is_root_equivalent_entity` so the
    // cross-integ-frame fence below treats Earth.inertial-as-root-
    // equivalent topology (a direct child of root with identity state)
    // as semantically root.
    frame_states: Query<(&FrameTransC, &FrameRotC, &FrameAngVelC)>,
    // Registered source frame entities. Used to verify that a body's
    // resolved live integ-frame entity is a *legal* integ-frame entity
    // (root or a registered source frame), matching the same fence
    // `frame_switch_system` enforces. Without this an attach with both
    // bodies misparented under the same arbitrary frame would otherwise
    // be silently accepted as "same integration frame".
    source_frames: Query<&FrameEntityC, With<GravitySourceC>>,
    mut integrators: Query<(
        &crate::MassBodyIdC,
        Option<&mut GaussJacksonStateC>,
        Option<&mut Abm4StateC>,
    )>,
    // `frame_origin` performs the per-body root-inertial lift required by
    // the cross-integration-frame attach path. The merge kernel composes
    // parent and child state through `omega × r` and
    // `T_inertial_struct.transpose()` shifts — both arithmetic-valid only
    // when both bodies' translational state lives in the same inertial
    // frame. The pre-attach states are lifted to root inertial via each
    // body's `IntegOrigin` (mirrors the runner's `mass_tree::attach_inner`
    // RF.10 shift site), the kernel runs in root coordinates, and the
    // merged result is lowered through the parent's `IntegOrigin` for
    // writeback into `TranslationalStateC`'s integration-frame storage.
    // The lift is identically zero only for root-integrated bodies; for
    // any body integrating in a non-root `PlanetInertial<P>` the lift is
    // non-zero. The same-integ-frame case is a no-op in *physical* terms
    // (parent and child share an `IntegOrigin`, so the post-lift kernel
    // arithmetic matches what the pre-lift integ-frame arithmetic would
    // have produced — the per-body shifts cancel in any inter-body
    // difference the kernel forms), but `frame_origin` is still consulted
    // for both bodies in that case rather than short-circuited.
    frame_origin: FrameOrigin,
    root_frame_entity: Option<Res<crate::RootFrameEntityR>>,
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
    //
    // `parent_position`/`parent_velocity` and `child_position`/
    // `child_velocity` are stored in **root-inertial** coordinates —
    // each side has been lifted through its own body's `IntegOrigin`
    // at capture time. The combine kernel composes states across
    // bodies (mass-weighted velocity, inertial-frame CoM shift,
    // ω×r over offsets), which is only arithmetic-valid when both
    // sides live in the same inertial frame. Storing the lifted
    // values keeps that invariant explicit; for root-integrated
    // bodies the lift is `IntegOrigin::zero()` and the captured
    // values are bit-identical to the raw `TranslationalStateC`
    // contents. Mirrors the runner's `attach`/`detach` snapshot
    // shape in `jeod_runner::Simulation`.
    //
    // `parent_integ_origin_pos`/`parent_integ_origin_vel` are the
    // parent's integ-origin in root-inertial, retained so the
    // writeback below can lower the merged result back into
    // integ-frame storage (`TranslationalStateC` is integ-frame).
    struct AttachWork {
        parent_entity: Entity,
        child_entity: Entity,
        parent_id: jeod_sim::MassBodyId,
        // Pre-attach snapshot for the kernel — lifted to root-inertial.
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
        // Parent's integ-origin in root-inertial (the displacement
        // from root to the body's integration frame). Used to lower
        // the merged result back into the parent's integ-frame
        // storage at the `TranslationalStateC` writeback. Identity
        // for root-integrated parents; load-bearing for non-root.
        parent_integ_origin_pos: glam::DVec3,
        parent_integ_origin_vel: glam::DVec3,
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
        // Cross-integration-frame attach metadata. `Some` when the
        // parent and child resolve to different integ-frame entities
        // (post root-equivalence fold). When set, the merge kernel
        // is run in root-inertial coordinates: the pre-attach states
        // are lifted through `parent_integ_origin_pos/vel` and
        // `child_integ_origin_pos/vel` on input, and the merged
        // composite is lowered back through the parent's integ
        // origin on writeback so the parent's `TranslationalStateC`
        // continues to store integration-frame coordinates. This
        // mirrors `jeod_runner::Simulation::attach_inner`'s lift /
        // lower around `combine_states_at_attach`. JEOD source
        // reference: `dyn_body_attach.cc::attach_establish_links` →
        // `dyn_body_integration.cc::set_integ_frame`.
        cross_integ: Option<CrossIntegFrameWork>,
    }

    // Cross-integration-frame attach metadata. Captured before the
    // mass tree is mutated so the integ-origin lifts at the kernel
    // boundary observe the *pre-attach* origins (the post-attach root
    // is the parent's root, so the lower step uses the parent's
    // origin, but the seed-time lifts use the per-body pre-attach
    // origins).
    struct CrossIntegFrameWork {
        // Parent's integ-frame entity position + velocity in
        // root-inertial coordinates. Zero when the parent is integrated
        // in root (the body-frame entity is `ChildOf(root)`); for any
        // body integrating in `PlanetInertial<P>` the shift is the only
        // thing that keeps the per-descendant numerical-rewrite below
        // from silently mixing coordinates across distinct integration
        // frames. RF.10 shift site, mirrors `mass_tree::attach_inner`'s
        // `body_integ_origins`-based shift. The seed-time root-inertial
        // lift consumed by `stage_attach_combine` is performed up-front
        // at `AttachWork` construction (parent_position_integ +
        // parent_integ_origin_pos), so the kernel-input lift uses the
        // `AttachWork` field directly and these fields are needed only
        // for the descendant numerical-rewrite step below.
        parent_integ_origin_pos: glam::DVec3,
        parent_integ_origin_vel: glam::DVec3,
        // The new integ-frame entity for the child + every kinematic
        // descendant of the child in the mass tree. Per JEOD's
        // `dyn_body_integration.cc::set_integ_frame` (lines 64-117)
        // this reparent recurses into `dyn_children` so all
        // descendants follow the child onto the parent's integ frame.
        // The corresponding Bevy `commands.entity(...).insert(ChildOf(...))`
        // calls are issued by the writeback loop after the kernel
        // runs, so the deferred-Commands flush sees a consistent
        // post-merge frame tree.
        new_parent_frame_entity: Entity,
        // Per-entity reparent work: the body-frame entity to reparent
        // under `new_parent_frame_entity`, the owning body entity, and
        // the body's pre-attach integ-frame origin in root-inertial
        // coordinates. Resolved before the kernel call so the reparent
        // (and the matching numerical rewrite of `TranslationalStateC`
        // / `FrameTransC`, see writeback loop below) can be issued as a
        // single batch alongside the merged-state writeback. Includes
        // the child plus every mass-tree descendant of the child that
        // has a registered `FrameEntityC` (mass-only descendants
        // without a frame node are skipped — they have no frame-tree
        // node to reparent and no `TranslationalStateC` consumer
        // inside the staging→propagate window to corrupt).
        reparent_entries: Vec<CrossIntegReparentEntry>,
    }

    // Per-entity payload for the cross-integ-frame reparent loop.
    // Each entry pairs a body-frame entity with its owning body entity
    // plus the pre-attach integ-frame origin in root-inertial
    // coordinates, enough to numerically rewrite the body's
    // `TranslationalStateC` (and the body-frame entity's
    // `FrameTransC`) so the stored coordinates remain consistent with
    // the frame-tree's interpretation after the reparent (per
    // `register_body_frames_system`'s docstring: the body's
    // `TranslationalStateC` is interpreted as already in integ-frame
    // coordinates, where "integ frame" is the body-frame entity's
    // current `ChildOf` parent). Without this rewrite, consumers
    // running between `staging_system` and the next
    // `propagate_state_from_root_system` (the entire `Interaction`
    // set: `aero_drag_system`, `gravity_torque_system`, the SRP
    // systems, plus `force_collection_system`) read the body's
    // pre-attach numerical state through the post-attach frame-tree
    // topology and silently mix coordinates across distinct integ
    // frames.
    struct CrossIntegReparentEntry {
        body_entity: Entity,
        body_frame_entity: Entity,
        old_integ_origin_pos: glam::DVec3,
        old_integ_origin_vel: glam::DVec3,
    }

    let mut attach_work: Vec<AttachWork> = Vec::new();
    // Per-detach work: captured pre-detach composite-body state to be
    // attached to the detached entity as `DetachedSubtreeStateC` once
    // the topology mutation is done.
    let mut detach_work: Vec<(Entity, jeod_sim::DetachedSubtreeState)> = Vec::new();

    // The set of registered source frame entities is invariant across
    // the entire `staging_system` call — collect it once here rather
    // than once per AttachEvent. Mirrors the optimization in
    // `frame_switch_system` (see lines 737-744) so a tick that drains
    // a batch of attaches doesn't pay the rebuild cost N times.
    let known_source_frames: std::collections::HashSet<Entity> =
        source_frames.iter().map(|fe| fe.0).collect();

    // One-shot mass-body-id → entity map built from a single pass over
    // the `bodies` query. The cross-integ-frame attach branch
    // (descendant subtree walk) and the detach handler both need
    // id-keyed entity lookups; without a shared map each event would
    // re-scan `bodies` once per descendant, giving an `O(subtree_size
    // × body_count)` cost per attach and an `O(body_count)` cost per
    // detach inside the per-event loops. Building the map once amortizes
    // both into a single `O(body_count)` scan plus `O(1)` membership
    // tests, matching the runner's `id_to_entity`-style indexed
    // lookups.
    let id_to_entity: std::collections::HashMap<jeod_sim::MassBodyId, Entity> = bodies
        .iter()
        .map(|(e, body_id, _, _, _)| (body_id.0, e))
        .collect();

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
        let (child_position_integ, child_velocity_integ) = child_trans
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
        let (parent_position_integ, parent_velocity_integ) = parent_trans
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

        // Cross-integration-frame attach is not yet supported for
        // bodies that participate in the frame tree. The unsupported
        // piece is the frame-entity reparenting + coordinate rewrite,
        // *not* the multi-step integrator reset (the IG.37 reset for
        // every affected body still runs later in this function via
        // the `affected_ids` walk) and *not* the mass-tree composite
        // recomputation (which is frame-agnostic and runs
        // unconditionally below). JEOD's
        // `dyn_body_attach.cc::attach_establish_links` calls
        // `set_integ_frame(*(dyn_parent->get_integ_frame()))` whenever
        // the child's integ frame differs from the parent's. JEOD's
        // `dyn_body_integration.cc::set_integ_frame` (lines 64-117)
        // reparents the child's `core_body`/`composite_body`/
        // `structure` frames + every registered vehicle point under
        // the new integ frame via low-level
        // `RefFrame::reset_parent()` calls; the in-source comment
        // "NOTE WELL: This uses the low-level reset_parent(). It does
        // not update state." makes explicit that the stored numbers
        // are NOT rewritten. Later propagation reinterprets the
        // existing coordinates against the new parent. Our staging
        // path performs neither the frame-entity `ChildOf` reparent
        // nor the coordinate rewrite, so allowing the merge to
        // proceed silently corrupts every downstream
        // `RelativeFrameState` walk. Per the Fail Loudly rule
        // (CLAUDE.md), surface the misconfiguration at the point of
        // detection rather than producing a wrong trajectory.
        //
        // The live integ-frame for each body is the `ChildOf` parent
        // of its body-frame entity, NOT the body's `IntegSourceC`
        // value. `frame_switch_system` mutates the body-frame
        // entity's `ChildOf` parent on each switch but intentionally
        // leaves `IntegSourceC` (the config-time intent) untouched —
        // comparing `IntegSourceC` would both miss real cross-frame
        // attaches (root-started body that switched to Moon: still
        // `None`) and falsely reject same-frame attaches (a body
        // switched into the parent's frame: stale `IntegSourceC`
        // differs from parent's).
        //
        // The fence has four semantic layers, applied in order so
        // legality is decided on the *original* parent rather than
        // its root-equivalent fold (otherwise an arbitrary entity
        // that happens to be a direct child of root with identity
        // state would silently fold to the root and pass the
        // legality check):
        //
        //   1. **Resolve the live integ-frame entity.** Bodies that
        //      participate in the frame tree carry `FrameEntityC`
        //      (inserted by `register_body_frames_system` for
        //      entities with both `DynamicsConfigC` and
        //      `TranslationalStateC`) and that frame entity must
        //      have a `ChildOf` parent. Mass-only attach
        //      participants (entities carrying only `MassBodyIdC` +
        //      `MassPropertiesC`, matching JEOD's
        //      `MassBody`-without-`DynBody` configuration that
        //      `AttachEvent`'s contract permits) have no
        //      `FrameEntityC` and therefore no frame tree to corrupt
        //      — the fence skips them. If `FrameEntityC` *is*
        //      present but the `ChildOf` is missing the frame tree
        //      itself is corrupt and the attach cannot be safely
        //      processed — panic per Fail Loudly.
        //
        //   1.5. **State-completeness on dynamic participants.** A
        //      body that resolved a frame entity must carry the
        //      full state-component set (`DynamicsConfigC` +
        //      `TranslationalStateC` + `RotationalStateC`). The
        //      kernel snaps any missing input to zero/identity and
        //      the combine-back-write is conditional on the same
        //      components — without them the merged state is
        //      silently dropped. Matches JEOD's
        //      `attach_validate_child` rejecting partial state with
        //      "<role> body has an incomplete state".
        //
        //   2. **Verify the live parent is a legal integ-frame
        //      entity.** Anything that is not the root frame entity
        //      and not a registered gravity source's frame entity
        //      (e.g. both bodies misparented under another body's
        //      frame entity by a buggy mission script) is rejected
        //      here, *before* root-equivalence folding. This matches
        //      `frame_switch_system`'s same-tick check at lines
        //      765-781 so the same misconfig is caught at attach
        //      time rather than only later when a switch evaluates.
        //      Falls back to comparing against `known_source_frames`
        //      when `RootFrameEntityR` is absent (low-level tests
        //      that drive `staging_system` directly without
        //      `JeodPlugin`).
        //
        //   3. **Normalize root-equivalent topology for equality.**
        //      In `jeod_runner` the central body's inertial frame
        //      *is* the root frame. The Bevy adapter instead
        //      registers every gravity source — including the
        //      central body — one level below a generic root, so
        //      `IntegSourceC(Some(earth))` lands the body's frame
        //      entity under `earth.inertial` (a direct child of root
        //      with identity state). Folding root-equivalent parents
        //      onto root for the equality comparison means an
        //      Earth-centered body and a root-integrated body
        //      ("`IntegSourceC(None)`") count as the same integ
        //      frame. Folding *only* drives the equality check —
        //      legality has already been decided in step 2 against
        //      the un-folded parent.
        //
        // The fail-loud structural and state-completeness checks below
        // (steps 1, 1.5, and 2) run unconditionally — they protect
        // invariants that hold without any reference to root-equivalence
        // semantics, so a low-level test (or a partial app build) that
        // drove `staging_system` directly without `JeodPlugin` still
        // sees the same misconfigurations rejected. Only the
        // root-equivalence equality fold (step 3) requires
        // `RootFrameEntityR` and is therefore conditional on its
        // presence; without root the equality fold is skipped, but no
        // production path reaches that branch with `RootFrameEntityR`
        // absent (`JeodPlugin::build` always inserts it).
        //
        // Skipped per-event when *both* bodies lack `FrameEntityC`
        // and neither is dynamic — see step 1's narrowed mass-only
        // carve-out. A missing `FrameEntityC` on a body that *would*
        // qualify for `register_body_frames_system` (carries both
        // `DynamicsConfigC` and `TranslationalStateC`) is a
        // registration race, not a mass-only configuration, and is
        // rejected fail-loud below.

        // Step 1: resolve the original `ChildOf` parent of each
        // body's frame entity (no folding yet). Returns `None`
        // only when the entity is intentionally mass-only — its
        // component set fails `register_body_frames_system`'s
        // eligibility filter, so `FrameEntityC` will never be
        // inserted and the entity has no node in the frame tree.
        //
        // An entity that *passes* the eligibility filter
        // (`DynamicsConfigC` + `TranslationalStateC`) but still
        // lacks `FrameEntityC` is a registration race — the body
        // was spawned mid-tick after `register_body_frames_system`
        // already ran, so its frame-tree node does not yet exist
        // even though the rest of the world expects one. Letting
        // the attach proceed would silently corrupt the frame
        // tree on the next register pass; per Fail Loudly we
        // panic with a diagnostic that names the broken
        // assumption (entity has the eligibility components but
        // ran the staging fence before registration).
        let resolve_original_parent = |body: Entity, role: &str| -> Option<Entity> {
            match body_frames.get(body) {
                Ok(frame_handle) => {
                    let child_of = parents.get(frame_handle.0).unwrap_or_else(|err| {
                        panic!(
                            "AttachEvent.{role} = {body:?}: body-frame entity \
                                 {fe:?} has no ChildOf parent. The body-frame entity \
                                 must be parented under its integration-frame entity \
                                 (root frame entity, or a registered source's frame \
                                 entity). `register_body_frames_system` inserts that \
                                 ChildOf when it runs in the JeodPlugin schedules \
                                 (Startup, PreUpdate, FixedUpdate); a missing parent \
                                 here means the frame tree is corrupt. Underlying \
                                 query error: {err:?}",
                            fe = frame_handle.0,
                        )
                    });
                    Some(child_of.parent())
                }
                Err(_) => {
                    // No `FrameEntityC`. Distinguish the two
                    // populations: mass-only (carve-out) vs
                    // registration race (fail-loud).
                    let (has_dyn_cfg, has_trans, _has_rot) =
                        eligibility.get(body).unwrap_or((false, false, false));
                    if has_dyn_cfg && has_trans {
                        panic!(
                            "AttachEvent.{role} = {body:?}: entity carries \
                             DynamicsConfigC + TranslationalStateC (the \
                             eligibility filter for register_body_frames_system) \
                             but does not yet carry FrameEntityC. This is a \
                             registration race — the body was spawned mid-tick \
                             after register_body_frames_system already ran in \
                             PreUpdate / FixedUpdate (before \
                             JeodSet::EphemerisUpdate), so its frame-tree node \
                             has not been spawned yet by the time staging_system \
                             runs. Spawn the body before the first FixedUpdate \
                             step (e.g. in Startup or PreUpdate ahead of \
                             register_body_frames_system), or defer the \
                             AttachEvent until the next tick so the registration \
                             pass has had a chance to run."
                        );
                    }
                    None
                }
            }
        };

        // Mass-only attach carve-out: both bodies must carry
        // `FrameEntityC` for the fence to apply. If either side
        // has no frame node (legitimate `MassBody`-without-
        // `DynBody` configuration permitted by `AttachEvent`'s
        // contract), the frame tree has no node to corrupt and
        // the equality / legality checks below have nothing to
        // enforce. The mass-tree composite recompute and IG.37
        // integrator reset still run unconditionally outside
        // this branch.
        //
        // One asymmetric case is rejected fail-loud: a dynamic
        // child (with `FrameEntityC`) attaching to a mass-only
        // parent (no `FrameEntityC`). JEOD's
        // `dyn_body_attach.cc::attach_validate_parent` rejects
        // this with "Dynamic attachments can only be made to
        // valid DynBodies" — and our combine-back-write below
        // only writes the merged composite into the parent's
        // `TranslationalStateC` / `RotationalStateC`, which a
        // mass-only parent does not carry. Without this guard
        // the merged state is silently dropped. The dual case
        // (mass-only child on dynamic parent) matches JEOD's
        // legitimate `add_mass_body` path and is allowed.
        let parent_orig = resolve_original_parent(evt.parent, "parent");
        let child_orig = resolve_original_parent(evt.child, "child");
        if parent_orig.is_none() && child_orig.is_some() {
            panic!(
                "AttachEvent: dynamic child {child:?} (carries FrameEntityC) \
                 cannot be attached to mass-only parent {parent:?} (no \
                 FrameEntityC). JEOD's dyn_body_attach.cc::attach_validate_parent \
                 rejects this with \"Dynamic attachments can only be made to \
                 valid DynBodies\" (Modified_data parents need both \
                 DynamicsConfigC and TranslationalStateC). The combine-back-write \
                 in this function only writes the merged composite into the \
                 parent's TranslationalStateC / RotationalStateC, which a \
                 mass-only parent does not carry — the merged state would be \
                 silently lost. Either promote the parent to a dynamic body \
                 (add DynamicsConfigC + TranslationalStateC + RotationalStateC) \
                 before the attach, or attach the parent to its own dynamic \
                 ancestor first so the composite has a free-flying root.",
                child = evt.child,
                parent = evt.parent,
            );
        }

        // Step 1.5: state-completeness for any body that *did*
        // resolve a frame entity. The kernel reads
        // `parent_position` / `parent_velocity` /
        // `parent_quaternion` / `parent_ang_vel_body` (and the
        // child analogs) from `TranslationalStateC` /
        // `RotationalStateC`, falling back to zero / identity when
        // those components are absent — and the combine-back-write
        // below only writes the merged composite back when the
        // same components are present. A body that carries
        // `FrameEntityC` (registration ran) but has had
        // `DynamicsConfigC` / `TranslationalStateC` removed since
        // is therefore in a miscomputing-attach state: missing
        // inputs silently snap to zero, and any merged result is
        // silently dropped. JEOD's `attach_validate_child`
        // (`dyn_body_attach.cc:121-180`) rejects partial state
        // with "Child body has an incomplete state" / "Root body
        // has an incomplete state"; we surface the same
        // misconfiguration here at the staging fence so the bug is
        // caught at the event boundary rather than silently
        // corrupting downstream state.
        //
        // `RotationalStateC` is required only when the attach
        // partner also has it: the bevy adapter supports a 3-DOF
        // configuration (`DynamicsConfigC` + `TranslationalStateC`
        // without `RotationalStateC`) — `register_body_frames_system`'s
        // filter mirrors this — and an attach between two 3-DOF
        // bodies merges translational state only, leaving rotation
        // identity on both sides consistently. The dangerous case
        // is *asymmetric* rotation: one body 6-DOF, the other 3-DOF,
        // where the 3-DOF side's rotation snaps to identity and the
        // merged attitude / angular momentum is silently wrong. We
        // reject the asymmetric case below.
        let parent_has_state =
            parent_orig.map(|_| eligibility.get(evt.parent).unwrap_or((false, false, false)));
        let child_has_state =
            child_orig.map(|_| eligibility.get(evt.child).unwrap_or((false, false, false)));
        let rotational_asymmetry = match (parent_has_state, child_has_state) {
            (Some((_, _, parent_rot)), Some((_, _, child_rot))) => parent_rot != child_rot,
            _ => false,
        };
        for (entity, orig, role) in [
            (evt.parent, parent_orig, "parent"),
            (evt.child, child_orig, "child"),
        ] {
            if orig.is_none() {
                continue;
            }
            let (has_dyn_cfg, has_trans, has_rot) =
                eligibility.get(entity).unwrap_or((false, false, false));
            let mut missing: Vec<&'static str> = Vec::new();
            if !has_dyn_cfg {
                missing.push("DynamicsConfigC");
            }
            if !has_trans {
                missing.push("TranslationalStateC");
            }
            if rotational_asymmetry && !has_rot {
                missing.push("RotationalStateC");
            }
            if !missing.is_empty() {
                let missing = missing.join(", ");
                panic!(
                    "AttachEvent.{role} = {entity:?}: dynamic body carries \
                     FrameEntityC (registration ran) but is missing required \
                     state component(s): {missing}. The stage_attach_combine \
                     kernel reads TranslationalStateC / RotationalStateC for \
                     pre-attach pose + velocity, and the merged composite is \
                     written back only into those same components — without \
                     them the kernel silently substitutes zero / identity for \
                     the missing input and the merged result is silently \
                     dropped. JEOD's dyn_body_attach.cc::attach_validate_child \
                     rejects this same case with \"Child body has an \
                     incomplete state\" / \"Root body has an incomplete state\". \
                     Re-insert the missing component(s) on the entity before \
                     firing the AttachEvent, or remove the body from the \
                     mass-tree before stripping its state. (Note: \
                     RotationalStateC is required only when the attach \
                     partner carries it — pure 3-DOF attach between two \
                     bodies that both lack RotationalStateC is allowed.)"
                );
            }
        }

        // Cross-integration-frame attach metadata, computed below
        // when both bodies resolved frame entities and the post-fold
        // parents differ. Stays `None` for the same-integ-frame case
        // (the common one) so the writeback loop bypasses the lift /
        // lower / reparent code paths bit-identically.
        let mut cross_integ: Option<CrossIntegFrameWork> = None;

        if let (Some(parent_orig), Some(child_orig)) = (parent_orig, child_orig) {
            // Step 2: legality is decided against the *original*
            // parent — never the root-equivalent fold. An arbitrary
            // entity that happens to satisfy root-equivalence (direct
            // child of root with identity state) but is not itself a
            // registered source frame must still be rejected, because
            // `frame_switch_system` will reject the same parent on the
            // very next tick. Match its legality check at lines
            // 765-781.
            //
            // The legality predicate uses `known_source_frames`
            // (built without `RootFrameEntityR`) directly, plus an
            // optional equality with the root entity when the
            // resource is present — so when `RootFrameEntityR` is
            // absent the check still rejects illegal parents (those
            // not under any registered source) instead of silently
            // bypassing.
            let root_e_opt = root_frame_entity.as_ref().map(|r| r.0);
            for (entity, integ_frame, role) in [
                (evt.parent, parent_orig, "parent"),
                (evt.child, child_orig, "child"),
            ] {
                let is_root = root_e_opt == Some(integ_frame);
                let is_legal = is_root || known_source_frames.contains(&integ_frame);
                assert!(
                    is_legal,
                    "AttachEvent.{role} = {entity:?}: live integration-frame \
                     entity {integ_frame:?} (the ChildOf parent of the body's \
                     frame entity) is neither the root frame entity \
                     ({root_e_opt:?}) nor a registered gravity source's frame \
                     entity. The body-frame entity must be parented under one \
                     of those — register the source via PlanetBundle (which \
                     inserts GravitySourceC and FrameEntityC) before spawning \
                     the body, or attach the body under the root frame entity."
                );
            }

            // Step 3: fold root-equivalent topology *only* for the
            // equality comparison below. Both `parent_orig` and
            // `child_orig` are now known to be legal integ frames, so
            // any fold to `root_e` happens on a registered source
            // (typically the central body's `*.inertial` frame).
            //
            // The fold (and the equality check it drives) requires
            // the root entity to be known. When `RootFrameEntityR`
            // is absent (low-level tests / partial app builds), we
            // skip just this final equality — the structural
            // fail-loud checks above have already run unconditionally,
            // and production paths always set the resource via
            // `JeodPlugin::build`.
            if let Some(root_e) = root_e_opt {
                let fold_root_equivalent = |parent: Entity| -> Entity {
                    if crate::validation::is_root_equivalent_entity(
                        parent,
                        root_e,
                        &parents,
                        &frame_states,
                    ) {
                        root_e
                    } else {
                        parent
                    }
                };
                let parent_frame = fold_root_equivalent(parent_orig);
                let child_frame = fold_root_equivalent(child_orig);

                if parent_frame != child_frame {
                    // Cross-integration-frame attach. Mirrors JEOD's
                    // `dyn_body_attach.cc::attach_establish_links`
                    // calling `set_integ_frame(*(dyn_parent->get_integ_frame()))`
                    // when the child's integ frame differs from the
                    // parent's. The merged body will integrate in the
                    // parent's frame post-attach — the runner's
                    // `mass_tree::attach_inner` keeps the same
                    // post-attach invariant (the merged composite is
                    // written back into the integrated tree root).
                    //
                    // Compute each body's pre-attach integ-frame
                    // origin in root-inertial coordinates. The lift
                    // is identically zero for any body whose folded
                    // integ frame is root (`parent_frame == root_e` /
                    // `child_frame == root_e`), so for the asymmetric
                    // case "parent in root + child in
                    // PlanetInertial<P>" only the child carries a
                    // non-zero shift; for the symmetric case "parent
                    // in PlanetInertial<P> + child in PlanetInertial<Q>"
                    // both lifts are non-zero and distinct. Note that
                    // `parent_frame`/`child_frame` are the *folded*
                    // values (root-equivalent topology mapped onto
                    // `root_e`); the unfolded `parent_orig` /
                    // `child_orig` may name a registered source frame
                    // that is itself root-equivalent.
                    let resolve_integ_origin = |frame: Entity| -> (glam::DVec3, glam::DVec3) {
                        if frame == root_e {
                            (glam::DVec3::ZERO, glam::DVec3::ZERO)
                        } else {
                            let (p, v) = frame_origin.origin_in_root(root_e, frame);
                            (p.raw_si(), v.raw_si())
                        }
                    };
                    let (parent_integ_origin_pos, parent_integ_origin_vel) =
                        resolve_integ_origin(parent_frame);

                    // Walk the child's mass-tree subtree (inclusive)
                    // and resolve a body-frame entity for each
                    // descendant. Mirrors JEOD's `set_integ_frame`
                    // recursing into `dyn_children`. Mass-only
                    // descendants (no `FrameEntityC`) are skipped —
                    // they have no frame-tree node to reparent.
                    //
                    // The walk uses the *pre-attach* mass tree: the
                    // child has not been linked to the parent yet, so
                    // walking from `child_id` collects the original
                    // child subtree (i.e. every body that was a
                    // descendant of the child before the attach
                    // mutation).
                    //
                    // Per descendant we also capture the body's old
                    // integ-frame origin in root-inertial coordinates
                    // (the body-frame entity's current `ChildOf`
                    // parent's frame state). The numerical rewrite at
                    // the reparent site uses this to shift the body's
                    // stored `TranslationalStateC` from old-frame to
                    // new-frame coordinates without arithmetically
                    // mixing the two; without it, consumers in the
                    // staging → propagate window read pre-attach
                    // numerics through post-attach topology.
                    let mut reparent_entries: Vec<CrossIntegReparentEntry> = Vec::new();
                    let mut subtree: Vec<jeod_sim::MassBodyId> = vec![child_id];
                    let mut idx = 0;
                    while idx < subtree.len() {
                        let id = subtree[idx];
                        for child in tree.children(id) {
                            subtree.push(*child);
                        }
                        idx += 1;
                    }
                    for id in subtree {
                        // O(1) id → entity lookup via the shared map
                        // built once at the top of `staging_system`,
                        // mirroring the runner's `id_to_entity`. A
                        // linear `bodies.iter()` scan per id would be
                        // O(subtree_size × body_count) per attach.
                        if let Some(&entity) = id_to_entity.get(&id) {
                            // A descendant can legitimately lack a
                            // body-frame entity *only* when it is a
                            // pure mass-only node — i.e. it has no
                            // `DynamicsConfigC` / `TranslationalStateC`
                            // and therefore no integ-frame
                            // interpretation to keep in sync with the
                            // post-attach topology. A descendant that
                            // *does* carry those components but is
                            // missing `FrameEntityC` is the same
                            // registration-race state we already
                            // fail-loud on for the attach participants
                            // (lines above, mirroring
                            // `attach_validate_child`). Skipping it
                            // would silently leave that body's stored
                            // `TranslationalStateC` interpreted under
                            // the pre-attach integ frame for every
                            // staging→propagate consumer in the same
                            // tick, so surface the misconfiguration
                            // here rather than letting the stale
                            // numerics propagate.
                            let fe = body_frames.get(entity).ok();
                            let body_frame_entity = match fe {
                                Some(fe) => fe.0,
                                None => {
                                    let (has_dyn_cfg, has_trans, _has_rot) =
                                        eligibility.get(entity).unwrap_or((false, false, false));
                                    assert!(
                                        !(has_dyn_cfg && has_trans),
                                        "staging_system: cross-integ-frame attach: \
                                         descendant body {entity:?} carries DynamicsConfigC \
                                         and TranslationalStateC (dynamic body) but has no \
                                         FrameEntityC — registration race vs \
                                         register_body_frames_system. The cross-integ-frame \
                                         reparent would otherwise leave this descendant's \
                                         stored TranslationalStateC interpreted under the \
                                         pre-attach integ frame while every other body in \
                                         the subtree gets shifted into the new frame, \
                                         silently mixing coordinates across distinct \
                                         integration frames. Spawn the body with a \
                                         registered IntegSourceC (or under the root frame) \
                                         and ensure register_body_frames_system has run \
                                         before firing the AttachEvent."
                                    );
                                    continue;
                                }
                            };
                            // Resolve this descendant's pre-attach
                            // integ-frame origin from its
                            // body-frame entity's current `ChildOf`
                            // parent (the live integ-frame source
                            // of truth, same fold rule used above
                            // for the child/parent equality check).
                            let descendant_integ_frame = parents
                                .get(body_frame_entity)
                                .unwrap_or_else(|err| {
                                    panic!(
                                        "staging_system: cross-integ-frame attach: \
                                         descendant body {entity:?} body-frame entity \
                                         {body_frame_entity:?} has no ChildOf parent \
                                         ({err:?}). Every body-frame entity must be \
                                         parented under its integration frame entity \
                                         (set by register_body_frames_system)."
                                    )
                                })
                                .parent();
                            let descendant_integ_frame_folded =
                                if crate::validation::is_root_equivalent_entity(
                                    descendant_integ_frame,
                                    root_e,
                                    &parents,
                                    &frame_states,
                                ) {
                                    root_e
                                } else {
                                    descendant_integ_frame
                                };
                            let (old_pos, old_vel) = if descendant_integ_frame_folded == root_e {
                                (glam::DVec3::ZERO, glam::DVec3::ZERO)
                            } else {
                                let (p, v) =
                                    frame_origin.origin_in_root(root_e, descendant_integ_frame);
                                (p.raw_si(), v.raw_si())
                            };
                            reparent_entries.push(CrossIntegReparentEntry {
                                body_entity: entity,
                                body_frame_entity,
                                old_integ_origin_pos: old_pos,
                                old_integ_origin_vel: old_vel,
                            });
                        }
                    }

                    // Resolve the new parent frame entity: in the
                    // root-equivalent case (parent's integ-frame
                    // entity folded onto `root_e` for the equality
                    // check above), the actual reparent target must
                    // be the *unfolded* parent frame entity — the
                    // child's body-frame entity becomes
                    // `ChildOf(parent_orig)`, mirroring exactly what
                    // `register_body_frames_system` would have done
                    // for a body spawned with the parent's
                    // `IntegSourceC`. Reparenting onto the folded
                    // root entity directly would bypass the central
                    // body's frame entity and silently break any
                    // consumer that walks the body's `ChildOf` chain
                    // expecting a registered source.
                    let new_parent_frame_entity = parent_orig;

                    cross_integ = Some(CrossIntegFrameWork {
                        parent_integ_origin_pos,
                        parent_integ_origin_vel,
                        new_parent_frame_entity,
                        reparent_entries,
                    });
                }
            }
        }

        // Lift each body's translational state from its own
        // integration frame to root-inertial before feeding the
        // combine kernel. `TranslationalStateC` is stored in the
        // body's `IntegrationFrame` (planet-relative for a non-root
        // integ source), but `combine_states_at_attach` does
        // cross-body composition (mass-weighted velocity, inertial
        // CoM shift, ω×r) which is only arithmetic-valid when both
        // sides live in the same inertial frame. Add each body's
        // `IntegOrigin` (its integ-frame origin in root-inertial) to
        // get root-inertial coordinates. For root-integrated bodies
        // the origin is identically zero so the lift is a numerical
        // no-op; for two bodies that integrate in distinct
        // `PlanetInertial<P>` frames (or one in root + one in a
        // planet) the lift is the only thing that prevents the
        // kernel from silently mixing coordinates across distinct
        // origins. Mirrors the runner's seed-time lift in
        // `jeod_runner::Simulation::attach`.
        //
        // JEOD_INV: RF.10 — `body.trans` is typed
        // `TranslationalStateTyped<IntegrationFrame>`; the only
        // safe transition to `RootInertial` is the integ-origin
        // shift, and the combine kernel is a root-inertial-shift
        // consumer.
        let parent_body_frame_capture = body_frames.get(evt.parent).ok();
        let (parent_integ_origin_pos_typed, parent_integ_origin_vel_typed) =
            body_integ_origin_in_root_lazy(
                parent_body_frame_capture,
                &parents,
                root_frame_entity.as_deref().map(|r| r.0),
                &frame_origin,
            );
        let parent_integ_origin_pos = parent_integ_origin_pos_typed.raw_si();
        let parent_integ_origin_vel = parent_integ_origin_vel_typed.raw_si();
        let child_body_frame_capture = body_frames.get(evt.child).ok();
        let (child_integ_origin_pos_typed, child_integ_origin_vel_typed) =
            body_integ_origin_in_root_lazy(
                child_body_frame_capture,
                &parents,
                root_frame_entity.as_deref().map(|r| r.0),
                &frame_origin,
            );
        let child_integ_origin_pos = child_integ_origin_pos_typed.raw_si();
        let child_integ_origin_vel = child_integ_origin_vel_typed.raw_si();
        let parent_position = parent_position_integ + parent_integ_origin_pos;
        let parent_velocity = parent_velocity_integ + parent_integ_origin_vel;
        let child_position = child_position_integ + child_integ_origin_pos;
        let child_velocity = child_velocity_integ + child_integ_origin_vel;

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
            parent_integ_origin_pos,
            parent_integ_origin_vel,
            child_was_detached,
            parent_was_detached,
            cross_integ,
        });

        // `tree.attach` takes raw structural-frame DVec3; drop the
        // typed phantom at this kernel boundary. The typed
        // `AttachEvent.offset` field guards the structural-frame
        // contract at the writer site.
        tree.attach(child_id, parent_id, evt.offset.raw_si(), evt.t_parent_child);
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

    // The mass-body-id → entity map built at the top of this system
    // (above the attach loop) is reused here. Mirrors the runner's
    // `detach_subtree` which indexes `self.bodies` by id directly.

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
        // same tick, after `composite_mass_system`), reading the
        // entity's `MassPropertiesC` component would yield the
        // just-reverted *core* mass instead of the live post-attach
        // composite — and the CoM-shift formula below would key off
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

        // Lift the parent's `TranslationalStateC` from its integration
        // frame to root-inertial before walking the rigid-body offset
        // chain. The storage convention pins `TranslationalStateC` to
        // the body's integration frame, so for a parent integrated in a
        // non-root `PlanetInertial<P>` the raw position/velocity are
        // planet-relative; running `propagate_forward` on planet-relative
        // coords would seed the walk in integ-frame and produce a
        // subtree state that lives in the same integ-frame, while the
        // captured `DetachedSubtreeState` is typed `Position/Velocity<
        // RootInertial>` and propagated as such by `step_ballistic`.
        // The runner mirrors this exact lift in
        // `crates/jeod_runner/src/simulation/mass_tree.rs:583-585`
        // (root-pre-state) and feeds the inertial seed to the same
        // `derive_subtree_composite_state` walk. Identity for root-
        // integrated parents (`integ_origin == zero`); load-bearing for
        // non-root.
        // JEOD_INV: RF.10 — root-inertial-shift consumer: the kernel
        // walks rigid-body composition in root-inertial coordinates.
        let parent_body_frame = body_frames.get(tree_root_entity).ok();
        let (parent_integ_origin_pos, parent_integ_origin_vel) = body_integ_origin_in_root_lazy(
            parent_body_frame,
            &parents,
            root_frame_entity.as_deref().map(|r| r.0),
            &frame_origin,
        );
        let parent_pre_position_inertial = parent_pre_position + parent_integ_origin_pos.raw_si();
        let parent_pre_velocity_inertial = parent_pre_velocity + parent_integ_origin_vel.raw_si();
        let parent_composite_state = jeod_sim::RefFrameState {
            trans: jeod_sim::RefFrameTrans {
                position: parent_pre_position_inertial,
                velocity: parent_pre_velocity_inertial,
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
    // it via the MassChildOf / mass-tree composition.
    // JEOD_INV: DB.14 — integration-frame switch on attach: when the
    // child's pre-attach integ frame differs from the parent's, the
    // child's body-frame entity (and every kinematic descendant) is
    // reparented under the parent's integ-frame entity here, mirroring
    // JEOD's `dyn_body_attach.cc::attach_establish_links` calling
    // `set_integ_frame(*(dyn_parent->get_integ_frame()))` and the
    // recursive `dyn_body_integration.cc::set_integ_frame` walk over
    // `core_body`/`composite_body`/`structure` + `dyn_children`. The
    // integrator-state reset (JEOD's `reset_integrators()`) is handled
    // independently below via the `affected_ids` IG.37 walk.
    // JEOD_INV: DB.21 — only unattached bodies integrate: after attach the
    // detached-subtree-state is removed from the child so it stops drifting
    // ballistically; the integrated body's state is the merged composite.
    // JEOD_INV: RF.10 — cross-integration-frame attach is a kernel-input
    // shift site. The combine kernel does cross-body composition
    // (`omega × r`, `T_inertial_struct.transpose()`) which is only
    // arithmetic-valid when both bodies' state lives in the same
    // inertial frame. Lift each body to root inertial via its
    // pre-attach integ origin on input, then lower the merged
    // composite back through the parent's integ origin on writeback so
    // the parent's `TranslationalStateC` continues to hold
    // integration-frame coordinates. Mirrors
    // `jeod_runner::Simulation::attach_inner`. For the same-integ-frame
    // case both lifts are identically zero so the kernel call and
    // writeback collapse to the previous bit-identical behaviour.
    for work in &attach_work {
        let combined_mass = tree.get(work.parent_id).composite_properties;

        // The kernel runs in root-inertial coordinates: `work.parent_position`
        // and `work.child_position` were already lifted from each body's
        // pre-attach integration frame through its own `IntegOrigin` at the
        // construction site above (`parent_position_integ +
        // parent_integ_origin_pos`). For root-integrated bodies the lift is
        // identically zero and the kernel input collapses bit-identically to
        // the integ-frame value. Mirrors `jeod_runner::Simulation::attach`'s
        // seed-time lift through `body_integ_origins`.
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

        // Lower the merged composite through the parent's integ
        // origin so the writeback into `TranslationalStateC` lands in
        // the parent's integration-frame coordinates. The parent's
        // integ frame is the new integ frame for the merged body — in
        // the runner this corresponds to writing the merged composite
        // back into the integrated tree root's `body.trans` (the
        // parent IS the tree root post-attach). For the
        // same-integ-frame case `parent_integ_origin_pos/vel` are
        // zero and the subtraction is bit-identically a no-op.
        let merged_position = merged.position - work.parent_integ_origin_pos;
        let merged_velocity = merged.velocity - work.parent_integ_origin_vel;

        if let Ok((_, _, _, mut trans, mut rot)) = bodies.get_mut(work.parent_entity) {
            if let Some(ref mut t) = trans {
                // Kernel returned the merged composite in root-inertial
                // (the captured snapshots were lifted before the call).
                // `TranslationalStateC` is integ-frame storage, so
                // lower back through the parent's `IntegOrigin` —
                // identity for root-integrated parents, load-bearing
                // for non-root. Symmetric partner of the seed-time
                // lift above; mirrors the runner's writeback in
                // `jeod_runner::Simulation::attach`.
                //
                // JEOD_INV: RF.10 — `body.trans` is typed
                // `TranslationalStateTyped<IntegrationFrame>`; the only
                // safe transition from `RootInertial` is the
                // integ-origin shift.
                t.0 =
                    // allowed: stage_attach_combine kernel boundary; the
                    // kernel returns untyped DVec3 by design, so re-wrapping
                    // as TranslationalStateTyped<PlanetInertial<P>> is the
                    // same typed↔untyped pattern as the
                    // From<TranslationalState> impl on TranslationalStateC.
                    jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<P>>::from_untyped_unchecked(
                        &jeod_sim::TranslationalState {
                            position: merged_position,
                            velocity: merged_velocity,
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

        // Reparent the child's body-frame entity (and every kinematic
        // descendant of the child in the mass tree) under the
        // parent's integ-frame entity, AND numerically rewrite the
        // body's stored translational state into the new
        // integ-frame's coordinates so the staged values stay
        // consistent with the frame-tree's post-reparent
        // interpretation. Mirrors JEOD's
        // `dyn_body_integration.cc::set_integ_frame` recursion over
        // `core_body`/`composite_body`/`structure` + `dyn_children`.
        //
        // Why both steps run together: per
        // `register_body_frames_system`'s docstring the body's
        // `TranslationalStateC` is interpreted as already in
        // integ-frame coordinates, where "integ frame" is the
        // body-frame entity's current `ChildOf` parent. After the
        // reparent the body-frame entity's parent has changed, so
        // the stored numerical value must be shifted by
        // `(old_integ_origin - new_integ_origin)` (root-inertial
        // coordinates) to keep this contract. Skipping the rewrite
        // would let consumers running between this system and the
        // next `propagate_state_from_root_system` (the entire
        // `JeodSet::Interaction` set — `aero_drag_system`,
        // `gravity_torque_system`, the SRP systems — plus
        // `force_collection_system` at the top of
        // `JeodSet::ForceCollection`) read pre-attach numerics
        // through post-attach topology and silently mix coordinates
        // across distinct integ frames. `frame_switch_system` does
        // the symmetric pair (reparent + state rewrite) for its own
        // distance-triggered frame transitions; this is the
        // cross-integ-frame attach analogue.
        //
        // The reparent itself is issued through deferred Commands so
        // the post-merge frame tree is consistent on the next system
        // flush. The matching `FrameTransC` rewrite goes through
        // `Commands::insert` in the same call so both land on the
        // same flush boundary. The body's `TranslationalStateC`
        // rewrite goes through the existing `&mut TranslationalStateC`
        // borrow on the `bodies` query — taking it from a fresh
        // `bodies.get_mut(...)` lookup keyed on the descendant's
        // body entity. After this same-tick rewrite,
        // `propagate_state_from_root_system` (later this tick) will
        // re-derive every kinematic child's `TranslationalStateC` /
        // `RotationalStateC` from the parent's freshly-merged
        // composite-body state composed through the `MassChildOf`
        // link, overwriting the staged value. The intermediate
        // rewrite is what keeps the staging → propagate window
        // arithmetic-correct.
        //
        // Rotational state is intentionally NOT rewritten here:
        // every legitimate integ-frame entity is non-rotating (root
        // inertial or `PlanetInertial<P>` — both are inertial and
        // co-aligned with root inertial axes by the frame-tree's
        // construction), so the body's attitude expressed
        // `parent → body` is identical in the old and new integ
        // frames. A rotating integ frame would require an
        // attitude/`ang_vel` rewrite analogous to the position /
        // velocity rewrite below; that case is structurally rejected
        // upstream by the cross-integ-frame fence (every legal
        // integ-frame entity is the root or a registered gravity
        // source, none of which are rotating).
        // JEOD_INV: DB.14, JEOD_INV: RF.10, JEOD_INV: RF.11 — child
        // frame-tree reparent following the integ-frame switch +
        // matching numerical state rewrite.
        if let Some(ci) = work.cross_integ.as_ref() {
            for entry in &ci.reparent_entries {
                let shift_pos = entry.old_integ_origin_pos - ci.parent_integ_origin_pos;
                let shift_vel = entry.old_integ_origin_vel - ci.parent_integ_origin_vel;

                // Reparent the body-frame entity and rewrite its
                // `FrameTransC` into the new parent frame's
                // coordinates in the same Commands batch so the
                // post-flush frame tree is internally consistent
                // (the stored `position`/`velocity` are
                // `parent-frame-relative` per `FrameTransC`'s
                // docstring; switching the parent without rewriting
                // the stored value would produce a discontinuity
                // exactly equal to `(old_origin - new_origin)` on
                // any frame-tree walk that goes through this entity).
                //
                // The `FrameTransC` write here is load-bearing for
                // the staging → integration window: `staging_system`
                // is ordered `.after(JeodSet::Environment).before(
                // JeodSet::Interaction)`, so within the attach tick
                // every consumer that reads frame state via
                // `RelativeFrameState` *after* staging — the
                // `JeodSet::Interaction` set (drag, SRP,
                // gravity-torque), `force_collection_system` /
                // `wrench_aggregation_system` in
                // `JeodSet::ForceCollection`, and `integration_system`
                // in `JeodSet::Integration` — sees this value.
                // `JeodSet::Environment` already ran for this tick
                // and operated on pre-attach `FrameTransC`; the
                // attach physics applies starting at the next
                // Environment pass (tick N+1). After integration,
                // `sync_body_to_frame_system` overwrites
                // `FrameTransC` from the freshly-updated
                // `TranslationalStateC`, so the late-tick value is
                // re-derived. Both writes carry the same physical
                // pose (the staging-time value comes from the
                // already-rewritten pre-integration state; the
                // post-integration value comes from the integrated
                // state), so the apparent "double write" produces a
                // single consistent trajectory. The Commands /
                // immediate-mutation split is dictated by
                // `frame_states` / `FrameOrigin` already holding a
                // shared read borrow on `FrameTransC`; making the
                // write immediate would require a `ParamSet` split
                // that doesn't pay back its complexity. Bevy 0.18's
                // `auto_insert_apply_deferred` (default-on) flushes
                // this `Commands` batch at the
                // `staging_system → JeodSet::Interaction` set
                // boundary, given `staging_system.before(
                // JeodSet::Interaction)`, so the deferred write is
                // observed by every post-staging consumer above
                // without a manual `ApplyDeferred`.
                let new_frame_trans_pos = frame_states.get(entry.body_frame_entity).map_or_else(
                    |_| {
                        // Defensive default: a body-frame entity
                        // without `FrameTransC` cannot exist in
                        // production (`register_body_frames_system`
                        // always inserts the triplet), but the
                        // query type signature still returns a
                        // `Result`. Falling back to identity here
                        // would corrupt the post-reparent state
                        // for any consumer that finds the entity;
                        // surface the misconfiguration loudly
                        // instead. Mirrors `sync_body_to_frame_system`'s
                        // unwrap_or_else panic for the same
                        // FrameTransC invariant.
                        panic!(
                            "staging_system: cross-integ-frame attach: body-frame \
                                 entity {fe:?} has no FrameTransC. Every body-frame \
                                 entity must be alive with FrameTransC attached \
                                 (spawned by register_body_frames_system).",
                            fe = entry.body_frame_entity,
                        )
                    },
                    |(t, _, _)| FrameTransC {
                        position: t.position + shift_pos,
                        velocity: t.velocity + shift_vel,
                    },
                );
                commands
                    .entity(entry.body_frame_entity)
                    .insert(ChildOf(ci.new_parent_frame_entity))
                    .insert(new_frame_trans_pos);

                // Rewrite the body's `TranslationalStateC` so the
                // typed integ-frame storage holds the new-frame
                // coordinates. Skips the parent's body entity — the
                // parent's `TranslationalStateC` was already written
                // above with the merged composite in
                // `parent_integ_origin`-relative coordinates, and
                // adding the shift again here would double-count it.
                // (The parent itself is never in `reparent_entries`
                // — that list is the *child's* subtree, the parent's
                // body-frame entity is the reparent *target*, not a
                // payload.)
                if let Ok((_, _, _, Some(mut t), _)) = bodies.get_mut(entry.body_entity) {
                    let old = t.0.to_untyped();
                    t.0 =
                        // allowed: cross-integ-frame numerical rewrite
                        // boundary; same typed↔untyped re-wrap pattern
                        // as the merged-composite writeback above.
                        // The shift is a pure translation between two
                        // inertial integ frames (planet-inertial
                        // origins differ but axes are co-aligned), so
                        // the post-shift value is still in
                        // integration-frame coordinates with the
                        // `<PlanetInertial<P>>` tag — bit-identical
                        // phantom relabel to the original storage type.
                        jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<P>>::from_untyped_unchecked(
                            &jeod_sim::TranslationalState {
                                position: old.position + shift_pos,
                                velocity: old.velocity + shift_vel,
                            },
                        );
                }
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
            // `merged` is already in root-inertial — both parent and
            // child snapshots were lifted through their own
            // `IntegOrigin` before feeding the kernel, so the kernel
            // produced the merged composite in root-inertial too.
            // `DetachedSubtreeState.composite_*` is typed
            // `Position/Velocity<RootInertial>` by witness, so this
            // is a direct relabel with no further shift. The runner
            // mirrors this contract by tracking `composite_state` in
            // root-inertial inside its detached-subtree map.
            //
            // JEOD_INV: DB.21 — detached subtrees keep advancing
            // ballistically post-attach; the merged composite simply
            // becomes the new "free-flying root" state.
            // JEOD_INV: RF.10 — root-inertial-shift consumer: the
            // typed `DetachedSubtreeState.composite_*` is
            // `Position/Velocity<RootInertial>`.
            use jeod_sim::Vec3Ext as _;
            let updated = jeod_sim::DetachedSubtreeState {
                composite_position: merged.position.m_at::<jeod_sim::RootInertial>(),
                composite_velocity: merged.velocity.m_per_s_at::<jeod_sim::RootInertial>(),
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
                // `PlanetInertial<P>` is the same convention as the
                // pre-detach value.
                jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<P>>::from_untyped_unchecked(
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
            //
            // `new_position` / `new_velocity` are computed by adding a
            // CoM-delta to `shift.parent_pre_position/velocity`, which
            // were captured directly from the parent's
            // `TranslationalStateC` (integration-frame coords). The
            // delta itself is frame-invariant (kinematic offset of a
            // CoM within rigid-body inertial space). To stamp the
            // `RootInertial` phantom for the typed
            // `DetachedSubtreeState`, lift through the parent's
            // `IntegOrigin`. Identity for root-integrated parents
            // (origin = zero); load-bearing for non-root.
            //
            // JEOD_INV: RF.10 — root-inertial-shift consumer: the
            // typed `DetachedSubtreeState.composite_*` is
            // `Position/Velocity<RootInertial>`.
            let parent_body_frame_shift = body_frames.get(shift.tree_root_entity).ok();
            let (parent_integ_origin_pos_shift, parent_integ_origin_vel_shift) =
                body_integ_origin_in_root_lazy(
                    parent_body_frame_shift,
                    &parents,
                    root_frame_entity.as_deref().map(|r| r.0),
                    &frame_origin,
                );
            use jeod_sim::Vec3Ext as _;
            let updated = jeod_sim::DetachedSubtreeState {
                composite_position: (new_position + parent_integ_origin_pos_shift.raw_si())
                    .m_at::<jeod_sim::RootInertial>(),
                composite_velocity: (new_velocity + parent_integ_origin_vel_shift.raw_si())
                    .m_per_s_at::<jeod_sim::RootInertial>(),
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
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn step_detached_system<P: Planet>(
    time: Res<Time<Fixed>>,
    sim_time: Res<SimulationTimeR>,
    mut detached: Query<(
        Entity,
        &mut crate::DetachedSubtreeStateC,
        Option<&mut TranslationalStateC<P>>,
        Option<&mut RotationalStateC>,
    )>,
    body_frames: Query<&FrameEntityC>,
    parents: Query<&ChildOf>,
    frame_origin: FrameOrigin,
    root_frame_entity: Option<Res<crate::RootFrameEntityR>>,
) {
    let dt = time.delta().as_secs_f64();
    if dt == 0.0 {
        return;
    }
    let integ_dt = dt * sim_time.0.time_scale_factor;
    for (entity, mut state, trans, rot) in &mut detached {
        state.0.step_ballistic(integ_dt);
        if let Some(mut t) = trans {
            // Lower the typed `Position/Velocity<RootInertial>` back
            // through the body's `IntegOrigin` to match
            // `TranslationalStateC`'s integration-frame storage
            // convention. For a root-integrated body the origin is
            // zero and the subtraction is bit-identical to a no-op;
            // for a body integrated in `PlanetInertial<P>` (set up
            // at config time via `IntegSourceC`) it is the only
            // thing that prevents stamping a root-inertial coord into
            // an integration-frame slot. Symmetric partner of the
            // staging-system lift; mirrors the runner's writeback in
            // `crates/jeod_runner/src/simulation/mass_tree.rs:681-688`.
            // JEOD_INV: RF.10 — root-inertial-shift consumer:
            // step-time writeback lowers from root-inertial to integ
            // frame.
            let body_frame = body_frames.get(entity).ok();
            let (integ_origin_pos, integ_origin_vel) = body_integ_origin_in_root_lazy(
                body_frame,
                &parents,
                root_frame_entity.as_deref().map(|r| r.0),
                &frame_origin,
            );
            let position = state.0.composite_position.raw_si() - integ_origin_pos.raw_si();
            let velocity = state.0.composite_velocity.raw_si() - integ_origin_vel.raw_si();
            t.0 =
                // allowed: DetachedSubtreeState kernel boundary; the
                // ballistic-step result is returned as raw DVec3 fields by
                // design — re-wrapping into TranslationalStateTyped is the
                // same typed↔untyped pattern as the
                // From<TranslationalState> impl on TranslationalStateC.
                jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<P>>::from_untyped_unchecked(
                    &jeod_sim::TranslationalState {
                        position,
                        velocity,
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
