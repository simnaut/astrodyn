//! Glue: frame-tree source / body / pfix registration and per-step
//! source ↔ body ↔ frame-entity synchronization.
//!
//! These systems run outside the seven [`JeodSet`](crate::JeodSet)
//! pipeline stages — they wire the frame-tree ECS hierarchy that the
//! pipeline systems read. Spawned in `Startup` / `PreUpdate` /
//! `FixedUpdate` (before [`JeodSet::EphemerisUpdate`](crate::JeodSet::EphemerisUpdate))
//! by [`crate::JeodPlugin`] so late-spawned entities are picked up
//! before any pipeline consumer reads their state.

use bevy::prelude::*;
use jeod_sim::Planet;

#[allow(unused_imports)] // intra-doc-link resolution
use super::super::integration::{frame_switch_system, integration_system};
use crate::components::*;

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
