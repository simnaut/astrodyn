//! Glue: frame-tree source / body / pfix registration and per-step
//! source ↔ body ↔ frame-entity synchronization.
//!
//! These systems run outside the seven [`AstrodynSet`](crate::AstrodynSet)
//! pipeline stages — they wire the frame-tree ECS hierarchy that the
//! pipeline systems read. Spawned in `Startup` / `PreUpdate` /
//! `FixedUpdate` (before [`AstrodynSet::EphemerisUpdate`](crate::AstrodynSet::EphemerisUpdate))
//! by [`crate::AstrodynPlugin`] so late-spawned entities are picked up
//! before any pipeline consumer reads their state.

use astrodyn::Planet;
use bevy::prelude::*;

#[allow(unused_imports)] // intra-doc-link resolution
use super::super::integration::{frame_switch_system, integration_system};
use crate::components::*;

/// Identity index for the ECS frame store (issue #664): `FrameUid` →
/// frame [`Entity`], the analogue of the arena's private `uid_index`.
/// Maintained by [`index_frame_uids_system`] /
/// [`deindex_frame_uids_system`]; the indexer rejects duplicate
/// identities loudly, mirroring the arena's `register_uid`, and the map
/// is the "addressable by stable identity" lookup the frame-document
/// apply path resolves through.
///
/// The map is **read-only outside this module** ([`Self::get`] /
/// [`Self::iter`]), carrying the arena's discipline across the
/// boundary: `uid_index` is private there because direct mutation
/// desyncs identity lookup from the store without ever tripping the
/// duplicate-identity rejection (RF.14). Mutation happens only through
/// the index/deindex systems, which observe the components themselves.
#[derive(Resource, Debug, Default)]
pub struct FrameUidIndexR(pub(crate) std::collections::HashMap<astrodyn::FrameUid, Entity>);

impl FrameUidIndexR {
    /// Resolve a frame identity to its frame entity, or `None` if no
    /// live frame entity carries it (never minted, or retired).
    pub fn get(&self, uid: &astrodyn::FrameUid) -> Option<Entity> {
        self.0.get(uid).copied()
    }

    /// Iterate the indexed `(identity, frame entity)` pairs, in
    /// unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (&astrodyn::FrameUid, Entity)> {
        self.0.iter().map(|(uid, &e)| (uid, e))
    }

    /// Number of indexed identities.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Index every newly-stamped frame entity's [`FrameUidC`] into
/// [`FrameUidIndexR`], rejecting duplicates at the point of
/// introduction. Only **frame entities** (carrying [`FrameTransC`]) are
/// indexed — source / body entities carry `FrameUidC` as the
/// identity-*carrier* the registration systems copy from, and indexing
/// them too would self-collide with their own frame entities.
///
/// Runs in the registration slots after the `register_*_frames_system`
/// chain; `Added<FrameUidC>` makes repeated passes cost one query
/// iteration. Frame entities stamped by a later-registered planet's
/// chain in the same pass are picked up on the next pass — duplicate
/// rejection is at-latest-next-tick, and the manual frame-document
/// apply path (which runs between ticks) always sees a current index.
// JEOD_INV: RF.14 — every frame identity maps to exactly one entity in
// the ECS store; a duplicate registration is rejected loudly, mirroring
// the arena's register_uid.
#[allow(clippy::type_complexity)]
pub fn index_frame_uids_system(
    mut index: ResMut<FrameUidIndexR>,
    added: Query<(Entity, &FrameUidC), (Added<FrameUidC>, With<FrameTransC>)>,
) {
    for (entity, uid) in &added {
        if let Some(&existing) = index.0.get(&uid.0) {
            if existing == entity {
                continue; // re-stamp of the same entity (pfix reuse path)
            }
            panic!(
                "FrameUidIndexR: duplicate frame identity `{}` — already registered \
                 at entity {existing:?}, cannot register it again at {entity:?}. Each \
                 FrameUid maps to exactly one frame entity. If these are genuinely \
                 distinct frames, mint distinct identities (different tag, role, or \
                 namespace).",
                uid.0
            );
        }
        index.0.insert(uid.0.clone(), entity);
    }
}

/// Drop index entries for frame entities whose [`FrameUidC`] was removed
/// (pfix retirement) or that were despawned — keeps
/// [`FrameUidIndexR`] consistent so a retired identity can be re-minted
/// on the next rotation-model toggle without colliding.
pub fn deindex_frame_uids_system(
    mut index: ResMut<FrameUidIndexR>,
    mut removed: RemovedComponents<FrameUidC>,
) {
    for entity in removed.read() {
        index.0.retain(|_, e| *e != entity);
    }
}

/// Auto-register every gravity-source entity (carrying [`GravitySourceC`])
/// by spawning its frame entity as a child of the root frame entity and
/// attaching [`FrameEntityC`] back to the source. The frame entity's
/// [`FrameTransC`] is initialized from [`SourceInertialPositionC`] and
/// (when present) [`SourceInertialVelocityC`].
///
/// This is the Bevy analog of `astrodyn_runner::Simulation::add_source` —
/// it makes the source state observable to gravity / integration via
/// [`crate::frame_param::FrameOrigin`] and to mission code via
/// [`crate::frame_param::RelativeFrameState`].
///
/// **Planet-agnostic since issue #664** (and registered exactly once,
/// not per planet): the frame entity's identity comes from the source
/// entity's carried [`FrameUidC`] value, not from a type parameter — a
/// `<P>`-generic version matched every source under *every* registered
/// planet's instantiation (the query never keyed on `<P>`), so a
/// two-planet App double-spawned an orphan frame entity per source.
/// The identity index turned that silent leak into a loud duplicate,
/// and value-carried identity removes the `<P>` entirely. Pfix child
/// frames are spawned by the per-planet
/// [`register_pfix_frames_system`], which is genuinely `<P>`-keyed
/// (`With<PlanetFixedRotationC<P>>`).
///
/// **Divergence from astrodyn_runner**: every source becomes a child of
/// the root frame, including the central body. `astrodyn_runner` renames
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
    sim_time: Res<crate::SimulationTimeR>,
    sources: Query<
        (
            Entity,
            Option<&Name>,
            Option<&FrameUidC>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
        ),
        (With<GravitySourceC>, Without<FrameEntityC>),
    >,
) {
    let frame_epoch = FrameEpochC(sim_time.tdb());
    for (entity, name, uid, pos, vel) in &sources {
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("source{:?}", entity));
        // Identity is required (issue #664) and travels as a value on
        // the source entity — this system cannot type-derive the
        // source's own planet (a Sun entity in an Earth-centered sim
        // carries `TranslationalStateC<Earth>`). The spawn site knows
        // the planet: `PlanetBundle<P>` stamps
        // `FrameUid::of::<PlanetInertial<P>>()`, `populate_app` carries
        // the builder's pre-minted identities, and raw spawns supply
        // one explicitly.
        let uid = uid.unwrap_or_else(|| {
            panic!(
                "register_source_frames_system: gravity source {entity:?} ({label}) \
                 has no FrameUidC — every gravity source must carry its \
                 inertial-frame identity, minted from the source's own planet \
                 marker. Spawn it via PlanetBundle or populate_app (both stamp \
                 it), or insert it explicitly before registration, e.g. \
                 FrameUidC(astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>())."
            )
        });
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
                // The source entity's carried identity becomes the frame
                // entity's identity (issue #664) — same value the runner
                // stamps on its source inertial node.
                uid.clone(),
                frame_epoch,
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

        // Pfix child frames are spawned by the per-planet
        // `register_pfix_frames_system` (the genuinely `<P>`-keyed half
        // of registration), which runs after this system and picks the
        // source up once its `FrameEntityC` lands — same pass for the
        // common Startup/PreUpdate flows.
    }
}

/// Register a [`PfixFrameEntityC`] for sources that were registered
/// without [`PlanetFixedRotationC`] and acquired it later, or for
/// sources whose [`RotationModelC`] just toggled back from
/// [`astrodyn::RotationModel::None`] to a rotating model.
/// [`register_source_frames_system`] filters by `Without<FrameEntityC>`,
/// so it cannot pick up an entity that gained `PlanetFixedRotationC`
/// after its initial registration.
///
/// Same registration semantics as [`register_source_frames_system`]'s
/// pfix branch: gated on [`PlanetFixedRotationC`], `EarthRNP` default
/// when [`RotationModelC`] is absent, no frame entity when the rotation
/// model is explicitly [`astrodyn::RotationModel::None`].
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
    sim_time: Res<crate::SimulationTimeR>,
    sources: Query<
        (
            Entity,
            Option<&Name>,
            // The source's carried inertial-frame identity (issue #664);
            // required — `register_source_frames_system` panics before
            // this system can see a source without one.
            &FrameUidC,
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
    let frame_epoch = FrameEpochC(sim_time.tdb());
    for (entity, name, uid, source_frame_entity, rotation_model, retired_entity) in &sources {
        let default_model = astrodyn::RotationModel::EarthRNP;
        let model_value = rotation_model.map_or(default_model, |m| m.0);
        if matches!(model_value, astrodyn::RotationModel::None) {
            continue;
        }
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("source{:?}", entity));
        let pfix_uid = FrameUidC(astrodyn::pfix_sibling_uid(&uid.0));

        let pfix_frame_entity = if let Some(retired_e) = retired_entity {
            // Reuse: restore canonical name (via Commands so we
            // don't need a `Query<&mut Name, With<PlanetFixedFrameMarker>>`
            // that would conflict with the outer query's
            // `Option<&Name>` access at runtime) and reset typed
            // state to identity. The orphan's
            // `ChildOf(source_frame_entity.0)` edge was preserved
            // across the toggle cycle, so the hierarchy is already
            // correct.
            commands.entity(retired_e.0).insert((
                Name::new(format!("{label}.frame.pfix")),
                // Re-mint the identity + epoch the retirement path
                // removed (issue #664): the reused entity becomes a
                // live tree node again, re-indexed via Added<FrameUidC>.
                pfix_uid.clone(),
                frame_epoch,
            ));
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
                    pfix_uid.clone(),
                    frame_epoch,
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
/// `astrodyn_runner::Simulation::update_ephemeris`'s post-DE4xx writeback —
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
/// Runs in `AstrodynSet::EphemerisUpdate` after `ephemeris_update_system`
/// (which writes the ECS components from DE4xx) so the frame-entity
/// sync sees the latest values.
#[allow(clippy::type_complexity)]
pub fn sync_source_to_frame_system<P: Planet>(
    sim_time: Res<crate::SimulationTimeR>,
    sources: Query<(
        &FrameEntityC,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TranslationalStateC<P>>,
        Has<EphemerisBodyC>,
    )>,
    mut frame_states: Query<&mut FrameTransC>,
    mut frame_epochs: Query<&mut FrameEpochC>,
) {
    // Frame-epoch re-stamp for ephemeris-driven sources only (RFS-603):
    // mirrors the runner's 2b loop, which re-stamps a source node's
    // epoch only when DE4xx rewrites it. Static sources keep their
    // registration-time epoch — the heterogeneous epochs must match the
    // runner's bit-for-bit for cross-backend document equality.
    let frame_epoch = FrameEpochC(sim_time.tdb());
    for (fe, pos, vel, trans, ephemeris_driven) in &sources {
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
        if ephemeris_driven {
            let mut epoch = frame_epochs.get_mut(fe.0).unwrap_or_else(|err| {
                panic!(
                    "sync_source_to_frame_system: source has FrameEntityC({:?}) \
                     but that entity has no FrameEpochC ({err:?}). Frame \
                     entities are stamped with FrameEpochC at registration \
                     (issue #664); do not strip it.",
                    fe.0
                )
            });
            *epoch = frame_epoch;
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
/// Runs at `Startup` and again before `AstrodynSet::EphemerisUpdate` to
/// catch dynamically-spawned bodies. Filters by
/// `Without<FrameEntityC>` so the registration is one-time per body.
#[allow(clippy::type_complexity)]
pub fn register_body_frames_system<P: Planet>(
    mut commands: Commands,
    // The ECS-side root frame entity, used as the body's frame
    // parent when no IntegSourceC is supplied.
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sim_time: Res<crate::SimulationTimeR>,
    sources: Query<(&FrameUidC, &FrameEntityC), With<GravitySourceC>>,
    bodies: Query<
        (
            Entity,
            Option<&Name>,
            Option<&FrameUidC>,
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
    let frame_epoch = FrameEpochC(sim_time.tdb());
    for (entity, name, uid, trans, integ_source, has_mass) in &bodies {
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("body{:?}", entity));
        // Identity is REQUIRED, never minted by accident (issue #664,
        // mirroring the runner's #662 add_body): the body's frame
        // identity is mission-supplied configuration, carried as a value
        // on the body entity.
        let body_uid = uid.unwrap_or_else(|| {
            panic!(
                "register_body_frames_system: body {entity:?} ({label}) has no \
                 FrameUidC — every dynamic body must carry a stable frame \
                 identity. Spawn it via VehicleConfig::spawn_bevy (which stamps \
                 config.frame_uid), or insert \
                 FrameUidC(astrodyn::named_body_frame_uid(\"<label>\")) on the \
                 body entity before registration."
            )
        });

        // Resolve the integration frame entity from the config-carried
        // identity (issue #668). Default: root inertial. Same-tick scan
        // over the registered sources' carried identities — a cached
        // index could be stale within the registration tick.
        let integ_frame_entity = match integ_source.and_then(|c| c.0.as_ref()) {
            Some(integ_uid) => sources
                .iter()
                .find(|(uid, _)| &uid.0 == integ_uid)
                .map(|(_, fe)| fe.0)
                .unwrap_or_else(|| {
                    panic!(
                        "register_body_frames_system: body {entity:?} has \
                         IntegSourceC(`{integ_uid}`), but no registered gravity \
                         source carries that identity (it is missing, or its \
                         frame was never registered). Spawn the source via \
                         PlanetBundle before the body, fix the identity, or \
                         remove IntegSourceC."
                    )
                }),
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
                // The mission-supplied identity carried by the body entity
                // becomes the frame entity's identity — the same value the
                // runner stamps from `VehicleConfig::frame_uid` (#662).
                body_uid.clone(),
                frame_epoch,
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
/// before `AstrodynSet::EphemerisUpdate`) so the invariant is restored
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
/// Runs in `AstrodynSet::Integration` after `integration_system` and
/// before `frame_switch_system`.
///
/// The `With<DynamicsConfigC>` filter narrows the iteration to actual
/// dynamic bodies — gravity-source entities (planets) also carry
/// `TranslationalStateC` + `FrameEntityC` post-registration but their
/// frame entity is updated by `sync_source_to_frame_system` from the
/// source-side state instead.
pub fn sync_body_to_frame_system<P: Planet>(
    sim_time: Res<crate::SimulationTimeR>,
    bodies: Query<(&TranslationalStateC<P>, &FrameEntityC), With<DynamicsConfigC>>,
    mut frame_states: Query<&mut FrameTransC>,
    mut frame_epochs: Query<&mut FrameEpochC>,
) {
    // Frame-epoch re-stamp every step (RFS-603) — mirrors the runner's
    // stage-8 body-node writeback stamping each body node per step.
    let frame_epoch = FrameEpochC(sim_time.tdb());
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
        let mut epoch = frame_epochs.get_mut(frame_entity.0).unwrap_or_else(|err| {
            panic!(
                "sync_body_to_frame_system: body has FrameEntityC({:?}) but \
                 that entity has no FrameEpochC ({err:?}). Frame entities are \
                 stamped with FrameEpochC at registration (issue #664); do \
                 not strip it.",
                frame_entity.0
            )
        });
        *epoch = frame_epoch;
    }
}

/// Resolve deferred [`ShadowBodyRefC`] references (issue #668):
/// `spawn_bevy` only holds `Commands`, so a vehicle's shadow-caster
/// declaration travels as a body-side `(FrameUid, radius)` ref; this
/// system resolves the identity to the registered source entity,
/// inserts [`ShadowBodyC`] on the *source*, and removes the ref.
///
/// Registered in the same slots as the `register_*_frames_system`
/// chain so dynamically-spawned vehicles resolve before the SRP stage
/// reads `ShadowBodyC`. A radius that disagrees with an
/// already-present `ShadowBodyC` on the same source panics — two
/// vehicles declaring different radii for one caster is a
/// misconfiguration the SRP eclipse geometry cannot honor (this is the
/// same agreement check `populate_app`'s eager pre-collection used to
/// perform before the deferred resolver replaced it).
pub fn resolve_shadow_body_ref_system(
    mut commands: Commands,
    refs: Query<(Entity, &ShadowBodyRefC)>,
    sources: Query<(Entity, &FrameUidC, Option<&ShadowBodyC>), With<GravitySourceC>>,
) {
    for (body_entity, sb_ref) in &refs {
        let (source_entity, _, existing) = sources
            .iter()
            .find(|(_, uid, _)| uid.0 == sb_ref.source)
            .unwrap_or_else(|| {
                panic!(
                    "resolve_shadow_body_ref_system: body {body_entity:?} declares \
                     shadow caster `{}`, but no registered gravity source carries \
                     that identity. Spawn the source (PlanetBundle / populate_app / \
                     explicit FrameUidC) before the vehicle, or fix the identity.",
                    sb_ref.source
                )
            });
        if let Some(existing) = existing {
            assert!(
                existing.radius.to_bits() == sb_ref.radius.to_bits(),
                "resolve_shadow_body_ref_system: shadow caster `{}` already carries \
                 ShadowBodyC with radius {} m, but body {body_entity:?} declares \
                 radius {} m. All bodies sharing a shadow caster must agree on its \
                 radius.",
                sb_ref.source,
                existing.radius,
                sb_ref.radius,
            );
        } else {
            commands.entity(source_entity).insert(ShadowBodyC {
                radius: sb_ref.radius,
            });
        }
        commands.entity(body_entity).remove::<ShadowBodyRefC>();
    }
}
