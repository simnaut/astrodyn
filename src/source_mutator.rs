//! Source-state read/write API for Bevy missions.
//!
//! `jeod_runner::Simulation` exposes `set_source_position`,
//! `set_source_state`, and `set_source_ephemeris` for runtime
//! gravity-source retargeting, and `source_position`,
//! `source_pfix_rotation`, `source_frame_id` for read-only access.
//! The Bevy adapter mirrors **the frame-state-touching read/write**
//! surface via the [`SourceMutator`] system parameter, which operates
//! directly on the source's frame entity (carrying [`FrameTransC`] and
//! optional [`PfixFrameEntityC`]) and additionally syncs the ECS
//! components ([`SourceInertialPositionC`] /
//! [`SourceInertialVelocityC`] / [`TranslationalStateC`]) on writes so
//! existing systems observe the mutation.
//!
//! `set_source_ephemeris` is intentionally not mirrored: it records a
//! `(target, observer)` mapping on a runner-private vector with no
//! frame-state mutation; the Bevy adapter expresses the same intent
//! via the [`crate::components::EphemerisBodyC`] component.
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_jeod::{components::GravitySourceC, SourceMutator};
//! use glam::DVec3;
//!
//! // Targets a gravity-source entity (e.g. one spawned via `PlanetBundle`
//! // or with an explicit `GravitySourceC` + `SourceInertialPositionC`).
//! // `SunBundle` / `SunMarker` entities are *not* gravity sources and do
//! // not carry `FrameEntityC`; calling the mutator on one panics.
//! fn retarget(mut mutator: SourceMutator, planet: Single<Entity, With<GravitySourceC>>) {
//!     mutator.set_source_state(*planet, DVec3::new(1.5e11, 0.0, 0.0), DVec3::ZERO);
//! }
//! ```
//!
//! The mutator runs against entities carrying a [`FrameEntityC`]
//! (auto-inserted on every gravity-source entity by
//! `register_source_frames_system`).

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use glam::{DMat3, DVec3};

use crate::components::{
    CentralSourceMarker, FrameEntityC, FrameRotC, FrameTransC, PfixFrameEntityC,
    SourceInertialPositionC, SourceInertialVelocityC, TranslationalStateC,
};

/// Bevy `SystemParam` exposing source-state read/write API analogous
/// to `jeod_runner::Simulation::source_*` / `set_source_*`. Operates
/// on entities that carry a [`FrameEntityC`] pointing at the source's
/// frame entity.
///
/// Mutators (`set_source_position`, `set_source_state`) update **both**
/// the source's frame entity (`FrameTransC`) and the ECS components
/// (`SourceInertialPositionC`, `SourceInertialVelocityC`,
/// `TranslationalStateC`) so any system observing those components
/// sees the change immediately. If a source entity lacks
/// [`SourceInertialVelocityC`] when [`Self::set_source_state`] is
/// called with a non-zero velocity, the component is auto-inserted
/// so the gravity / PPN code that reads it sees the new value on the
/// next step. (Without this auto-insert, `set_source_state` would
/// silently no-op on the velocity write for sources spawned via
/// `PlanetBundle::point_mass`, which doesn't include
/// [`SourceInertialVelocityC`] by default.)
///
/// Read-only accessors (`source_position`, `source_pfix_rotation`,
/// `source_frame_entity`) read the source's frame entity directly,
/// returning the same numerics the gravity / integration / mission
/// systems consume via [`crate::frame_param::FrameOrigin`] and
/// [`crate::frame_param::RelativeFrameState`].
///
/// **Central-body protection**: `jeod_runner::Simulation` rejects
/// mutations of the *root* source (the central body, since the root
/// frame must stay identity) via `assert_ne!(fid, root_frame_id, …)`.
/// The Bevy adapter never maps any source to the root
/// (`register_source_frames_system` always adds sources as children),
/// so that structural-root assertion has no analog. To restore the
/// user-facing protection in a normal Bevy app, attach
/// [`CentralSourceMarker`] to the gravity-source entity that mission
/// code treats as the pinned origin (e.g. Earth in an Earth-centered
/// scenario). The mutator panics if the target entity carries the
/// marker — same outcome as `jeod_runner`'s root-source rejection,
/// just opt-in.
#[derive(SystemParam)]
pub struct SourceMutator<'w, 's> {
    /// Commands for auto-inserting [`SourceInertialVelocityC`] on
    /// sources that lack it when [`Self::set_source_state`] is called.
    commands: Commands<'w, 's>,
    frame_entities: Query<'w, 's, &'static FrameEntityC>,
    pfix_frame_entities: Query<'w, 's, &'static PfixFrameEntityC>,
    frame_trans: Query<'w, 's, &'static mut FrameTransC>,
    frame_rots: Query<'w, 's, &'static FrameRotC>,
    positions: Query<'w, 's, &'static mut SourceInertialPositionC>,
    velocities: Query<'w, 's, &'static mut SourceInertialVelocityC>,
    translational: Query<'w, 's, &'static mut TranslationalStateC>,
    central: Query<'w, 's, (), With<CentralSourceMarker>>,
    names: Query<'w, 's, &'static Name>,
}

impl SourceMutator<'_, '_> {
    /// Get the source's frame entity. Mirrors
    /// `jeod_sim::source_state::source_frame_id` but returns a Bevy
    /// `Entity` — the post-PR4 replacement for the arena's
    /// `FrameId`.
    ///
    /// # Panics
    ///
    /// Panics if `source` does not carry a [`FrameEntityC`] (i.e. it
    /// isn't a registered gravity source).
    pub fn source_frame_entity(&self, source: Entity) -> Entity {
        self.fetch_frame_entity(source, "source_frame_entity")
    }

    /// Get the inertial-frame position of `source` relative to the
    /// root inertial frame, in root-frame coordinates. Mirrors
    /// `jeod_sim::source_state::source_position`.
    ///
    /// Reads the source's frame entity's [`FrameTransC`] directly —
    /// the same storage [`crate::frame_param::FrameOrigin`] and
    /// [`crate::frame_param::RelativeFrameState`] consume.
    ///
    /// # Panics
    ///
    /// Panics if `source` does not carry a [`FrameEntityC`].
    pub fn source_position(&self, source: Entity) -> DVec3 {
        let fe = self.fetch_frame_entity(source, "source_position");
        self.frame_trans
            .get(fe)
            .map(|t| t.position)
            .unwrap_or_else(|err| {
                panic!(
                    "SourceMutator::source_position: {label} has \
                     FrameEntityC({fe:?}) but that entity has no \
                     FrameTransC ({err:?}). The source's frame entity \
                     must be alive with FrameTransC attached (spawned \
                     by register_source_frames_system).",
                    label = self.entity_label(source),
                )
            })
    }

    /// Get the planet-fixed rotation matrix for `source` (the
    /// `t_parent_this` of the source's pfix frame entity). Returns
    /// `None` if the source has no pfix frame entity (i.e. no
    /// [`PfixFrameEntityC`] — non-rotating source). Mirrors
    /// `jeod_sim::source_state::source_pfix_rotation`.
    ///
    /// # Panics
    ///
    /// Panics if `source` does not carry a [`FrameEntityC`] (i.e.
    /// isn't a registered gravity source). Returns `None` (not panic)
    /// when the source is registered but has no pfix child — same
    /// contract as the arena helper.
    pub fn source_pfix_rotation(&self, source: Entity) -> Option<DMat3> {
        // Verify the entity is a registered gravity source first.
        let _ = self.fetch_frame_entity(source, "source_pfix_rotation");
        let pfix_fe = self.pfix_frame_entities.get(source).ok()?;
        let rot = self.frame_rots.get(pfix_fe.0).unwrap_or_else(|err| {
            panic!(
                "SourceMutator::source_pfix_rotation: {label} has \
                 PfixFrameEntityC({fe:?}) but that entity has no \
                 FrameRotC ({err:?}). The pfix frame entity must be \
                 alive with FrameRotC attached (spawned by \
                 register_pfix_frames_system).",
                fe = pfix_fe.0,
                label = self.entity_label(source),
            )
        });
        Some(rot.t_parent_this)
    }

    /// Set the inertial position of `source` and sync to the source's
    /// frame entity and ECS components. Velocity is not modified —
    /// prefer [`Self::set_source_state`] when the new velocity is
    /// also known.
    ///
    /// # Panics
    ///
    /// - `source` does not carry a [`FrameEntityC`] (i.e. it isn't a
    ///   registered gravity source — spawn it via [`crate::PlanetBundle`]
    ///   so `register_source_frames_system` registers it).
    /// - `source` carries [`CentralSourceMarker`]: mission code has opted
    ///   that entity into central-body protection (mirrors
    ///   `jeod_runner::Simulation::set_source_position`'s root-source
    ///   rejection).
    pub fn set_source_position(&mut self, source: Entity, position: DVec3) {
        // Verify the entity is a registered gravity source first;
        // that's the more fundamental misconfiguration to surface
        // (a non-source entity carrying CentralSourceMarker hits the
        // FrameEntityC panic before the marker panic, which is the
        // diagnostic ordering a debugging user actually wants).
        let fe = self.fetch_frame_entity(source, "set_source_position");
        self.assert_not_central(source, "set_source_position");
        // Capture the label before the mutable borrow so the panic
        // closure doesn't need to re-borrow `self`.
        let label = self.entity_label(source);

        // Write to the source's frame entity's FrameTransC. The
        // referenced entity must carry FrameTransC; fail loud
        // otherwise (mirrors the sync-system contract).
        let mut frame_trans = self.frame_trans.get_mut(fe).unwrap_or_else(|err| {
            panic!(
                "SourceMutator::set_source_position: {label} has \
                 FrameEntityC({fe:?}) but that entity has no \
                 FrameTransC ({err:?}). The source's frame entity \
                 must be alive with FrameTransC attached (spawned by \
                 register_source_frames_system)."
            )
        });
        frame_trans.position = position;

        // SourceMutator's public API takes a raw user-supplied DVec3
        // (mirroring `jeod_runner::Simulation::set_source_position`);
        // this is the typed-API boundary for the user → ECS conversion.
        let typed_pos = jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(position); // allowed: user-DVec3 → typed boundary

        // SourceInertialPositionC and TranslationalStateC must exist
        // for a registered gravity source — `register_source_frames_system`
        // queries `&SourceInertialPositionC` to spawn the source's
        // frame entity (so its absence is impossible for a registered
        // source), `PlanetBundle` includes both, and
        // `sync_source_to_frame_system` reads `SourceInertialPositionC`
        // every step to overwrite the frame entity's position. Silently
        // skipping the write here would let the frame_trans update get
        // overwritten back on the next sync, producing a vehicle-visible
        // discontinuity from the user's perspective. Fail loud so a
        // misconfiguration (source without these components) is
        // diagnosed at the mutation site rather than surfacing as a
        // mysterious next-tick reset.
        let mut pos_c = self.positions.get_mut(source).unwrap_or_else(|err| {
            panic!(
                "SourceMutator::set_source_position: {label} is a \
                 registered gravity source (has FrameEntityC) but \
                 lacks SourceInertialPositionC ({err:?}). Spawn the \
                 source via PlanetBundle (which includes both \
                 components) — without SourceInertialPositionC, \
                 `sync_source_to_frame_system` would overwrite the \
                 frame entity's position back on the next step."
            )
        });
        pos_c.0 = typed_pos;
        let mut ts = self.translational.get_mut(source).unwrap_or_else(|err| {
            panic!(
                "SourceMutator::set_source_position: {label} is a \
                 registered gravity source (has FrameEntityC) but \
                 lacks TranslationalStateC ({err:?}). Spawn the \
                 source via PlanetBundle (which includes \
                 TranslationalStateC) so per-step systems observe a \
                 consistent position across the source's components."
            )
        });
        ts.0.position = typed_pos;
    }

    /// Set the inertial position and velocity of `source`. Mirrors
    /// `jeod_runner::Simulation::set_source_state`.
    ///
    /// # Panics
    ///
    /// - `source` does not carry a [`FrameEntityC`].
    /// - `source` carries [`CentralSourceMarker`].
    pub fn set_source_state(&mut self, source: Entity, position: DVec3, velocity: DVec3) {
        let fe = self.fetch_frame_entity(source, "set_source_state");
        self.assert_not_central(source, "set_source_state");
        let label = self.entity_label(source);

        let mut frame_trans = self.frame_trans.get_mut(fe).unwrap_or_else(|err| {
            panic!(
                "SourceMutator::set_source_state: {label} has \
                 FrameEntityC({fe:?}) but that entity has no \
                 FrameTransC ({err:?}). The source's frame entity \
                 must be alive with FrameTransC attached (spawned by \
                 register_source_frames_system)."
            )
        });
        frame_trans.position = position;
        frame_trans.velocity = velocity;

        // SourceMutator's public API takes raw user-supplied DVec3s
        // (mirroring `jeod_runner::Simulation::set_source_state`);
        // this is the typed-API boundary for the user → ECS conversion.
        let typed_pos = jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(position); // allowed: user-DVec3 → typed boundary
        let typed_vel = jeod_sim::Velocity::<jeod_sim::RootInertial>::from_raw_si(velocity); // allowed: user-DVec3 → typed boundary

        // Position + translational writes are part of the registered-source
        // contract: SourceInertialPositionC is required by
        // register_source_frames_system, PlanetBundle includes
        // TranslationalStateC, and sync_source_to_frame_system reads
        // SourceInertialPositionC each step to overwrite the frame
        // entity's position. Silently skipping these would let the
        // frame_trans update get overwritten back on the next sync.
        // Fail loud — same diagnostic ordering as set_source_position.
        let mut pos_c = self.positions.get_mut(source).unwrap_or_else(|err| {
            panic!(
                "SourceMutator::set_source_state: {label} is a \
                 registered gravity source (has FrameEntityC) but \
                 lacks SourceInertialPositionC ({err:?}). Spawn the \
                 source via PlanetBundle (which includes both \
                 components) — without SourceInertialPositionC, \
                 `sync_source_to_frame_system` would overwrite the \
                 frame entity's position back on the next step."
            )
        });
        pos_c.0 = typed_pos;
        // Auto-insert SourceInertialVelocityC if the source doesn't carry
        // one — `PlanetBundle::point_mass` doesn't include it by default,
        // and without auto-insert the velocity write would silently
        // no-op. This is the one component on the source-mutation path
        // that is genuinely optional at registration time, so the
        // best-effort branch stays here (unlike SourceInertialPositionC
        // / TranslationalStateC above, which are contract-required).
        match self.velocities.get_mut(source) {
            Ok(mut vc) => vc.0 = typed_vel,
            Err(_) => {
                self.commands
                    .entity(source)
                    .insert(SourceInertialVelocityC(typed_vel));
            }
        }
        let mut ts = self.translational.get_mut(source).unwrap_or_else(|err| {
            panic!(
                "SourceMutator::set_source_state: {label} is a \
                 registered gravity source (has FrameEntityC) but \
                 lacks TranslationalStateC ({err:?}). Spawn the \
                 source via PlanetBundle (which includes \
                 TranslationalStateC) so per-step systems observe a \
                 consistent state across the source's components."
            )
        });
        ts.0.position = typed_pos;
        ts.0.velocity = typed_vel;
    }

    fn assert_not_central(&self, source: Entity, method: &str) {
        if self.central.get(source).is_ok() {
            panic!(
                "SourceMutator::{method}: {label} carries CentralSourceMarker \
                 — the central body's state is pinned by convention. Remove \
                 the marker (or target a different gravity source) if \
                 retargeting the central body is really intended.",
                label = self.entity_label(source),
            );
        }
    }

    /// Format a user-facing label for `entity` — `"Earth (Entity {…})"` if a
    /// `Name` component is present, falling back to the raw `Entity` debug
    /// form. Used by the panic-formatting helpers so diagnostics name the
    /// gravity source the way mission code spelled it.
    fn entity_label(&self, entity: Entity) -> String {
        match self.names.get(entity) {
            Ok(name) => format!("{name} ({entity:?})"),
            Err(_) => format!("{entity:?}"),
        }
    }

    fn fetch_frame_entity(&self, source: Entity, method: &str) -> Entity {
        self.frame_entities
            .get(source)
            .map(|c| c.0)
            .unwrap_or_else(|err| {
                panic!(
                    "SourceMutator::{method}: {label} is not a registered \
                 gravity source (missing FrameEntityC). Spawn it via PlanetBundle \
                 (or insert GravitySourceC + SourceInertialPositionC) and let \
                 `register_source_frames_system` register the frame entity before \
                 mutating it. Underlying error: {err:?}",
                    label = self.entity_label(source),
                )
            })
    }
}
