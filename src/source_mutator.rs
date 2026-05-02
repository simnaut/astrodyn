//! Source-state mutation API for Bevy missions (issue #71 item 5).
//!
//! `jeod_runner::Simulation` exposes `set_source_position`,
//! `set_source_state`, and `set_source_ephemeris` for runtime gravity-source
//! retargeting. The Bevy adapter mirrors **the frame-tree-touching
//! mutators** (`set_source_position`, `set_source_state`) via the
//! [`SourceMutator`] system parameter, which wraps the lifted helpers in
//! [`jeod_sim::source_state`] and additionally syncs the legacy ECS
//! components ([`SourceInertialPositionC`] / [`SourceInertialVelocityC`])
//! so existing systems observe the mutation. `set_source_ephemeris` is
//! intentionally not mirrored: it records a `(target, observer)` mapping
//! on a runner-private vector with no frame-tree mutation; the Bevy
//! adapter expresses the same intent via the
//! [`crate::components::EphemerisBodyC`] component.
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_jeod::{components::GravitySourceC, SourceMutator};
//! use glam::DVec3;
//!
//! // Targets a gravity-source entity (e.g. one spawned via `PlanetBundle`
//! // or with an explicit `GravitySourceC` + `SourceInertialPositionC`).
//! // `SunBundle` / `SunMarker` entities are *not* gravity sources and do
//! // not carry `SourceFrameIdC`; calling the mutator on one panics.
//! fn retarget(mut mutator: SourceMutator, planet: Single<Entity, With<GravitySourceC>>) {
//!     mutator.set_source_state(*planet, DVec3::new(1.5e11, 0.0, 0.0), DVec3::ZERO);
//! }
//! ```
//!
//! The mutator runs against [`crate::FrameTreeR`] + entities carrying a
//! [`crate::SourceFrameIdC`] (auto-inserted on every gravity source
//! entity by `register_source_frames_system`).

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{set_source_position, set_source_state, FrameId, SourceFrameIds};

use crate::components::{
    CentralSourceMarker, SourceFrameIdC, SourceInertialPositionC, SourceInertialVelocityC,
    TranslationalStateC,
};
use crate::{FrameTreeR, RootFrameIdR};

/// Bevy `SystemParam` exposing source-state mutation analogous to
/// `jeod_runner::Simulation::set_source_*`. Operates on entities that
/// carry a [`SourceFrameIdC`] pointing into [`FrameTreeR`].
///
/// The mutator updates **both** the frame-tree node and the legacy ECS
/// components (`SourceInertialPositionC`, `SourceInertialVelocityC`,
/// `TranslationalStateC`) so any system observing those components sees
/// the change immediately. If a source entity lacks
/// [`SourceInertialVelocityC`] when [`Self::set_source_state`] is called
/// with a non-zero velocity, the component is auto-inserted so the
/// gravity / PPN code that reads it sees the new value on the next
/// step. (Closes a footgun raised in PR #260 review:
/// `PlanetBundle::point_mass` doesn't include
/// [`SourceInertialVelocityC`] by default; without auto-insert,
/// `set_source_state` would silently no-op on the velocity write.)
///
/// **Central-body protection**: `jeod_runner::Simulation` rejects
/// mutations of the *root* source (the central body, since the root
/// frame must stay identity) via `assert_ne!(fid, root_frame_id, …)`.
/// The Bevy adapter never maps any source to the root
/// (`register_source_frames_system` always adds sources as children),
/// so that structural-root assertion only fires for entities that
/// manually attach `SourceFrameIdC(root_id)`. To restore the
/// user-facing protection in a normal Bevy app, attach
/// [`CentralSourceMarker`] to the gravity-source entity that mission
/// code treats as the pinned origin (e.g. Earth in an Earth-centered
/// scenario). The mutator panics if the target entity carries the
/// marker — same outcome as `jeod_runner`'s root-source rejection,
/// just opt-in.
#[derive(SystemParam)]
pub struct SourceMutator<'w, 's> {
    /// The simulation frame tree resource.
    pub frame_tree: ResMut<'w, FrameTreeR>,
    /// Root inertial frame ID (used to refuse mutation of the root-mapped
    /// central source — matches `jeod_runner::Simulation::set_source_position`).
    pub root: Res<'w, RootFrameIdR>,
    /// Commands for auto-inserting [`SourceInertialVelocityC`] on sources
    /// that lack it when [`Self::set_source_state`] is called.
    commands: Commands<'w, 's>,
    frame_ids: Query<'w, 's, &'static SourceFrameIdC>,
    positions: Query<'w, 's, &'static mut SourceInertialPositionC>,
    velocities: Query<'w, 's, &'static mut SourceInertialVelocityC>,
    translational: Query<'w, 's, &'static mut TranslationalStateC>,
    central: Query<'w, 's, (), With<CentralSourceMarker>>,
    names: Query<'w, 's, &'static Name>,
}

impl SourceMutator<'_, '_> {
    /// Set the inertial position of `source` and sync to the frame tree
    /// and ECS components. Velocity is not modified — prefer
    /// [`Self::set_source_state`] when the new velocity is also known.
    ///
    /// # Panics
    ///
    /// - `source` does not carry a [`SourceFrameIdC`] (i.e. it isn't a
    ///   registered gravity source — spawn it via [`crate::PlanetBundle`]
    ///   so `register_source_frames_system` registers it).
    /// - `source` carries [`CentralSourceMarker`]: mission code has opted
    ///   that entity into central-body protection (mirrors
    ///   `jeod_runner::Simulation::set_source_position`'s root-source
    ///   rejection).
    /// - `source` maps to the root frame: the root frame must remain
    ///   identity, so its position cannot be retargeted. (Only reachable
    ///   if `SourceFrameIdC(root_id)` is attached manually — Bevy's
    ///   `register_source_frames_system` never maps a source to root.)
    pub fn set_source_position(&mut self, source: Entity, position: DVec3) {
        // Verify the entity is a registered gravity source first; that's
        // the more fundamental misconfiguration to surface (a non-source
        // entity carrying CentralSourceMarker hits the SourceFrameIdC
        // panic before the marker panic, which is the diagnostic ordering
        // a debugging user actually wants — PR #267 review).
        let fid = self.fetch_frame_id(source, "set_source_position");
        self.assert_not_central(source, "set_source_position");
        let source_frames = [SourceFrameIds {
            inertial: fid,
            pfix: None,
        }];
        set_source_position(
            &mut self.frame_tree.0,
            &source_frames,
            self.root.0,
            0,
            position,
        );

        // SourceMutator's public API takes a raw user-supplied DVec3
        // (mirroring `jeod_runner::Simulation::set_source_position`);
        // this is the typed-API boundary for the user → ECS conversion.
        let typed_pos = jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(position); // allowed: user-DVec3 → typed boundary
        if let Ok(mut pos_c) = self.positions.get_mut(source) {
            pos_c.0 = typed_pos;
        }
        if let Ok(mut ts) = self.translational.get_mut(source) {
            ts.0.position = typed_pos;
        }
    }

    /// Set the inertial position and velocity of `source`. Mirrors
    /// `jeod_runner::Simulation::set_source_state`.
    ///
    /// # Panics
    ///
    /// - `source` does not carry a [`SourceFrameIdC`].
    /// - `source` carries [`CentralSourceMarker`].
    /// - `source` maps to the root frame.
    pub fn set_source_state(&mut self, source: Entity, position: DVec3, velocity: DVec3) {
        let fid = self.fetch_frame_id(source, "set_source_state");
        self.assert_not_central(source, "set_source_state");
        let source_frames = [SourceFrameIds {
            inertial: fid,
            pfix: None,
        }];
        set_source_state(
            &mut self.frame_tree.0,
            &source_frames,
            self.root.0,
            0,
            position,
            velocity,
        );

        // SourceMutator's public API takes raw user-supplied DVec3s
        // (mirroring `jeod_runner::Simulation::set_source_state`);
        // this is the typed-API boundary for the user → ECS conversion.
        let typed_pos = jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(position); // allowed: user-DVec3 → typed boundary
        let typed_vel = jeod_sim::Velocity::<jeod_sim::RootInertial>::from_raw_si(velocity); // allowed: user-DVec3 → typed boundary
        if let Ok(mut pos_c) = self.positions.get_mut(source) {
            pos_c.0 = typed_pos;
        }
        // Auto-insert SourceInertialVelocityC if the source doesn't carry
        // one — `PlanetBundle::point_mass` doesn't include it by default,
        // and without auto-insert the velocity write would silently no-op
        // (footgun raised in PR #260 review).
        match self.velocities.get_mut(source) {
            Ok(mut vc) => vc.0 = typed_vel,
            Err(_) => {
                self.commands
                    .entity(source)
                    .insert(SourceInertialVelocityC(typed_vel));
            }
        }
        if let Ok(mut ts) = self.translational.get_mut(source) {
            ts.0.position = typed_pos;
            ts.0.velocity = typed_vel;
        }
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
    /// gravity source the way mission code spelled it (PR #267 review).
    fn entity_label(&self, entity: Entity) -> String {
        match self.names.get(entity) {
            Ok(name) => format!("{name} ({entity:?})"),
            Err(_) => format!("{entity:?}"),
        }
    }

    fn fetch_frame_id(&self, source: Entity, method: &str) -> FrameId {
        self.frame_ids
            .get(source)
            .map(|c| c.0)
            .unwrap_or_else(|err| {
                panic!(
                    "SourceMutator::{method}: {label} is not a registered \
                 gravity source (missing SourceFrameIdC). Spawn it via PlanetBundle \
                 (or insert GravitySourceC + SourceInertialPositionC) and let \
                 `register_source_frames_system` register the frame node before \
                 mutating it. Underlying error: {err:?}",
                    label = self.entity_label(source),
                )
            })
    }
}
