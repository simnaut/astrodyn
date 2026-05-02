//! Source-state mutation API for Bevy missions (issue #71 item 5).
//!
//! `jeod_runner::Simulation` exposes `set_source_position`,
//! `set_source_state`, and `set_source_ephemeris` for runtime gravity-source
//! retargeting. The Bevy adapter mirrors that surface via the
//! [`SourceMutator`] system parameter, which wraps the lifted helpers in
//! [`jeod_sim::source_state`] and additionally syncs the legacy ECS
//! components ([`SourceInertialPositionC`] / [`SourceInertialVelocityC`])
//! so existing systems observe the mutation.
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_jeod::SourceMutator;
//! use glam::DVec3;
//!
//! fn retarget_sun(mut mutator: SourceMutator, sun: Single<Entity, With<bevy_jeod::SunMarker>>) {
//!     mutator.set_source_state(*sun, DVec3::new(1.5e11, 0.0, 0.0), DVec3::ZERO);
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
    SourceFrameIdC, SourceInertialPositionC, SourceInertialVelocityC, TranslationalStateC,
};
use crate::{FrameTreeR, RootFrameIdR};

/// Bevy `SystemParam` exposing source-state mutation analogous to
/// `jeod_runner::Simulation::set_source_*`. Operates on entities that
/// carry a [`SourceFrameIdC`] pointing into [`FrameTreeR`].
///
/// The mutator updates **both** the frame-tree node and the legacy ECS
/// components (`SourceInertialPositionC`, `SourceInertialVelocityC`,
/// `TranslationalStateC`) so any system observing those components sees
/// the change immediately.
#[derive(SystemParam)]
pub struct SourceMutator<'w, 's> {
    /// The simulation frame tree resource.
    pub frame_tree: ResMut<'w, FrameTreeR>,
    /// Root inertial frame ID (used to refuse mutation of the root-mapped
    /// central source — matches `jeod_runner::Simulation::set_source_position`).
    pub root: Res<'w, RootFrameIdR>,
    frame_ids: Query<'w, 's, &'static SourceFrameIdC>,
    positions: Query<'w, 's, &'static mut SourceInertialPositionC>,
    velocities: Query<'w, 's, &'static mut SourceInertialVelocityC>,
    translational: Query<'w, 's, &'static mut TranslationalStateC>,
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
    /// - `source` maps to the root frame (central body): the root frame
    ///   must remain identity, so its position cannot be retargeted.
    pub fn set_source_position(&mut self, source: Entity, position: DVec3) {
        let fid = self.fetch_frame_id(source, "set_source_position");
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

        let typed_pos = jeod_sim::Position::<jeod_sim::Inertial>::from_raw_si(position);
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
    /// - `source` maps to the root frame.
    pub fn set_source_state(&mut self, source: Entity, position: DVec3, velocity: DVec3) {
        let fid = self.fetch_frame_id(source, "set_source_state");
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

        let typed_pos = jeod_sim::Position::<jeod_sim::Inertial>::from_raw_si(position);
        let typed_vel = jeod_sim::Velocity::<jeod_sim::Inertial>::from_raw_si(velocity);
        if let Ok(mut pos_c) = self.positions.get_mut(source) {
            pos_c.0 = typed_pos;
        }
        if let Ok(mut vc) = self.velocities.get_mut(source) {
            vc.0 = typed_vel;
        }
        if let Ok(mut ts) = self.translational.get_mut(source) {
            ts.0.position = typed_pos;
            ts.0.velocity = typed_vel;
        }
    }

    fn fetch_frame_id(&self, source: Entity, method: &str) -> FrameId {
        self.frame_ids
            .get(source)
            .map(|c| c.0)
            .unwrap_or_else(|err| {
                panic!(
                    "SourceMutator::{method}: entity {source:?} is not a registered \
                 gravity source (missing SourceFrameIdC). Spawn it via PlanetBundle \
                 (or insert GravitySourceC + SourceInertialPositionC) and let \
                 `register_source_frames_system` register the frame node before \
                 mutating it. Underlying error: {err:?}"
                )
            })
    }
}
