//! Gravity-source registration and accessors for [`super::Simulation`].
//!
//! Methods: `add_source`, `set_source_ephemeris`, `frame_tree`,
//! `num_sources`, `source_frame`, `source_position`, `set_source_position`,
//! `set_source_state`, `source_pfix_rotation`, `source_tidal_config_mut`,
//! `source_delta_c20`.

use glam::{DMat3, DVec3};

use jeod_frames::{FrameId, FrameTree, RefFrameKind, RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_sim::{GravitySourceEntry, JeodQuat, RotationModel};

use super::types::{GravityData, SourceFrameIds};
use super::Simulation;

impl Simulation {
    /// Add a gravity source. Returns its index for use in `GravityControls`.
    ///
    /// Sources with `central: true` (set by [`GravitySourceEntry::central_body`]
    /// and [`GravitySourceEntry::central_body_sh`]) are mapped to the root frame.
    /// Non-central sources get child inertial frames under the root.
    ///
    /// Only one central source may be added; a second will panic.
    ///
    /// If the source has a rotation model, a planet-fixed child frame is also
    /// created under the source's inertial frame.
    pub fn add_source(&mut self, name: impl Into<String>, entry: GravitySourceEntry) -> usize {
        let idx = self.gravity_data.len();
        let name = name.into();

        // Central bodies map to the root frame; third bodies get child frames.
        // Only one central source is allowed (the root can't be shared).
        let inertial_name = format!("{name}.inertial");
        let inertial_id = if entry.central {
            assert!(
                !self
                    .source_frame_ids
                    .iter()
                    .any(|sf| sf.inertial == self.root_frame_id),
                "add_source: a central source already maps to root_frame_id. \
                 Only one central source is allowed per simulation."
            );
            assert!(
                entry.position == DVec3::ZERO,
                "add_source: central sources must have zero position because they map \
                 directly to root_frame_id."
            );
            // Central body: use the root frame directly. Rename to match.
            // `entry.velocity` is stored in `gravity_data` for relativistic
            // corrections, but is not applied as root-frame kinematics.
            self.frame_tree.get_mut(self.root_frame_id).name = inertial_name;
            self.root_frame_id
        } else {
            self.frame_tree.add_child(
                self.root_frame_id,
                inertial_name,
                RefFrameKind::Inertial,
                RefFrameState {
                    trans: RefFrameTrans {
                        position: entry.position,
                        velocity: entry.velocity,
                    },
                    rot: RefFrameRot::default(),
                },
            )
        };

        // Create a planet-fixed child when the source has a rotation model or
        // an explicit inertial-to-pfix transform. This ensures a fixed initial
        // orientation is not silently ignored when rotation_model is None.
        let pfix_id =
            if entry.rotation_model != RotationModel::None || entry.t_inertial_pfix.is_some() {
                let pfix_name = format!("{name}.pfix");
                let rot = if let Some(t) = entry.t_inertial_pfix {
                    RefFrameRot {
                        q_parent_this: JeodQuat::left_quat_from_transformation(&t),
                        t_parent_this: t,
                        ang_vel_this: DVec3::ZERO,
                    }
                } else {
                    RefFrameRot::default()
                };
                Some(self.frame_tree.add_child(
                    inertial_id,
                    pfix_name,
                    RefFrameKind::PlanetFixed,
                    RefFrameState {
                        trans: RefFrameTrans::default(),
                        rot,
                    },
                ))
            } else {
                None
            };

        // Tidal ΔC20 requires a planet-fixed frame for the rotation matrix.
        assert!(
            entry.tidal_config.is_none() || pfix_id.is_some(),
            "add_source: tidal_config requires a planet-fixed frame \
             (set rotation_model or t_inertial_pfix on the source)."
        );

        self.source_frame_ids.push(SourceFrameIds {
            inertial: inertial_id,
            pfix: pfix_id,
        });
        self.gravity_data.push(GravityData {
            source: entry.source,
            velocity: entry.velocity,
            delta_c20: entry.delta_c20,
            tidal_config: entry.tidal_config,
            rotation_model: entry.rotation_model,
            planet_omega: entry.planet_omega,
        });
        self.source_ephem_bodies.push(None);
        idx
    }

    /// Configure ephemeris-based position updates for a source.
    /// Each step, the source's position and velocity will be updated from DE4xx.
    ///
    /// `target` is the body this source represents (e.g., `EphemerisBody::Sun`).
    /// `observer` is the integration frame center (e.g., `EphemerisBody::Earth`).
    pub fn set_source_ephemeris(
        &mut self,
        source_idx: usize,
        target: jeod_sim::EphemerisBody,
        observer: jeod_sim::EphemerisBody,
    ) {
        assert!(
            source_idx < self.source_ephem_bodies.len(),
            "set_source_ephemeris: source_idx {source_idx} out of bounds (len = {})",
            self.source_ephem_bodies.len()
        );
        // Root-frame conflict is caught by validate() → EphemerisOnRootSource.
        // We don't panic here so that all misconfiguration errors are reported
        // together in a single validate() pass rather than aborting on the first.
        self.source_ephem_bodies[source_idx] = Some((target, observer));
    }

    /// Read-only access to the reference frame tree.
    pub fn frame_tree(&self) -> &FrameTree {
        &self.frame_tree
    }

    /// Number of gravity sources.
    pub fn num_sources(&self) -> usize {
        self.gravity_data.len()
    }

    /// Get the inertial frame ID for a gravity source.
    pub fn source_frame(&self, source_idx: usize) -> FrameId {
        self.source_frame_ids
            .get(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_frame: source index {source_idx} is out of range; \
                     {} source frame(s) configured",
                    self.num_sources()
                )
            })
            .inertial
    }

    /// Get the current position of a gravity source relative to the root
    /// inertial frame. Returns `DVec3::ZERO` for the root-mapped central source.
    pub fn source_position(&self, source_idx: usize) -> DVec3 {
        let fid = self.source_frame(source_idx);
        if fid == self.root_frame_id {
            DVec3::ZERO
        } else {
            self.frame_tree.get(fid).state.trans.position
        }
    }

    /// Set the position of a gravity source relative to the root inertial frame.
    pub fn set_source_position(&mut self, source_idx: usize, position: DVec3) {
        assert!(
            source_idx < self.source_frame_ids.len(),
            "set_source_position: source index {source_idx} out of range; \
             {} source(s) configured",
            self.source_frame_ids.len()
        );
        let fid = self.source_frame_ids[source_idx].inertial;
        assert_ne!(
            fid, self.root_frame_id,
            "set_source_position: cannot set position of the root (central body) source"
        );
        self.frame_tree.get_mut(fid).state.trans.position = position;
    }

    /// Set the position and velocity of a gravity source relative to the root inertial frame.
    ///
    /// Prefer this over [`set_source_position`](Simulation::set_source_position)
    /// when velocity is also available, to keep position and velocity consistent.
    pub fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3) {
        assert!(
            source_idx < self.source_frame_ids.len(),
            "set_source_state: source index {source_idx} out of range; \
             {} source(s) configured",
            self.source_frame_ids.len()
        );
        let fid = self.source_frame_ids[source_idx].inertial;
        assert_ne!(
            fid, self.root_frame_id,
            "set_source_state: cannot set state of the root (central body) source"
        );
        let node = self.frame_tree.get_mut(fid);
        node.state.trans.position = position;
        node.state.trans.velocity = velocity;
        // Keep gravity_data velocity in sync for relativistic corrections (PPN).
        self.gravity_data[source_idx].velocity = velocity;
    }

    /// Get the planet-fixed rotation matrix for a gravity source. Returns `None`
    /// if the source has no rotation model (no pfix frame).
    pub fn source_pfix_rotation(&self, source_idx: usize) -> Option<DMat3> {
        self.source_frame_ids
            .get(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_pfix_rotation: source index {source_idx} out of range; \
                     {} source(s) configured",
                    self.source_frame_ids.len()
                )
            })
            .pfix
            .map(|pfix_id| self.frame_tree.get(pfix_id).state.rot.t_parent_this)
    }

    /// Get mutable access to a source's tidal configuration.
    pub fn source_tidal_config_mut(
        &mut self,
        source_idx: usize,
    ) -> Option<&mut jeod_gravity::tides::TidalConfig> {
        let len = self.gravity_data.len();
        self.gravity_data
            .get_mut(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_tidal_config_mut: source index {source_idx} out of range; \
                     {len} source(s) configured",
                )
            })
            .tidal_config
            .as_mut()
    }

    /// Get the current ΔC20 tidal correction for a gravity source.
    pub fn source_delta_c20(&self, source_idx: usize) -> f64 {
        assert!(
            source_idx < self.gravity_data.len(),
            "source_delta_c20: source index {source_idx} out of range; \
             {} source(s) configured",
            self.gravity_data.len()
        );
        self.gravity_data[source_idx].delta_c20
    }
}
