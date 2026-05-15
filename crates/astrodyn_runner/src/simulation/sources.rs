//! Gravity-source registration and accessors for [`super::Simulation`].
//!
//! Methods: `add_source`, `set_source_ephemeris`, `frame_tree`,
//! `num_sources`, `source_frame`, `source_position`, `set_source_position`,
//! `set_source_state`, `source_pfix_rotation`, `source_tidal_config_mut`,
//! `source_delta_c20`.

use glam::{DMat3, DVec3};

use astrodyn::{
    set_source_position as sim_set_source_position, set_source_state as sim_set_source_state,
    source_pfix_rotation as sim_source_pfix_rotation, source_position as sim_source_position,
    FrameId, FrameTree, GravitySourceEntry, JeodQuat, RefFrameKind, RefFrameRot, RefFrameState,
    RefFrameTrans, RotationModel, SourceFrameIds,
};

use super::types::GravityData;
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
                entry.position.raw_si() == DVec3::ZERO,
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
                        // RefFrameTrans is the runtime arena's untyped storage —
                        // unwrap the typed entry value at this documented
                        // boundary (RF.10 catalog row notes the frame tree is
                        // runtime-typed by design).
                        position: entry.position.raw_si(),
                        velocity: entry.velocity.raw_si(),
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
            // `GravityData.velocity` is runner-internal cache; unwrap the
            // typed entry value at this documented runner boundary.
            velocity: entry.velocity.raw_si(),
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
        target: astrodyn::EphemerisBody,
        observer: astrodyn::EphemerisBody,
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
    ///
    /// Prefer [`source_position_typed`](Self::source_position_typed) for the
    /// typed-result path.
    // ESCAPE_HATCH: pending H4 typed-migration follow-up. The typed sibling
    // `source_position_typed` exists; this raw variant is preserved for
    // backwards compatibility until the parity-lockstep follow-on PR (see
    // #485 task 18) removes it alongside the Bevy mirrors.
    pub fn source_position(&self, source_idx: usize) -> DVec3 {
        sim_source_position(
            &self.frame_tree,
            &self.source_frame_ids,
            self.root_frame_id,
            source_idx,
        )
    }

    /// Typed sibling of [`Self::source_position`] returning the source
    /// position already wrapped as `Position<RootInertial>`. The
    /// frame phantom matches the runner's root-inertial convention; per-
    /// step pipelines that consume gravity / SRP / shadow inputs in
    /// `RootInertial` should call this rather than re-lifting the raw
    /// `DVec3` at every body.
    pub fn source_position_typed(
        &self,
        source_idx: usize,
    ) -> astrodyn::Position<astrodyn::RootInertial> {
        // allowed: typed-sibling accessor boundary. The frame phantom
        // is the runner-wide convention (root frame is inertial); the
        // raw kernel returns a frame-erased `DVec3` and the only source
        // of truth for the phantom is the runner's documented root
        // configuration.
        astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(self.source_position(source_idx))
    }

    /// Set the position of a gravity source relative to the root inertial frame.
    pub fn set_source_position(&mut self, source_idx: usize, position: DVec3) {
        sim_set_source_position(
            &mut self.frame_tree,
            &self.source_frame_ids,
            self.root_frame_id,
            source_idx,
            position,
        );
    }

    /// Set the position and velocity of a gravity source relative to the root inertial frame.
    ///
    /// Prefer this over [`set_source_position`](Simulation::set_source_position)
    /// when velocity is also available, to keep position and velocity consistent.
    pub fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3) {
        sim_set_source_state(
            &mut self.frame_tree,
            &self.source_frame_ids,
            self.root_frame_id,
            source_idx,
            position,
            velocity,
        );
        // Keep gravity_data velocity in sync for relativistic corrections (PPN).
        // Bounds already validated inside `sim_set_source_state` — `gravity_data`
        // is parallel to `source_frame_ids`, so the same index is in range.
        self.gravity_data[source_idx].velocity = velocity;
    }

    /// Get the planet-fixed rotation matrix for a gravity source. Returns `None`
    /// if the source has no rotation model (no pfix frame).
    ///
    /// Prefer [`source_pfix_rotation_typed`](Self::source_pfix_rotation_typed)
    /// for typed `FrameTransform<RootInertial, PlanetFixed<P>>`.
    // ESCAPE_HATCH: pending H4 typed-migration follow-up. Raw `DMat3`
    // return is deferred to the parity-lockstep follow-on PR (see #485
    // task 18) to be removed alongside the Bevy mirror in
    // `SourceReader::source_pfix_rotation`.
    pub fn source_pfix_rotation(&self, source_idx: usize) -> Option<DMat3> {
        sim_source_pfix_rotation(&self.frame_tree, &self.source_frame_ids, source_idx)
    }

    /// Typed sibling of [`source_pfix_rotation`](Self::source_pfix_rotation).
    ///
    /// Returns the source's planet-fixed rotation as a typed
    /// `FrameTransform<RootInertial, PlanetFixed<P>>`, matching the canonical
    /// shape used by `PlanetFixedRotationC<P>` on the Bevy adapter side. The
    /// caller asserts via the turbofish that `source_idx` resolves to the
    /// planet `P`'s pfix frame — no runtime check is performed (mirroring
    /// `source_position_typed`'s phantom-attachment boundary).
    pub fn source_pfix_rotation_typed<P: astrodyn::Planet>(
        &self,
        source_idx: usize,
    ) -> Option<astrodyn::FrameTransform<astrodyn::RootInertial, astrodyn::PlanetFixed<P>>> {
        let raw = sim_source_pfix_rotation(&self.frame_tree, &self.source_frame_ids, source_idx);
        // allowed: typed-sibling accessor boundary. The kernel returns a
        // frame-erased `DMat3`; the phantom is the caller's pinned-planet
        // convention attached unchecked, matching `source_position_typed`.
        raw.map(astrodyn::FrameTransform::<astrodyn::RootInertial, astrodyn::PlanetFixed<P>>::from_matrix)
    }

    /// Get the inertial-frame [`FrameId`] for a gravity source.
    ///
    /// Useful for `Simulation::attach_to_frame(body_idx, parent_frame_id, …)`
    /// callers that want to attach a body to (e.g.) Earth's inertial frame.
    /// For the central body this returns the simulation's root frame.
    pub fn source_inertial_frame_id(&self, source_idx: usize) -> FrameId {
        let len = self.source_frame_ids.len();
        self.source_frame_ids
            .get(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_inertial_frame_id: source index {source_idx} out of \
                     range; {len} source(s) configured"
                )
            })
            .inertial
    }

    /// Get the planet-fixed [`FrameId`] for a gravity source. Returns
    /// `None` if the source has no rotation model (no pfix frame).
    ///
    /// The companion to [`source_inertial_frame_id`](Self::source_inertial_frame_id)
    /// for the rotating-frame attachment use case (JEOD's
    /// `attach_to_frame("Earth.pfix")` pattern from
    /// `verif/SIM_dyncomp/SET_test/RUN_attach_to_ref_frame/input.py`).
    pub fn source_pfix_frame_id(&self, source_idx: usize) -> Option<FrameId> {
        let len = self.source_frame_ids.len();
        self.source_frame_ids
            .get(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_pfix_frame_id: source index {source_idx} out of \
                     range; {len} source(s) configured"
                )
            })
            .pfix
    }

    /// Get mutable access to a source's tidal configuration.
    pub fn source_tidal_config_mut(
        &mut self,
        source_idx: usize,
    ) -> Option<&mut astrodyn::TidalConfig> {
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
