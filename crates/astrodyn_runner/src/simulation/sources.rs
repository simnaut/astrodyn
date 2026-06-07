//! Gravity-source registration and accessors for [`super::Simulation`].
//!
//! Methods: `add_source`, `set_source_ephemeris`, `frame_tree`,
//! `num_sources`, `source_frame`, `source_position`, `set_source_position`,
//! `set_source_state`, `source_pfix_rotation`, `source_tidal_config_mut`,
//! `source_delta_c20`.

use astrodyn::FrameUid;
use glam::DVec3;

use astrodyn::{
    set_source_position as sim_set_source_position, set_source_state as sim_set_source_state,
    source_pfix_rotation as sim_source_pfix_rotation, source_position as sim_source_position,
    FrameId, FrameTree, GravitySourceEntry, JeodQuat, RefFrameRot, RefFrameState, RefFrameTrans,
    RotationModel, SourceFrameIds,
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
        let name = name.into();
        // The name→identity table is shared code
        // (`astrodyn::sealed_planet_inertial_uid`), not a per-host match —
        // a drifted copy would mint different identities for the same
        // source name across backends. The pfix sibling derivation is
        // pinned equal to the typed mint for all six sealed planets.
        let inertial_uid = astrodyn::sealed_planet_inertial_uid(&name).unwrap_or_else(|| {
            panic!(
                "add_source: unknown gravity-source name {name:?} — the string-dispatch \
                 add_source only knows the six sealed planets (Earth, Moon, Sun, Mars, \
                 Jupiter, Saturn). For any other body, call add_source_typed::<P>(name, \
                 entry) with a Planet marker defined via define_planet! so the source \
                 frames get real stamped identities."
            )
        });
        let pfix_uid = astrodyn::pfix_sibling_uid(&inertial_uid);
        self.add_source_with_uids(name, entry, inertial_uid, pfix_uid)
    }

    /// Typed sibling of [`Self::add_source`]: the source's frame identities
    /// are derived from the compile-time planet marker `P`
    /// (`FrameUid::of::<PlanetInertial<P>>()` for the inertial child,
    /// `FrameUid::of::<PlanetFixed<P>>()` for the optional planet-fixed
    /// child). Use this directly for planets minted downstream via
    /// `define_planet!`; the string-dispatch `add_source` routes the six
    /// built-in planets here.
    ///
    /// The `name` argument labels the frame nodes for diagnostics and
    /// `find_by_name`; the stamped identity comes from `P`, not from
    /// `name`. Passing a name that disagrees with `P::NAME` is allowed
    /// (the label is cosmetic) but discouraged. `marker_only` sources
    /// flow through the same path and are stamped identically.
    ///
    /// # Panics
    /// - a second central source, or a central source with non-zero
    ///   position (pre-existing invariants);
    /// - a duplicate source identity: registering the same planet twice
    ///   trips the frame tree's duplicate-identity rejection.
    pub fn add_source_typed<P: astrodyn::Planet>(
        &mut self,
        name: impl Into<String>,
        entry: GravitySourceEntry,
    ) -> usize {
        self.add_source_with_uids(
            name,
            entry,
            FrameUid::of::<astrodyn::PlanetInertial<P>>(),
            FrameUid::of::<astrodyn::PlanetFixed<P>>(),
        )
    }

    /// Value-level core of source registration: the caller supplies the
    /// minted identities for the inertial and (optional) planet-fixed
    /// frames. [`Self::add_source_typed`] mints them from a planet marker;
    /// the `SimulationBuilder` path carries pre-minted identities as
    /// configuration data — identity travels as a value through config,
    /// exactly like body identity (issue #662).
    ///
    /// The `pfix_uid` is registered only when the source actually creates
    /// a planet-fixed frame (rotation model or explicit transform).
    pub fn add_source_with_uids(
        &mut self,
        name: impl Into<String>,
        entry: GravitySourceEntry,
        inertial_uid: FrameUid,
        pfix_uid: FrameUid,
    ) -> usize {
        let idx = self.gravity_data.len();
        let name = name.into();

        // Allocate a distinct inertial frame node as a child of root for
        // every gravity source — including the central body, which has
        // zero position relative to root. Mirrors JEOD's canonical
        // `BasePlanet { inertial, pfix }` structure
        // (`models/environment/planet/include/base_planet.hh:109,121`)
        // and the Bevy adapter's existing layout.
        //
        // Pre-#567 the runner aliased the central source's inertial frame
        // to `root_frame_id` as a one-hop optimization, which gave the
        // runner a structurally-different frame tree than Bevy. With
        // `JEOD_INV: RF.13` (#566) in place — the identity-rotation
        // fast-path in `RefFrameState::incr_right` / `::negate` — the
        // extra hop through an identity intermediate preserves f64 bits,
        // so the canonical-JEOD structural alignment between the
        // runtimes is safe.
        //
        // Only one central source is allowed per simulation: enforce
        // via the `central` flag on `SourceFrameIds` rather than the
        // pre-#567 root-id alias check.
        if entry.central {
            assert!(
                !self.source_frame_ids.iter().any(|sf| sf.central),
                "add_source: a central source has already been added. \
                 Only one central source is allowed per simulation."
            );
            assert!(
                entry.position.raw_si() == DVec3::ZERO,
                "add_source: central sources must have zero position relative to root."
            );
        }
        let inertial_name = format!("{name}.inertial");
        let inertial_id = self.frame_tree.add_child_uid(
            self.root_frame_id,
            inertial_uid.clone(),
            inertial_name,
            RefFrameState {
                trans: RefFrameTrans {
                    // RefFrameTrans is the runtime arena's untyped storage —
                    // unwrap the typed entry value at this documented
                    // boundary (RF.10 catalog row notes the frame tree is
                    // runtime-typed by design).
                    position: entry.position.raw_si(),
                    velocity: entry.velocity.raw_si(),
                },
                // Central sources land at identity rotation (zero position,
                // identity quaternion); non-central sources do the same by
                // default and are written into by ephemeris updates each
                // step. Identity rotation here triggers `RF.13`'s
                // composition fast-paths during frame-tree walks.
                rot: RefFrameRot::default(),
            },
            Some(self.time.tdb()),
        );

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
                Some(self.frame_tree.add_child_uid(
                    inertial_id,
                    pfix_uid,
                    pfix_name,
                    RefFrameState {
                        trans: RefFrameTrans::default(),
                        rot,
                    },
                    Some(self.time.tdb()),
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

        // The single handle↔uid boundary (issue #668): config-carried
        // identities resolve to source indices through this map at every
        // consumption site. `add_child_uid` above already rejected a
        // duplicate identity (RF.14), so this insert cannot clobber.
        self.source_uid_to_idx.insert(inertial_uid, idx);

        self.source_frame_ids.push(SourceFrameIds {
            inertial: inertial_id,
            pfix: pfix_id,
            central: entry.central,
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

    /// True if `frame_id` is the simulation's root frame or the inertial
    /// frame node of the central source — both are semantically the same
    /// integration origin. Post-#567 the central source's inertial frame
    /// is structurally distinct from root, but it sits at identity
    /// rotation and zero position relative to root, and `JEOD_INV: RF.13`'s
    /// composition fast-path keeps walks through it bit-identical to walks
    /// through root. Callers that previously checked
    /// `frame == self.root_frame_id` as a "is this the root inertial
    /// origin?" predicate should use this helper instead.
    pub fn is_root_equivalent_frame(&self, frame_id: FrameId) -> bool {
        if frame_id == self.root_frame_id {
            return true;
        }
        self.source_frame_ids
            .iter()
            .any(|sf| sf.central && sf.inertial == frame_id)
    }

    /// The inertial-frame identity of a gravity source (issue #668) —
    /// the value mission config references the source by
    /// (`GravityControl::new_spherical(sim.source_uid(idx).clone(), …)`,
    /// `FrameSwitchConfig { target, … }`). The reverse of the boundary
    /// map maintained by [`Self::add_source_with_uids`].
    ///
    /// # Panics
    /// Panics if `source_idx` is out of range.
    pub fn source_uid(&self, source_idx: usize) -> &astrodyn::FrameUid {
        let sf = self.source_frame_ids.get(source_idx).unwrap_or_else(|| {
            panic!(
                "source_uid: source index {source_idx} is out of range; \
                 {} source(s) configured",
                self.num_sources()
            )
        });
        self.frame_tree.get(sf.inertial).uid()
    }

    /// Resolve a source identity to its registration index, or `None`
    /// if no source carries it. The single uid → handle resolution the
    /// runner's consumption sites go through (issue #668).
    pub(crate) fn source_idx_for_uid(&self, uid: &astrodyn::FrameUid) -> Option<usize> {
        self.source_uid_to_idx.get(uid).copied()
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
    /// inertial frame, already wrapped as `Position<RootInertial>`. The
    /// frame phantom matches the runner's root-inertial convention; per-
    /// step pipelines that consume gravity / SRP / shadow inputs in
    /// `RootInertial` should call this rather than re-lifting the raw
    /// `DVec3` at every body. Returns the typed zero for the root-mapped
    /// central source.
    pub fn source_position_typed(
        &self,
        source_idx: usize,
    ) -> astrodyn::Position<astrodyn::RootInertial> {
        // allowed: typed-sibling accessor boundary. The frame phantom
        // is the runner-wide convention (root frame is inertial); the
        // raw kernel returns a frame-erased `DVec3` and the only source
        // of truth for the phantom is the runner's documented root
        // configuration.
        astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(sim_source_position(
            &self.frame_tree,
            &self.source_frame_ids,
            self.root_frame_id,
            source_idx,
        ))
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

    /// Get the planet-fixed rotation for a gravity source as a typed
    /// `FrameTransform<RootInertial, PlanetFixed<P>>`, matching the canonical
    /// shape used by `PlanetFixedRotationC<P>` on the Bevy adapter side. Returns
    /// `None` if the source has no rotation model (no pfix frame). The
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
    /// Post-#567 every source — central or not — has its own inertial frame
    /// node as a distinct child of root. The central source's inertial node
    /// sits at identity rotation and zero position relative to root, so
    /// attaching to it is semantically the same as attaching to root (see
    /// [`is_root_equivalent_frame`](Self::is_root_equivalent_frame) and
    /// `JEOD_INV: RF.13`).
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
