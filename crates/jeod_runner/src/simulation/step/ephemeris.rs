//! Stages 2 + 2b of [`super::super::Simulation::step_internal`]:
//! ephemeris updates — planet-fixed rotations + frame-tree sync, then
//! source positions from DE4xx. Self-contained; mutates `frame_tree`
//! and `gravity_data` in place.

use glam::{DMat3, DVec3};

use jeod_sim::{JeodQuat, RotationModel};

use super::super::Simulation;
use crate::error::StepError;

impl Simulation {
    /// Sync a planet-fixed frame node's rotation state from a computed matrix.
    ///
    /// Sets `t_parent_this`, derives `q_parent_this` from it, and sets
    /// `ang_vel_this = [0, 0, planet_omega]` matching JEOD's `planet_rnp.cc`.
    /// The `planet_omega` value comes from `PlanetConfig::omega` via
    /// `GravityData::planet_omega`.
    fn sync_pfix_rotation(
        frame_tree: &mut jeod_frames::FrameTree,
        pfix_id: jeod_frames::FrameId,
        rotation: DMat3,
        planet_omega: f64,
    ) {
        let node = frame_tree.get_mut(pfix_id);
        node.state.rot.t_parent_this = rotation;
        node.state.rot.q_parent_this = JeodQuat::left_quat_from_transformation(&rotation);
        // JEOD sets ang_vel_this = [0, 0, planet_omega] in planet_rnp.cc.
        // This is used by compute_relative_state velocity composition.
        node.state.rot.ang_vel_this = DVec3::new(0.0, 0.0, planet_omega);
    }

    /// Stages 2 + 2b — update planet-fixed rotations and DE4xx source
    /// positions in the frame tree, plus tidal ΔC20 on each gravity
    /// source. JEOD_INV: DM.13 — ephemeris updated before gravity.
    pub(super) fn update_ephemeris(&mut self) -> Result<(), StepError> {
        // Per-source rotation dispatch: each source has its own rotation model.
        // Lazy-compute Earth RNP only if needed (most common case).
        let mut earth_rotation: Option<DMat3> = Option::None;
        for (i, grav) in self.gravity_data.iter_mut().enumerate() {
            match grav.rotation_model {
                RotationModel::None => {}
                RotationModel::EarthRNP => {
                    let rotation = *earth_rotation.get_or_insert_with(|| {
                        jeod_sim::compute_t_parent_this_from_tjt_with_polar(
                            self.time.gmst_seconds,
                            self.time.tt_tjt(),
                            self.polar_motion,
                        )
                    });
                    // Sync to frame tree pfix node.
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
                    }
                }
                RotationModel::MarsIAU => {
                    // JEOD's RNPMars receives TT seconds since J2000 (time_tt.seconds).
                    let tt_s_since_j2000 = (self.time.tt_tjt() - jeod_time::epoch::J2000_TT_TJT)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
                    }
                }
                RotationModel::MoonIAU => {
                    let tdb_jd = self.time.tdb_julian_date();
                    let tdb_s_since_j2000 = (tdb_jd - jeod_time::epoch::J2000_TT_JD)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
                    }
                }
                RotationModel::MoonDE421 => {
                    let eph = self.ephemeris.as_ref().expect(
                        "MoonDE421 rotation requires ephemeris with BPC. \
                         Set sim.ephemeris = Some(eph) after calling eph.load_bpc().",
                    );
                    let tdb_jd = self.time.tdb_julian_date();
                    let rotation = eph
                        .get_body_rotation(jeod_sim::EphemerisBody::Moon, tdb_jd)
                        .expect("Moon DE421 BPC rotation query failed");
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
                    }
                }
            }
            // Compute tidal ΔC20 if configured; otherwise clear any stale value.
            if let Some(ref config) = grav.tidal_config {
                let pfix_id = self.source_frame_ids[i]
                    .pfix
                    .expect("tidal_config requires a planet-fixed frame (set rotation_model or t_inertial_pfix).");
                let rotation = self.frame_tree.get(pfix_id).state.rot.t_parent_this;
                grav.delta_c20 = jeod_gravity::tides::compute_delta_c20(config, &rotation);
            } else {
                grav.delta_c20 = 0.0;
            }
        }

        // ── 2b. Ephemeris update — source positions from DE4xx ──
        // Update source positions from ephemeris each step and sync to frame tree.
        if let Some(ref eph) = self.ephemeris {
            let tdb_jd = self.time.tdb_julian_date();
            for i in 0..self.source_ephem_bodies.len() {
                if let Some(Some((target, observer))) = self.source_ephem_bodies.get(i) {
                    let (pos_typed, vel_typed) = eph
                        .get_state_typed(*target, *observer, tdb_jd)
                        .map_err(|e| StepError::EphemerisLookup {
                            source_idx: i,
                            target: *target,
                            observer: *observer,
                            tdb_jd,
                            message: e.to_string(),
                        })?;
                    let (pos, vel) = (pos_typed.raw_si(), vel_typed.raw_si());
                    // Root-mapped sources cannot consume ephemeris position updates:
                    // the root frame must remain identity, so accepting such a
                    // mapping would silently ignore `pos` and yield an incorrect
                    // source position.
                    let fid = self.source_frame_ids[i].inertial;
                    assert!(
                        fid != self.root_frame_id,
                        "Invalid ephemeris mapping for source {i} \
                         ({target:?} wrt {observer:?}): source inertial frame is the root frame, \
                         whose state must remain identity. Root-mapped sources cannot use \
                         ephemeris position updates."
                    );
                    // Update frame tree node with ephemeris position/velocity.
                    let node = self.frame_tree.get_mut(fid);
                    node.state.trans.position = pos;
                    node.state.trans.velocity = vel;
                    // Also update gravity_data velocity for relativistic corrections.
                    self.gravity_data[i].velocity = vel;
                }
            }
        }

        Ok(())
    }
}
