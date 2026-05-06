//! Stages 2 + 2b of [`super::super::Simulation::step_internal`]:
//! ephemeris updates — planet-fixed rotations + frame-tree sync, then
//! source positions from DE4xx. Self-contained; mutates `frame_tree`
//! and `gravity_data` in place.

use glam::DMat3;

use jeod_frames::{
    nutation_j2000::nutation,
    precession_j2000::precession_matrix,
    rotation_j2000::{gast_rotation_matrix, polar_motion_matrix},
};
use jeod_math::matrix3x3_product_transpose_transpose;
use jeod_sim::{sync_pfix_rotation, RotationModel};

use super::super::Simulation;
use crate::error::StepError;

impl Simulation {
    /// Stages 2 + 2b — update planet-fixed rotations and DE4xx source
    /// positions in the frame tree, plus tidal ΔC20 on each gravity
    /// source. JEOD_INV: DM.13 — ephemeris updated before gravity.
    pub(super) fn update_ephemeris(&mut self) -> Result<(), StepError> {
        // Resolve the Earth RNP rotation up-front so the per-source loop
        // below can run with `gravity_data` borrowed mutably without
        // contending with the cache field on `self`. Honours
        // `earth_rnp_refresh_cadence_s`: 0.0 (default) recomputes
        // unconditionally per step; positive cadence reuses the cached
        // matrix until that many simulated seconds have elapsed since
        // the previous refresh — matching Trick's
        // `(LOW_RATE_ENV, "environment")` job-class scheduling on
        // `Earth_GGM05C_SimObject::rnp.update_rnp`
        // (`Base/earth_GGM05C_baseline.sm:114`). See JEOD_INV: PF.06
        // — RNP refresh cadence is configurable per simulation.
        //
        // Computing once here when *any* source uses `EarthRNP` is a
        // bit-identical refactor of the previous lazy
        // `Option::get_or_insert_with` pattern: the inertial → pfix
        // matrix is a function of (gmst_seconds, tt_tjt, polar_motion)
        // alone, so all `EarthRNP`-tagged sources receive the same
        // value either way. The `needs_earth_rnp` short-circuit keeps
        // configurations without an `EarthRNP` source from doing any
        // RNP work.
        let needs_earth_rnp = self
            .gravity_data
            .iter()
            .any(|g| matches!(g.rotation_model, RotationModel::EarthRNP));
        let earth_rotation: Option<DMat3> = if needs_earth_rnp {
            Some(self.refresh_earth_rnp())
        } else {
            None
        };
        for (i, grav) in self.gravity_data.iter_mut().enumerate() {
            match grav.rotation_model {
                RotationModel::None => {}
                RotationModel::EarthRNP => {
                    let rotation =
                        earth_rotation.expect("needs_earth_rnp implies earth_rotation is Some");
                    // Sync to frame tree pfix node.
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        sync_pfix_rotation(
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
                        sync_pfix_rotation(
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
                        sync_pfix_rotation(
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
                        sync_pfix_rotation(
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

    /// Resolve the Earth inertial → pfix rotation matrix at the current
    /// `simtime`, honouring [`Self::earth_rnp_refresh_cadence_s`].
    ///
    /// Mirrors JEOD's two-cadence RNP scheduling
    /// (`Base/earth_GGM05C_baseline.sm:114, 119`):
    ///
    /// - **Cadence == 0.0** (default): recompute the full RNP via
    ///   `compute_t_parent_this_from_tjt_with_polar` every call. The
    ///   cache is left untouched so the per-step path stays
    ///   bit-identical to the pre-cadence behaviour (no extra state,
    ///   no chance for the cache to drift from the freshly-computed
    ///   value if cadence is later flipped to non-zero).
    /// - **Cadence > 0.0**: refresh the slowly-varying components
    ///   (precession, nutation, polar motion → `pm^T × NP` and
    ///   `equa_of_equi`) at the configured cadence, but recompute the
    ///   GAST rotation and final composition every call. This matches
    ///   JEOD's `(LOW_RATE_ENV, "environment")` job class on
    ///   `update_rnp` plus `P_ENV("derivative")` job class on
    ///   `update_axial_rotation`. The first call (cache `None`) always
    ///   refreshes — the cache is populated lazily on demand. Time
    ///   reversal (`time_scale_factor < 0`) is handled by the
    ///   absolute value: an identical cache hit in either direction.
    // JEOD_INV: PF.06 — RNP refresh cadence configurable per simulation
    fn refresh_earth_rnp(&mut self) -> DMat3 {
        let cadence = self.earth_rnp_refresh_cadence_s;
        if cadence == 0.0 {
            // Fast path: bit-identical to the pre-cadence implementation.
            // `compute_t_parent_this_from_tjt_with_polar` does the full
            // composition (precession, nutation, GAST, polar motion)
            // each call.
            return jeod_sim::compute_t_parent_this_from_tjt_with_polar(
                self.time.gmst_seconds,
                self.time.tt_tjt(),
                self.polar_motion,
            );
        }

        let now = self.time.simtime;
        let needs_refresh = match self.earth_rnp_cache {
            Some(cache) => (now - cache.simtime_at_refresh).abs() >= cadence,
            None => true,
        };
        if needs_refresh {
            // Refresh the slowly-varying components: precession,
            // nutation, polar motion, and the cached `equa_of_equi`
            // that the per-step GAST recomputation consumes.
            // Convert TT TJT to Julian centuries since J2000:
            //   tt_centuries = (tt_tjt - 11544.5) / 36525.0.
            let tt_centuries = (self.time.tt_tjt() - 11544.5) / 36525.0;
            let prec = precession_matrix(tt_centuries);
            let nut = nutation(tt_centuries);
            let np = matrix3x3_product_transpose_transpose(&nut.rotation, &prec);
            let polar_t = self
                .polar_motion
                .map(|(xp, yp)| polar_motion_matrix(xp, yp).transpose());
            self.earth_rnp_cache = Some(super::super::EarthRnpCache {
                simtime_at_refresh: now,
                np,
                polar_t,
                equa_of_equi: nut.equa_of_equi,
            });
        }

        // Per-step GAST recomposition (mirrors JEOD's
        // `P_ENV("derivative") rnp.update_axial_rotation`):
        //   T_parent_this = polar^T × gast^T × NP
        //                 = gast^T × NP        (when polar motion absent)
        // The GAST angle is `(gmst_seconds + cached_equa_of_equi) / 240`
        // (degrees), recomputed every call from the freshly-advanced
        // `gmst_seconds`. `np` and `polar_t` are reused from the cache
        // until the next refresh boundary.
        let cache = self
            .earth_rnp_cache
            .expect("cache populated above when cadence > 0");
        let gast_t = gast_rotation_matrix(self.time.gmst_seconds, cache.equa_of_equi).transpose();
        match cache.polar_t {
            Some(polar_t) => polar_t * gast_t * cache.np,
            None => gast_t * cache.np,
        }
    }
}
