//! Stage 9 of [`super::super::Simulation::step_internal`]: derived
//! states (orbital elements, Euler angles, LVLH frame, geodetic state,
//! solar beta, earth lighting). Runs after integration; reads the
//! post-integrated body state and writes per-body derived-state fields.

use glam::DVec3;

use super::super::Simulation;

impl Simulation {
    /// Stage 9 — derived state computation. Reads `sun_pos` / `moon_pos`
    /// (computed once in stage 6 and threaded through) so solar-beta /
    /// earth-lighting evaluations don't re-resolve them here. No output;
    /// mutates per-body derived-state fields in place.
    pub(super) fn compute_derived_states(
        &mut self,
        sun_pos: Option<DVec3>,
        moon_pos: Option<DVec3>,
    ) {
        let gravity_data = &self.gravity_data;

        for body in &mut self.bodies {
            // Orbital elements
            if let Some(src_idx) = body.orbital_elements_source {
                if let Some(mu) = gravity_data.get(src_idx).map(|g| g.source.mu) {
                    body.orbital_elements = jeod_sim::compute_orbital_elements(
                        mu,
                        body.trans.position,
                        body.trans.velocity,
                    )
                    .ok();
                } else {
                    body.orbital_elements = None;
                }
            }

            // Euler angles
            if let Some(seq) = body.euler_sequence {
                if let Some(ref rot) = body.rot {
                    body.euler_angles = Some(jeod_sim::compute_body_euler_angles(rot, seq));
                } else {
                    body.euler_angles = None;
                }
            }

            // LVLH frame
            if body.compute_lvlh {
                body.lvlh_frame = Some(jeod_sim::compute_body_lvlh_frame(
                    body.trans.position,
                    body.trans.velocity,
                ));
            }

            // Geodetic state
            if let Some((src_idx, r_eq, r_pol)) = body.geodetic_planet {
                let pfix_rot = self
                    .source_frame_ids
                    .get(src_idx)
                    .and_then(|sfids| sfids.pfix)
                    .map(|pfix_id| self.frame_tree.get(pfix_id).state.rot.t_parent_this);
                if let Some(t_pfix) = pfix_rot {
                    body.geodetic_state = Some(jeod_sim::compute_body_geodetic(
                        body.trans.position,
                        &t_pfix,
                        r_eq,
                        r_pol,
                    ));
                } else {
                    body.geodetic_state = None;
                }
            }

            // Solar beta
            if body.compute_solar_beta {
                if let Some(sp) = sun_pos {
                    body.solar_beta = Some(jeod_sim::compute_body_solar_beta(
                        body.trans.position,
                        body.trans.velocity,
                        sp,
                    ));
                } else {
                    body.solar_beta = None;
                }
            }

            // Earth lighting
            if let Some((earth_r, moon_r, sun_r)) = body.earth_lighting_config {
                if let (Some(sp), Some(mp)) = (sun_pos, moon_pos) {
                    body.earth_lighting =
                        Some(jeod_interactions::earth_lighting::compute_earth_lighting(
                            body.trans.position,
                            sp,
                            mp,
                            sun_r,
                            earth_r,
                            moon_r,
                        ));
                } else {
                    body.earth_lighting = None;
                }
            }
        }
    }
}
