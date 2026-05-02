//! Stage 9 of [`super::super::Simulation::step_internal`]: derived
//! states (orbital elements, Euler angles, LVLH frame, geodetic state,
//! solar beta, earth lighting). Runs after integration; reads the
//! post-integrated body state and writes per-body derived-state fields.

use jeod_sim::{IntegOrigin, Position, RootInertial};

use super::super::Simulation;

impl Simulation {
    /// Stage 9 — derived state computation. Reads `sun_pos` / `moon_pos`
    /// (computed once in stage 6 and threaded through) so solar-beta /
    /// earth-lighting evaluations don't re-resolve them here. No output;
    /// mutates per-body derived-state fields in place.
    ///
    /// `sun_pos` / `moon_pos` are typed `Position<RootInertial>` so any
    /// site that mixes them with body integration-frame state fails to
    /// compile (RF.10 structural guard).
    pub(super) fn compute_derived_states(
        &mut self,
        sun_pos: Option<Position<RootInertial>>,
        moon_pos: Option<Position<RootInertial>>,
        body_integ_origins: &[IntegOrigin],
    ) {
        let gravity_data = &self.gravity_data;

        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            // RF.10 distinction:
            //
            // - Orbital elements / LVLH / geodetic compute around a
            //   single gravitating body using its planet-inertial frame.
            //   In realistic configs the body's integration frame IS that
            //   planet's inertial frame, so `body.trans.position` is the
            //   correct input. NOT a shift site.
            //
            // - Solar beta and earth lighting consume `sun_pos` / `moon_pos`
            //   which are root-inertial; the body must be shifted to root
            //   for the geometry to compose correctly. Shift sites.
            //
            // The shift is computed lazily so non-shift sites pay nothing.
            let integ_origin = &body_integ_origins[body_idx];

            // Orbital elements (NOT a shift site)
            if let Some(src_idx) = body.orbital_elements_source {
                if let Some(mu) = gravity_data.get(src_idx).map(|g| g.source.mu) {
                    body.orbital_elements = jeod_sim::compute_orbital_elements(
                        mu,
                        body.trans.position.raw_si(),
                        body.trans.velocity.raw_si(),
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

            // LVLH frame (NOT a shift site)
            if body.compute_lvlh {
                body.lvlh_frame = Some(jeod_sim::compute_body_lvlh_frame(
                    body.trans.position.raw_si(),
                    body.trans.velocity.raw_si(),
                ));
            }

            // Geodetic state (NOT a shift site)
            if let Some((src_idx, r_eq, r_pol)) = body.geodetic_planet {
                let pfix_rot = self
                    .source_frame_ids
                    .get(src_idx)
                    .and_then(|sfids| sfids.pfix)
                    .map(|pfix_id| self.frame_tree.get(pfix_id).state.rot.t_parent_this);
                if let Some(t_pfix) = pfix_rot {
                    body.geodetic_state = Some(jeod_sim::compute_body_geodetic(
                        body.trans.position.raw_si(),
                        &t_pfix,
                        r_eq,
                        r_pol,
                    ));
                } else {
                    body.geodetic_state = None;
                }
            }

            // Solar beta — JEOD_INV: RF.10 — uses root-inertial sun_pos.
            // The typed sibling `compute_body_solar_beta_typed` takes
            // `Position<RootInertial>` and `Velocity<RootInertial>`, so any
            // attempt to pass a body integration-frame value is a compile
            // error. Structural enforcement of the shift.
            if body.compute_solar_beta {
                if let Some(sp) = sun_pos {
                    let inertial_state = body.trans.to_inertial(integ_origin);
                    body.solar_beta = Some(
                        jeod_sim::compute_body_solar_beta_typed(
                            inertial_state.position,
                            inertial_state.velocity,
                            sp,
                        )
                        .get::<uom::si::angle::radian>(),
                    );
                } else {
                    body.solar_beta = None;
                }
            }

            // Earth lighting — JEOD_INV: RF.10 — uses root-inertial sun_pos / moon_pos.
            // Typed sibling enforces matching `RootInertial` frames at all
            // three position arguments.
            if let Some((earth_r, moon_r, sun_r)) = body.earth_lighting_config {
                if let (Some(sp), Some(mp)) = (sun_pos, moon_pos) {
                    let inertial_pos_typed = body.trans.to_inertial(integ_origin).position;
                    body.earth_lighting = Some(jeod_interactions::compute_earth_lighting_typed(
                        inertial_pos_typed,
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
