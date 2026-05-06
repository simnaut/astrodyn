//! Stages 4 + 4b + 5 of [`super::super::Simulation::step_internal`]:
//! gravity (Newtonian + post-Newtonian PPN relativistic correction)
//! and atmosphere evaluation. Reads source positions and frame tree
//! (stage 2/2b output); writes per-body `gravity_accel` and
//! `atmospheric_state`.

use glam::DVec3;

use jeod_quantities::IntegOrigin;
use jeod_sim::atmosphere::evaluate_atmosphere;
use jeod_sim::gravity::accumulate_gravity;

use super::super::Simulation;

impl Simulation {
    /// Stages 4 + 4b + 5 — gravity (Newtonian and post-Newtonian
    /// relativistic) plus atmosphere evaluation. `body_integ_origins`
    /// is the per-body integration-frame origin resolved against the
    /// root inertial frame; it is also reused by stage 8 integration,
    /// so the caller pre-computes it once.
    pub(super) fn update_environment(&mut self, body_integ_origins: &[IntegOrigin]) {
        // ── 4. Environment — gravity ──
        // Helper: resolve source to gravity data via frame tree.
        let gravity_data = &self.gravity_data;
        let source_frame_ids = &self.source_frame_ids;
        let frame_tree = &self.frame_tree;
        let root_fid = self.root_frame_id;
        let resolve_source = |source_id: usize| -> Option<jeod_sim::ResolvedSource<'_>> {
            let grav = gravity_data.get(source_id)?;
            let sfids = &source_frame_ids[source_id];
            let src_node = frame_tree.get(sfids.inertial);
            let position = if sfids.inertial == root_fid {
                DVec3::ZERO
            } else {
                src_node.state.trans.position
            };
            let rotation = sfids
                .pfix
                .map(|pfix_id| &frame_tree.get(pfix_id).state.rot.t_parent_this);
            Some(jeod_sim::ResolvedSource {
                source: &grav.source,
                rotation,
                position,
                delta_c20: grav.delta_c20,
                has_delta_coeffs: grav.tidal_config.is_some(),
            })
        };

        // JEOD_INV: RF.10 — shift body position from integration frame to
        // root inertial before passing to gravity / relativistic
        // consumers. (Atmosphere is a non-shift site — see the atmosphere
        // block below.) `IntegOrigin::zero()` is a no-op (bit-identical)
        // when the body integrates in the root frame.
        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let o = &body_integ_origins[body_idx];
            let inertial_state = body.trans.to_inertial(o);
            body.gravity_accel = accumulate_gravity(
                inertial_state.position.raw_si(),
                &body.gravity_controls,
                o.position.raw_si(),
                resolve_source,
            );
        }

        // ── 4b. Relativistic corrections ──
        // After Newtonian gravity, apply post-Newtonian PPN correction for
        // any source with `relativistic: true`. Folkner eq 27 (β=γ=1).
        // PPN uses inertial coordinates — convert from integration frame.
        let resolve_rel_source =
            |source_id: usize| -> Option<jeod_sim::ResolvedRelativisticSource> {
                let grav = gravity_data.get(source_id)?;
                let sfids = &source_frame_ids[source_id];
                let position = if sfids.inertial == root_fid {
                    DVec3::ZERO
                } else {
                    frame_tree.get(sfids.inertial).state.trans.position
                };
                Some(jeod_sim::ResolvedRelativisticSource {
                    mu: grav.source.mu,
                    position,
                    // Use velocity from gravity_data, not the tree node, because
                    // central bodies at the root frame have zero tree velocity
                    // but may have physical velocity for relativistic corrections.
                    velocity: grav.velocity,
                })
            };

        // JEOD_INV: RF.10 — relativistic correction needs both position and
        // velocity in root inertial coordinates.
        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let inertial_state = body.trans.to_inertial(&body_integ_origins[body_idx]);
            body.gravity_accel.grav_accel += jeod_sim::accumulate_relativistic_corrections(
                inertial_state.position.raw_si(),
                inertial_state.velocity.raw_si(),
                &body.gravity_controls,
                resolve_rel_source,
            );
        }

        // ── 5. Environment — atmosphere ──
        if let Some(ref atmos_config) = self.atmosphere {
            let t_pfix = self
                .atmosphere_planet_source
                .and_then(|idx| self.source_frame_ids.get(idx))
                .and_then(|sfids| sfids.pfix)
                .map(|pfix_id| &self.frame_tree.get(pfix_id).state.rot.t_parent_this);
            let tai_tjt = Some(self.time.tai_tjt);

            // RF.10 NOTE: atmosphere is *not* in the shift class.
            // `evaluate_atmosphere` rotates by `t_pfix` (the atmosphere
            // planet's `inertial → pfix` rotation) and computes geodetic
            // altitude relative to that planet's center. The required
            // input frame is *the atmosphere planet's inertial frame*,
            // not the root inertial frame. In every realistic config the
            // body's integration frame already is that planet's inertial
            // frame (e.g., body integrates in `Earth.inertial`,
            // atmosphere planet = Earth). So `body.trans.position`
            // (integration-frame coords) is the correct input. Adding
            // `body_integ_origins[idx].position` would shift to root and
            // produce wrong altitude for any non-root-integrated body.
            for body in &mut self.bodies {
                if body.atmospheric_state.is_some() {
                    body.atmospheric_state = Some(evaluate_atmosphere(
                        atmos_config,
                        body.trans.position.raw_si(),
                        t_pfix,
                        tai_tjt,
                    ));
                }
            }
        }
    }
}
