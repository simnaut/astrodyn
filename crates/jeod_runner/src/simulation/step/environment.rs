//! Stages 4 + 4b + 5 of [`super::super::Simulation::step_internal`]:
//! gravity (Newtonian + post-Newtonian PPN relativistic correction)
//! and atmosphere evaluation. Reads source positions and frame tree
//! (stage 2/2b output); writes per-body `gravity_accel` and
//! `atmospheric_state`.

use glam::DVec3;

use jeod_sim::atmosphere::evaluate_atmosphere;
use jeod_sim::gravity::accumulate_gravity;

use super::super::Simulation;

impl Simulation {
    /// Stages 4 + 4b + 5 — gravity (Newtonian and post-Newtonian
    /// relativistic) plus atmosphere evaluation. `body_integ_origins`
    /// is the per-body integration-frame origin (position + velocity)
    /// resolved against the root frame; it is also reused by stage 8
    /// integration, so the caller pre-computes it once.
    pub(super) fn update_environment(&mut self, body_integ_origins: &[(DVec3, DVec3)]) {
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

        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let integ_origin = body_integ_origins[body_idx].0;
            body.gravity_accel = accumulate_gravity(
                body.trans.position + integ_origin,
                &body.gravity_controls,
                integ_origin,
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

        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let (origin, origin_vel) = body_integ_origins[body_idx];
            body.gravity_accel.grav_accel += jeod_sim::accumulate_relativistic_corrections(
                body.trans.position + origin,
                body.trans.velocity + origin_vel,
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

            for body in &mut self.bodies {
                if body.atmospheric_state.is_some() {
                    body.atmospheric_state = Some(evaluate_atmosphere(
                        atmos_config,
                        body.trans.position,
                        t_pfix,
                        tai_tjt,
                    ));
                }
            }
        }
    }
}
