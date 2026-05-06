//! Stages 4 + 4b + 5 of [`super::super::Simulation::step_internal`]:
//! gravity (Newtonian + post-Newtonian PPN relativistic correction)
//! and atmosphere evaluation. Reads source positions and frame tree
//! (stage 2/2b output); writes per-body `gravity_accel` and
//! `atmospheric_state`.

use glam::DVec3;

use jeod_sim::atmosphere::evaluate_atmosphere;
use jeod_sim::gravity::{run_gravity_stage, GravityBodyInputs};
use jeod_sim::IntegOrigin;

use super::super::Simulation;

impl Simulation {
    /// Stages 4 + 4b + 5 — gravity (Newtonian and post-Newtonian
    /// relativistic) plus atmosphere evaluation. `body_integ_origins`
    /// is the per-body integration-frame origin resolved against the
    /// root inertial frame; it is also reused by stage 8 integration,
    /// so the caller pre-computes it once.
    pub(super) fn update_environment(&mut self, body_integ_origins: &[IntegOrigin]) {
        // ── 4 + 4b. Environment — gravity (Newtonian + relativistic) ──
        // Per-body gravity composition (RF.10 shift, Newtonian
        // accumulation, post-Newtonian correction) lives in the shared
        // `evaluate_body_gravity` kernel; both this runner and the Bevy
        // adapter call it inside their respective body loops, paying the
        // adapter cost once at the boundary (here: source resolvers
        // closing over the frame tree and gravity data).
        let gravity_data = &self.gravity_data;
        let source_frame_ids = &self.source_frame_ids;
        let frame_tree = &self.frame_tree;
        let root_fid = self.root_frame_id;
        let resolve_source =
            |_body_idx: usize, source_id: usize| -> Option<jeod_sim::ResolvedSource<'_>> {
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
        let resolve_rel_source =
            |_body_idx: usize, source_id: usize| -> Option<jeod_sim::ResolvedRelativisticSource> {
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
                    // Velocity comes from gravity_data, not the tree node,
                    // because central bodies at the root frame have zero
                    // tree velocity but may still have a physical velocity
                    // for relativistic corrections.
                    velocity: grav.velocity,
                })
            };

        // Project each `(idx, &mut SimBody)` row into the
        // `(key, inputs, store)` triple `run_gravity_stage` expects.
        // The store closure captures `&mut body.gravity_accel` and
        // lowers the typed kernel result back to raw `GravityAcceleration`
        // until #364 migrates `SimBody.gravity_accel` to the typed
        // sibling — at which point the lowering goes away and both
        // adapters write the typed value through unchanged.
        let body_iter = self.bodies.iter_mut().enumerate().map(|(body_idx, body)| {
            let inputs = GravityBodyInputs {
                position: body.trans.position,
                velocity: body.trans.velocity,
                integ_origin: body_integ_origins[body_idx],
                controls: &body.gravity_controls,
            };
            let gravity_accel_slot = &mut body.gravity_accel;
            let store =
                move |result: jeod_sim::GravityAccelerationTyped<jeod_sim::RootInertial>| {
                    gravity_accel_slot.grav_accel = result.grav_accel.raw_si();
                    gravity_accel_slot.grav_grad = result.grav_grad;
                    gravity_accel_slot.grav_pot = result.grav_pot;
                };
            (body_idx, inputs, store)
        });

        run_gravity_stage(body_iter, resolve_source, resolve_rel_source);

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
