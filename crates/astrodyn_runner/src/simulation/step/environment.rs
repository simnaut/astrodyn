//! Stages 4 + 4b + 5 of [`super::super::Simulation::step_internal`]:
//! gravity (Newtonian + post-Newtonian PPN relativistic correction)
//! and atmosphere evaluation. Reads source positions and frame tree
//! (stage 2/2b output); writes per-body `gravity_accel` and
//! `atmospheric_state`.
//!
//! JEOD_INV: TS.01 — this module is the atmosphere-stage adapter for
//! the runner's planet-erased storage. `SimBody.atmospheric_state` is
//! `AtmosphereState<SelfPlanet>` because the runner's atmosphere
//! planet identity is keyed by a runtime source index
//! (`atmosphere_planet_source: Option<usize>`), not by a compile-time
//! `<P: Planet>` parameter. The `<SelfPlanet>`-tagged kernel call and
//! the `Position<PlanetInertial<SelfPlanet>>` relabel below are the
//! storage-boundary uses sanctioned by row TS.01.

use astrodyn::atmosphere::{run_atmosphere_stage, AtmosphereBodyInputs};
use astrodyn::gravity::{run_gravity_stage, GravityBodyInputs};
use astrodyn::IntegOrigin;
use astrodyn::SelfPlanet;

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
        // Identity → index boundary map (issue #668).
        let uid_to_idx = &self.source_uid_to_idx;
        // Pre-#567 these closures short-circuited the position lookup when
        // `sfids.inertial == root_fid` (the central body alias). With #567
        // making every source's inertial frame structurally distinct from
        // root, the frame-tree lookup returns the same `DVec3::ZERO` for
        // the central body anyway — read directly.
        let resolve_source =
            |_body_idx: usize, uid: &astrodyn::FrameUid| -> Option<astrodyn::ResolvedSource<'_>> {
                let source_id = *uid_to_idx.get(uid)?;
                let grav = gravity_data.get(source_id)?;
                let sfids = &source_frame_ids[source_id];
                let src_node = frame_tree.get(sfids.inertial);
                let position = src_node.state.trans.position;
                let rotation = sfids
                    .pfix
                    .map(|pfix_id| &frame_tree.get(pfix_id).state.rot.t_parent_this);
                Some(astrodyn::ResolvedSource {
                    source: &grav.source,
                    rotation,
                    position,
                    delta_c20: grav.delta_c20,
                    has_delta_coeffs: grav.tidal_config.is_some(),
                })
            };
        let resolve_rel_source = |_body_idx: usize,
                                  uid: &astrodyn::FrameUid|
         -> Option<astrodyn::ResolvedRelativisticSource> {
            let source_id = *uid_to_idx.get(uid)?;
            let grav = gravity_data.get(source_id)?;
            let sfids = &source_frame_ids[source_id];
            let position = frame_tree.get(sfids.inertial).state.trans.position;
            Some(astrodyn::ResolvedRelativisticSource {
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
        // moves the typed kernel result through unchanged — both
        // adapters now share the same
        // `GravityAccelerationTyped<RootInertial>` storage type.
        let body_iter = self.bodies.iter_mut().enumerate().map(|(body_idx, body)| {
            let inputs = GravityBodyInputs {
                position: body.trans.position,
                velocity: body.trans.velocity,
                integ_origin: body_integ_origins[body_idx],
                controls: &body.gravity_controls,
            };
            let gravity_accel_slot = &mut body.gravity_accel;
            let store =
                move |result: astrodyn::GravityAccelerationTyped<astrodyn::RootInertial>| {
                    *gravity_accel_slot = result;
                };
            (body_idx, inputs, store)
        });

        run_gravity_stage(body_iter, resolve_source, resolve_rel_source);

        // ── 5. Environment — atmosphere ──
        // RF.10 NOTE: atmosphere is an RF.10 *non-shift* site, the
        // opposite of the gravity stage above. The atmosphere kernel
        // rotates by `t_pfix` (the atmosphere planet's
        // `inertial → pfix` rotation) and computes geodetic altitude
        // relative to that planet's center. The required input frame
        // is *the atmosphere planet's inertial frame*, not the root
        // inertial frame. In every realistic config the body's
        // integration frame already is that planet's inertial frame
        // (e.g., body integrates in `Earth.inertial`, atmosphere
        // planet = Earth), so the `Position<IntegrationFrame>`
        // relabel-to-`PlanetInertial<SelfPlanet>` at the boundary is
        // bit-identical. Adding `body_integ_origins[idx].position`
        // would shift to root inertial and produce a wrong altitude
        // for any non-root-integrated body — so the atmosphere
        // driver explicitly does *not* take an `IntegOrigin`.
        if let Some(ref atmos_config) = self.atmosphere {
            let t_pfix = self
                .atmosphere_planet_source
                .and_then(|idx| self.source_frame_ids.get(idx))
                .and_then(|sfids| sfids.pfix)
                .map(|pfix_id| &self.frame_tree.get(pfix_id).state.rot.t_parent_this);
            let tai_tjt = Some(self.time.tai_tjt);

            // Project each `(idx, &mut SimBody)` row whose drag config
            // is active into the `(key, inputs, store)` triple
            // `run_atmosphere_stage` expects. `body.drag.is_some()` is
            // the runner analog of "entity has `DragConfigC`
            // (auto-inserts `AtmosphericStateC`)" in the Bevy
            // adapter — both gate the kernel call so non-drag bodies
            // keep their stored default. The store closure captures
            // `&mut body.atmospheric_state` and moves the typed
            // `<SelfPlanet>` result through unchanged; the Bevy
            // adapter, which stores `AtmosphericStateC<P>`, moves the
            // planet-tagged value through unchanged.
            let body_iter =
                self.bodies
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, body)| body.drag.is_some())
                    .map(|(body_idx, body)| {
                        let inputs = AtmosphereBodyInputs {
                        position: astrodyn::Position::<astrodyn::PlanetInertial<SelfPlanet>>::from_raw_si(body.trans.position.raw_si()), // allowed: atmosphere RF.10 non-shift boundary — body integrates in the atmosphere planet's inertial frame, so relabel `IntegrationFrame` → `PlanetInertial<SelfPlanet>` is bit-identical
                    };
                        let slot = &mut body.atmospheric_state;
                        let store = move |result: astrodyn::AtmosphereState<SelfPlanet>| {
                            *slot = result;
                        };
                        (body_idx, inputs, store)
                    });

            run_atmosphere_stage::<SelfPlanet, _, _, _>(body_iter, atmos_config, t_pfix, tai_tjt);
        }
    }
}
