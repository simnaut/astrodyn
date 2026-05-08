//! Stage 9 of [`super::super::Simulation::step_internal`]: derived
//! states (orbital elements, Euler angles, LVLH frame, geodetic state,
//! solar beta, earth lighting). Runs after integration; reads the
//! post-integrated body state and writes per-body derived-state fields.

use astrodyn::{Position, RootInertial};
use astrodyn_quantities::IntegOrigin;

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
                    body.orbital_elements = astrodyn::compute_orbital_elements(
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
                    body.euler_angles = Some(astrodyn::compute_body_euler_angles(rot, seq));
                } else {
                    body.euler_angles = None;
                }
            }

            // LVLH frame (NOT a shift site)
            if body.compute_lvlh {
                body.lvlh_frame = Some(astrodyn::compute_body_lvlh_frame(
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
                    body.geodetic_state = Some(astrodyn::compute_body_geodetic(
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
                        astrodyn::compute_body_solar_beta_typed(
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
                    body.earth_lighting =
                        Some(astrodyn_interactions::compute_earth_lighting_typed(
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

#[cfg(test)]
mod tests {
    use crate::Simulation;
    use astrodyn::{
        default_leap_second_table, DerivedStateConfig, FrameSwitchConfig, GravityControl,
        GravityControls, GravityModel, GravitySource, GravitySourceEntry, MassProperties, Position,
        RotationModel, RotationalState, SimulationTime, SwitchSense, TranslationalState,
        VehicleConfig, Velocity,
    };
    use astrodyn_math::JeodQuat;
    use glam::DVec3;

    /// Stage 8b's frame-switch evaluation can change `body.integ_frame_id`
    /// and rewrite `body.trans` into the new integration frame's
    /// coordinates. Stage 9's derived-state computation must read the
    /// matching new-frame integ origin when calling
    /// `body.trans.to_inertial(integ_origin)` (solar beta, earth lighting),
    /// not the pre-step snapshot — otherwise the body-to-root shift
    /// mixes new-frame body coords with the old-frame origin and
    /// silently produces a root-inertial value off by the inter-source
    /// separation distance.
    ///
    /// This test pins the fix by constructing a config where stage 8b
    /// is guaranteed to fire on the first step and the inter-source
    /// separation (1e9 m) dwarfs any other position scale, so the
    /// stale-origin solar beta is geometrically far from the correct
    /// post-switch one. With the bug, the recorded `solar_beta` would
    /// be computed from a body position 1e9 m away from the truth.
    #[test]
    fn frame_switch_recomputes_integ_origins_for_derived_states() {
        // Sources: Origin at root, Far offset by +1e9 m in root inertial.
        // Both `mu = 0` so no gravity perturbs the body — the test isolates
        // the stage-8b → stage-9 origin handoff.
        let dt = 0.1;
        let time = SimulationTime::at_j2000(default_leap_second_table());
        let mut sim = Simulation::new(time, dt);
        let origin_src = sim.add_source(
            "Origin",
            GravitySourceEntry {
                source: GravitySource {
                    mu: 0.0,
                    model: GravityModel::PointMass,
                },
                position: Position::zero(),
                velocity: Velocity::zero(),
                t_inertial_pfix: None,
                rotation_model: RotationModel::None,
                delta_c20: 0.0,
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );
        // Source 1: 1e9 m offset in +X. Marked non-central so the body's
        // gravity_controls flip on switch (mirrors realistic frame-switch
        // wiring; the test does not depend on the flip).
        let far_src = sim.add_source(
            "Far",
            GravitySourceEntry {
                source: GravitySource {
                    mu: 0.0,
                    model: GravityModel::PointMass,
                },
                position: Position::from_raw_si(DVec3::new(1.0e9, 0.0, 0.0)),
                velocity: Velocity::zero(),
                t_inertial_pfix: None,
                rotation_model: RotationModel::None,
                delta_c20: 0.0,
                tidal_config: None,
                planet_omega: 0.0,
                central: false,
            },
        );
        // Sun used for solar beta. Placed far in +X so the sun direction
        // (sun − body) is dominated by +X. Combined with the body's Y/Z
        // velocity components below, this makes `solar_beta = asin(h_hat
        // · sun_hat)` highly sensitive to the body's true root-inertial
        // X coordinate — the bug case (X ≈ 50 m) and the correct case
        // (X ≈ 1e9 m) produce h directions that differ by ~50 degrees.
        let sun_src = sim.add_source(
            "Sun",
            GravitySourceEntry {
                source: GravitySource {
                    mu: 0.0,
                    model: GravityModel::PointMass,
                },
                position: Position::from_raw_si(DVec3::new(1.5e11, 0.0, 0.0)),
                velocity: Velocity::zero(),
                t_inertial_pfix: None,
                rotation_model: RotationModel::None,
                delta_c20: 0.0,
                tidal_config: None,
                planet_omega: 0.0,
                central: false,
            },
        );
        sim.sun_source = Some(sun_src);

        // Body integrates in Origin (root). Place it close to Far's
        // origin in absolute root-inertial coords so OnApproach fires
        // on the very first stage-8b evaluation. The body's Y position
        // and the velocity's Z component give an `h = r × v` whose
        // direction depends materially on the X component of `r`,
        // making `asin(h_hat · sun_hat)` differ by ~1 rad between the
        // bug case (root-inertial X ≈ 50) and the correct case
        // (X ≈ 1e9 + 50).
        let body_root_pos = DVec3::new(1.0e9 + 50.0, 100.0, 0.0);
        let body_root_vel = DVec3::new(0.0, 7500.0, 1000.0);
        let body_idx = sim.add_body(VehicleConfig {
            trans: TranslationalState {
                position: body_root_pos,
                velocity: body_root_vel,
            }
            .into(),
            rot: Some(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
            mass: Some(MassProperties::new(1.0)),
            // Both sources controlled so the post-switch flip has
            // somewhere to write — the dynamics don't depend on these
            // values (mu=0 on both).
            gravity_controls: GravityControls {
                controls: vec![
                    GravityControl::new_spherical(origin_src, false),
                    GravityControl::new_spherical(far_src, true),
                ],
            },
            integ_source: None,
            frame_switches: vec![FrameSwitchConfig {
                target_source: far_src,
                switch_sense: SwitchSense::OnApproach,
                switch_distance: 1.0e3,
                active: true,
            }],
            derived: DerivedStateConfig {
                solar_beta: true,
                ..Default::default()
            },
            ..Default::default()
        });

        sim.step().expect("step() must succeed");

        // Confirm the switch actually happened — otherwise the test is
        // not exercising the regression path.
        let post_integ_fid = sim.bodies[body_idx].integ_frame_id;
        let (post_origin_pos, _post_origin_vel) = sim.frame_origin(post_integ_fid);
        assert!(
            (post_origin_pos.x - 1.0e9).abs() < 1.0,
            "frame switch did not fire: post-step integ origin = {post_origin_pos:?}; \
             expected ~(1e9, 0, 0). Test setup would not exercise the regression."
        );

        // Independently compute the expected solar beta using the
        // *post-switch* root-inertial body state. Reuses the same
        // `compute_body_solar_beta_typed` kernel as production so the
        // test pins the integ-origin handoff, not the kernel itself.
        // After one RK4 step of zero gravity / zero force, the
        // root-inertial body state is body_root_pos + body_root_vel * dt
        // (free-flight). We evaluate solar beta at that state.
        let expected_root_pos = body_root_pos + body_root_vel * dt;
        let expected_root_vel = body_root_vel;
        let sun_pos = DVec3::new(1.5e11, 0.0, 0.0);
        let expected_beta = astrodyn::compute_body_solar_beta_typed(
            Position::from_raw_si(expected_root_pos),
            Velocity::from_raw_si(expected_root_vel),
            Position::from_raw_si(sun_pos),
        )
        .get::<uom::si::angle::radian>();

        let actual_beta = sim.bodies[body_idx]
            .solar_beta
            .expect("solar_beta must be populated when compute_solar_beta = true");

        assert!(
            (actual_beta - expected_beta).abs() < 1e-12,
            "solar_beta does not match the post-switch root-inertial geometry. \
             actual = {actual_beta}, expected = {expected_beta}. Likely cause: \
             stage 9 received the pre-integration `body_integ_origins` snapshot \
             instead of the post-stage-8b origins, so `body.trans` (now in Far \
             coords) was shifted with the Origin root snapshot."
        );

        // Defense-in-depth: independently compute the bug-shape solar
        // beta — what would be reported if stage 9 used the *old* root
        // origin (zero) with the new-frame `body.trans` coords. This
        // value must be measurably different from the correct one;
        // otherwise the regression assert above is vacuous.
        let bug_root_pos = sim.bodies[body_idx].trans.position.raw_si(); // in Far coords ≈ (50 + dv·dt, ...)
        let bug_beta = astrodyn::compute_body_solar_beta_typed(
            Position::from_raw_si(bug_root_pos),
            Velocity::from_raw_si(expected_root_vel),
            Position::from_raw_si(sun_pos),
        )
        .get::<uom::si::angle::radian>();
        assert!(
            (bug_beta - expected_beta).abs() > 1e-3,
            "test setup invariant: bug-shape solar beta must differ from the correct \
             one by a measurable amount. bug = {bug_beta}, expected = {expected_beta}. \
             A weaker separation would let a regression slip past the primary assert."
        );
    }
}
