//! End-to-end exercise of the branded [`Simulation::run`] surface (#163).
//!
//! Pins that the rounded-out `BrandedSimulation` covers the common
//! index-using methods: source registration, body registration, body
//! state setters, contact-pair registration (the most important
//! brand-protected entry point — two body indices that the type
//! system now refuses to mix across simulations), plus the non-index
//! forwarders (`num_bodies`, `num_sources`, `frame_tree`,
//! `body_frame_id`, `body_mass_id`, `source_frame`).
//!
//! Bit-exact behavior is not the point — the unbranded API already has
//! Tier 3 coverage. This test is the proof-of-life that the branded
//! path compiles, runs, and produces the same shape of result.

use astrodyn::{
    default_leap_second_table, ContactFacet, ContactMaterial, GravityControl, GravityControls,
    GravityGradient, GravityModel, GravitySource, GravitySourceEntry, IntegratorType, Position,
    RootInertial, RotationModel, SimulationTime, TranslationalStateTyped, VehicleConfig, Velocity,
    EARTH,
};
use astrodyn_runner::Simulation;
use glam::DVec3;

fn earth_central_source() -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu: EARTH.shape.mu,
            model: GravityModel::PointMass,
        },
        position: Position::<RootInertial>::zero(),
        velocity: Velocity::<RootInertial>::zero(),
        t_inertial_pfix: None,
        rotation_model: RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
        marker_only: false,
    }
}

fn leo_body(gravity_source_idx: usize, x_offset_m: f64) -> VehicleConfig {
    let altitude_m = 700_000.0;
    let radius_m = EARTH.shape.r_eq() + altitude_m;
    let speed_m_s = 7_504.567;
    VehicleConfig {
        trans: TranslationalStateTyped::<RootInertial> {
            position: Position::<RootInertial>::from_raw_si(DVec3::new(
                radius_m + x_offset_m,
                0.0,
                0.0,
            )),
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, speed_m_s, 0.0)),
        },
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                gravity_source_idx,
                GravityGradient::Skip,
            )],
        },
        ..Default::default()
    }
}

#[test]
fn run_exercises_rounded_out_branded_surface() {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let final_state = Simulation::run(time, 1.0, |mut sim| {
        let earth_idx = sim.add_source("Earth", earth_central_source());
        let body_a = sim.add_body(leo_body(earth_idx.into_raw(), 0.0));
        let body_b = sim.add_body(leo_body(earth_idx.into_raw(), 100.0));

        // Non-index forwarders + branded getters.
        assert_eq!(sim.num_sources(), 1);
        assert_eq!(sim.num_bodies(), 2);
        let earth_frame = sim.source_frame(earth_idx);
        let frame_a = sim.body_frame_id(body_a);
        let frame_b = sim.body_frame_id(body_b);
        assert_ne!(frame_a, frame_b, "each body owns a distinct frame node");
        assert_ne!(frame_a, earth_frame, "body frames are not the source frame");

        // Mass-tree id is `None` for non-tree-registered bodies — verify
        // the wrap returns the inner `None` rather than panicking.
        assert!(sim.body_mass_id(body_a).is_none());

        // SRP slot is absent for these bodies — wrap returns inner `None`.
        assert!(sim.srp_plate_temperatures(body_a).is_none());

        // Validate before stepping.
        sim.validate()
            .expect("the two-body LEO configuration must validate");

        sim.step().expect("one RK4 step on a clean LEO config");

        // Read back the integrated state via `body(idx)` — the snapshot
        // accessor that the branded API surfaces.
        let out_a = sim.body(body_a);
        let out_b = sim.body(body_b);
        let pos_a = out_a.trans.position.raw_si();
        let pos_b = out_b.trans.position.raw_si();
        assert!(
            pos_a.x > 6_000_000.0 && pos_b.x > 6_000_000.0,
            "both bodies remain on LEO radii after one step (got {pos_a:?}, {pos_b:?})"
        );

        // Hand back the computed delta so the closure's return value
        // actually depends on the branded surface.
        pos_b.x - pos_a.x
    });

    // Body B started 100 m further out; one step at LEO velocities does
    // not significantly close that gap.
    assert!(
        (final_state - 100.0).abs() < 10.0,
        "expected ~100 m radial separation post-step, got {final_state}"
    );
}

#[test]
fn run_register_contact_pair_takes_branded_body_indices() {
    // Contact-pair registration is the most important brand-protected
    // entry point: two body indices flow IN, and the type system now
    // refuses to mix them across simulations. Verify the wrap accepts
    // branded indices and delegates correctly.
    let time = SimulationTime::at_j2000(default_leap_second_table());
    Simulation::run(time, 1.0, |mut sim| {
        let earth_idx = sim.add_source("Earth", earth_central_source());
        let body_a = sim.add_body(leo_body(earth_idx.into_raw(), 0.0));
        let body_b = sim.add_body(leo_body(earth_idx.into_raw(), 100.0));

        let mat = ContactMaterial::jeod_spring(1.0, 1.0, 0.5);
        let facet = ContactFacet::point(DVec3::ZERO, 1.0, mat);

        // Branded inputs; same '`sim` brand on both indices is required
        // for the call to typecheck.
        sim.register_contact_pair(body_a, facet, body_b, facet);

        // Receipt is `()` — there is no contact-pair reader by index,
        // so branding the return would be pure type theater. The
        // unbranded `num_contact_pairs()` is reachable via the
        // `unbranded()` escape hatch when needed.
        assert_eq!(sim.unbranded().num_contact_pairs(), 1);
    });
}
