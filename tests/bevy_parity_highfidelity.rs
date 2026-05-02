//! Bevy-vs-Simulation parity tests: high-fidelity physics (spherical harmonics,
//! tidal effects, polar motion, Gauss-Jackson integrator).

mod common;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, GaussJacksonStateC, GravityControlsC, GravitySourceC, IntegratorTypeC,
    PlanetFixedRotationC, PolarMotionR, SourceInertialPositionC, TidalConfigC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_runner::RotationModel;
use jeod_sim::{
    GaussJacksonConfig, GaussJacksonState, GravityControl, GravityControls, GravityModel,
    GravitySource, IntegratorType, TidalBody, TidalConfig, TranslationalState,
};
use jeod_sim::{GravitySourceEntry, VehicleConfig};

use common::*;

// ── Scenario F: Spherical harmonics 4x4 + RNP (requires JEOD_HOME) ──

#[test]
fn tier3_bevy_sh4x4_rnp() {
    println!("Scenario F: Spherical harmonics 4x4 + RNP");

    let sh_data = jeod_test_data::gravity_fixtures::load_ggm02c();
    let mu = sh_data.mu;

    let sh_source = GravitySource {
        mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(bevy_jeod::JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(sh_source.clone()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(jeod_sim::FrameTransform::from_matrix(DMat3::IDENTITY)),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            DynamicsConfigC(jeod_sim::DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_nonspherical(planet, 4, 4, false)],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: sh_source,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: jeod_sim::planet_config::EARTH.omega,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(earth_idx, 4, 4, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_state = sim.body(0).trans;

    assert_trans_eq("Bevy vs Sim (SH 4x4)", &bevy_state, &sim_state);
}

// ── Scenario J: Solid body tides ──

#[test]
fn tier3_bevy_tidal_sh4x4() {
    println!("Scenario J: SH 4x4 + RNP + solid body tides");

    let sh_data = jeod_test_data::gravity_fixtures::load_ggm02c();
    let mu = sh_data.mu;
    let radius = sh_data.radius;

    let moon_pos = DVec3::new(2.0e8, 3.0e8, 1.0e8);
    let sun_pos = DVec3::new(1.0e11, 0.5e11, 0.2e11);

    let tidal_config = TidalConfig {
        k2: jeod_sim::EARTH_K2,
        mu_primary: mu,
        radius_primary: radius,
        tidal_bodies: vec![
            TidalBody {
                mu: jeod_sim::MOON.shape.mu,
                position_inertial: moon_pos,
            },
            TidalBody {
                mu: jeod_sim::SUN.shape.mu,
                position_inertial: sun_pos,
            },
        ],
    };

    let sh_source = GravitySource {
        mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(bevy_jeod::JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(sh_source.clone()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(jeod_sim::FrameTransform::from_matrix(DMat3::IDENTITY)),
            TidalConfigC::from_untyped(&tidal_config),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            DynamicsConfigC(jeod_sim::DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_nonspherical(planet, 4, 4, false)],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: sh_source,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(DMat3::IDENTITY),
            rotation_model: RotationModel::EarthRNP,
            delta_c20: 0.0,
            tidal_config: Some(tidal_config),
            planet_omega: jeod_sim::planet_config::EARTH.omega,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(earth_idx, 4, 4, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_state = sim.body(0).trans;

    assert_trans_eq("Bevy vs Sim (SH 4x4 + tides)", &bevy_state, &sim_state);
    println!("  Bevy vs Sim SH 4x4 + tides: bit-identical");
}

// ── Polar motion parity ──

#[test]
fn tier3_bevy_run2p_polar_motion() {
    println!("Run2p polar motion parity");
    const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0);
    let xp = 0.06806 * ARCSEC_TO_RAD;
    let yp = 0.24156 * ARCSEC_TO_RAD;

    let mut app = new_bevy_app(DT);
    app.insert_resource(PolarMotionR { xp, yp });
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);

    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(earth_source(), DVec3::ZERO, None);
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    sim.polar_motion = Some((xp, yp));

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    assert_trans_eq(
        "Bevy vs Sim (polar motion)",
        &bevy_trans,
        &sim.body(0).trans,
    );
}

// ── Gauss-Jackson parity ──

fn run_gj_parity(label: &str, config: GaussJacksonConfig, dt: f64, n_steps: usize) {
    let gj_trans = TranslationalState {
        position: DVec3::new(9e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 8000.0, 0.0),
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(dt));
    app.add_plugins(bevy_jeod::JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            DynamicsConfigC::default(),
            TranslationalStateC::from(gj_trans),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            IntegratorTypeC(IntegratorType::GaussJackson(config)),
            GaussJacksonStateC(GaussJacksonState::new(config)),
        ))
        .id();

    step_bevy_dt(&mut app, n_steps, dt);
    let bevy_trans = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, dt);
    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);

    sim.add_body(VehicleConfig {
        trans: gj_trans,
        integrator: IntegratorType::GaussJackson(config),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(n_steps).expect("step_n failed");

    let sim_trans = sim.body(0).trans;

    assert_trans_eq(label, &bevy_trans, &sim_trans);
    println!("  {label}: bit-identical");
}

#[test]
fn tier3_bevy_gj_point_mass() {
    println!("Scenario I: GJ order 8, dt=10s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 8)",
        GaussJacksonConfig::with_order(8),
        DT,
        NUM_STEPS,
    );
}

#[test]
fn tier3_bevy_gj_order4() {
    println!("Scenario I-b: GJ order 4, dt=10s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 4)",
        GaussJacksonConfig::with_order(4),
        DT,
        NUM_STEPS,
    );
}

#[test]
fn tier3_bevy_gj_order12() {
    println!("Scenario I-c: GJ order 12, dt=10s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 12)",
        GaussJacksonConfig::with_order(12),
        DT,
        NUM_STEPS,
    );
}

#[test]
fn tier3_bevy_gj_dt1() {
    println!("Scenario I-d: GJ order 8, dt=1s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 8, dt=1s)",
        GaussJacksonConfig::with_order(8),
        1.0,
        1000,
    );
}
