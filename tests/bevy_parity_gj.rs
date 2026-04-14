//! Bevy App vs jeod_runner::Simulation parity for Gauss-Jackson integrator.
//!
//! Mirrors each test in `crates/jeod_sim/tests/tier3_sim_gj.rs` with a
//! Bevy-vs-Simulation bit-identical assertion, establishing:
//!   Bevy ≡ Simulation ≈ JEOD
//!
//! Also covers bootstrap (ndoubling > 0) and time_scale_factor != 1.0 paths
//! that the older cross_parity.rs GJ tests (Scenario I) do not exercise.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, GaussJacksonStateC, GravityControlsC, GravitySourceC, IntegratorTypeC,
    JeodPlugin, SimulationTimeR, SourceInertialPositionC, TranslationalStateC,
};
use glam::DVec3;
use jeod_runner::{GravitySourceEntry, Simulation, VehicleConfig};
use jeod_sim::{
    GaussJacksonConfig, GaussJacksonState, GravityControl, GravityControls, GravityModel,
    GravitySource, IntegratorType, TranslationalState,
};

/// Non-standard μ matching SIM_GJ_test (same as tier3_sim_gj.rs).
const MU_GJ_TEST: f64 = 5.76e14;

/// GJ test initial state matching SIM_GJ_test: r₀=[9e6,0,0], v₀=[0,8000,0].
fn gj_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(9e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 8000.0, 0.0),
    }
}

// ── Helpers ──

fn step_bevy(app: &mut App, n: usize, dt: f64) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

fn read_trans(world: &World, entity: Entity) -> TranslationalState {
    world.get::<TranslationalStateC>(entity).unwrap().0
}

fn assert_bits_eq(label: &str, component: &str, a: f64, b: f64) {
    assert!(
        a.to_bits() == b.to_bits(),
        "{label} {component} not bit-identical:\n  \
         A: {a} (bits={:#018x})\n  \
         B: {b} (bits={:#018x})",
        a.to_bits(),
        b.to_bits(),
    );
}

fn assert_trans_eq(label: &str, a: &TranslationalState, b: &TranslationalState) {
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("position[{i}]"),
            a.position[i],
            b.position[i],
        );
        assert_bits_eq(
            label,
            &format!("velocity[{i}]"),
            a.velocity[i],
            b.velocity[i],
        );
    }
    println!("  {label}: bit-identical (all 6 components)");
}

/// Run a GJ Bevy-vs-Simulation parity test.
///
/// Builds identical Bevy App and Simulation with the given GJ config,
/// sim_dt, time_scale_factor, and step count; asserts bit-identical
/// translational state.
fn run_gj_parity(
    label: &str,
    config: GaussJacksonConfig,
    sim_dt: f64,
    time_scale_factor: f64,
    n_steps: usize,
) {
    let trans = gj_trans();

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(JeodPlugin);

    // Set time_scale_factor on the SimulationTimeR resource.
    app.world_mut()
        .resource_mut::<SimulationTimeR>()
        .0
        .time_scale_factor = time_scale_factor;

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Planet"),
            GravitySourceC(GravitySource {
                mu: MU_GJ_TEST,
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
            TranslationalStateC(trans),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            IntegratorTypeC(IntegratorType::GaussJackson(config)),
            GaussJacksonStateC(GaussJacksonState::new(config)),
        ))
        .id();

    step_bevy(&mut app, n_steps, sim_dt);
    let bevy_trans = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let mut time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    time.time_scale_factor = time_scale_factor;
    let mut sim = Simulation::new(time, sim_dt);

    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_GJ_TEST,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);

    sim.add_body(VehicleConfig {
        trans,
        integrator: IntegratorType::GaussJackson(config),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(n_steps);

    let sim_trans = sim.body(0).trans;
    assert_trans_eq(label, &bevy_trans, &sim_trans);
}

// ── Tests mirroring tier3_sim_gj.rs ──

/// Mirrors `tier3_simulation_gj_order8`: GJ order 8, dt=1s, tsf=1.0.
#[test]
fn tier3_bevy_parity_gj_order8() {
    run_gj_parity(
        "GJ order 8, dt=1s",
        GaussJacksonConfig::with_order(8),
        1.0,
        1.0,
        1000,
    );
}

/// Mirrors `tier3_simulation_gj_order4`: GJ order 4, dt=1s, tsf=1.0.
#[test]
fn tier3_bevy_parity_gj_order4() {
    run_gj_parity(
        "GJ order 4, dt=1s",
        GaussJacksonConfig::with_order(4),
        1.0,
        1.0,
        1000,
    );
}

/// Mirrors `tier3_simulation_gj_order12`: GJ order 12, dt=1s, tsf=1.0.
#[test]
fn tier3_bevy_parity_gj_order12() {
    run_gj_parity(
        "GJ order 12, dt=1s",
        GaussJacksonConfig::with_order(12),
        1.0,
        1.0,
        1000,
    );
}

/// Mirrors `tier3_simulation_gj_dt10`: GJ order 8, sim_dt=1s, tsf=10.
/// Exercises time_scale_factor through both Bevy and Simulation pipelines.
#[test]
fn tier3_bevy_parity_gj_dt10() {
    run_gj_parity(
        "GJ order 8, sim_dt=1s, tsf=10",
        GaussJacksonConfig::with_order(8),
        1.0,
        10.0,
        1000,
    );
}

// ── Bootstrap tests (ndoubling > 0) ──

/// GJ with default config (initial=4, final=12, ndoubling=4).
/// Exercises full bootstrap subcycling through both pipelines.
#[test]
fn tier3_bevy_parity_gj_bootstrap_default() {
    run_gj_parity(
        "GJ default (ndoubling=4), dt=1s",
        GaussJacksonConfig::default(),
        1.0,
        1.0,
        500,
    );
}

/// GJ with standard config (initial=8, final=12, ndoubling=2).
#[test]
fn tier3_bevy_parity_gj_bootstrap_standard() {
    run_gj_parity(
        "GJ standard (ndoubling=2), dt=10s",
        GaussJacksonConfig::standard(),
        10.0,
        1.0,
        100,
    );
}

/// Bootstrap + time_scale_factor: default config with tsf=2.0.
/// Exercises both subcycling and time scaling through both pipelines.
#[test]
fn tier3_bevy_parity_gj_bootstrap_tsf() {
    run_gj_parity(
        "GJ default (ndoubling=4), dt=0.5s, tsf=2.0",
        GaussJacksonConfig::default(),
        0.5,
        2.0,
        1000,
    );
}
