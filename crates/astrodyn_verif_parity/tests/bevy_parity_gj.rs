//! Bevy ↔ runner parity for the SIM_GJ_test Gauss-Jackson family.
//!
//! Two flavors of test live in this file:
//!
//! 1. **Trajectory parity** (orderN / dt10): the four variants whose
//!    JEOD reference CSV exists. Each is a one-liner over a
//!    `sim_gj::*` recipe — the parity trait
//!    ([`VerificationCaseParityExt::run_and_assert_parity`]) drives both
//!    the runner and a Bevy `App` from the same scenario factory and
//!    asserts bit-identical translational state at every reference
//!    checkpoint. Recipes live in
//!    `crates/astrodyn_verif_jeod/src/run_verification/sim_gj.rs`.
//!
//! 2. **Bootstrap-only parity** (`bootstrap_*`): three variants that
//!    exercise GJ's `ndoubling > 0` priming subcycle. There is no JEOD
//!    reference for these — they're a Bevy-mechanism stress test that
//!    pins the runner-vs-bevy correctness of GJ's bootstrap path
//!    independent of trajectory cross-validation. They stay
//!    hand-rolled because they don't fit the [`VerificationCase`]
//!    shape (no reference CSV, no JEOD oracle).
//!
//! See `crates/astrodyn_verif_jeod/tests/tier3_sim_gj.rs` for the
//! runner-vs-JEOD oracle that supplies the transitivity argument.
//!
//! [`VerificationCase`]: astrodyn_verif_jeod::verification::VerificationCase
//! [`VerificationCaseParityExt::run_and_assert_parity`]: astrodyn_verif_parity::VerificationCaseParityExt::run_and_assert_parity

use std::time::Duration;

use astrodyn::{
    GaussJacksonConfig, GaussJacksonState, GravityControl, GravityControls, GravityModel,
    GravityRole, GravitySource, IntegratorType, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_bevy::{
    AstrodynPlugin, DynamicsConfigC, GaussJacksonStateC, GravityControlsC, GravitySourceC,
    IntegratorTypeC, SimulationTimeR, SourceInertialPositionC, TranslationalStateC,
};
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_gj;
use astrodyn_verif_parity::VerificationCaseParityExt;
use bevy::prelude::*;

// ── Trajectory variants (recipe-based, mirroring tier3_sim_gj.rs) ──

/// Mirrors `tier3_simulation_gj_order8` — GJ order 8, dt=1 s, tsf=1.0.
#[test]
fn tier3_bevy_parity_gj_order8() {
    sim_gj::gj_order8().run_and_assert_parity::<astrodyn::Earth>();
}

/// Mirrors `tier3_simulation_gj_order4` — GJ order 4, dt=1 s, tsf=1.0.
#[test]
fn tier3_bevy_parity_gj_order4() {
    sim_gj::gj_order4().run_and_assert_parity::<astrodyn::Earth>();
}

/// Mirrors `tier3_simulation_gj_order12` — GJ order 12, dt=1 s, tsf=1.0.
#[test]
fn tier3_bevy_parity_gj_order12() {
    sim_gj::gj_order12().run_and_assert_parity::<astrodyn::Earth>();
}

/// Mirrors `tier3_simulation_gj_dt10` — GJ order 8, sim_dt=1 s, tsf=10.
/// Exercises `time_scale_factor` through both pipelines.
#[test]
fn tier3_bevy_parity_gj_dt10() {
    sim_gj::gj_dt10().run_and_assert_parity::<astrodyn::Earth>();
}

// ── Bootstrap-only variants (no JEOD reference; hand-rolled) ──

fn step_bevy(app: &mut App, n: usize, dt: f64) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

fn read_trans(world: &World, entity: Entity) -> TranslationalState {
    astrodyn::typed_bridge::trans_typed_to_raw(
        &world
            .get::<TranslationalStateC<astrodyn::Earth>>(entity)
            .unwrap()
            .0,
    )
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

/// Run a GJ Bevy-vs-Simulation parity test for a config that has no JEOD
/// reference (bootstrap variants). Builds identical Bevy App and
/// Simulation and asserts bit-identical translational state after
/// `n_steps` integration ticks.
fn run_gj_bootstrap_parity(
    label: &str,
    config: GaussJacksonConfig,
    sim_dt: f64,
    time_scale_factor: f64,
    n_steps: usize,
) {
    let trans = sim_gj::gj_initial_state();

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(AstrodynPlugin);
    app.world_mut()
        .resource_mut::<SimulationTimeR>()
        .0
        .time_scale_factor = time_scale_factor;

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Planet"),
            GravitySourceC(GravitySource {
                mu: sim_gj::MU_GJ_TEST,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();
    let vehicle = app
        .world_mut()
        .spawn((
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, GravityRole::Central)],
            }),
            IntegratorTypeC(IntegratorType::GaussJackson(config)),
            GaussJacksonStateC(GaussJacksonState::new(config)),
        ))
        .id();
    step_bevy(&mut app, n_steps, sim_dt);
    let bevy_trans = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let mut time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    time.time_scale_factor = time_scale_factor;
    let mut sim = Simulation::new(time, sim_dt);
    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: sim_gj::MU_GJ_TEST,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&trans),
        integrator: IntegratorType::GaussJackson(config),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth_idx,
                GravityRole::Central,
            )],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(n_steps).expect("step_n failed");
    let sim_trans = astrodyn::typed_bridge::trans_typed_to_raw(&sim.body(0).trans);

    assert_trans_eq(label, &bevy_trans, &sim_trans);
}

/// GJ with default config (initial=4, final=12, ndoubling=4).
/// Exercises full bootstrap subcycling through both pipelines.
#[test]
fn tier3_bevy_parity_gj_bootstrap_default() {
    run_gj_bootstrap_parity(
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
    run_gj_bootstrap_parity(
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
    run_gj_bootstrap_parity(
        "GJ default (ndoubling=4), dt=0.5s, tsf=2.0",
        GaussJacksonConfig::default(),
        0.5,
        2.0,
        1000,
    );
}
