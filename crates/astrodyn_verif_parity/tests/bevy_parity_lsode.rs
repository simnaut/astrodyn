//! Bevy ↔ runner parity for the `RUN_lsode` portion of JEOD's
//! `SIM_integ_test` — the variable-order/variable-step LSODE method.
//!
//! The runner-side oracle is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_lsode.rs::tier3_simulation_lsode_default`,
//! which cross-validates our ported LSODE (`ImplicitAdamsNonStiff`,
//! `rtol = 2.3e-16`, `atol = 0`) against JEOD's LSODE on a Kepler orbit
//! using Trick's `TimeDyn::scale_factor` (≈ 15.54 dyn-seconds per
//! sim-second, one degree of orbital phase per sim step).
//!
//! This wrapper carries the `bevy ≡ runner` half of the
//! `bevy ≡ runner ≈ JEOD` transitivity argument for the LSODE method:
//! both runtimes integrate the same Kepler orbit with the same
//! `LsodeConfig` and the same `time_scale_factor`, and bit-identity at
//! every checkpoint over the full 80 000 s window is the contract.
//! Unlike the companion `bevy_parity_lsode_abm4.rs` (which mirrors
//! `RUN_abm4` only — LSODE was unported when it was written), this file
//! exercises the `LsodeStateC` Bevy component path landed in #200 Phase
//! 6B: the integrator's persistent Nordsieck history must thread through
//! the ECS exactly as it does through the runner's `SimBody::lsode_state`.
//!
//! ## Why not a recipe?
//!
//! `SIM_integ_test` derives μ from `sma`/`mean_motion`, takes initial
//! conditions from the deterministically-rotated `prop_integ_state` row
//! (the t=0 line of the reference CSV — a JEOD source value), and applies
//! a `time_scale_factor`. None of these match a recipe preset; the
//! runner-side test in `tier3_sim_lsode.rs` is annotated `non-recipe:`
//! for the same reason. This wrapper mirrors the bootstrap pattern in
//! `bevy_parity_lsode_abm4.rs`: build identical `SimulationBuilder`s for
//! both runtimes and assert bit-identity at every checkpoint, with no
//! `VerificationCase` factory in between.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy state fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test step counts fit exactly in f64 mantissa and usize"
)]

use std::path::Path;
use std::time::Duration;

use astrodyn::{
    default_leap_second_table, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, GravitySourceEntry, IntegratorType, LsodeConfig, Position, RootInertial,
    SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig, Velocity,
};
use astrodyn_bevy::{SimulationBuilderBevyExt, TranslationalStateC};
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use bevy::prelude::*;
use glam::DVec3;

/// Orbital parameters from `TranslationTestOrbit` defaults
/// (`translation_test.hh` member initializers); duplicated here from
/// `tier3_sim_lsode.rs` so the parity wrapper is standalone.
const SMA: f64 = 6_811_137.0; // m
const MDOT: f64 = 1.123_154_395_240_404_1e-3; // rad/s
/// Dynamics timestep used by `SIM_integ_test` (`S_define`: DYNAMICS = 1.00).
const SIM_DT: f64 = 1.0;

/// `IntegrationTest::initialize` (integration_test.cc:167-172) computes
/// `delta_t = omega_dt / omega` and `time_scale = delta_t / sim_dt` so each
/// sim-step represents 1° of orbital phase.
fn compute_time_scale() -> f64 {
    // omega_dt = 1° = π/180 rad, omega = MDOT rad/s, sim_dt = 1 s.
    (std::f64::consts::PI / 180.0) / MDOT / SIM_DT
}

/// μ derived from `sma` and mean motion (matches `TranslationTestOrbit`
/// in JEOD; the artificial value is intentional — the verification sim
/// picks it so the integrator is exercised in isolation, not gravitational
/// model matching).
fn compute_mu() -> f64 {
    SMA * SMA * SMA * MDOT * MDOT
}

/// Load the t=0 `prop_integ_state` row from JEOD's `integ_lsode_integ.csv`.
///
/// The integrator frame is deterministically rotated from the canonical
/// Kepler solution (`IntegrationTest` uses a regression-mode RNG), so the
/// reference CSV's t=0 line is the only legitimate IC source for parity:
/// it matches the JEOD source seed the runner-side test consumes, and
/// both runtimes step forward independently from there.
fn load_t0_initial_state(path: &Path) -> TranslationalState {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD SIM_integ_test CSV at {}: {e}",
            path.display()
        )
    });
    let first_data_line = content
        .lines()
        .nth(1)
        .expect("integ_lsode_integ.csv must have at least one data row after the header");
    let f: Vec<&str> = first_data_line.split(',').collect();
    assert!(
        f.len() >= 7,
        "integ_lsode_integ.csv t=0 row expected >=7 columns (time + pos[3] + vel[3]), got {}",
        f.len()
    );
    let parse = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
    let t0: f64 = parse(0);
    assert!(
        t0 == 0.0,
        "first data row of integ_lsode_integ.csv must be t=0 (got t={t0})",
    );
    TranslationalState {
        position: DVec3::new(parse(1), parse(2), parse(3)),
        velocity: DVec3::new(parse(4), parse(5), parse(6)),
    }
}

/// Build the SIM_integ_test LSODE scenario as a fresh `SimulationBuilder`.
/// Both runtime sides consume an independently-constructed builder from
/// this factory so neither builder is moved twice. The returned builder
/// carries the matched `time_scale_factor`, derived μ, the same IC, and
/// the `RUN_lsode` configuration (`ImplicitAdamsNonStiff`, rtol = 2.3e-16,
/// atol = 0 — JEOD `integ_option_int = 140`).
fn build_lsode_scenario(initial_state: TranslationalState) -> SimulationBuilder {
    let mu = compute_mu();
    let time_scale = compute_time_scale();

    let mut time = SimulationTime::at_j2000(default_leap_second_table());
    time.set_scale_factor(time_scale);
    let mut b = SimulationBuilder::new(time, SIM_DT);

    let earth = GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: Position::<RootInertial>::zero(),
        velocity: Velocity::<RootInertial>::zero(),
        t_inertial_pfix: None,
        rotation_model: astrodyn::RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
        marker_only: false,
    };
    let earth_idx = b.add_source("Earth", earth);

    let cfg = LsodeConfig::non_stiff_adams().with_tolerances(2.3e-16, 0.0);
    b.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&initial_state),
        integrator: IntegratorType::Lsode(cfg),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth_idx,
                GravityGradient::Skip,
            )],
        },
        ..Default::default()
    });
    b
}

/// Compare cadence in seconds. `SIM_integ_test` logs every 200 sim-seconds
/// (`Log_data/log_suite.py`); aligning the parity checkpoint to the same
/// 200 s cadence keeps the checkpoint set a strict subset of the per-tick
/// trajectory and matches the cadence the runner-side test asserts at.
/// 200 s / 1 s sim_dt = 200 ticks per checkpoint.
const CHECKPOINT_CADENCE_S: f64 = 200.0;

/// Full reference window: 80 000 sim-seconds = 401 records = ~14 orbits.
const PARITY_WINDOW_S: f64 = 80_000.0;

#[test]
fn bevy_parity_lsode() {
    let csv_path = test_data_path("integ_lsode_integ.csv");
    assert!(
        csv_path.exists(),
        "JEOD SIM_integ_test reference CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display(),
    );
    let initial_state = load_t0_initial_state(&csv_path);

    // Runner side — own builder instance.
    let runner_builder = build_lsode_scenario(initial_state);
    let dt = runner_builder.dt;
    let mut runner = runner_builder
        .build()
        .expect("runner build for SIM_integ_test LSODE");

    // Bevy side — independent builder instance from the same factory,
    // materialized into a fresh App under `<Earth>`. `populate_app`
    // honors the builder's `SimulationTime.time_scale_factor` via the
    // `SimulationTimeR` insertion in `scenario.rs`, and inserts the
    // `LsodeStateC` component (scenario.rs auto-init arm).
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let handles = build_lsode_scenario(initial_state)
        .populate_app::<astrodyn::Earth>(&mut app)
        .expect("bevy populate_app under <Earth>");
    let vehicle = handles.body_entities[0];

    // Run startup so per-source frame trees / source-frame-id resources
    // are wired before stepping. `MinimalPlugins` does not auto-run
    // `Startup`; the parity loop drives `FixedUpdate` directly.
    app.world_mut().run_schedule(Startup);

    let steps_per_checkpoint = (CHECKPOINT_CADENCE_S / dt).round() as usize;
    assert!(
        steps_per_checkpoint >= 1,
        "checkpoint cadence ({CHECKPOINT_CADENCE_S}s) must be a positive multiple of dt ({dt}s)"
    );

    let mut t = 0.0_f64;
    while t + CHECKPOINT_CADENCE_S <= PARITY_WINDOW_S {
        runner
            .step_n(steps_per_checkpoint)
            .expect("runner step_n during SIM_integ_test LSODE parity loop");
        for _ in 0..steps_per_checkpoint {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(dt));
            app.world_mut().run_schedule(FixedUpdate);
        }
        t += CHECKPOINT_CADENCE_S;

        let r_pos = runner.body(0).trans.position.raw_si();
        let r_vel = runner.body(0).trans.velocity.raw_si();
        let bevy_trans = app
            .world()
            .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
            .expect("vehicle entity carries TranslationalStateC<Earth>")
            .0;
        let b_pos = bevy_trans.position.raw_si();
        let b_vel = bevy_trans.velocity.raw_si();

        for i in 0..3 {
            assert!(
                r_pos[i].to_bits() == b_pos[i].to_bits(),
                "SIM_integ_test LSODE translational bit-parity broke at t={t} on position[{i}]: \
                 runner={} bevy={}",
                r_pos[i],
                b_pos[i],
            );
            assert!(
                r_vel[i].to_bits() == b_vel[i].to_bits(),
                "SIM_integ_test LSODE translational bit-parity broke at t={t} on velocity[{i}]: \
                 runner={} bevy={}",
                r_vel[i],
                b_vel[i],
            );
        }
    }
}
