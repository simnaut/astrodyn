//! Bevy ↔ runner parity for the `mercury_relativistic` recipe.
//!
//! Drives `astrodyn::recipes::scenarios::mercury::mercury_relativistic()`
//! through both runtimes — [`astrodyn_runner::Simulation`] and the Bevy
//! `populate_app::<Sun>` bridge — and asserts bit-identical translational
//! state at every checkpoint over a Mercury-period parity window.
//!
//! Single-planet bridge note: every body integrates in
//! `PlanetInertial<Sun>`. The Sun is the central gravity source and the
//! integration frame; no third bodies. The runner-side counterpart is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_mercury.rs` (the
//! relativistic-effect / perihelion-advance tests). The parity wrapper
//! validates that propagation through the Bevy bridge tracks the
//! runner bit-for-bit, so the runner's GR-perihelion-advance result
//! transfers transitively to the Bevy adapter.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy state fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test step counts fit exactly in f64 mantissa and usize"
)]

use std::time::Duration;

use astrodyn::recipes::scenarios::mercury::mercury_relativistic;
use astrodyn_bevy::{SimulationBuilderBevyExt, TranslationalStateC};
use astrodyn_runner::SimulationBuilderExt;
use bevy::prelude::*;

/// Parity-window cap, in seconds of simulation time.
///
/// One full Mercury orbit is ~88 days. Bit-identity divergence between
/// two runtimes that share the same `astrodyn_*` math is monotonic, so
/// a fraction of an orbit is sufficient to catch any drift introduced
/// by the bridge — pin the window at ~10 days = 8.64e5 s, ~9600 ticks
/// at the recipe's `dt = 100 s`. A few seconds of CI runtime.
const PARITY_WINDOW_S: f64 = 8.64e5;

/// Compare cadence in seconds. Picked to land cleanly on the recipe's
/// `dt = 100 s` integration step so each checkpoint is exactly a whole
/// number of fixed-update ticks ahead of the previous one.
const CHECKPOINT_CADENCE_S: f64 = 86_400.0;

#[test]
fn bevy_parity_mercury() {
    // Runner side.
    let runner_builder = mercury_relativistic();
    let dt = runner_builder.dt;
    let mut runner = runner_builder.build().expect("runner build");

    // Bevy side — same recipe factory, materialized into a fresh App under <Sun>.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let handles = mercury_relativistic()
        .populate_app::<astrodyn::Sun>(&mut app)
        .expect("populate_app under <Sun>");
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
        // Step both runtimes forward by exactly one checkpoint window.
        runner.step_n(steps_per_checkpoint).expect("runner step_n");
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
            .get::<TranslationalStateC<astrodyn::Sun>>(vehicle)
            .expect("vehicle entity carries TranslationalStateC<Sun>")
            .0;
        let b_pos = bevy_trans.position.raw_si();
        let b_vel = bevy_trans.velocity.raw_si();

        for i in 0..3 {
            assert!(
                r_pos[i].to_bits() == b_pos[i].to_bits(),
                "mercury translational bit-parity broke at t={t} on position[{i}]: \
                 runner={} bevy={}",
                r_pos[i],
                b_pos[i],
            );
            assert!(
                r_vel[i].to_bits() == b_vel[i].to_bits(),
                "mercury translational bit-parity broke at t={t} on velocity[{i}]: \
                 runner={} bevy={}",
                r_vel[i],
                b_vel[i],
            );
        }
    }
}
