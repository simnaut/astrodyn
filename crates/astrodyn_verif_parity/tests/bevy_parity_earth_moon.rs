//! Bevy ↔ runner parity for the SIM_Earth_Moon Clementine scenario.
//!
//! Drives [`astrodyn_verif_jeod::setups::earth_moon_clem::earth_moon_clem`]
//! through both runtimes — [`astrodyn_runner::Simulation`] and the Bevy
//! `populate_app::<Moon>` bridge — and asserts bit-identical
//! translational state at every checkpoint over a Moon-orbit parity
//! window.
//!
//! Single-planet bridge note: every body integrates in
//! `PlanetInertial<Moon>` — Moon is the central gravity source (LP150Q
//! 60×60 + DE421 BPC libration), Earth and Sun are point-mass third
//! bodies with per-step DE421 ephemeris updates, cannonball SRP. The
//! runner-side counterpart is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_earth_moon.rs`. The
//! parity wrapper validates that propagation through the Bevy bridge
//! tracks the runner bit-for-bit, so the runner's 7-day Clementine
//! cross-validation result transfers transitively to the Bevy adapter.

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

use astrodyn_bevy::{SimulationBuilderBevyExt, TranslationalStateC};
use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::setups::earth_moon_clem::earth_moon_clem;
use astrodyn_verif_jeod::setups::earth_moon_rosetta::earth_moon_rosetta;
use bevy::prelude::*;

/// Integration timestep matching the runner-side tier3 test (32 Hz RK4).
const DT: f64 = 0.03125;

/// Parity-window cap, in seconds of simulation time.
///
/// The runner-side tier3 test runs the full 7-day Clementine reference
/// (≈ 604 800 s, ≈ 19.4 M ticks at dt = 0.03125 s). Bit-identity
/// divergence between two runtimes that share the same `astrodyn_*`
/// math is monotonic, so a fraction of the orbit is sufficient to
/// catch any drift introduced by the bridge — pin the window at a few
/// minutes of simulation time so the parity wrapper exercises the
/// LP150Q 60×60 + DE421 BPC libration + 3rd-body ephemeris path
/// through both runtimes without spending tens of minutes per CI run.
/// The heavy bucket (`test-parity-trajectory-full`) can lift this
/// later if longer-horizon coverage becomes load-bearing.
const PARITY_WINDOW_S: f64 = 300.0;

/// Compare cadence in seconds. Picked to land cleanly on `DT` so each
/// checkpoint is exactly a whole number of fixed-update ticks ahead of
/// the previous one. 30 s = 960 ticks at dt = 0.03125 s.
const CHECKPOINT_CADENCE_S: f64 = 30.0;

#[test]
fn bevy_parity_earth_moon_clem() {
    // Runner side — build the canonical Clementine scenario.
    let runner_builder = earth_moon_clem(DT, None);
    let mut runner = runner_builder.build().expect("runner build");

    // Bevy side — same factory, materialized into a fresh App under <Moon>.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let handles = earth_moon_clem(DT, None)
        .populate_app::<astrodyn::Moon>(&mut app)
        .expect("populate_app under <Moon>");
    let vehicle = handles.body_entities[0];

    // Run startup so per-source frame trees / source-frame-id resources
    // are wired before stepping. `MinimalPlugins` does not auto-run
    // `Startup`; the parity loop drives `FixedUpdate` directly.
    app.world_mut().run_schedule(Startup);

    let steps_per_checkpoint = (CHECKPOINT_CADENCE_S / DT).round() as usize;
    assert!(
        steps_per_checkpoint >= 1,
        "checkpoint cadence ({CHECKPOINT_CADENCE_S}s) must be a positive multiple of dt ({DT}s)"
    );

    let mut t = 0.0_f64;
    while t + CHECKPOINT_CADENCE_S <= PARITY_WINDOW_S {
        // Step both runtimes forward by exactly one checkpoint window.
        runner.step_n(steps_per_checkpoint).expect("runner step_n");
        for _ in 0..steps_per_checkpoint {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(DT));
            app.world_mut().run_schedule(FixedUpdate);
        }
        t += CHECKPOINT_CADENCE_S;

        let r_pos = runner.body(0).trans.position.raw_si();
        let r_vel = runner.body(0).trans.velocity.raw_si();
        let bevy_trans = app
            .world()
            .get::<TranslationalStateC<astrodyn::Moon>>(vehicle)
            .expect("vehicle entity carries TranslationalStateC<Moon>")
            .0;
        let b_pos = bevy_trans.position.raw_si();
        let b_vel = bevy_trans.velocity.raw_si();

        for i in 0..3 {
            assert!(
                r_pos[i].to_bits() == b_pos[i].to_bits(),
                "earth_moon_clem translational bit-parity broke at t={t} on position[{i}]: \
                 runner={} bevy={}",
                r_pos[i],
                b_pos[i],
            );
            assert!(
                r_vel[i].to_bits() == b_vel[i].to_bits(),
                "earth_moon_clem translational bit-parity broke at t={t} on velocity[{i}]: \
                 runner={} bevy={}",
                r_vel[i],
                b_vel[i],
            );
        }
    }
}

/// Rosetta Earth swing-by parity: Earth is the central gravity source
/// (point-mass + J2), Moon and Sun point-mass third bodies, cannonball
/// SRP — every body integrates in `PlanetInertial<Earth>`. Drives
/// [`earth_moon_rosetta`] through both runtimes and asserts bit-identical
/// translational state, transferring the runner-side RUN_rosetta
/// cross-validation to the Bevy adapter.
#[test]
fn bevy_parity_earth_moon_rosetta() {
    let runner_builder = earth_moon_rosetta(DT, None);
    let mut runner = runner_builder.build().expect("runner build");

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let handles = earth_moon_rosetta(DT, None)
        .populate_app::<astrodyn::Earth>(&mut app)
        .expect("populate_app under <Earth>");
    let vehicle = handles.body_entities[0];

    app.world_mut().run_schedule(Startup);

    let steps_per_checkpoint = (CHECKPOINT_CADENCE_S / DT).round() as usize;
    assert!(steps_per_checkpoint >= 1);

    let mut t = 0.0_f64;
    while t + CHECKPOINT_CADENCE_S <= PARITY_WINDOW_S {
        runner.step_n(steps_per_checkpoint).expect("runner step_n");
        for _ in 0..steps_per_checkpoint {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(DT));
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
                "earth_moon_rosetta translational bit-parity broke at t={t} on position[{i}]: \
                 runner={} bevy={}",
                r_pos[i],
                b_pos[i],
            );
            assert!(
                r_vel[i].to_bits() == b_vel[i].to_bits(),
                "earth_moon_rosetta translational bit-parity broke at t={t} on velocity[{i}]: \
                 runner={} bevy={}",
                r_vel[i],
                b_vel[i],
            );
        }
    }
}
