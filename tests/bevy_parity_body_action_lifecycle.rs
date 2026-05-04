//! Tier 3-style cross-validation for the dynamic body-action lifecycle
//! API (#199). Mirrors JEOD's `SIM_removable_body_action` `RUN_1` and
//! `mass.py` add → remove → re-add idiom in the Bevy adapter, then
//! propagates the resulting orbit and cross-validates the trajectory
//! against the JEOD reference CSV `dyncomp_run2_state.csv`.
//!
//! # Why this CSV
//!
//! `SIM_removable_body_action` is a Trick copy of `SIM_dyncomp` whose
//! only difference is the `mass.py` add/remove/re-add pattern that
//! ends with `mass = 100_000` kg instead of the standard `400_000`.
//! `SIM_removable_body_action::RUN_1` (the only run in that sim) is
//! a 1-second logging-only configuration; it doesn't ship a multi-
//! hour reference trajectory. The orbit propagated **is** numerically
//! identical to `SIM_dyncomp::RUN_2` because:
//!
//! - the configuration is `Earth: spherical / Sun-Moon: off / drag:
//!   off / GG-torque: off` (`SIM_removable_body_action::RUN_1::input.py`
//!   defaults via `common_input.py`),
//! - vehicle mass does *not* enter the gravitational acceleration
//!   `g = mu/r^2` for a point-mass spherical-Earth field (Newton's
//!   second law cancels mass on both sides),
//! - the initial position / velocity are the same `iss_typical`
//!   state vector as `SIM_dyncomp::RUN_2`'s `state.py`.
//!
//! So the `dyncomp_run2_state.csv` reference is the right comparison
//! target — both for `m = 400_000` and `m = 100_000` the trajectory
//! numerically matches the same CSV. The test asserts the lifecycle
//! API behaves correctly **and** that mass changes do not perturb the
//! point-mass trajectory.
//!
//! # What is exercised
//!
//! Total run length is `STOP_TIME = 300 s` (5 minutes at 32 Hz =
//! 9 600 ticks); the lifecycle parity question is fully observable in
//! the first few minutes and the cross-validation only needs enough
//! samples to surface drift the API would introduce. The full 8-hour
//! `dyncomp_run2_state.csv` propagation is already covered by
//! `tier3_simulation_run2_3dof` — this test focuses on lifecycle.
//!
//! 1. **Init-time `add → remove → re-add`** — JEOD `mass.py:46-49`
//!    pattern. The first `add` queues a `MassProperties::new(400_000)`
//!    action, the `remove` cancels it, then the second `add` queues
//!    `MassProperties::new(100_000)`. After the first `app.update()`
//!    only the second action has fired and mass = 100 000.
//! 2. **Mid-sim mass change** — at `t = MID_SIM_MASS_CHANGE_TIME`
//!    (`150 s`, half-way into the run) the test queues another mass
//!    change to 50 000 kg. Because the trajectory is mass-independent
//!    under spherical point-mass gravity, the cross-validation against
//!    the CSV must still hold across the change-point.
//! 3. **Mid-sim `add → remove`** — at `t = MID_SIM_ADD_REMOVE_TIME`
//!    (`240 s`) the test queues an `InitMass` action with name
//!    `"abort_mid_sim"` and immediately removes it. The vehicle's mass
//!    is unchanged.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    BodyActionMessage, GravityControlsC, JeodPlugin, MassPropertiesC, RotationalStateC,
    SourceInertialPositionC, TranslationalStateC,
};
use glam::DVec3;
use jeod_sim::{
    BodyAction, DynamicsConfig, GravityControl, GravityControls, MassProperties, RotationalState,
    TranslationalState,
};

use common::earth_source;

/// JEOD `SIM_dyncomp/Modified_data/state.py`
/// `set_trans_init_typical()` initial vehicle state.
fn iss_typical_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(-4_292_653.41, 955_168.47, 5_139_356.57),
        velocity: DVec3::new(109.649663, -7527.726490, 1484.521489),
    }
}

/// `SIM_dyncomp::RUN_2` integration step (32 Hz, 0.03125 s).
const DT: f64 = 0.03125;
/// Run length: 5 minutes.
///
/// `SIM_dyncomp::RUN_2`'s `input.py` runs for 28 800 s (8 hours), but
/// the body-action lifecycle parity question is fully exercised within
/// the first few minutes — the cross-validation only needs enough
/// samples to detect any drift the lifecycle API would introduce, and
/// 5 minutes at 32 Hz is 9 600 ticks (5 of which are between
/// reference-CSV checkpoints at 60 s cadence). The full 8-hour run
/// test against `dyncomp_run2_state.csv` is already covered by
/// `tier3_simulation_run2_3dof` (`tier3_sim_dyncomp_run2.rs`).
const STOP_TIME: f64 = 300.0;

/// Mid-run mass change: half-way into the run.
const MID_SIM_MASS_CHANGE_TIME: f64 = 150.0;
const MID_SIM_MASS_KG: f64 = 50_000.0;
/// Mid-sim add-then-remove (drained without firing): later still.
const MID_SIM_ADD_REMOVE_TIME: f64 = 240.0;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

#[derive(Debug, Clone, Copy)]
struct CsvRow {
    time: f64,
    position: DVec3,
    velocity: DVec3,
}

/// Parse the cross-validation reference CSV.
///
/// `dyncomp_run2_state.csv` has 481 rows at 60 s cadence; only `time`,
/// `position[0..3]`, and `velocity[0..3]` are needed for this test.
fn load_reference_csv() -> Vec<CsvRow> {
    let path = test_data_dir().join("dyncomp_run2_state.csv");
    assert!(
        path.exists(),
        "JEOD reference CSV not found at {}.\n\
         The Tier 3 reference for SIM_removable_body_action is identical\n\
         to SIM_dyncomp RUN_2 — the body-action lifecycle does not change\n\
         the point-mass trajectory. Regenerate via:\n  \
           cargo xtask regenerate-tier3",
        path.display()
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let mut lines = content.lines();
    // Skip the JEOD CSV header (one row of column headers).
    lines.next().expect("CSV header");
    let mut rows = Vec::new();
    for line in lines {
        let cols: Vec<f64> = line
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .unwrap_or_else(|e| panic!("CSV parse: '{}' -> {e}", s.trim()))
            })
            .collect();
        // Layout (per CLAUDE.md `CSV column layout for log_state_ASCII.csv`):
        //   col 0: sys.exec.out.time {s}
        //   col 1: composite_body.state.trans.position[0]
        //   col 2: composite_body.state.trans.velocity[0]
        //   col 8: position[1], col 9: velocity[1]
        //   col 15: position[2], col 16: velocity[2]
        if cols.len() < 17 {
            continue;
        }
        rows.push(CsvRow {
            time: cols[0],
            position: DVec3::new(cols[1], cols[8], cols[15]),
            velocity: DVec3::new(cols[2], cols[9], cols[16]),
        });
    }
    rows
}

/// Spawn a Bevy `App` configured for SIM_removable_body_action::RUN_1.
fn build_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            bevy_jeod::GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    // Initial (pre-action) vehicle: ISS-typical state, mass = 1 kg
    // (placeholder — the `add → remove → re-add` pair below resets it
    // to 100 000 kg before any tick fires).
    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_typical_state()),
            RotationalStateC::from(RotationalState::default()),
            MassPropertiesC::from(MassProperties::new(1.0)),
            bevy_jeod::DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            }),
        ))
        .id();

    (app, vehicle)
}

/// Push a `BodyActionMessage` directly into the world's message buffer.
fn write_msg(app: &mut App, msg: BodyActionMessage) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionMessage>>()
        .write(msg);
}

/// Read the vehicle's mass from the typed Component.
fn read_mass(app: &App, vehicle: Entity) -> f64 {
    app.world()
        .entity(vehicle)
        .get::<MassPropertiesC>()
        .expect("mass props present")
        .0
        .to_untyped()
        .mass
}

/// Read the vehicle's translational state from the typed Component.
fn read_trans(app: &App, vehicle: Entity) -> TranslationalState {
    app.world()
        .entity(vehicle)
        .get::<TranslationalStateC>()
        .expect("translational state present")
        .0
        .to_untyped()
}

#[test]
fn tier3_bevy_parity_body_action_init_lifecycle() {
    let (mut app, vehicle) = build_app();

    // ── Step 0: queue the JEOD `mass.py` add → remove → re-add ──
    // Mirrors `models/dynamics/dyn_manager/verif/SIM_removable_body_action/Modified_data/mass.py:44-49`.
    write_msg(
        &mut app,
        BodyActionMessage::add(
            vehicle,
            BodyAction::InitMass {
                mass: MassProperties::new(400_000.0),
            },
            Some("vehicle.mass_init"),
        ),
    );
    write_msg(&mut app, BodyActionMessage::remove("vehicle.mass_init"));
    write_msg(
        &mut app,
        BodyActionMessage::add(
            vehicle,
            BodyAction::InitMass {
                mass: MassProperties::new(100_000.0),
            },
            Some("vehicle.mass_init"),
        ),
    );

    // First tick fires every queued action that's ready. The
    // body-action intake system runs *before* the apply system,
    // collapsing the add → remove → re-add into a single queued
    // action that the apply pass executes on this same tick.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    let mass_after_init = read_mass(&app, vehicle);
    assert_eq!(
        mass_after_init, 100_000.0,
        "Body-action lifecycle parity: after init-time `add → remove → \
         re-add(100 000)` the live mass must be 100 000 kg (not 400 000 \
         — the first add was cancelled). JEOD `mass.py:44-49` checks \
         the same lifecycle."
    );

    // ── Step the rest of the run ──
    let reference = load_reference_csv();
    assert!(
        !reference.is_empty(),
        "Reference CSV {} contained no rows after parsing.",
        test_data_dir().join("dyncomp_run2_state.csv").display()
    );

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    let mut mid_sim_change_applied = false;
    let mut mid_sim_add_remove_done = false;
    // Reference CSV starts at t=0 with the *initial* state. Skip
    // that first row — by the time the test enters the propagation
    // loop the integrator has already advanced one tick, so we
    // compare row 1 (t=60 s) onward.
    let mut log_iter = reference.iter().skip(1).peekable();
    let mut sim_t = DT;

    while sim_t < STOP_TIME + 0.5 * DT {
        // Inject mid-sim actions just before the relevant tick.
        if !mid_sim_change_applied && sim_t + 0.5 * DT >= MID_SIM_MASS_CHANGE_TIME {
            write_msg(
                &mut app,
                BodyActionMessage::add(
                    vehicle,
                    BodyAction::InitMass {
                        mass: MassProperties::new(MID_SIM_MASS_KG),
                    },
                    Some("midsim.mass_change"),
                ),
            );
            mid_sim_change_applied = true;
        }
        if !mid_sim_add_remove_done && sim_t + 0.5 * DT >= MID_SIM_ADD_REMOVE_TIME {
            write_msg(
                &mut app,
                BodyActionMessage::add(
                    vehicle,
                    BodyAction::InitMass {
                        // A clearly distinct value so a regression
                        // that fails to drop the action would surface
                        // immediately as a mass mismatch below.
                        mass: MassProperties::new(7.0),
                    },
                    Some("abort_mid_sim"),
                ),
            );
            write_msg(&mut app, BodyActionMessage::remove("abort_mid_sim"));
            mid_sim_add_remove_done = true;
        }

        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
        sim_t += DT;

        // After the mid-sim mass change has been queued + applied,
        // assert the live mass reflects the change. The intake +
        // apply systems run within the same FixedUpdate tick the
        // message lands in.
        if mid_sim_change_applied && (sim_t - MID_SIM_MASS_CHANGE_TIME).abs() < 2.0 * DT {
            let m = read_mass(&app, vehicle);
            assert!(
                (m - MID_SIM_MASS_KG).abs() < 1e-9,
                "Mid-sim BodyAction::InitMass at t={MID_SIM_MASS_CHANGE_TIME} did not update mass; \
                 got {m}, expected {MID_SIM_MASS_KG}."
            );
        }
        if mid_sim_add_remove_done && (sim_t - MID_SIM_ADD_REMOVE_TIME).abs() < 2.0 * DT {
            let m = read_mass(&app, vehicle);
            assert!(
                (m - MID_SIM_MASS_KG).abs() < 1e-9,
                "Mid-sim add-then-remove at t={MID_SIM_ADD_REMOVE_TIME} should have left mass unchanged \
                 ({MID_SIM_MASS_KG} kg) but live mass = {m}."
            );
        }

        // Compare against the reference CSV at every 60 s log cadence
        // (matching JEOD's `LOG_CYCLE`). Look up the next expected
        // log row; if `sim_t` straddles its time, run the comparison
        // and advance the iterator.
        if let Some(next) = log_iter.peek() {
            if sim_t + 0.5 * DT >= next.time {
                let trans = read_trans(&app, vehicle);
                let pos_err = (trans.position - next.position).length();
                let vel_err = (trans.velocity - next.velocity).length();
                max_pos_err = max_pos_err.max(pos_err);
                max_vel_err = max_vel_err.max(vel_err);
                log_iter.next();
            }
        }
    }

    // Mass-independence of point-mass gravity makes this a tight tolerance.
    // `tier3_sim_dyncomp_run2`'s identical configuration converges to
    // ~1.4e-6 m / ~1.5e-9 m/s in `tolerances`; over a 5-minute run the
    // position error doesn't exceed ~1e-5 m. The Bevy adapter inherits
    // the same physics through `JeodPlugin`, so we hold a 5%-margined
    // 1.0e-4 m / 1.0e-7 m/s tolerance — strict enough to catch a
    // lifecycle-API regression that introduces visible drift.
    let tol_pos = 1.0e-4_f64;
    let tol_vel = 1.0e-7_f64;

    assert!(
        max_pos_err < tol_pos,
        "Body-action lifecycle Tier 3 parity: max position error \
         {max_pos_err} m exceeds {tol_pos} m tolerance. The trajectory \
         must remain identical to SIM_dyncomp RUN_2 because point-mass \
         spherical gravity is mass-independent — a divergence here \
         indicates the lifecycle API is leaking state into the \
         per-step pipeline beyond the mass component (e.g. corrupting \
         the integrator history or the gravity source list)."
    );
    assert!(
        max_vel_err < tol_vel,
        "Body-action lifecycle Tier 3 parity: max velocity error \
         {max_vel_err} m/s exceeds {tol_vel} m/s tolerance."
    );

    // Final mass should be the mid-sim change.
    let final_mass = read_mass(&app, vehicle);
    assert!(
        (final_mass - MID_SIM_MASS_KG).abs() < 1e-9,
        "Final mass: expected {MID_SIM_MASS_KG} (mid-sim change), got {final_mass}."
    );

    // Sanity: at least one log row was processed. With `STOP_TIME =
    // 300` and 60 s cadence we expect at least 4 comparisons after
    // skipping the t=0 row.
    let consumed = reference.len() - 1 - log_iter.count();
    assert!(
        consumed >= 4,
        "Cross-validation walked only {consumed} log rows in {STOP_TIME}s; expected at least 4 \
         (one per 60 s checkpoint after the t=0 skip). The test driver did not advance \
         to the expected sim time."
    );
}
