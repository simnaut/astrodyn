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
//!    `MassProperties::new(100_000)`. After the first `FixedUpdate`
//!    tick (driven via `Time::<Fixed>::advance_by` +
//!    `run_schedule(FixedUpdate)` rather than `app.update()`) only the
//!    second action has fired and mass = 100 000.
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

use astrodyn::{
    BodyAction, DynamicsConfig, GravityControl, GravityControls, GravityRole, MassProperties,
    RotationalState, TranslationalState,
};
use astrodyn_bevy::{
    AstrodynPlugin, BodyActionEvent, GravityControlsC, MassPropertiesC, RotationalStateC,
    SourceInertialPositionC, TranslationalStateC,
};
use astrodyn_verif_jeod::dyncomp_csv::{load_dyncomp_csv, DyncompRecord};
use bevy::prelude::*;
use glam::DVec3;

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
/// samples to detect any drift the lifecycle API would introduce. At
/// 32 Hz a 5-minute run is 9 600 integration ticks and crosses 5
/// reference-CSV checkpoints (the CSV logs at 60 s cadence, so within
/// the 300 s after t=0 the checkpoints land at t = 60, 120, 180, 240,
/// 300 s). The full 8-hour run against `dyncomp_run2_state.csv` is
/// already covered by `tier3_simulation_run2_3dof`
/// (`tier3_sim_dyncomp_run2.rs`).
const STOP_TIME: f64 = 300.0;

/// Mid-run mass change: half-way into the run.
const MID_SIM_MASS_CHANGE_TIME: f64 = 150.0;
const MID_SIM_MASS_KG: f64 = 50_000.0;
/// Mid-sim add-then-remove (drained without firing): later still.
const MID_SIM_ADD_REMOVE_TIME: f64 = 240.0;

fn test_data_dir() -> PathBuf {
    // Resolve via the canonical astrodyn_verif_jeod resolver, which knows
    // tier-3 reference CSVs live under `crates/astrodyn_verif_jeod/test_data/`.
    // Use a known fixture and strip the filename to recover the directory.
    astrodyn_verif_jeod::tier3_csv::test_data_path("dyncomp_run2_state.csv")
        .parent()
        .expect("test_data_path returns a file inside test_data/")
        .to_path_buf()
}

/// Load the cross-validation reference CSV via the workspace's canonical
/// `astrodyn_verif_jeod::dyncomp_csv::load_dyncomp_csv` parser.
///
/// Reusing the canonical loader keeps the column layout in one place and
/// inherits its fail-loud handling (missing file, parse failure, truncated
/// row with fewer than 23 columns all panic with diagnostic messages).
/// Only `time`, `composite_body.position`, and `composite_body.velocity`
/// are consumed by this test.
fn load_reference_csv() -> Vec<DyncompRecord> {
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
    load_dyncomp_csv(&path)
}

/// Spawn a Bevy `App` configured for SIM_removable_body_action::RUN_1.
fn build_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            astrodyn_bevy::GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    // Initial (pre-action) vehicle: ISS-typical state, mass = 1 kg
    // (placeholder — the `add → remove → re-add` pair below resets it
    // to 100 000 kg before any tick fires).
    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from_untyped(iss_typical_state()),
            RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                &(RotationalState::default()),
            )),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(1.0)),
            )),
            astrodyn_bevy::DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
            }),
        ))
        .id();

    (app, vehicle)
}

/// Push a `BodyActionEvent` directly into the world's message buffer.
fn write_msg(app: &mut App, msg: BodyActionEvent) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
        .write(msg);
}

/// Read the vehicle's mass from the typed Component.
fn read_mass(app: &App, vehicle: Entity) -> f64 {
    astrodyn::typed_bridge::mass_typed_to_raw(
        &app.world()
            .entity(vehicle)
            .get::<MassPropertiesC>()
            .expect("mass props present")
            .0,
    )
    .mass
}

/// Read the vehicle's translational state from the typed Component.
fn read_trans(app: &App, vehicle: Entity) -> TranslationalState {
    astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .entity(vehicle)
            .get::<TranslationalStateC<astrodyn::Earth>>()
            .expect("translational state present")
            .0,
    )
}

#[test]
fn tier3_bevy_parity_body_action_init_lifecycle() {
    let (mut app, vehicle) = build_app();

    // ── Step 0: queue the JEOD `mass.py` add → remove → re-add ──
    // Mirrors `models/dynamics/dyn_manager/verif/SIM_removable_body_action/Modified_data/mass.py:44-49`.
    write_msg(
        &mut app,
        BodyActionEvent::add(
            vehicle,
            BodyAction::InitMass {
                mass: MassProperties::new(400_000.0),
            },
            Some("vehicle.mass_init"),
        ),
    );
    write_msg(&mut app, BodyActionEvent::remove("vehicle.mass_init"));
    write_msg(
        &mut app,
        BodyActionEvent::add(
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
    // `sim_t` is the post-tick time. The first FixedUpdate tick has
    // already fired above (line ~234), so `sim_t == DT` here means
    // 1 tick has been integrated. Every loop iteration runs one more
    // tick; for the documented `STOP_TIME = 300 s` (= 9600 ticks at
    // 32 Hz) we need exactly 9599 loop iterations on top of that
    // first tick, so the bound must exclude the iteration that would
    // start at `sim_t == STOP_TIME` (which would push integration
    // past the documented horizon). `< STOP_TIME - 0.5 * DT` keeps
    // the iteration that brings `sim_t` to exactly `STOP_TIME` and
    // drops the one that would overshoot.
    let mut sim_t = DT;
    let mut tick_count: usize = 1; // counts the pre-loop tick

    while sim_t < STOP_TIME - 0.5 * DT {
        // Inject mid-sim actions just before the relevant tick.
        if !mid_sim_change_applied && sim_t + 0.5 * DT >= MID_SIM_MASS_CHANGE_TIME {
            write_msg(
                &mut app,
                BodyActionEvent::add(
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
                BodyActionEvent::add(
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
            write_msg(&mut app, BodyActionEvent::remove("abort_mid_sim"));
            mid_sim_add_remove_done = true;
        }

        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
        sim_t += DT;
        tick_count += 1;

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
                let pos_err = (trans.position - next.composite_body.position).length();
                let vel_err = (trans.velocity - next.composite_body.velocity).length();
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
    // the same physics through `AstrodynPlugin`, so we hold a 5%-margined
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

    // The loop must integrate exactly `STOP_TIME / DT` FixedUpdate
    // ticks — 9 600 ticks for the documented 300 s / 32 Hz run.
    // Counting them and asserting against the closed-form expectation
    // is what catches an off-by-one in the loop bound or initial
    // `sim_t` (the kind of mismatch that previously let the loop
    // execute one extra tick beyond `STOP_TIME`).
    let expected_ticks = (STOP_TIME / DT).round() as usize;
    assert_eq!(
        tick_count, expected_ticks,
        "Body-action lifecycle Tier 3 parity: integrated {tick_count} FixedUpdate ticks but \
         the documented run is {expected_ticks} (STOP_TIME = {STOP_TIME} s at DT = {DT} s)."
    );

    // All 5 reference checkpoints (t = 60, 120, 180, 240, 300 s) must
    // have been consumed. The reference CSV has rows at 60 s cadence
    // and we skipped the t=0 row before the loop.
    let consumed = reference.len() - 1 - log_iter.count();
    assert_eq!(
        consumed, 5,
        "Cross-validation walked {consumed} log rows in {STOP_TIME}s; expected exactly 5 \
         (one per 60 s checkpoint at t = 60, 120, 180, 240, 300 after the t=0 skip)."
    );
}

/// IG.37 regression for mid-sim `BodyAction::InitTrans`.
///
/// `body_action_system` overwrites a body's translational state when an
/// `InitTrans` action lands. Multi-step integrators (Gauss–Jackson / ABM4)
/// accumulate predictor / corrector history that becomes inconsistent
/// with the new state — the same class of bug the attach/detach reset
/// path closed for in #274 / IG.37. This test wires an ABM4 integrator
/// onto a body, lets it run long enough for the integrator to leave
/// its priming window, then queues an `InitTrans` action and verifies:
///
/// 1. The integrator returns to priming (i.e. `reset_for_topology_change`
///    fired) after the action lands.
/// 2. The next several FixedUpdate ticks do not panic. The IG.37 assert
///    inside `abm4_translational_step` panics if `topology_dirty` is
///    still set when `integrate` is called, so a missing reset hook
///    would surface as a fail-loud panic instead of silent corruption.
#[test]
fn bevy_parity_body_action_init_trans_resets_abm4_history() {
    use astrodyn::{
        Abm4State, GravityControl, GravityControls, GravityModel, GravityRole, GravitySource,
        IntegratorType,
    };
    use astrodyn_bevy::{
        Abm4StateC, AstrodynPlugin, DynamicsConfigC, GravitySourceC, IntegratorTypeC,
        SourceInertialPositionC,
    };

    // Tight integration step so ABM4 fills its 4-sample priming window
    // quickly and we can observe the reset within a short test run.
    const SIM_DT: f64 = 0.5;
    const MU: f64 = 3.986_004_415e14;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(SIM_DT));
    app.add_plugins(AstrodynPlugin);

    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    // ABM4 does its own RK4 priming for the first 4 samples; once
    // `is_priming` flips false the predictor / corrector history is
    // live and a state overwrite without a reset would corrupt the
    // next integrate.
    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from_untyped(iss_typical_state()),
            RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                &(RotationalState::default()),
            )),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(400_000.0)),
            )),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
            }),
            IntegratorTypeC(IntegratorType::Abm4),
            Abm4StateC(Abm4State::new()),
        ))
        .id();

    // Step long enough for ABM4 to leave the priming window. ABM4
    // primes on its first four `integrate` calls; we drive 16 ticks
    // to give a comfortable margin (and to accumulate non-trivial
    // predictor history before the reset).
    for _ in 0..16 {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(SIM_DT));
        app.world_mut().run_schedule(FixedUpdate);
    }
    let priming_before = app
        .world()
        .entity(vehicle)
        .get::<Abm4StateC>()
        .expect("Abm4StateC present")
        .0
        .is_priming();
    assert!(
        !priming_before,
        "Pre-condition: ABM4 should have left the priming window after 16 ticks of dt={SIM_DT}; \
         test setup is wrong if it hasn't."
    );

    // Queue a mid-sim `InitTrans` action. The translational state is a
    // small offset from the current one — large enough to be non-trivial,
    // small enough that the next integrate is well-conditioned.
    let new_trans = TranslationalState {
        position: DVec3::new(-4_292_653.41, 955_168.47, 5_139_356.57) + DVec3::new(1.0e3, 0.0, 0.0),
        velocity: DVec3::new(109.649663, -7527.726490, 1484.521489),
    };
    write_msg(
        &mut app,
        BodyActionEvent::add(
            vehicle,
            BodyAction::InitTrans { state: new_trans },
            Some("midsim.trans_change"),
        ),
    );

    // Single FixedUpdate tick: intake + apply systems run, body_action_system
    // mutates `TranslationalStateC` and resets ABM4 history; integration
    // then runs against the fresh (now-priming) state.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(SIM_DT));
    app.world_mut().run_schedule(FixedUpdate);

    // Note: by the end of this one tick the integrator has already
    // consumed one sample of the new priming run, so `is_priming`
    // is still true (priming consumes 4 samples). Subsequent ticks
    // would re-arm history correctly because the reset cleared the
    // dirty flag.
    let priming_after = app
        .world()
        .entity(vehicle)
        .get::<Abm4StateC>()
        .expect("Abm4StateC present")
        .0
        .is_priming();
    assert!(
        priming_after,
        "IG.37: BodyAction::InitTrans must reset ABM4 history (`is_priming` should be true \
         after the action lands). A missing reset would leave the integrator running with \
         predictor history pointing at the pre-action state."
    );

    // Drive several more ticks to surface the IG.37 panic
    // (`abm4_translational_step` asserts `!topology_dirty` on every
    // integrate). A missing reset would have left `topology_dirty`
    // set and the very next tick would have panicked here.
    for _ in 0..20 {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(SIM_DT));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

/// IG.37 regression for mid-sim `BodyAction::InitMass`.
///
/// `body_action_system` updates `MassPropertiesC` when an `InitMass`
/// action lands. The acceleration the multi-step integrator predicts
/// is `force / mass`: a mid-sim mass change makes the predictor /
/// corrector history (recorded under the prior mass) inconsistent
/// with the new dynamics whenever any non-gravitational force is
/// present. Per the IG.37 attach/detach precedent, a topology-class
/// change must fire `reset_integrators()` so the next integrate
/// re-primes from fresh samples.
///
/// This test wires an ABM4 integrator onto a body, lets it run long
/// enough for the integrator to leave its priming window, then queues
/// an `InitMass` action that halves the vehicle mass and verifies:
///
/// 1. The integrator returns to priming (i.e. `reset_for_topology_change`
///    fired) after the action lands.
/// 2. The next several FixedUpdate ticks do not panic. The IG.37 assert
///    inside `abm4_translational_step` panics if `topology_dirty` is
///    still set when `integrate` is called, so a missing reset hook
///    would surface as a fail-loud panic instead of silent corruption.
#[test]
fn bevy_parity_body_action_init_mass_resets_abm4_history() {
    use astrodyn::{
        Abm4State, GravityControl, GravityControls, GravityModel, GravityRole, GravitySource,
        IntegratorType,
    };
    use astrodyn_bevy::{
        Abm4StateC, AstrodynPlugin, DynamicsConfigC, GravitySourceC, IntegratorTypeC,
        SourceInertialPositionC,
    };

    // Tight integration step so ABM4 fills its 4-sample priming window
    // quickly and we can observe the reset within a short test run.
    const SIM_DT: f64 = 0.5;
    const MU: f64 = 3.986_004_415e14;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(SIM_DT));
    app.add_plugins(AstrodynPlugin);

    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    // Same priming setup as the `InitTrans` test above: ABM4 primes on
    // its first four `integrate` calls; once `is_priming` flips false
    // the predictor / corrector history is live and a mass overwrite
    // without a reset would leave the next predictor extrapolating an
    // acceleration computed under the old mass.
    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from_untyped(iss_typical_state()),
            RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                &(RotationalState::default()),
            )),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(400_000.0)),
            )),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
            }),
            IntegratorTypeC(IntegratorType::Abm4),
            Abm4StateC(Abm4State::new()),
        ))
        .id();

    // Step long enough for ABM4 to leave the priming window. Same
    // 16-tick margin as the `InitTrans` regression test.
    for _ in 0..16 {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(SIM_DT));
        app.world_mut().run_schedule(FixedUpdate);
    }
    let priming_before = app
        .world()
        .entity(vehicle)
        .get::<Abm4StateC>()
        .expect("Abm4StateC present")
        .0
        .is_priming();
    assert!(
        !priming_before,
        "Pre-condition: ABM4 should have left the priming window after 16 ticks of dt={SIM_DT}; \
         test setup is wrong if it hasn't."
    );

    // Queue a mid-sim `InitMass` action that halves the mass. The
    // numerical value is unimportant — IG.37 is structural: any mass
    // overwrite must clear `topology_dirty`.
    write_msg(
        &mut app,
        BodyActionEvent::add(
            vehicle,
            BodyAction::InitMass {
                mass: MassProperties::new(200_000.0),
            },
            Some("midsim.mass_change"),
        ),
    );

    // Single FixedUpdate tick: intake + apply systems run,
    // body_action_system mutates `MassPropertiesC` and resets ABM4
    // history; integration then runs against the now-priming state.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(SIM_DT));
    app.world_mut().run_schedule(FixedUpdate);

    // Confirm the mass actually changed (sanity that the action fired).
    let mass_after = read_mass(&app, vehicle);
    assert_eq!(
        mass_after, 200_000.0,
        "Mid-sim BodyAction::InitMass at the priming-window-exit boundary did not update mass; \
         the action did not fire and IG.37 enforcement cannot be observed."
    );

    let priming_after = app
        .world()
        .entity(vehicle)
        .get::<Abm4StateC>()
        .expect("Abm4StateC present")
        .0
        .is_priming();
    assert!(
        priming_after,
        "IG.37: BodyAction::InitMass must reset ABM4 history (`is_priming` should be true \
         after the action lands). A missing reset would leave the integrator running with \
         predictor history recorded under the pre-action mass — the next non-gravitational \
         force sample would be inconsistent with `accel = force / mass_new`."
    );

    // Drive several more ticks to surface the IG.37 panic
    // (`abm4_translational_step` asserts `!topology_dirty` on every
    // integrate). A missing reset would have left `topology_dirty`
    // set and the very next tick would have panicked here.
    for _ in 0..20 {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(SIM_DT));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

/// Regression for the dual-schedule double-fire bug.
///
/// `body_action_intake_system` and `body_action_system` are pinned to
/// the `FixedUpdate` schedule only — they are NOT registered in
/// `Startup`. Each registration site gets its own
/// `Local<MessageCursor<BodyActionEvent>>`, so a dual-schedule wiring
/// would let an anonymous fire-once `BodyActionEvent::Add` apply
/// twice within the first `app.update()` call (Bevy's double-buffered
/// `Messages` keeps writes alive across the buffer swap that
/// `message_update_system` performs in `First`, so the FixedUpdate
/// cursor would re-read the same write the Startup cursor already
/// consumed).
///
/// The test drives schedules manually so the two reads fall at
/// observable boundaries:
///
/// 1. Build app with the full plugin, spawn a vehicle with mass = M0.
/// 2. Write a single anonymous `BodyActionEvent::Add(InitMass(K))`.
/// 3. Run `Startup` — assert mass is still M0 (no body-action
///    processing in `Startup`; this is the assertion that fails if a
///    Startup intake / apply registration is reintroduced).
/// 4. Manually set mass to a sentinel S between the two schedule runs.
/// 5. Run `First` (the message buffer swap) followed by
///    `FixedUpdate` — the apply runs here, mass becomes K (the
///    sentinel was overwritten by the FixedUpdate apply).
/// 6. Manually set mass to a second sentinel S2.
/// 7. Run another `FixedUpdate` cycle (no new messages written) —
///    mass must remain S2; the action was drained on tick 1 and must
///    not re-fire.
#[test]
fn bevy_parity_body_action_startup_message_applies_exactly_once() {
    use astrodyn::{GravityModel, GravitySource};
    use astrodyn_bevy::{DynamicsConfigC, GravitySourceC};

    const SIM_DT: f64 = 0.5;
    const MU: f64 = 3.986_004_415e14;
    const M0: f64 = 400_000.0;
    const K: f64 = 100_000.0;
    // Distinct positive sentinels (mass must be > 0 per `MassProperties::new`,
    // MA.02). The values are chosen to be obviously not part of any
    // legitimate code path so a regression that overwrites them is
    // unambiguous.
    const SENTINEL: f64 = 12_345.0;
    const SENTINEL2: f64 = 67_890.0;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(SIM_DT));
    app.add_plugins(AstrodynPlugin);

    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from_untyped(iss_typical_state()),
            RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                &(RotationalState::default()),
            )),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(M0)),
            )),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
            }),
        ))
        .id();

    // Single anonymous fire-once `Add`. With a dual-schedule
    // intake / apply registration, this would fire once at the end of
    // `Startup` and again on the first `FixedUpdate` tick.
    write_msg(
        &mut app,
        BodyActionEvent::add(
            vehicle,
            BodyAction::InitMass {
                mass: MassProperties::new(K),
            },
            None,
        ),
    );

    // Run only `Startup`. With body-action processing pinned to
    // `FixedUpdate`, mass is unchanged from spawn-time M0.
    app.world_mut().run_schedule(Startup);
    let mass_after_startup = read_mass(&app, vehicle);
    assert_eq!(
        mass_after_startup, M0,
        "Body-action systems must not run in `Startup` — registering them \
         in both `Startup` and `FixedUpdate` would give each registration \
         site its own `Local<MessageCursor<BodyActionEvent>>` and a fire-\
         once `Add` would apply twice on the first `app.update()` cycle. \
         Expected mass = {M0} (untouched by Startup), got {mass_after_startup}."
    );

    // Sentinel between the two schedule passes. The FixedUpdate apply
    // below should overwrite this with K (the action firing once);
    // a regression that prevented the FixedUpdate apply would surface
    // here as the sentinel surviving.
    *app.world_mut()
        .entity_mut(vehicle)
        .get_mut::<MassPropertiesC>()
        .expect("mass props present") = MassPropertiesC::from(
        astrodyn::typed_bridge::mass_raw_to_self_ref(&(MassProperties::new(SENTINEL))),
    );

    // First runs `message_update_system` (the buffer swap). FixedUpdate
    // intake reads the message from the back buffer, applies it.
    app.world_mut().run_schedule(First);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(SIM_DT));
    app.world_mut().run_schedule(FixedUpdate);
    let mass_after_first_fixed = read_mass(&app, vehicle);
    assert_eq!(
        mass_after_first_fixed, K,
        "FixedUpdate intake / apply must consume the Startup-era message \
         and overwrite the sentinel. Expected mass = {K} (action applied), \
         got {mass_after_first_fixed}."
    );

    // Second sentinel. With the action drained on tick 1, no further
    // FixedUpdate tick should re-apply it.
    *app.world_mut()
        .entity_mut(vehicle)
        .get_mut::<MassPropertiesC>()
        .expect("mass props present") = MassPropertiesC::from(
        astrodyn::typed_bridge::mass_raw_to_self_ref(&(MassProperties::new(SENTINEL2))),
    );

    app.world_mut().run_schedule(First);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(SIM_DT));
    app.world_mut().run_schedule(FixedUpdate);
    let mass_after_second_fixed = read_mass(&app, vehicle);
    assert_eq!(
        mass_after_second_fixed, SENTINEL2,
        "Action must fire exactly once — the message has already been \
         drained on the previous FixedUpdate tick and must not re-apply. \
         Expected mass = {SENTINEL2} (sentinel preserved), got \
         {mass_after_second_fixed}; a non-sentinel value here means the \
         action fired again, indicating either the message buffer was not \
         advanced (cursor regression) or a duplicate intake registration \
         re-pushed it onto `BodyActionsR.pending`."
    );
}
