//! Bevy parity test for `BodyAction::InitLvlhRot` (port of JEOD's
//! `DynBodyInitLvlhRotState`).
//!
//! Spawns a Bevy vehicle with a placeholder rotational state, fires a
//! `BodyAction::InitLvlhRot` with a known reference orbit and a known
//! LVLH→body attitude / LVLH-relative angular velocity, and asserts the
//! post-apply `RotationalStateC` matches what
//! `astrodyn_dynamics::body_init::init_rot_from_lvlh` (the underlying
//! kernel) computes for the same inputs.
//!
//! Tier 3 cross-validation against a JEOD verif sim is deferred: the
//! only JEOD verif that exercises `DynBodyInitLvlhRotState`
//! (`models/dynamics/body_action/verif/SIM_orbinit`) is a static
//! initialization test (no propagation of the resulting rotational
//! state through Trick), so a CSV-based trajectory comparison would
//! not exercise additional code beyond what this test already covers.
//! When a JEOD sim that propagates LVLH-rot-init attitude appears, the
//! Tier 3 test should land at that point.

mod common;

use std::time::Duration;

use astrodyn::{
    BodyAction, DynamicsConfig, JeodQuat, LvlhAngularVelocityFrame, MassProperties, RotationalState,
};
use astrodyn_bevy::{
    AstrodynPlugin, BodyActionEvent, GravitySourceC, MassPropertiesC, RotationalStateC,
    SourceInertialPositionC, TranslationalStateC,
};
use astrodyn_dynamics::body_init::{
    init_rot_from_lvlh, LvlhAngularVelocityFrame as KernelLvlhFrame,
};
use bevy::prelude::*;
use glam::DVec3;

use common::earth_source;

/// ISS-like inclined circular reference orbit (matches the LVLH frame
/// the kernel will construct).
fn reference_orbit() -> (DVec3, DVec3) {
    const EARTH_MU: f64 = 3.986_004_415e14;
    const EARTH_R_EQ: f64 = 6_378_137.0;
    let r = EARTH_R_EQ + 400_000.0;
    let v = (EARTH_MU / r).sqrt();
    let inc = 51.6_f64.to_radians();
    let pos = DVec3::new(r * 0.6, r * 0.8, 0.0);
    let vel = DVec3::new(-v * 0.8 * inc.cos(), v * 0.6 * inc.cos(), v * inc.sin());
    (pos, vel)
}

/// `FixedUpdate` step duration. The body-action systems are pinned to
/// `FixedUpdate` by `AstrodynPlugin`, so the test must drive that
/// schedule (not the default `Update`).
const DT: f64 = 0.03125;

/// Build a minimal `App` with the JEOD plugin, an Earth gravity source,
/// and a single vehicle entity that owns translational + rotational +
/// mass components. Returns the vehicle entity.
fn build_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();
    let _ = earth;

    let vehicle = app
        .world_mut()
        .spawn((
            // Placeholder translational state — `InitLvlhRot` only
            // touches the rotational component; the trans state is
            // unused by the action and irrelevant to the assertion.
            TranslationalStateC::<astrodyn::Earth>::default(),
            // Placeholder rotational state we expect the action to
            // overwrite on `update`.
            RotationalStateC::from(RotationalState::default()),
            MassPropertiesC::from(MassProperties::new(1_000.0)),
            astrodyn_bevy::DynamicsConfigC(DynamicsConfig {
                translational_dynamics: false,
                rotational_dynamics: true,
                three_dof: false,
            }),
        ))
        .id();

    (app, vehicle)
}

fn write_msg(app: &mut App, msg: BodyActionEvent) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
        .write(msg);
}

fn read_rot(app: &App, vehicle: Entity) -> RotationalState {
    app.world()
        .entity(vehicle)
        .get::<RotationalStateC>()
        .expect("rot state present")
        .0
        .to_untyped()
}

#[test]
fn bevy_parity_init_lvlh_rot_writes_rotational_state_in_body_frame() {
    let (mut app, vehicle) = build_app();

    let (ref_pos, ref_vel) = reference_orbit();
    // Non-trivial LVLH→body attitude: 1.0 rad about a non-axis-aligned
    // direction. Picked the same shape as the kernel's
    // `lvlh_rot_nontrivial_attitude_round_trips` so a regression in the
    // composition order surfaces as an attitude mismatch here too.
    let axis = DVec3::new(1.0, 2.0, -1.0).normalize();
    let q_lvlh_body = JeodQuat::left_quat_from_eigen_rotation(1.0, axis);
    let ang_vel_lvlh_to_body = DVec3::new(0.005, -0.01, 0.02);

    // Compute the expected post-apply state via the kernel directly.
    // The Bevy `body_action_system` should call the same kernel.
    let expected = init_rot_from_lvlh(
        q_lvlh_body,
        ang_vel_lvlh_to_body,
        KernelLvlhFrame::Body,
        ref_pos,
        ref_vel,
    );

    write_msg(
        &mut app,
        BodyActionEvent::add(
            vehicle,
            BodyAction::InitLvlhRot {
                q_lvlh_body,
                ang_vel_lvlh_to_body,
                ang_vel_frame: LvlhAngularVelocityFrame::Body,
                reference_position: ref_pos,
                reference_velocity: ref_vel,
            },
            Some("vehicle.lvlh_rot_init"),
        ),
    );

    // `AstrodynPlugin` pins the body-action systems to `FixedUpdate`
    // (not `Update`) so a `MinimalPlugins`-only test must drive the
    // fixed schedule explicitly: advance the `Time::<Fixed>` clock by
    // one DT, then run that schedule.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    let state = read_rot(&app, vehicle);
    let dq: f64 = (0..4)
        .map(|i| (state.quaternion.data[i] - expected.quaternion.data[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        dq < 1e-14,
        "post-apply quaternion must match kernel output: dq = {dq}, applied = {:?}, expected = {:?}",
        state.quaternion.data,
        expected.quaternion.data
    );
    let dw = (state.ang_vel_body - expected.ang_vel_body).length();
    assert!(
        dw < 1e-14,
        "post-apply ang vel must match kernel output: dw = {dw}, applied = {:?}, expected = {:?}",
        state.ang_vel_body,
        expected.ang_vel_body,
    );
}

#[test]
fn bevy_parity_init_lvlh_rot_lvlh_rate_frame_dispatches() {
    // Same as the body-frame test, but with the user supplying the
    // angular-velocity input in the LVLH frame instead of the body
    // frame. This exercises the `rate_in_parent` branch of the kernel
    // (the JEOD `apply_user_inputs` rotational arm) end-to-end through
    // the Bevy adapter.
    let (mut app, vehicle) = build_app();

    let (ref_pos, ref_vel) = reference_orbit();
    let axis = DVec3::new(0.0, 0.0, 1.0);
    let q_lvlh_body = JeodQuat::left_quat_from_eigen_rotation(0.7, axis);
    let ang_vel_in_lvlh = DVec3::new(0.001, 0.002, -0.003);

    let expected = init_rot_from_lvlh(
        q_lvlh_body,
        ang_vel_in_lvlh,
        KernelLvlhFrame::Lvlh,
        ref_pos,
        ref_vel,
    );

    write_msg(
        &mut app,
        BodyActionEvent::add(
            vehicle,
            BodyAction::InitLvlhRot {
                q_lvlh_body,
                ang_vel_lvlh_to_body: ang_vel_in_lvlh,
                ang_vel_frame: LvlhAngularVelocityFrame::Lvlh,
                reference_position: ref_pos,
                reference_velocity: ref_vel,
            },
            None,
        ),
    );
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    let state = read_rot(&app, vehicle);
    let dq: f64 = (0..4)
        .map(|i| (state.quaternion.data[i] - expected.quaternion.data[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(dq < 1e-14, "lvlh-rate-frame quaternion mismatch: dq = {dq}");
    let dw = (state.ang_vel_body - expected.ang_vel_body).length();
    assert!(dw < 1e-14, "lvlh-rate-frame ang vel mismatch: dw = {dw}");
}
