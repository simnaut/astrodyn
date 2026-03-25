//! Bevy App 6-DOF parity test.
//!
//! Validates that the Bevy integration_system produces identical state to
//! calling rk4_sixdof_step() directly. Tests system wiring, not physics.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod_dynamics::{
    DynamicsConfigC, GravityAccelerationC, GravityControlsC, GravitySourceC, MassPropertiesC,
    RotationalStateC, TotalForceC, TranslationalStateC,
};
use bevy_jeod_dynamics::JeodDynamicsPlugin;
use bevy_jeod_gravity::JeodGravityPlugin;
use bevy_jeod_time::JeodTimePlugin;
use glam::{DMat3, DVec3};
use jeod_dynamics::{
    DynamicsConfig, MassProperties, RotationalState, SixDofState, TranslationalState,
};
use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource};
use jeod_math::JeodQuat;

const MU_EARTH: f64 = 3.986004418e14;
const DT: f64 = 10.0;
const NUM_STEPS: usize = 100;

/// ISS-like initial translational state: 400 km circular orbit.
fn initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

/// Non-trivial initial rotational state with tumble.
fn initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.001, 0.0, 0.001),
    }
}

/// ISS-like mass properties with realistic diagonal inertia.
fn mass_props() -> MassProperties {
    MassProperties::with_inertia(
        400_000.0,
        DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
        DVec3::ZERO,
    )
}

/// Build a minimal Bevy App with the JEOD dynamics, gravity, and time plugins.
/// Returns the App and the Entity ID of the spawned vehicle.
fn build_app() -> (App, Entity, Entity) {
    let mut app = App::new();

    // Minimal plugins for headless operation (includes TimePlugin for Time<Fixed>).
    app.add_plugins(MinimalPlugins);

    // Set fixed timestep before adding JEOD plugins.
    app.insert_resource(Time::<Fixed>::from_seconds(DT));

    // JEOD plugins: sets up system ordering, gravity computation, integration, and time.
    app.add_plugins(JeodDynamicsPlugin);
    app.add_plugins(JeodGravityPlugin);
    app.add_plugins(JeodTimePlugin);

    // Spawn planet entity (gravity source at origin).
    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            }),
            TranslationalStateC(TranslationalState::default()),
        ))
        .id();

    // Spawn vehicle entity with all required components for 6-DOF integration.
    let controls = GravityControls {
        controls: vec![GravityControl::new(planet, false)],
    };

    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("Vehicle"),
            TranslationalStateC(initial_trans()),
            RotationalStateC(initial_rot()),
            MassPropertiesC(mass_props()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(controls),
            GravityAccelerationC::default(),
            TotalForceC::default(),
        ))
        .id();

    (app, planet, vehicle)
}

/// Run 100 integration steps through the Bevy ECS and return the final
/// translational and rotational state from the vehicle entity.
fn run_bevy_steps(app: &mut App, vehicle: Entity) -> SixDofState {
    for _ in 0..NUM_STEPS {
        // Manually advance Time<Fixed> by one timestep so that
        // integration_system sees a non-zero delta_secs_f64().
        // This bypasses the virtual-time accumulation path and gives
        // deterministic control over exactly how many fixed steps execute.
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));

        // Run the FixedUpdate schedule directly (contains gravity computation,
        // force collection, integration, time advance, etc.).
        app.world_mut().run_schedule(FixedUpdate);
    }

    let world = app.world();
    let trans = world.get::<TranslationalStateC>(vehicle).unwrap();
    let rot = world.get::<RotationalStateC>(vehicle).unwrap();
    SixDofState {
        trans: trans.0,
        rot: rot.0,
    }
}

/// Run 100 integration steps using the pure rk4_sixdof_step function with
/// identical gravity computation, returning the final state.
fn run_pure_steps() -> SixDofState {
    let mp = mass_props();
    let mu = MU_EARTH;

    let mut state = SixDofState {
        trans: initial_trans(),
        rot: initial_rot(),
    };

    for _ in 0..NUM_STEPS {
        state = jeod_dynamics::rk4_sixdof_step(
            &state,
            |s| {
                // Replicate exactly what integration_system computes:
                // jeod_gravity::gravitation with PointMass, DMat3::IDENTITY,
                // None degree/order, perturbing_only=false.
                // This reduces to: -mu/r^3 * r
                let r = s.trans.position.length();
                -mu / (r * r * r) * s.trans.position
            },
            |_| DVec3::ZERO, // No external torque
            &mp,
            DT,
        );
    }

    state
}

#[test]
fn bevy_integration_matches_pure_rk4_sixdof() {
    let (mut app, _planet, vehicle) = build_app();

    let bevy_state = run_bevy_steps(&mut app, vehicle);
    let pure_state = run_pure_steps();

    // Position parity: should be identical to machine precision since both
    // paths call the same rk4_sixdof_step with the same gravity formula.
    // Observed: ~5e-10 m (consistent with f64 ULP at ~7e6 m scale over 100 steps).
    let pos_diff = (bevy_state.trans.position - pure_state.trans.position).length();
    assert!(
        pos_diff < 1e-8,
        "Position difference between Bevy and pure RK4: {} m (exceeds 1e-8 m)\n\
         Bevy:  {:?}\n\
         Pure:  {:?}",
        pos_diff,
        bevy_state.trans.position,
        pure_state.trans.position,
    );

    // Velocity parity.
    // Observed: ~9e-13 m/s.
    let vel_diff = (bevy_state.trans.velocity - pure_state.trans.velocity).length();
    assert!(
        vel_diff < 1e-11,
        "Velocity difference between Bevy and pure RK4: {} m/s (exceeds 1e-11 m/s)\n\
         Bevy:  {:?}\n\
         Pure:  {:?}",
        vel_diff,
        bevy_state.trans.velocity,
        pure_state.trans.velocity,
    );

    // Quaternion parity.
    let q_bevy = bevy_state.rot.quaternion.data;
    let q_pure = pure_state.rot.quaternion.data;
    let q_diff: f64 = (0..4).map(|i| (q_bevy[i] - q_pure[i]).powi(2)).sum::<f64>().sqrt();
    assert!(
        q_diff < 1e-14,
        "Quaternion difference between Bevy and pure RK4: {} (exceeds 1e-14)\n\
         Bevy:  {:?}\n\
         Pure:  {:?}",
        q_diff,
        q_bevy,
        q_pure,
    );

    // Angular velocity parity.
    let omega_diff = (bevy_state.rot.ang_vel_body - pure_state.rot.ang_vel_body).length();
    assert!(
        omega_diff < 1e-14,
        "Angular velocity difference between Bevy and pure RK4: {} rad/s (exceeds 1e-14)\n\
         Bevy:  {:?}\n\
         Pure:  {:?}",
        omega_diff,
        bevy_state.rot.ang_vel_body,
        pure_state.rot.ang_vel_body,
    );
}
