//! Bevy ECS Kepler orbit example.
//!
//! Spawns an Earth gravity source and a satellite in a 400 km circular orbit,
//! then propagates for approximately one orbital period using JEOD's RK4
//! integrator in FixedUpdate. Prints orbital state every 100 steps and exits
//! after ~1 orbit.
//!
//! Virtual time is sped up by a large factor so that FixedUpdate steps
//! accumulate rapidly even though wall-clock time advances slowly.

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, FrameDerivativesC, GravityAccelerationC, GravityControlsC, GravitySourceC,
    JeodPlugin, JeodSet, MassPropertiesC, TotalForceC, TranslationalStateC,
};
use glam::DVec3;
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, MassProperties,
    TranslationalState,
};
use std::time::Duration;

const MU_EARTH: f64 = 3.986004418e14;

fn eccentricity(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    let h = position.cross(velocity);
    let e_vec = velocity.cross(h) / mu - position.normalize();
    e_vec.length()
}

fn main() {
    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(0))))
        .insert_resource(Time::<Fixed>::from_seconds(10.0))
        .add_plugins(JeodPlugin)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, print_state.after(JeodSet::Integration))
        .insert_resource(StepCounter(0))
        .run();
}

#[derive(Resource)]
struct StepCounter(usize);

fn setup(mut commands: Commands, mut time: ResMut<Time<Virtual>>) {
    let mu_earth = MU_EARTH;
    let r0 = 6_778_137.0_f64;
    let v0 = (mu_earth / r0).sqrt();

    // Speed up virtual time so FixedUpdate accumulates steps rapidly.
    // Each real-time frame contributes ~1ms of wall-clock delta; with a
    // relative speed of 1e6, that becomes ~1000s of virtual time per frame,
    // yielding ~100 fixed-update steps (dt=10s) per frame.
    time.set_relative_speed_f64(1e6);

    // Spawn Earth as gravity source.
    let earth = commands
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            }),
        ))
        .id();

    // Spawn satellite with all required dynamics components.
    let controls = GravityControls {
        controls: vec![GravityControl::new_spherical(earth, false)],
    };

    commands.spawn((
        Name::new("Satellite"),
        TranslationalStateC(TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        }),
        MassPropertiesC(MassProperties::new(100.0)),
        GravityAccelerationC::default(),
        TotalForceC::default(),
        FrameDerivativesC::default(),
        DynamicsConfigC::default(),
        GravityControlsC(controls),
    ));

    println!("Bevy JEOD Kepler Orbit Example");
    println!("==============================");
    println!("Initial altitude: {:.1} km", (r0 - 6_378_137.0) / 1000.0);
}

fn print_state(
    query: Query<(&Name, &TranslationalStateC)>,
    mut counter: ResMut<StepCounter>,
    mut exit: MessageWriter<AppExit>,
) {
    // Stop counting once we have already requested exit.
    if counter.0 >= 560 {
        return;
    }

    counter.0 += 1;
    let mu_earth = MU_EARTH;

    for (name, state) in &query {
        if name.as_str() != "Satellite" {
            continue;
        }

        if counter.0 % 100 == 0 || counter.0 <= 1 {
            let v = state.velocity.length();
            let alt_km = (state.position.length() - 6_378_137.0) / 1000.0;

            let e_mag = eccentricity(mu_earth, state.position, state.velocity);
            println!(
                "step={:5}  t={:8.0}s  alt={:7.1}km  v={:.1}m/s  e={:.2e}",
                counter.0,
                counter.0 as f64 * 10.0,
                alt_km,
                v,
                e_mag
            );
        }

        // Run for ~1 orbit (~560 steps at dt=10s for 400 km altitude).
        if counter.0 >= 560 {
            println!("Completed ~1 orbit. Exiting.");
            exit.write(AppExit::Success);
            return;
        }
    }
}
