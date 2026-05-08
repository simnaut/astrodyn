//! Bevy App vs astrodyn_runner::Simulation parity test.
//!
//! Validates that the Bevy ECS pipeline and the standalone Simulation runner
//! produce bit-identical state from the same initial conditions. Both go
//! through the same `astrodyn` per-body functions, so any difference means
//! the Bevy glue layer is wiring something incorrectly.
//!
//! Combined with Tier 3 tests (Simulation vs JEOD CSV), this establishes:
//!   Bevy App ≡ Simulation ≈ JEOD

use std::time::Duration;

use astrodyn::{
    DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource, IntegratorType,
    JeodQuat, MassProperties, RotationalState, SixDofState, TranslationalState,
};
use astrodyn_bevy::{
    AstrodynPlugin, DynamicsConfigC, GravityControlsC, GravitySourceC, IntegratorTypeC,
    MassPropertiesC, RotationalStateC, SourceInertialPositionC, TranslationalStateC,
};
use bevy::prelude::*;
use glam::{DMat3, DVec3};

const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;
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

/// Build a minimal Bevy App with the JEOD plugin.
/// Returns the App and the Entity ID of the spawned vehicle.
fn build_app() -> (App, Entity, Entity) {
    let mut app = App::new();

    // Minimal plugins for headless operation (includes TimePlugin for Time<Fixed>).
    app.add_plugins(MinimalPlugins);

    // Set fixed timestep before adding JEOD plugins.
    app.insert_resource(Time::<Fixed>::from_seconds(DT));

    // Unified JEOD plugin: sets up system ordering, gravity, integration, time, etc.
    app.add_plugins(AstrodynPlugin);

    // Spawn planet entity (gravity source at origin).
    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState::default()),
        ))
        .id();

    // Spawn vehicle entity with all required components for 6-DOF integration.
    let controls = GravityControls {
        controls: vec![GravityControl::new_spherical(planet, false)],
    };

    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("Vehicle"),
            TranslationalStateC::<astrodyn::Earth>::from(initial_trans()),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(mass_props()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(controls),
        ))
        .id();

    (app, planet, vehicle)
}

/// Run 100 integration steps through the Bevy ECS and return the final
/// translational and rotational state from the vehicle entity.
fn run_bevy_steps(app: &mut App, vehicle: Entity) -> SixDofState {
    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let world = app.world();
    let trans = world
        .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
        .unwrap();
    let rot = world.get::<RotationalStateC>(vehicle).unwrap();
    SixDofState {
        trans: trans.0.to_untyped(),
        rot: rot.0.to_untyped(),
    }
}

/// Run 100 integration steps via astrodyn_runner::Simulation with identical
/// initial conditions and gravity setup.
fn run_simulation_steps() -> SixDofState {
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);

    let mut earth_entry = astrodyn::GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth = sim.add_source("Earth", earth_entry);

    sim.add_body(astrodyn::VehicleConfig {
        trans: initial_trans().into(),
        rot: Some(initial_rot()),
        mass: Some(mass_props()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    }
}

/// Assert two f64 values are bit-identical.
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

fn assert_sixdof_bit_identical(label: &str, a: &SixDofState, b: &SixDofState) {
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("pos[{i}]"),
            a.trans.position[i],
            b.trans.position[i],
        );
        assert_bits_eq(
            label,
            &format!("vel[{i}]"),
            a.trans.velocity[i],
            b.trans.velocity[i],
        );
        assert_bits_eq(
            label,
            &format!("omega[{i}]"),
            a.rot.ang_vel_body[i],
            b.rot.ang_vel_body[i],
        );
    }
    for i in 0..4 {
        assert_bits_eq(
            label,
            &format!("quat[{i}]"),
            a.rot.quaternion.data[i],
            b.rot.quaternion.data[i],
        );
    }
}

/// Bevy App vs astrodyn_runner::Simulation — bit-identical output required.
///
/// Both paths go through `accumulate_gravity` → `collect_and_resolve_forces`
/// → `integrate_body` with the same `astrodyn_*` functions underneath. Any
/// difference — even a single ULP — means the Bevy wiring is wrong.
#[test]
fn tier3_bevy_matches_simulation_bit_identical() {
    let (mut app, _planet, vehicle) = build_app();

    let bevy_state = run_bevy_steps(&mut app, vehicle);
    let sim_state = run_simulation_steps();

    assert_sixdof_bit_identical("Bevy vs Sim", &bevy_state, &sim_state);
}

/// Same as above but with RKF45 integrator — verifies IntegratorTypeC dispatch.
#[test]
fn tier3_bevy_rkf45_matches_simulation_bit_identical() {
    // Build Bevy app with RKF45
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState::default()),
        ))
        .id();

    let controls = GravityControls {
        controls: vec![GravityControl::new_spherical(planet, false)],
    };

    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("Vehicle"),
            TranslationalStateC::<astrodyn::Earth>::from(initial_trans()),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(mass_props()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(controls),
            IntegratorTypeC(IntegratorType::Rkf45),
        ))
        .id();

    let bevy_state = run_bevy_steps(&mut app, vehicle);

    // Run Simulation with RKF45
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);

    let mut earth_entry = astrodyn::GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth = sim.add_source("Earth", earth_entry);

    sim.add_body(astrodyn::VehicleConfig {
        trans: initial_trans().into(),
        rot: Some(initial_rot()),
        mass: Some(mass_props()),
        integrator: IntegratorType::Rkf45,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_bit_identical("Bevy RKF45 vs Sim RKF45", &bevy_state, &sim_state);
}
