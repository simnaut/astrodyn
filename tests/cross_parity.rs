//! Cross-parity: Bevy App vs jeod_sim::Simulation for every tier 3 physics scenario.
//!
//! Each test sets up identical initial conditions in both execution paths,
//! runs the same number of steps at the same dt, and asserts bit-level
//! agreement on all state variables. This proves that a non-Bevy ECS using
//! `jeod_sim` produces the same output as the Bevy pipeline.
//!
//! Scenarios mirror the physics exercised by tier 3 tests:
//!   A. Point-mass gravity, 6-DOF (tier3_jeod_trajectory / tier3_sixdof)
//!   B. Exponential atmosphere + ballistic drag, 6-DOF (tier3_drag_trajectory)
//!   C. Solar radiation pressure, 3-DOF (tier3_srp_trajectory)
//!   D. Gravity gradient torque, 6-DOF (tier3_sixdof_torque)
//!   E. Full stack: drag + SRP + gravity torque (combined)

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    AerodynamicForceC, AtmosphereConfig, AtmosphereModel, AtmosphereModelR, DragConfig,
    DragConfigC, DynamicsConfig, DynamicsConfigC, GravityAccelerationC, GravityControl,
    GravityControls, GravityControlsC, GravityModel, GravitySourceC, GravityTorqueC,
    JeodAtmospherePlugin, JeodDynamicsPlugin, JeodFramesPlugin, JeodGravityPlugin,
    JeodInteractionsPlugin, JeodTimePlugin, MassProperties, MassPropertiesC, RadiationForceC,
    RotationalStateC, SrpConfig, SrpConfigC, SunMarker, TotalForceC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_atmosphere::exponential::ExponentialAtmosphere;
use jeod_dynamics::{GravityAcceleration, RotationalState, SixDofState, TranslationalState};
use jeod_gravity::GravitySource;
use jeod_math::JeodQuat;
use jeod_sim::{GravitySourceEntry, SimBody, Simulation};

const MU_EARTH: f64 = 3.986004418e14;
const DT: f64 = 10.0;
const NUM_STEPS: usize = 100;

// ── Shared initial conditions ──

fn iss_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

fn tumble_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5),
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    }
}

fn iss_mass() -> MassProperties {
    MassProperties::with_inertia(
        400_000.0,
        DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
        DVec3::ZERO,
    )
}

fn earth_source() -> GravitySource {
    GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    }
}

// ── Bevy helpers ──

fn step_bevy(app: &mut App, n: usize) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

fn read_sixdof(world: &World, entity: Entity) -> SixDofState {
    SixDofState {
        trans: world.get::<TranslationalStateC>(entity).unwrap().0,
        rot: world.get::<RotationalStateC>(entity).unwrap().0,
    }
}

fn read_trans(world: &World, entity: Entity) -> TranslationalState {
    world.get::<TranslationalStateC>(entity).unwrap().0
}

fn assert_sixdof_eq(label: &str, a: &SixDofState, b: &SixDofState) {
    let pos_diff = (a.trans.position - b.trans.position).length();
    let vel_diff = (a.trans.velocity - b.trans.velocity).length();
    let q_a = a.rot.quaternion.data;
    let q_b = b.rot.quaternion.data;
    let q_diff: f64 = (0..4)
        .map(|i| (q_a[i] - q_b[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let omega_diff = (a.rot.ang_vel_body - b.rot.ang_vel_body).length();

    println!(
        "  {label}: pos={pos_diff:.2e} m  vel={vel_diff:.2e} m/s  \
         quat={q_diff:.2e}  omega={omega_diff:.2e} rad/s"
    );

    assert!(
        pos_diff < 1e-8,
        "{label}: position diff {pos_diff} m exceeds 1e-8 m"
    );
    assert!(
        vel_diff < 1e-11,
        "{label}: velocity diff {vel_diff} m/s exceeds 1e-11 m/s"
    );
    assert!(
        q_diff < 1e-14,
        "{label}: quaternion diff {q_diff} exceeds 1e-14"
    );
    assert!(
        omega_diff < 1e-14,
        "{label}: omega diff {omega_diff} rad/s exceeds 1e-14"
    );
}

fn assert_trans_eq(label: &str, a: &TranslationalState, b: &TranslationalState) {
    let pos_diff = (a.position - b.position).length();
    let vel_diff = (a.velocity - b.velocity).length();

    println!("  {label}: pos={pos_diff:.2e} m  vel={vel_diff:.2e} m/s");

    assert!(
        pos_diff < 1e-8,
        "{label}: position diff {pos_diff} m exceeds 1e-8 m"
    );
    assert!(
        vel_diff < 1e-11,
        "{label}: velocity diff {vel_diff} m/s exceeds 1e-11 m/s"
    );
}

fn new_sim_body_sixdof(earth_idx: usize, gradient: bool) -> SimBody {
    SimBody {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, gradient)],
        },
        drag: None,
        srp: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    }
}

// ── Scenario A: Point-mass 6-DOF ──
// Mirrors: tier3_jeod_trajectory, tier3_sixdof

#[test]
fn cross_parity_point_mass_sixdof() {
    println!("Scenario A: Point-mass gravity, 6-DOF");

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins((JeodTimePlugin, JeodDynamicsPlugin, JeodGravityPlugin));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            TranslationalStateC::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            RotationalStateC(tumble_rot()),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time =
        jeod_time::SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });
    sim.add_body(new_sim_body_sixdof(earth_idx, false));
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim", &bevy_state, &sim_state);
}

// ── Scenario B: Exponential atmosphere + drag, 6-DOF ──
// Mirrors: tier3_drag_trajectory

#[test]
fn cross_parity_drag_atmosphere_sixdof() {
    println!("Scenario B: Exponential atmosphere + drag, 6-DOF");

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1000.0,
    };
    let exp_atmos = ExponentialAtmosphere::default();

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins((
        JeodTimePlugin,
        JeodDynamicsPlugin,
        JeodGravityPlugin,
        JeodFramesPlugin,
        JeodAtmospherePlugin,
        JeodInteractionsPlugin,
    ));

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: 6_378_137.0,
            r_pol: 6_356_752.314_245,
            planet_omega: 0.0, // no wind for simplicity
        },
        planet_entity: None, // no rotation for exponential
    });

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            TranslationalStateC::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            RotationalStateC(tumble_rot()),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
            DragConfigC(drag_config),
            bevy_jeod::AtmosphericStateDynC::default(),
            AerodynamicForceC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time =
        jeod_time::SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: 6_378_137.0,
        r_pol: 6_356_752.314_245,
        planet_omega: 0.0,
    });

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.drag = Some(drag_config);
    body.atmospheric_state = Some(Default::default());
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (drag)", &bevy_state, &sim_state);
}

// ── Scenario C: Solar radiation pressure, 3-DOF ──
// Mirrors: tier3_srp_trajectory

#[test]
fn cross_parity_srp_3dof() {
    println!("Scenario C: Solar radiation pressure, 3-DOF");

    let srp_config = SrpConfig {
        cx_area: 100.0,
        rad_coeff: 1.5,
    };
    // Sun at ~1 AU along +X
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins((
        JeodTimePlugin,
        JeodDynamicsPlugin,
        JeodGravityPlugin,
        JeodInteractionsPlugin,
    ));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            TranslationalStateC::default(),
        ))
        .id();

    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
            SrpConfigC(srp_config),
            RadiationForceC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time =
        jeod_time::SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });
    let sun_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: sun_pos,
        t_inertial_pfix: None,
    });
    sim.sun_source = Some(sun_idx);

    sim.add_body(SimBody {
        trans: iss_trans(),
        rot: None,
        mass: Some(iss_mass()),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        drag: None,
        srp: Some(srp_config),
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_state = sim.body(0).trans;

    assert_trans_eq("Bevy vs Sim (SRP)", &bevy_state, &sim_state);
}

// ── Scenario D: Gravity gradient torque, 6-DOF ──
// Mirrors: tier3_sixdof_torque

#[test]
fn cross_parity_gravity_torque_sixdof() {
    println!("Scenario D: Gravity gradient torque, 6-DOF");

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins((
        JeodTimePlugin,
        JeodDynamicsPlugin,
        JeodGravityPlugin,
        JeodInteractionsPlugin,
    ));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            TranslationalStateC::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            RotationalStateC(tumble_rot()),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, true)], // gradient=true
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time =
        jeod_time::SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });

    let mut body = new_sim_body_sixdof(earth_idx, true); // gradient=true
    body.compute_gravity_torque = true;
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (grav torque)", &bevy_state, &sim_state);
}

// ── Scenario E: Full stack — drag + SRP + gravity torque ──
// Mirrors: combined tier 3 physics

#[test]
fn cross_parity_full_stack_sixdof() {
    println!("Scenario E: Full stack — drag + SRP + gravity torque, 6-DOF");

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1000.0,
    };
    let srp_config = SrpConfig {
        cx_area: 100.0,
        rad_coeff: 1.5,
    };
    let exp_atmos = ExponentialAtmosphere::default();
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins((
        JeodTimePlugin,
        JeodDynamicsPlugin,
        JeodGravityPlugin,
        JeodFramesPlugin,
        JeodAtmospherePlugin,
        JeodInteractionsPlugin,
    ));

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: 6_378_137.0,
            r_pol: 6_356_752.314_245,
            planet_omega: 7.292_115_146_706_388e-5, // Earth co-rotation wind
        },
        planet_entity: None,
    });

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            TranslationalStateC::default(),
        ))
        .id();

    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            RotationalStateC(tumble_rot()),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, true)], // gradient for torque
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
            // Drag
            DragConfigC(drag_config),
            bevy_jeod::AtmosphericStateDynC::default(),
            AerodynamicForceC::default(),
            // SRP
            SrpConfigC(srp_config),
            RadiationForceC::default(),
            // Gravity torque
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time =
        jeod_time::SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });
    let sun_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: sun_pos,
        t_inertial_pfix: None,
    });
    sim.sun_source = Some(sun_idx);
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: 6_378_137.0,
        r_pol: 6_356_752.314_245,
        planet_omega: 7.292_115_146_706_388e-5,
    });

    let mut body = new_sim_body_sixdof(earth_idx, true); // gradient=true
    body.drag = Some(drag_config);
    body.srp = Some(srp_config);
    body.compute_gravity_torque = true;
    body.atmospheric_state = Some(Default::default());
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (full stack)", &bevy_state, &sim_state);

    // Print intermediates for diagnostic visibility
    println!(
        "  Gravity accel magnitude: {:.6e} m/s^2",
        body.gravity_accel.grav_accel.length()
    );
    println!(
        "  Total non-grav force:    {:.6e} N",
        body.total_force.force.length()
    );
    println!(
        "  Total torque:            {:.6e} N*m",
        body.total_force.torque.length()
    );
}
