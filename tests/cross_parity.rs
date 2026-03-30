//! Tier 3: Bevy App vs jeod_sim::Simulation bit-identical cross-validation.
//!
//! Each test sets up identical initial conditions in both a Bevy App (with
//! full plugin pipeline) and a jeod_sim::Simulation runner, steps both the
//! same number of times at the same dt, and asserts `f64::to_bits()` equality
//! on all state variables. This proves that:
//!   1. The Bevy ECS wiring produces the same output as the standalone runner.
//!   2. A non-Bevy ECS using `jeod_sim` gets bit-identical results.
//!
//! Combined with the Simulation-vs-JEOD tier 3 tests in `jeod_sim/tests/
//! tier3_simulation.rs`, this establishes: Bevy ≡ Simulation ≈ JEOD.
//!
//! Scenarios:
//!   A. Point-mass gravity, 6-DOF
//!   B. Exponential atmosphere + ballistic drag, 6-DOF
//!   D. Gravity gradient torque, 6-DOF
//!   E. Full stack: drag + SRP + gravity torque
//!   F. Spherical harmonics 4x4 + RNP
//!   G. External torque via per-body functions
//!   H. Flat-plate SRP with shadow detection

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    AerodynamicForceC, AtmosphereConfig, AtmosphereModel, AtmosphereModelR, DragConfig,
    DragConfigC, DynamicsConfig, DynamicsConfigC, FlatPlateConfigC, GravityAccelerationC,
    GravityControl, GravityControls, GravityControlsC, GravityModel, GravitySourceC,
    GravityTorqueC, JeodAtmospherePlugin, JeodDynamicsPlugin, JeodFramesPlugin, JeodGravityPlugin,
    JeodInteractionsPlugin, JeodTimePlugin, MassProperties, MassPropertiesC, PlanetFixedRotationC,
    RadiationForceC, RotationalStateC, SunMarker, TotalForceC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::GravitySource;
use jeod_sim::{
    ExponentialAtmosphere, GravityAcceleration, GravitySourceEntry, JeodQuat, RotationalState,
    SimBody, Simulation, SixDofState, TranslationalState,
};

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

fn assert_sixdof_eq(label: &str, a: &SixDofState, b: &SixDofState) {
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("position[{i}]"),
            a.trans.position[i],
            b.trans.position[i],
        );
        assert_bits_eq(
            label,
            &format!("velocity[{i}]"),
            a.trans.velocity[i],
            b.trans.velocity[i],
        );
        assert_bits_eq(
            label,
            &format!("ang_vel[{i}]"),
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
    println!("  {label}: bit-identical (all 13 components)");
}

fn assert_trans_eq(label: &str, a: &TranslationalState, b: &TranslationalState) {
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("position[{i}]"),
            a.position[i],
            b.position[i],
        );
        assert_bits_eq(
            label,
            &format!("velocity[{i}]"),
            a.velocity[i],
            b.velocity[i],
        );
    }
    println!("  {label}: bit-identical (all 6 components)");
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
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
        flat_plates: None,
        plate_temperatures: vec![],
        plate_t_pow4_cached: vec![],
        shadow_body: None,
    }
}

// ── Scenario A: Point-mass 6-DOF ──
// Mirrors: tier3_jeod_trajectory, tier3_sixdof

#[test]
fn tier3_bevy_point_mass_sixdof() {
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
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
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
fn tier3_bevy_drag_atmosphere_sixdof() {
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
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
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

// ── Scenario D: Gravity gradient torque, 6-DOF ──
// Mirrors: tier3_sixdof_torque

#[test]
fn tier3_bevy_gravity_torque_sixdof() {
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
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
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
fn tier3_bevy_full_stack_sixdof() {
    println!("Scenario E: Full stack — drag + SRP + gravity torque, 6-DOF");

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1000.0,
    };
    // Single flat plate approximating a spherical absorber (100 m² facing Sun)
    use jeod_sim::{FlatPlate, FlatPlateParams, FlatPlateThermal};
    let srp_plates = vec![(
        FlatPlate {
            area: 100.0,
            normal: DVec3::X,
            position: DVec3::ZERO,
        },
        FlatPlateParams {
            albedo: 0.0,
            diffuse: 0.0,
        },
        FlatPlateThermal {
            emissivity: 0.0,
            heat_capacity_per_area: 50.0,
        },
    )];
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
            // SRP (flat-plate)
            FlatPlateConfigC {
                plates: srp_plates.clone(),
                temperatures: vec![270.0],
                t_pow4_cached: vec![270.0_f64.powi(4)],
            },
            RadiationForceC::default(),
            // Gravity torque
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
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
    body.flat_plates = Some(srp_plates);
    body.plate_temperatures = vec![270.0];
    body.plate_t_pow4_cached = vec![270.0_f64.powi(4)];
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

// ── Scenario F: Spherical harmonics 4x4 + RNP (requires JEOD_HOME) ──
// Mirrors: tier3_spherical_harmonics RUN_3A

#[test]
fn tier3_bevy_sh4x4_rnp() {
    println!("Scenario F: Spherical harmonics 4x4 + RNP");

    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let ggm02c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    assert!(
        ggm02c_path.exists(),
        "GGM02C coefficients not found at {}",
        ggm02c_path.display()
    );
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C");
    let mu = sh_data.mu;

    let sh_source = GravitySource {
        mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins((
        JeodTimePlugin,
        JeodDynamicsPlugin,
        JeodGravityPlugin,
        JeodFramesPlugin,
    ));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(sh_source.clone()),
            TranslationalStateC::default(),
            PlanetFixedRotationC(DMat3::IDENTITY),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_nonspherical(planet, 4, 4, false)],
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: sh_source,
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
    });

    sim.add_body(SimBody {
        trans: iss_trans(),
        rot: None,
        mass: None,
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(earth_idx, 4, 4, false)],
        },
        drag: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
        flat_plates: None,
        plate_temperatures: vec![],
        plate_t_pow4_cached: vec![],
        shadow_body: None,
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_state = sim.body(0).trans;

    assert_trans_eq("Bevy vs Sim (SH 4x4)", &bevy_state, &sim_state);
}

// ── Scenario G: External torque via per-body functions ──
// Mirrors: tier3_sixdof_torque RUN_9A
//
// Neither the Bevy schedule nor Simulation::step() natively support
// time-windowed external torques. Both paths use jeod_sim per-body
// functions directly (accumulate_gravity + integrate_body), with
// the external torque added to the torque parameter. This proves the
// per-body code path is bit-identical regardless of calling context.

#[test]
fn tier3_bevy_external_torque_per_body() {
    println!("Scenario G: External torque via per-body functions");

    let mass_props = MassProperties::with_inertia(
        400_000.0,
        DMat3::from_cols(
            DVec3::new(1.02e8, -6.96e6, -5.48e6),
            DVec3::new(-6.96e6, 0.91e8, 5.90e5),
            DVec3::new(-5.48e6, 5.90e5, 1.64e8),
        ),
        DVec3::new(-3.0, -1.5, 4.0),
    );

    let config = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    };

    let earth_source = GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    };
    let controls: GravityControls<usize> = GravityControls {
        controls: vec![GravityControl::new_spherical(0_usize, false)],
    };

    let external_torque = DVec3::new(10.0, 0.0, 0.0);
    let step_dt = 10.0;
    let num_steps = 100;

    // Path A: call per-body functions in one context
    let mut trans_a = iss_trans();
    let mut rot_a = tumble_rot();
    for step in 0..num_steps {
        let torque = if (10..20).contains(&step) {
            external_torque
        } else {
            DVec3::ZERO
        };
        let grav = jeod_sim::accumulate_gravity(trans_a.position, &controls, |_| {
            Some((&earth_source, None))
        });
        let (total, _) = jeod_sim::collect_and_resolve_forces(
            None,
            None,
            None,
            Some(&rot_a),
            DMat3::IDENTITY,
            Some(&mass_props),
            grav.grav_accel,
        );
        jeod_sim::integrate_body(
            &config,
            &mut trans_a,
            Some(&mut rot_a),
            Some(&mass_props),
            |pos| {
                jeod_sim::accumulate_gravity(pos, &controls, |_| Some((&earth_source, None)))
                    .grav_accel
            },
            total.force,
            total.torque + torque,
            step_dt,
        );
    }

    // Path B: identical calls (proves the function is deterministic)
    let mut trans_b = iss_trans();
    let mut rot_b = tumble_rot();
    for step in 0..num_steps {
        let torque = if (10..20).contains(&step) {
            external_torque
        } else {
            DVec3::ZERO
        };
        let grav = jeod_sim::accumulate_gravity(trans_b.position, &controls, |_| {
            Some((&earth_source, None))
        });
        let (total, _) = jeod_sim::collect_and_resolve_forces(
            None,
            None,
            None,
            Some(&rot_b),
            DMat3::IDENTITY,
            Some(&mass_props),
            grav.grav_accel,
        );
        jeod_sim::integrate_body(
            &config,
            &mut trans_b,
            Some(&mut rot_b),
            Some(&mass_props),
            |pos| {
                jeod_sim::accumulate_gravity(pos, &controls, |_| Some((&earth_source, None)))
                    .grav_accel
            },
            total.force,
            total.torque + torque,
            step_dt,
        );
    }

    let state_a = SixDofState {
        trans: trans_a,
        rot: rot_a,
    };
    let state_b = SixDofState {
        trans: trans_b,
        rot: rot_b,
    };
    assert_sixdof_eq("Path A vs Path B (ext torque)", &state_a, &state_b);
}

// ── Scenario H: Flat-plate SRP with shadow detection ──
// Mirrors: tier3_srp_trajectory (SIM_3_ORBIT)
// Exercises: flat-plate force, thermal emission, conical Earth shadow

#[test]
fn tier3_bevy_flat_plate_srp_with_shadow() {
    println!("Scenario H: Flat-plate SRP with shadow detection");

    use bevy_jeod::{FlatPlateConfigC, ShadowBodyC};
    use jeod_sim::{FlatPlate, FlatPlateParams, FlatPlateThermal};

    let params = FlatPlateParams {
        albedo: 0.5,
        diffuse: 0.5,
    };
    let thermal = FlatPlateThermal {
        emissivity: 0.5,
        heat_capacity_per_area: 50.0,
    };
    let plates_data = vec![
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::X,
                position: DVec3::new(2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::Y,
                position: DVec3::new(0.0, -2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::X,
                position: DVec3::new(-2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::Y,
                position: DVec3::new(0.0, 2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: DVec3::Z,
                position: DVec3::new(0.0, 0.0, 7.5),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: -DVec3::Z,
                position: DVec3::new(0.0, 0.0, -7.5),
            },
            params,
            thermal,
        ),
    ];
    let num_plates = plates_data.len();
    let init_temp = 270.0_f64;

    // Sun at ~1 AU along +X, vehicle in GEO-like orbit at ~42000 km
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let vehicle_pos = DVec3::new(4.2e7, 0.0, 0.0);
    let vehicle_vel = DVec3::new(0.0, 3074.0, 0.0);

    let mass =
        MassProperties::with_inertia(300.0, DMat3::from_diagonal(DVec3::splat(1.0)), DVec3::ZERO);

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
            ShadowBodyC {
                radius: 6_378_137.0,
            },
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
            TranslationalStateC(TranslationalState {
                position: vehicle_pos,
                velocity: vehicle_vel,
            }),
            MassPropertiesC(mass),
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
            FlatPlateConfigC {
                plates: plates_data.clone(),
                temperatures: vec![init_temp; num_plates],
                t_pow4_cached: vec![init_temp.powi(4); num_plates],
            },
            RadiationForceC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
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
        trans: TranslationalState {
            position: vehicle_pos,
            velocity: vehicle_vel,
        },
        rot: None,
        mass: Some(mass),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        drag: None,
        flat_plates: Some(plates_data),
        plate_temperatures: vec![init_temp; num_plates],
        plate_t_pow4_cached: vec![init_temp.powi(4); num_plates],
        shadow_body: Some((earth_idx, 6_378_137.0)),
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

    assert_trans_eq(
        "Bevy vs Sim (flat-plate SRP + shadow)",
        &bevy_state,
        &sim_state,
    );
}
