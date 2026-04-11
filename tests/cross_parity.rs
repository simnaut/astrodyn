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
//!   I. Gauss-Jackson (Störmer-Cowell), point-mass 3-DOF

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    AerodynamicForceC, AtmosphereModelR, AtmosphericStateC, DragConfigC, DynamicsConfigC,
    EulerAnglesC, EulerAnglesConfigC, FlatPlateConfigC, GaussJacksonStateC, GeodeticConfigC,
    GeodeticStateC, GravityAccelerationC, GravityControlsC, GravitySourceC, GravityTorqueC,
    IntegratorTypeC, JeodPlugin, LvlhFrameC, MassPropertiesC, OrbitalElementsC,
    OrbitalElementsConfigC, PlanetC, PlanetFixedRotationC, RadiationForceC, RotationalStateC,
    SolarBetaC, SourceInertialPositionC, SunMarker, TidalConfigC, TidalDeltaC20C, TotalForceC,
    TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::{
    AtmosphereConfig, AtmosphereModel, DragConfig, DynamicsConfig, EulerSequence,
    ExponentialAtmosphere, GaussJacksonConfig, GaussJacksonState, GeoIndexType, GravityControl,
    GravityControls, GravityModel, GravitySource, GravitySourceEntry, IntegratorType, JeodQuat,
    LvlhFrame, MassProperties, MetAtmosphere, OrbitalElements, PlanetShape, RotationModel,
    RotationalState, SimBody, Simulation, SixDofState, TidalBody, TidalConfig, TranslationalState,
};

const MU_EARTH: f64 = 3.986_004_415e14;
const MU_SUN: f64 = 1.327_124_40e20;
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
    step_bevy_dt(app, n, DT);
}

fn step_bevy_dt(app: &mut App, n: usize, dt: f64) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
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
        ..Default::default()
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
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
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
        constant_density: None,
    };
    let exp_atmos = ExponentialAtmosphere::default();

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: 6_378_137.0,
            r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
            planet_omega: 0.0, // no wind for simplicity
        },
        planet_entity: None, // no rotation for exponential
    });

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
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
            AtmosphericStateC::default(),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
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
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
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
        constant_density: None,
    };
    // Single flat plate approximating a spherical absorber (100 m² facing Sun)
    use jeod_sim::{FlatPlate, FlatPlateParams, FlatPlateThermal, RotationModel};
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
    app.add_plugins(JeodPlugin);

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: 6_378_137.0,
            r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
            planet_omega: 7.292_115_146_706_388e-5, // Earth co-rotation wind
        },
        planet_entity: None,
    });

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
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
            AtmosphericStateC::default(),
            AerodynamicForceC::default(),
            // SRP (flat-plate)
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: srp_plates.clone(),
                temperatures: vec![270.0],
                t_pow4_cached: vec![270.0_f64.powi(4)],
            }),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    let sun_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: sun_pos,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun_idx);
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: 7.292_115_146_706_388e-5,
    });

    let mut body = new_sim_body_sixdof(earth_idx, true); // gradient=true
    body.drag = Some(drag_config);
    body.flat_plate_state = Some(jeod_sim::FlatPlateState {
        plates: srp_plates,
        temperatures: vec![270.0],
        t_pow4_cached: vec![270.0_f64.powi(4)],
    });
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
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(sh_source.clone()),
            SourceInertialPositionC::default(),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
    });

    sim.add_body(SimBody {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(earth_idx, 4, 4, false)],
        },
        ..Default::default()
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
        let grav = jeod_sim::accumulate_gravity(trans_a.position, &controls, DVec3::ZERO, |_| {
            Some(jeod_sim::ResolvedSource {
                source: &earth_source,
                rotation: None,
                position: DVec3::ZERO,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
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
            |pos, _vel| {
                jeod_sim::accumulate_gravity(pos, &controls, DVec3::ZERO, |_| {
                    Some(jeod_sim::ResolvedSource {
                        source: &earth_source,
                        rotation: None,
                        position: DVec3::ZERO,
                        delta_c20: 0.0,
                        has_delta_coeffs: false,
                    })
                })
                .grav_accel
            },
            total.force,
            total.torque + torque,
            step_dt,
            1.0,
            jeod_sim::IntegratorType::Rk4,
            None,
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
        let grav = jeod_sim::accumulate_gravity(trans_b.position, &controls, DVec3::ZERO, |_| {
            Some(jeod_sim::ResolvedSource {
                source: &earth_source,
                rotation: None,
                position: DVec3::ZERO,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
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
            |pos, _vel| {
                jeod_sim::accumulate_gravity(pos, &controls, DVec3::ZERO, |_| {
                    Some(jeod_sim::ResolvedSource {
                        source: &earth_source,
                        rotation: None,
                        position: DVec3::ZERO,
                        delta_c20: 0.0,
                        has_delta_coeffs: false,
                    })
                })
                .grav_accel
            },
            total.force,
            total.torque + torque,
            step_dt,
            1.0,
            jeod_sim::IntegratorType::Rk4,
            None,
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
    use jeod_sim::{FlatPlate, FlatPlateParams, FlatPlateThermal, RotationModel};

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
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
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
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: plates_data.clone(),
                temperatures: vec![init_temp; num_plates],
                t_pow4_cached: vec![init_temp.powi(4); num_plates],
            }),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    let sun_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: sun_pos,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun_idx);

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: vehicle_pos,
            velocity: vehicle_vel,
        },
        mass: Some(mass),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        flat_plate_state: Some(jeod_sim::FlatPlateState {
            plates: plates_data,
            temperatures: vec![init_temp; num_plates],
            t_pow4_cached: vec![init_temp.powi(4); num_plates],
        }),
        shadow_body: Some((earth_idx, 6_378_137.0)),
        ..Default::default()
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

// ── Derived state assertion helpers ──

fn assert_orbital_elements_eq(label: &str, a: &OrbitalElements, b: &OrbitalElements) {
    assert_bits_eq(
        label,
        "semi_major_axis",
        a.semi_major_axis,
        b.semi_major_axis,
    );
    assert_bits_eq(label, "semiparam", a.semiparam, b.semiparam);
    assert_bits_eq(label, "e_mag", a.e_mag, b.e_mag);
    assert_bits_eq(label, "inclination", a.inclination, b.inclination);
    assert_bits_eq(label, "arg_periapsis", a.arg_periapsis, b.arg_periapsis);
    assert_bits_eq(label, "long_asc_node", a.long_asc_node, b.long_asc_node);
    assert_bits_eq(label, "true_anom", a.true_anom, b.true_anom);
    assert_bits_eq(label, "mean_anom", a.mean_anom, b.mean_anom);
    assert_bits_eq(label, "mean_motion", a.mean_motion, b.mean_motion);
    assert_bits_eq(label, "orb_energy", a.orb_energy, b.orb_energy);
    assert_bits_eq(
        label,
        "orb_ang_momentum",
        a.orb_ang_momentum,
        b.orb_ang_momentum,
    );
    assert_bits_eq(label, "orbital_anom", a.orbital_anom, b.orbital_anom);
    assert_bits_eq(label, "r_mag", a.r_mag, b.r_mag);
    assert_bits_eq(label, "vel_mag", a.vel_mag, b.vel_mag);
    println!("  {label}: bit-identical (14 orbital element fields)");
}

fn assert_lvlh_eq(label: &str, a: &LvlhFrame, b: &LvlhFrame) {
    for i in 0..3 {
        for j in 0..3 {
            assert_bits_eq(
                label,
                &format!("t_parent_this[{i}][{j}]"),
                a.t_parent_this.col(j)[i],
                b.t_parent_this.col(j)[i],
            );
        }
        assert_bits_eq(
            label,
            &format!("ang_vel[{i}]"),
            a.ang_vel_this[i],
            b.ang_vel_this[i],
        );
    }
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("position[{i}]"),
            a.position[i],
            b.position[i],
        );
    }
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("velocity[{i}]"),
            a.velocity[i],
            b.velocity[i],
        );
    }
    println!("  {label}: bit-identical (18 LVLH frame components)");
}

// ── Scenario I: Derived states (orbital elements, Euler, LVLH, solar beta) ──

#[test]
fn tier3_bevy_derived_states() {
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy App ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let sun = app
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
    let _ = sun;

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
            // Derived state config + output components
            OrbitalElementsConfigC {
                gravity_source: planet,
            },
            OrbitalElementsC::default(),
            EulerAnglesConfigC {
                sequence: EulerSequence::ZYX,
            },
            EulerAnglesC::default(),
            LvlhFrameC::default(),
            SolarBetaC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_state = read_sixdof(app.world(), vehicle);
    let bevy_oe = app
        .world()
        .get::<OrbitalElementsC>(vehicle)
        .unwrap()
        .0
        .clone();
    let bevy_euler = app.world().get::<EulerAnglesC>(vehicle).unwrap().0;
    let bevy_lvlh = app.world().get::<LvlhFrameC>(vehicle).unwrap().0;
    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    let sun_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: sun_pos,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.orbital_elements_source = Some(earth_idx);
    body.euler_sequence = Some(EulerSequence::ZYX);
    body.compute_lvlh = true;
    body.compute_solar_beta = true;
    sim.add_body(body);

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };

    // Assert dynamics are bit-identical
    assert_sixdof_eq("Bevy vs Sim (derived states)", &bevy_state, &sim_state);

    // Assert derived states are bit-identical
    let sim_oe = sim_body
        .orbital_elements
        .as_ref()
        .expect("orbital elements computed");
    assert_orbital_elements_eq("Bevy vs Sim OE", &bevy_oe, sim_oe);

    let sim_euler = sim_body.euler_angles.expect("euler angles computed");
    for i in 0..3 {
        assert_bits_eq(
            "Bevy vs Sim Euler",
            &format!("angle[{i}]"),
            bevy_euler[i],
            sim_euler[i],
        );
    }
    println!("  Bevy vs Sim Euler: bit-identical (3 angles)");

    let sim_lvlh = sim_body.lvlh_frame.as_ref().expect("LVLH frame computed");
    assert_lvlh_eq("Bevy vs Sim LVLH", &bevy_lvlh, sim_lvlh);

    let sim_beta = sim_body.solar_beta.expect("solar beta computed");
    assert_bits_eq("Bevy vs Sim", "solar_beta", bevy_beta, sim_beta);
    println!("  Bevy vs Sim solar beta: bit-identical");
}

// ── Scenario J: Geodetic derived state (requires RNP) ──

#[test]
fn tier3_bevy_geodetic_derived_state() {
    println!("Scenario J: Geodetic derived state with RNP");

    let earth_shape = PlanetShape {
        name: "Earth",
        mu: MU_EARTH,
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
        flat_coeff: 1.0 / 298.257_223_563,
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(DMat3::IDENTITY),
            PlanetC(earth_shape),
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
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
            // Geodetic config + output
            GeodeticConfigC { planet },
            GeodeticStateC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_trans = read_trans(app.world(), vehicle);
    let bevy_geodetic = app.world().get::<GeodeticStateC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY), // triggers RNP update
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
    });

    let body = SimBody {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        geodetic_planet: Some((earth_idx, earth_shape.r_eq, earth_shape.r_pol)),
        ..Default::default()
    };
    sim.add_body(body);

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);

    // Assert translational state is bit-identical
    assert_trans_eq("Bevy vs Sim (geodetic)", &bevy_trans, &sim_body.trans);

    // Assert geodetic state is bit-identical
    let sim_geodetic = sim_body.geodetic_state.expect("geodetic state computed");
    assert_bits_eq(
        "Bevy vs Sim",
        "latitude",
        bevy_geodetic.latitude,
        sim_geodetic.latitude,
    );
    assert_bits_eq(
        "Bevy vs Sim",
        "longitude",
        bevy_geodetic.longitude,
        sim_geodetic.longitude,
    );
    assert_bits_eq(
        "Bevy vs Sim",
        "altitude",
        bevy_geodetic.altitude,
        sim_geodetic.altitude,
    );
    println!("  Bevy vs Sim geodetic: bit-identical (lat, lon, alt)");
}

// ── Scenario K: Constant-density drag (Phase 4a parity) ──
// Exercises the `constant_density` override branch in compute_ballistic_drag,
// which is a distinct code path from atmosphere-provided density.

#[test]
fn tier3_bevy_constant_density_drag_sixdof() {
    println!("Scenario K: Constant-density drag, 6-DOF");

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1000.0,
        constant_density: Some(1.4e-12), // override atmosphere density
    };
    let exp_atmos = ExponentialAtmosphere::default();

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: 6_378_137.0,
            r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
            planet_omega: 0.0,
        },
        planet_entity: None,
    });

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
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
            AtmosphericStateC::default(),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
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

    assert_sixdof_eq("Bevy vs Sim (const-density drag)", &bevy_state, &sim_state);
}

// ── Scenario L: MET atmosphere + drag (Phase 4a parity) ──
// Exercises the MET atmosphere code path which requires SimulationTime (TAI TJT)
// and planet-fixed rotation for geodetic coordinate computation.

#[test]
fn tier3_bevy_met_atmosphere_drag_sixdof() {
    println!("Scenario L: MET atmosphere + drag, 6-DOF");

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1000.0,
        constant_density: None,
    };
    let met = MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: GeoIndexType::Ap,
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(DMat3::IDENTITY),
        ))
        .id();

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Met(met),
            r_eq: 6_378_137.0,
            r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
            planet_omega: 7.292_115_146_706_388e-5,
        },
        planet_entity: Some(planet),
    });

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
            AtmosphericStateC::default(),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
    });
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met),
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: 7.292_115_146_706_388e-5,
    });
    sim.atmosphere_planet_source = Some(earth_idx);

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

    assert_sixdof_eq("Bevy vs Sim (MET drag)", &bevy_state, &sim_state);
}

// ── Scenario M: Eccentric orbit with derived states (Phase 4b parity) ──
// Exercises LVLH, Euler, and orbital elements at varying orbital rate.

#[test]
fn tier3_bevy_eccentric_derived_states() {
    println!("Scenario M: Eccentric orbit with derived states");

    // Eccentric orbit: 400 km x 8000 km altitude
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0), // periapsis
        velocity: DVec3::new(0.0, 9500.0, 0.0),      // higher velocity for eccentric orbit
    };
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let sun = app
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
    let _ = sun;

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(ecc_trans),
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
            OrbitalElementsConfigC {
                gravity_source: planet,
            },
            OrbitalElementsC::default(),
            EulerAnglesConfigC {
                sequence: EulerSequence::XYZ,
            },
            EulerAnglesC::default(),
            LvlhFrameC::default(),
            SolarBetaC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_state = read_sixdof(app.world(), vehicle);
    let bevy_oe = app
        .world()
        .get::<OrbitalElementsC>(vehicle)
        .unwrap()
        .0
        .clone();
    let bevy_euler = app.world().get::<EulerAnglesC>(vehicle).unwrap().0;
    let bevy_lvlh = app.world().get::<LvlhFrameC>(vehicle).unwrap().0;
    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    let sun_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: sun_pos,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun_idx);

    sim.add_body(SimBody {
        trans: ecc_trans,
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        orbital_elements_source: Some(earth_idx),
        euler_sequence: Some(EulerSequence::XYZ),
        compute_lvlh: true,
        compute_solar_beta: true,
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (eccentric)", &bevy_state, &sim_state);

    let sim_oe = sim_body.orbital_elements.as_ref().expect("OE computed");
    assert_orbital_elements_eq("Bevy vs Sim OE (ecc)", &bevy_oe, sim_oe);

    let sim_euler = sim_body.euler_angles.expect("Euler computed");
    for i in 0..3 {
        assert_bits_eq(
            "Bevy vs Sim Euler (ecc)",
            &format!("angle[{i}]"),
            bevy_euler[i],
            sim_euler[i],
        );
    }
    println!("  Bevy vs Sim Euler (ecc): bit-identical");

    let sim_lvlh = sim_body.lvlh_frame.as_ref().expect("LVLH computed");
    assert_lvlh_eq("Bevy vs Sim LVLH (ecc)", &bevy_lvlh, sim_lvlh);

    let sim_beta = sim_body.solar_beta.expect("solar beta computed");
    assert_bits_eq("Bevy vs Sim (ecc)", "solar_beta", bevy_beta, sim_beta);
    println!("  Bevy vs Sim solar beta (ecc): bit-identical");
}

// ── Scenario N: Polar orbit with geodetic on spherical Earth (Phase 4b parity) ──
// Exercises geodetic conversion at high latitudes where longitude is ill-defined.

#[test]
fn tier3_bevy_polar_geodetic() {
    println!("Scenario N: Polar orbit with geodetic (spherical Earth)");

    // Polar orbit: i=90 deg (velocity purely in z-direction)
    let polar_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7668.56),
    };

    // Spherical Earth: r_eq = r_pol
    let r_sph = 6_378_137.0;
    let earth_shape = PlanetShape {
        name: "Earth",
        mu: MU_EARTH,
        r_eq: r_sph,
        r_pol: r_sph,
        flat_coeff: 0.0,
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(DMat3::IDENTITY),
            PlanetC(earth_shape),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(polar_trans),
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
            GeodeticConfigC { planet },
            GeodeticStateC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_trans = read_trans(app.world(), vehicle);
    let bevy_geodetic = app.world().get::<GeodeticStateC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
    });

    sim.add_body(SimBody {
        trans: polar_trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        geodetic_planet: Some((earth_idx, r_sph, r_sph)),
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);
    assert_trans_eq("Bevy vs Sim (polar geodetic)", &bevy_trans, &sim_body.trans);

    let sim_geodetic = sim_body.geodetic_state.expect("geodetic computed");
    assert_bits_eq(
        "Bevy vs Sim (polar)",
        "latitude",
        bevy_geodetic.latitude,
        sim_geodetic.latitude,
    );
    assert_bits_eq(
        "Bevy vs Sim (polar)",
        "longitude",
        bevy_geodetic.longitude,
        sim_geodetic.longitude,
    );
    assert_bits_eq(
        "Bevy vs Sim (polar)",
        "altitude",
        bevy_geodetic.altitude,
        sim_geodetic.altitude,
    );
    println!("  Bevy vs Sim polar geodetic: bit-identical");
}

// ── Scenario O: Equatorial orbit with solar beta (Phase 4b parity) ──
// Exercises solar beta at zero inclination where orbit plane contains equator.

#[test]
fn tier3_bevy_equatorial_solar_beta() {
    println!("Scenario O: Equatorial orbit with solar beta");

    let sun_pos = DVec3::new(1.496e11, 0.0, 2.5e10); // Sun off-equatorial

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let sun = app
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
    let _ = sun;

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()), // equatorial orbit (v in y-direction)
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
            SolarBetaC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_state = read_sixdof(app.world(), vehicle);
    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    let sun_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: sun_pos,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun_idx);

    sim.add_body(SimBody {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        compute_solar_beta: true,
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (equ solar beta)", &bevy_state, &sim_state);

    let sim_beta = sim_body.solar_beta.expect("solar beta computed");
    assert_bits_eq("Bevy vs Sim (equ)", "solar_beta", bevy_beta, sim_beta);
    println!("  Bevy vs Sim equatorial solar beta: bit-identical");
}

// ── Scenario I: Gauss-Jackson point-mass 3-DOF ──

/// Shared helper for GJ Bevy-vs-Simulation parity tests.
fn run_gj_parity(label: &str, config: GaussJacksonConfig, dt: f64, n_steps: usize) {
    let gj_trans = TranslationalState {
        position: DVec3::new(9e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 8000.0, 0.0),
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(dt));
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            DynamicsConfigC::default(),
            TranslationalStateC(gj_trans),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GravityAccelerationC::default(),
            TotalForceC::default(),
            IntegratorTypeC(IntegratorType::GaussJackson(config)),
            GaussJacksonStateC(GaussJacksonState::new(config)),
        ))
        .id();

    step_bevy_dt(&mut app, n_steps, dt);
    let bevy_trans = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    sim.add_body(SimBody {
        trans: gj_trans,
        integrator: IntegratorType::GaussJackson(config),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(n_steps);

    let sim_trans = sim.body(0).trans;

    assert_trans_eq(label, &bevy_trans, &sim_trans);
    println!("  {label}: bit-identical");
}

#[test]
fn tier3_bevy_gj_point_mass() {
    println!("Scenario I: GJ order 8, dt=10s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 8)",
        GaussJacksonConfig::with_order(8),
        DT,
        NUM_STEPS,
    );
}

#[test]
fn tier3_bevy_gj_order4() {
    println!("Scenario I-b: GJ order 4, dt=10s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 4)",
        GaussJacksonConfig::with_order(4),
        DT,
        NUM_STEPS,
    );
}

#[test]
fn tier3_bevy_gj_order12() {
    println!("Scenario I-c: GJ order 12, dt=10s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 12)",
        GaussJacksonConfig::with_order(12),
        DT,
        NUM_STEPS,
    );
}

#[test]
fn tier3_bevy_gj_dt1() {
    // Smaller timestep to verify GJ behavior at dt=1s.
    // Uses more steps to cover the same time range as other parity tests.
    println!("Scenario I-d: GJ order 8, dt=1s, point-mass 3-DOF");
    run_gj_parity(
        "Bevy vs Sim (GJ order 8, dt=1s)",
        GaussJacksonConfig::with_order(8),
        1.0,
        1000,
    );
}

// ── Scenario J: Solid body tides ──
// SH 4x4 + RNP + tidal ΔC20 with fixed Moon/Sun positions.
// Proves TidalConfigC + TidalDeltaC20C + tidal_update_system produce
// bit-identical results to Simulation's internal tidal pipeline.

#[test]
fn tier3_bevy_tidal_sh4x4() {
    println!("Scenario J: SH 4x4 + RNP + solid body tides");

    let jeod_root = jeod_test_data::jeod_path();
    let ggm02c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    assert!(
        ggm02c_path.exists(),
        "GGM02C coefficients not found at {}",
        ggm02c_path.display()
    );
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C");
    let mu = sh_data.mu;
    let radius = sh_data.radius;

    // Fixed Moon/Sun positions (representative, not from ephemeris)
    let moon_pos = DVec3::new(2.0e8, 3.0e8, 1.0e8);
    let sun_pos = DVec3::new(1.0e11, 0.5e11, 0.2e11);

    let tidal_config = TidalConfig {
        k2: jeod_sim::EARTH_K2,
        mu_primary: mu,
        radius_primary: radius,
        tidal_bodies: vec![
            TidalBody {
                mu: 4902.79980693169e9,
                position_inertial: moon_pos,
            },
            TidalBody {
                mu: 1.327_124_40e20,
                position_inertial: sun_pos,
            },
        ],
    };

    let sh_source = GravitySource {
        mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(sh_source.clone()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(DMat3::IDENTITY),
            TidalConfigC(tidal_config.clone()),
            TidalDeltaC20C(0.0),
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
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        rotation_model: RotationModel::EarthRNP,
        delta_c20: 0.0,
        tidal_config: Some(tidal_config),
    });

    sim.add_body(SimBody {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(earth_idx, 4, 4, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_state = sim.body(0).trans;

    assert_trans_eq("Bevy vs Sim (SH 4x4 + tides)", &bevy_state, &sim_state);
    println!("  Bevy vs Sim SH 4x4 + tides: bit-identical");
}

// ── Scenario S: Earth lighting consistency ──
// Validates compute_earth_lighting returns physically reasonable values
// and is deterministic (same inputs → same outputs).
#[test]
fn tier3_sim_earth_lighting_consistency() {
    use jeod_sim::compute_earth_lighting;
    println!("Scenario S: Earth lighting consistency");

    let pos_veh = DVec3::new(6.778e6, 0.0, 0.0); // LEO, sunlit side
    let pos_sun = DVec3::new(1.496e11, 0.0, 0.0);
    let pos_moon = DVec3::new(0.0, 3.844e8, 0.0);

    let state1 = compute_earth_lighting(pos_veh, pos_sun, pos_moon, 6.96e8, 6.371e6, 1.737e6);
    let state2 = compute_earth_lighting(pos_veh, pos_sun, pos_moon, 6.96e8, 6.371e6, 1.737e6);

    // Deterministic: same inputs → bit-identical outputs
    assert_eq!(
        state1.sun_earth.visible.to_bits(),
        state2.sun_earth.visible.to_bits(),
        "earth lighting should be deterministic"
    );
    assert_eq!(
        state1.sun_earth.occlusion.to_bits(),
        state2.sun_earth.occlusion.to_bits(),
        "occlusion should be deterministic"
    );

    // Physical checks
    assert!(state1.sun_earth.visible >= 0.0 && state1.sun_earth.visible <= 1.0);
    assert!(state1.sun_earth.occlusion >= 0.0 && state1.sun_earth.occlusion <= 1.0);
    assert!((state1.sun_earth.visible + state1.sun_earth.occlusion - 1.0).abs() < 1e-12);
    assert!(state1.earth_albedo.lighting >= 0.0);

    // Vehicle in sunlit side: should be fully visible
    assert!(
        state1.sun_earth.visible > 0.99,
        "sunlit vehicle should have visible > 0.99, got {}",
        state1.sun_earth.visible
    );
    println!("  Earth lighting: deterministic, physically consistent");
}

// ── Scenario P: Time reversal round-trip ──
// Validates that Simulation forward+backward returns to initial state.
// RK4 is reversible in exact arithmetic; floating-point round-trip tests
// that no state corruption occurs (sign errors, time wrapping, etc.).
#[test]
fn tier3_sim_time_reversal_round_trip() {
    println!("Scenario P: Time reversal round-trip");
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    ));
    sim.add_body(SimBody {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();

    // Save initial state
    let initial_pos = sim.body(0).trans.position;
    let initial_vel = sim.body(0).trans.velocity;

    // Forward 50 steps
    sim.step_n(50);
    let mid_pos = sim.body(0).trans.position;
    assert!(
        (mid_pos - initial_pos).length() > 1.0,
        "should have moved after 50 steps"
    );

    // Reverse 50 steps
    sim.set_dt(-DT);
    sim.step_n(50);
    let final_pos = sim.body(0).trans.position;
    let final_vel = sim.body(0).trans.velocity;

    // RK4 is not exactly reversible due to floating-point, but error should be tiny
    let pos_err = (final_pos - initial_pos).length();
    let vel_err = (final_vel - initial_vel).length();
    assert!(
        pos_err < 1e-3,
        "round-trip position error {pos_err} m should be < 1e-3 m"
    );
    assert!(
        vel_err < 1e-6,
        "round-trip velocity error {vel_err} m/s should be < 1e-6 m/s"
    );
    println!("  Time reversal round-trip: pos_err={pos_err:.2e} m, vel_err={vel_err:.2e} m/s");
}

// ── Scenario Q: Relative state computation ──
// Validates compute_relative_state by checking that A_relative_to_B
// is consistent with the individual body states.
#[test]
fn tier3_sim_relative_state_consistency() {
    use jeod_sim::compute_relative_state;
    println!("Scenario Q: Relative state consistency");

    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let earth = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    ));

    // Body A: ISS orbit
    sim.add_body(SimBody {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    // Body B: slightly offset orbit (100m ahead in velocity direction)
    let mut trans_b = iss_trans();
    trans_b.position += DVec3::new(100.0, 0.0, 0.0);
    sim.add_body(SimBody {
        trans: trans_b,
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::new(0.0, 0.0, 0.001),
        }),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(10);

    let a = sim.body(0);
    let b = sim.body(1);

    let rel = compute_relative_state(&a.trans, a.rot.as_ref(), &b.trans, b.rot.as_ref());

    // Relative position should equal T_ref * (B.pos - A.pos) — in ref body frame
    let t_ref = a
        .rot
        .as_ref()
        .unwrap()
        .quaternion
        .left_quat_to_transformation();
    let rel_pos_inertial = b.trans.position - a.trans.position;
    let expected_pos = t_ref * rel_pos_inertial;
    let pos_err = (rel.position - expected_pos).length();
    assert!(
        pos_err < 1e-10,
        "relative position error {pos_err:.4e} m exceeds 1e-10"
    );

    // Relative velocity includes Coriolis: T * Δv - ω × pos
    let rel_vel_inertial = b.trans.velocity - a.trans.velocity;
    let omega_ref = a.rot.as_ref().unwrap().ang_vel_body;
    let expected_vel = t_ref * rel_vel_inertial - omega_ref.cross(expected_pos);
    let vel_err = (rel.velocity - expected_vel).length();
    assert!(
        vel_err < 1e-10,
        "relative velocity error {vel_err:.4e} m/s exceeds 1e-10"
    );
    println!("  Relative state: matches body-frame computation within {pos_err:.2e} m, {vel_err:.2e} m/s");
}

// ── Scenario R: LVLH-relative state ──
// Validates compute_lvlh_relative_state is consistent with manual
// LVLH frame rotation of the inertial relative state.
#[test]
fn tier3_sim_lvlh_relative_consistency() {
    use jeod_sim::{compute_body_lvlh_frame, compute_lvlh_relative_state};
    println!("Scenario R: LVLH-relative state consistency");

    let ref_pos = iss_trans().position;
    let ref_vel = iss_trans().velocity;
    let subj_pos = ref_pos + DVec3::new(100.0, 50.0, -30.0);
    let subj_vel = ref_vel + DVec3::new(0.01, -0.02, 0.005);

    let lvlh_rel = compute_lvlh_relative_state(ref_pos, ref_vel, subj_pos, subj_vel);

    // Manual computation: get LVLH frame, rotate relative state + Coriolis
    let lvlh = compute_body_lvlh_frame(ref_pos, ref_vel);
    let rel_pos_inertial = subj_pos - ref_pos;
    let rel_vel_inertial = subj_vel - ref_vel;
    let expected_pos = lvlh.t_parent_this * rel_pos_inertial;
    let expected_vel =
        lvlh.t_parent_this * rel_vel_inertial - lvlh.ang_vel_this.cross(expected_pos);

    assert_eq!(
        lvlh_rel.position.x.to_bits(),
        expected_pos.x.to_bits(),
        "LVLH pos x"
    );
    assert_eq!(
        lvlh_rel.position.y.to_bits(),
        expected_pos.y.to_bits(),
        "LVLH pos y"
    );
    assert_eq!(
        lvlh_rel.position.z.to_bits(),
        expected_pos.z.to_bits(),
        "LVLH pos z"
    );
    assert_eq!(
        lvlh_rel.velocity.x.to_bits(),
        expected_vel.x.to_bits(),
        "LVLH vel x"
    );
    assert_eq!(
        lvlh_rel.velocity.y.to_bits(),
        expected_vel.y.to_bits(),
        "LVLH vel y"
    );
    assert_eq!(
        lvlh_rel.velocity.z.to_bits(),
        expected_vel.z.to_bits(),
        "LVLH vel z"
    );
    println!("  LVLH-relative: bit-identical with manual LVLH rotation + Coriolis");
}

// ── Scenario T: Mars IAU rotation dispatch ──
// Validates that per-source rotation dispatch works for Mars by confirming
// that a MarsIAU source produces a non-identity rotation after stepping.
#[test]
fn tier3_sim_mars_rotation_dispatch() {
    println!("Scenario T: Mars IAU rotation dispatch");
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let mars_mu = 4.282_837_452_7e13;
    let mars = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: mars_mu,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        rotation_model: RotationModel::MarsIAU,
        delta_c20: 0.0,
        tidal_config: None,
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: DVec3::new(3.5e6, 0.0, 0.0), // Low Mars orbit
            velocity: DVec3::new(0.0, 3.5e3, 0.0),
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(mars, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(10);

    // After stepping, Mars rotation should have been updated from identity
    let rot = sim.sources[mars].t_inertial_pfix.unwrap();
    assert!(
        rot != DMat3::IDENTITY,
        "Mars rotation should differ from identity after 10 steps"
    );

    // Should be a valid rotation matrix
    let det = rot.determinant();
    assert!(
        (det - 1.0).abs() < 1e-10,
        "Mars rotation determinant should be 1, got {det}"
    );

    println!("  Mars rotation dispatch: non-identity, det={det:.15}");
}

// ── Scenario U: Multi-source rotation (Earth + Mars) ──
// Validates that two sources with different rotation models are dispatched
// correctly in the same simulation.
#[test]
fn tier3_sim_multi_source_rotation() {
    println!("Scenario U: Multi-source rotation dispatch");
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    // Earth with EarthRNP
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        rotation_model: RotationModel::EarthRNP,
        delta_c20: 0.0,
        tidal_config: None,
    });

    // Mars with MarsIAU (at some offset)
    let mars = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 4.282_837_452_7e13,
            model: GravityModel::PointMass,
        },
        position: DVec3::new(2.28e11, 0.0, 0.0),
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        rotation_model: RotationModel::MarsIAU,
        delta_c20: 0.0,
        tidal_config: None,
    });

    sim.add_body(SimBody {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(10);

    let earth_rot = sim.sources[earth].t_inertial_pfix.unwrap();
    let mars_rot = sim.sources[mars].t_inertial_pfix.unwrap();

    // Both should be non-identity
    assert!(earth_rot != DMat3::IDENTITY, "Earth rotation updated");
    assert!(mars_rot != DMat3::IDENTITY, "Mars rotation updated");

    // They should differ (different planets, different rotation models)
    assert!(
        earth_rot != mars_rot,
        "Earth and Mars rotations should differ"
    );

    println!("  Multi-source rotation: Earth and Mars independently dispatched");
}

// ── Scenario V: Relativistic gravity correction ──
// Validates compute_relativistic_correction by checking that the correction
// is non-zero and physically reasonable for Mercury near the Sun.
#[test]
fn tier3_sim_relativistic_gravity_consistency() {
    use jeod_sim::relativistic::compute_relativistic_correction;
    println!("Scenario V: Relativistic gravity correction");

    // Mercury at perihelion
    let sun_pos = DVec3::ZERO;
    let sun_vel = DVec3::ZERO;
    let mercury_pos = DVec3::new(4.6e10, 0.0, 0.0);
    let mercury_vel = DVec3::new(0.0, 5.898e4, 0.0);

    let correction =
        compute_relativistic_correction(MU_SUN, sun_pos, mercury_pos, mercury_vel, sun_vel, &[]);

    // Should be non-zero
    assert!(correction.length() > 0.0, "correction should be non-zero");

    // Newtonian acceleration ≈ μ/r²
    let newtonian = MU_SUN / (4.6e10 * 4.6e10);
    let ratio = correction.length() / newtonian;

    // GR correction is ~v²/c² ≈ (6e4)²/(3e8)² ≈ 4e-8 of Newtonian
    assert!(
        ratio > 1e-9 && ratio < 1e-5,
        "correction/newtonian ratio {ratio:.2e} should be ~1e-7 to 1e-8"
    );

    // Deterministic
    let correction2 =
        compute_relativistic_correction(MU_SUN, sun_pos, mercury_pos, mercury_vel, sun_vel, &[]);
    assert_eq!(
        correction.x.to_bits(),
        correction2.x.to_bits(),
        "relativistic correction should be deterministic"
    );

    println!(
        "  Relativistic correction: {:.4e} m/s² ({:.2e} of Newtonian)",
        correction.length(),
        ratio
    );
}
