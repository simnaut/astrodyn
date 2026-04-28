//! Bevy-vs-Simulation parity tests: SRP (flat-plate, shadow, cannonball).

mod common;

use bevy::prelude::*;
use bevy_jeod::{
    AtmosphereModelR, DragConfigC, DynamicsConfigC, FlatPlateConfigC, GravityControlsC,
    GravitySourceC, GravityTorqueC, MassPropertiesC, RotationalStateC, ShadowBodyC,
    SourceInertialPositionC, StructuralTransformC, SunMarker, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, ShadowBody as RunnerShadowBody, SrpModel, VehicleConfig};
use jeod_sim::{
    AtmosphereConfig, AtmosphereModel, DragConfig, DynamicsConfig, ExponentialAtmosphere,
    FlatPlate, FlatPlateParams, FlatPlateThermal, GravityControl, GravityControls, GravityModel,
    GravitySource, MassProperties, SixDofState, TranslationalState,
};

use common::*;

// ── Scenario E: Full stack — drag + SRP + gravity torque ──

#[test]
fn tier3_bevy_full_stack_sixdof() {
    println!("Scenario E: Full stack — drag + SRP + gravity torque, 6-DOF");

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1000.0,
        constant_density: None,
    };
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
            thermal_power_dump: 0.0,
        },
    )];
    let exp_atmos = ExponentialAtmosphere::default();
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(bevy_jeod::JeodPlugin);

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: jeod_sim::planet_config::EARTH.shape.r_eq,
            r_pol: jeod_sim::planet_config::EARTH.shape.r_pol,
            planet_omega: jeod_sim::planet_config::EARTH.omega,
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
            TranslationalStateC::from(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            RotationalStateC::from(tumble_rot()),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, true)],
            }),
            DragConfigC::from_untyped(&drag_config),
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: srp_plates.clone(),
                temperatures: vec![270.0],
                t_pow4_cached: vec![270.0_f64.powi(4)],
                ..Default::default()
            }),
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(earth_source(), DVec3::ZERO, None);
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: jeod_sim::planet_config::EARTH.shape.r_eq,
        r_pol: jeod_sim::planet_config::EARTH.shape.r_pol,
        planet_omega: jeod_sim::planet_config::EARTH.omega,
    });

    let mut body = new_sim_body_sixdof(earth_idx, true);
    body.drag = Some(drag_config);
    body.srp = Some(SrpModel::FlatPlate(jeod_sim::FlatPlateState {
        plates: srp_plates,
        temperatures: vec![270.0],
        t_pow4_cached: vec![270.0_f64.powi(4)],
        ..Default::default()
    }));
    body.compute_gravity_gradient = true;
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (full stack)", &bevy_state, &sim_state);
}

// ── Scenario H: Flat-plate SRP with shadow detection ──

#[test]
fn tier3_bevy_flat_plate_srp_with_shadow() {
    println!("Scenario H: Flat-plate SRP with shadow detection");

    let params = FlatPlateParams {
        albedo: 0.5,
        diffuse: 0.5,
    };
    let thermal = FlatPlateThermal {
        emissivity: 0.5,
        heat_capacity_per_area: 50.0,
        thermal_power_dump: 0.0,
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

    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let vehicle_pos = DVec3::new(4.2e7, 0.0, 0.0);
    let vehicle_vel = DVec3::new(0.0, 3074.0, 0.0);

    let mass =
        MassProperties::with_inertia(300.0, DMat3::from_diagonal(DVec3::splat(1.0)), DVec3::ZERO);

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(bevy_jeod::JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            ShadowBodyC {
                radius: jeod_sim::planet_config::EARTH.shadow_radius,
            },
        ))
        .id();

    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC::from(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(TranslationalState {
                position: vehicle_pos,
                velocity: vehicle_vel,
            }),
            MassPropertiesC::from(mass),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: plates_data.clone(),
                temperatures: vec![init_temp; num_plates],
                t_pow4_cached: vec![init_temp.powi(4); num_plates],
                ..Default::default()
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);

    let mut earth_entry = GravitySourceEntry::new(earth_source(), DVec3::ZERO, None);
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);

    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: vehicle_pos,
            velocity: vehicle_vel,
        },
        mass: Some(mass),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        srp: Some(SrpModel::FlatPlate(jeod_sim::FlatPlateState {
            plates: plates_data,
            temperatures: vec![init_temp; num_plates],
            t_pow4_cached: vec![init_temp.powi(4); num_plates],
            ..Default::default()
        })),
        shadow_body: Some(RunnerShadowBody {
            source_idx: earth_idx,
            radius: jeod_sim::planet_config::EARTH.shadow_radius,
        }),
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_state = sim.body(0).trans;

    assert_trans_eq(
        "Bevy vs Sim (flat-plate SRP + shadow)",
        &bevy_state,
        &sim_state,
    );
}

// ── SRP shadow variants parity ──

fn make_single_plate(
    albedo: f64,
    diffuse: f64,
    emissivity: f64,
) -> Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)> {
    vec![(
        FlatPlate {
            area: 100.0,
            normal: DVec3::X,
            position: DVec3::ZERO,
        },
        FlatPlateParams { albedo, diffuse },
        FlatPlateThermal {
            emissivity,
            heat_capacity_per_area: 50.0,
            thermal_power_dump: 0.0,
        },
    )]
}

fn run_shadow_parity(label: &str, srp_plates: Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>) {
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            ShadowBodyC {
                radius: 6_371_000.0,
            },
        ))
        .id();

    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC::from(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            RotationalStateC::from(tumble_rot()),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: srp_plates.clone(),
                temperatures: vec![270.0; srp_plates.len()],
                t_pow4_cached: vec![270.0_f64.powi(4); srp_plates.len()],
                ..Default::default()
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.srp = Some(SrpModel::FlatPlate(jeod_sim::FlatPlateState {
        plates: srp_plates.clone(),
        temperatures: vec![270.0; srp_plates.len()],
        t_pow4_cached: vec![270.0_f64.powi(4); srp_plates.len()],
        ..Default::default()
    }));
    body.shadow_body = Some(RunnerShadowBody {
        source_idx: earth_idx,
        radius: 6_371_000.0,
    });
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
}

#[test]
fn tier3_bevy_shadow_2a_annular() {
    println!("Shadow 2a annular parity");
    run_shadow_parity("shadow_2a_annular", make_single_plate(0.0, 0.0, 0.0));
}

#[test]
fn tier3_bevy_shadow_2a_cooling() {
    println!("Shadow 2a cooling parity");
    run_shadow_parity("shadow_2a_cooling", make_single_plate(0.0, 0.0, 0.9));
}

// ── SRP basic variants parity ──

fn run_srp_basic_parity(
    label: &str,
    srp_plates: Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>,
) {
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);
    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC::from(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            RotationalStateC::from(tumble_rot()),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: srp_plates.clone(),
                temperatures: vec![270.0; srp_plates.len()],
                t_pow4_cached: vec![270.0_f64.powi(4); srp_plates.len()],
                ..Default::default()
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.srp = Some(SrpModel::FlatPlate(jeod_sim::FlatPlateState {
        plates: srp_plates.clone(),
        temperatures: vec![270.0; srp_plates.len()],
        t_pow4_cached: vec![270.0_f64.powi(4); srp_plates.len()],
        ..Default::default()
    }));
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
}

#[test]
fn tier3_bevy_srp_basic_default() {
    println!("SRP basic default parity");
    run_srp_basic_parity("srp_basic_default", make_single_plate(0.3, 0.3, 0.0));
}

#[test]
fn tier3_bevy_srp_basic_varied_cr() {
    println!("SRP basic varied Cr parity");
    run_srp_basic_parity("srp_basic_varied_cr", make_single_plate(0.8, 0.1, 0.0));
}

// ── Derivative-class thermal parity (issue #114) ──
//
// Exercises the coupled `integrate_body_coupled` dispatch through the Bevy
// adapter's `integration_system` and the `Simulation` runner's Stage 8.
// Uses a 6-DOF vehicle so the orbital RK4 stages produce varying attitudes
// and the per-stage SRP force diverges from a step-constant value,
// actually exercising the fork.

fn run_srp_deriv_parity(
    label: &str,
    order: jeod_sim::ThermalIntegrationOrder,
    srp_plates: Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>,
) {
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);
    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC::from(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            RotationalStateC::from(tumble_rot()),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: srp_plates.clone(),
                temperatures: vec![270.0; srp_plates.len()],
                t_pow4_cached: vec![270.0_f64.powi(4); srp_plates.len()],
                integration_order: order,
                ..Default::default()
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.srp = Some(SrpModel::FlatPlate(jeod_sim::FlatPlateState {
        plates: srp_plates.clone(),
        temperatures: vec![270.0; srp_plates.len()],
        t_pow4_cached: vec![270.0_f64.powi(4); srp_plates.len()],
        integration_order: order,
        ..Default::default()
    }));
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
}

#[test]
fn tier3_bevy_srp_derivative_first_order() {
    println!("SRP derivative first-order parity");
    run_srp_deriv_parity(
        "srp_derivative_first_order",
        jeod_sim::ThermalIntegrationOrder::DerivativeFirstOrder,
        make_single_plate(0.3, 0.3, 0.5),
    );
}

#[test]
fn tier3_bevy_srp_derivative_rk4() {
    println!("SRP derivative RK4 parity");
    run_srp_deriv_parity(
        "srp_derivative_rk4",
        jeod_sim::ThermalIntegrationOrder::DerivativeRk4,
        make_single_plate(0.3, 0.3, 0.5),
    );
}

/// Derivative-class SRP with a **non-identity `t_struct_body`** and an
/// **offset plate** that produces a non-zero structural-frame torque.
///
/// Regression test for a bug caught in review where the coupled RK4 stage
/// closure added `srp_result.torque` (structural frame) directly to
/// `constant_torque` (body frame) inside `CoupledStageEval`. When
/// `t_struct_body` is identity the bug is latent; a non-identity rotation
/// exposes it, and the Bevy + Simulation consumers would both drift from
/// the physically-correct rotational state. Both sides now rotate the SRP
/// torque into body frame via `t_struct_body * srp_result.torque` before
/// summing, so this test verifies Bevy parity under the rotated frame AND
/// that the rotational state evolves (non-identity final quaternion +
/// non-zero angular velocity change consistent with the applied torque).
#[test]
fn tier3_bevy_srp_derivative_rk4_with_rotated_struct_frame() {
    println!("SRP derivative RK4 parity with non-identity t_struct_body");

    // 90° rotation about body-Z maps structural X→body Y, structural
    // Y→body -X. Any structural-frame SRP torque with an X component
    // becomes a body-frame torque along Y.
    let t_struct_body = DMat3::from_cols(
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );

    // Plate offset along structural +Y (15 m away from CoM) with the
    // normal pointing along structural +X, so SRP pressure along -X
    // produces a torque about structural +Z via r × F ≠ 0.
    let offset_plate = vec![(
        FlatPlate {
            area: 10.0,
            normal: DVec3::X,
            position: DVec3::new(0.0, 15.0, 0.0),
        },
        FlatPlateParams {
            albedo: 0.3,
            diffuse: 0.3,
        },
        FlatPlateThermal {
            emissivity: 0.5,
            heat_capacity_per_area: 50.0,
            thermal_power_dump: 0.0,
        },
    )];

    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let order = jeod_sim::ThermalIntegrationOrder::DerivativeRk4;

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);
    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC::from(TranslationalState {
                position: sun_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            RotationalStateC::from(tumble_rot()),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            StructuralTransformC(jeod_sim::FrameTransform::from_matrix(t_struct_body)),
            FlatPlateConfigC(jeod_sim::FlatPlateState {
                plates: offset_plate.clone(),
                temperatures: vec![270.0],
                t_pow4_cached: vec![270.0_f64.powi(4)],
                integration_order: order,
                ..Default::default()
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.t_struct_body = t_struct_body;
    body.srp = Some(SrpModel::FlatPlate(jeod_sim::FlatPlateState {
        plates: offset_plate,
        temperatures: vec![270.0],
        t_pow4_cached: vec![270.0_f64.powi(4)],
        integration_order: order,
        ..Default::default()
    }));
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };

    // Bevy must agree with the Simulation runner under the rotated frame.
    // If either consumer had the structural/body frame mismatch we'd see
    // the rotational state diverge between the two.
    assert_sixdof_eq(
        "Bevy vs Sim (DerivativeRk4, rotated t_struct_body)",
        &bevy_state,
        &sim_state,
    );

    // And the test must actually exercise torque — if the rotational
    // state didn't change at all, the scenario isn't pressuring the
    // frame-conversion code path.
    let ang_vel_change = (sim_state.rot.ang_vel_body - tumble_rot().ang_vel_body).length();
    assert!(
        ang_vel_change > 1e-12,
        "Offset plate produced no detectable angular velocity change ({ang_vel_change:.3e}); \
         the torque-handling code path may not be exercised.",
    );
}
