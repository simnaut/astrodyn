//! Bevy-vs-Simulation parity tests: drag (exponential, constant-density, MET atmosphere).

mod common;

use astrodyn::GravitySourceEntry;
use astrodyn::{
    AtmosphereConfig, AtmosphereModel, DragConfig, DynamicsConfig, ExponentialAtmosphere,
    GeoIndexType, GravityControl, GravityControls, MetAtmosphere, SixDofState,
};
use astrodyn_bevy::{
    AtmosphereModelR, DragConfigC, DynamicsConfigC, GravityControlsC, GravitySourceC,
    MassPropertiesC, PlanetFixedRotationC, RotationalStateC, SourceInertialPositionC,
    TranslationalStateC,
};
use bevy::prelude::*;
use glam::DMat3;

use common::*;

// ── Scenario B: Exponential atmosphere + drag, 6-DOF ──

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
    app.add_plugins(astrodyn_bevy::AstrodynPlugin);

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
            r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
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
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
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
            DragConfigC::from_untyped(&drag_config),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
        r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
        planet_omega: 0.0,
    });

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.drag = Some(drag_config);
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (drag)", &bevy_state, &sim_state);
}

// ── Scenario K: Constant-density drag (Phase 4a parity) ──

#[test]
fn tier3_bevy_constant_density_drag_sixdof() {
    println!("Scenario K: Constant-density drag, 6-DOF");

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1000.0,
        constant_density: Some(1.4e-12),
    };
    let exp_atmos = ExponentialAtmosphere::default();

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(astrodyn_bevy::AstrodynPlugin);

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Exponential(exp_atmos),
            r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
            r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
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
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
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
            DragConfigC::from_untyped(&drag_config),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(exp_atmos),
        r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
        r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
        planet_omega: 0.0,
    });

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.drag = Some(drag_config);
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (const-density drag)", &bevy_state, &sim_state);
}

// ── Scenario L: MET atmosphere + drag (Phase 4a parity) ──

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
    app.add_plugins(astrodyn_bevy::AstrodynPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
            PlanetFixedRotationC::<astrodyn::Earth>(astrodyn::FrameTransform::from_matrix(
                DMat3::IDENTITY,
            )),
        ))
        .id();

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Met(met),
            r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
            r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
            planet_omega: astrodyn::planet_config::EARTH.omega,
        },
        planet_entity: Some(planet),
    });

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
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
            DragConfigC::from_untyped(&drag_config),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source(),
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: astrodyn::planet_config::EARTH.omega,
            central: true,
        },
    );
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met),
        r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
        r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
        planet_omega: astrodyn::planet_config::EARTH.omega,
    });
    sim.atmosphere_planet_source = Some(earth_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.drag = Some(drag_config);
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (MET drag)", &bevy_state, &sim_state);
}

// ── MET atmosphere drag parity (run5a) ──

#[test]
fn tier3_bevy_met_run5a() {
    println!("MET run5a parity: minimum solar");
    let met = MetAtmosphere {
        f10: 70.0,
        f10b: 70.0,
        geo_index: 0.0,
        geo_index_type: GeoIndexType::Ap,
    };

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
            PlanetFixedRotationC::<astrodyn::Earth>(astrodyn::FrameTransform::from_matrix(
                DMat3::IDENTITY,
            )),
        ))
        .id();

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Met(met),
            r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
            r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
            planet_omega: astrodyn::planet_config::EARTH.omega,
        },
        planet_entity: Some(planet),
    });

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
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
            DragConfigC::from_untyped(&DragConfig {
                cd: 2.2,
                area: 1000.0,
                constant_density: None,
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source(),
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: astrodyn::planet_config::EARTH.omega,
            central: true,
        },
    );
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met),
        r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
        r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
        planet_omega: astrodyn::planet_config::EARTH.omega,
    });
    sim.atmosphere_planet_source = Some(earth_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.drag = Some(DragConfig {
        cd: 2.2,
        area: 1000.0,
        constant_density: None,
    });
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq("Bevy vs Sim (MET run5a)", &bevy_state, &sim_state);
}

// ── Drag run6b parity (MET atmosphere + drag) ──

#[test]
fn tier3_bevy_drag_run6b() {
    println!("Drag run6b parity: MET atmosphere + drag");
    let met = MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: GeoIndexType::Ap,
    };
    let drag_config = DragConfig {
        cd: 0.02,
        area: 1.0,
        constant_density: None,
    };

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
            PlanetFixedRotationC::<astrodyn::Earth>(astrodyn::FrameTransform::from_matrix(
                DMat3::IDENTITY,
            )),
        ))
        .id();

    app.insert_resource(AtmosphereModelR {
        config: AtmosphereConfig {
            model: AtmosphereModel::Met(met),
            r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
            r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
            planet_omega: astrodyn::planet_config::EARTH.omega,
        },
        planet_entity: Some(planet),
    });

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
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
            DragConfigC::from_untyped(&drag_config),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source(),
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: astrodyn::planet_config::EARTH.omega,
            central: true,
        },
    );
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met),
        r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
        r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
        planet_omega: astrodyn::planet_config::EARTH.omega,
    });
    sim.atmosphere_planet_source = Some(earth_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.drag = Some(drag_config);
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq("Bevy vs Sim (drag run6b)", &bevy_state, &sim_state);
}
