//! Bevy-vs-Simulation parity tests: derived states (orbital elements, Euler angles,
//! LVLH frame, geodetic, solar beta).

mod common;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, EulerAnglesC, EulerAnglesConfigC, GeodeticConfigC, GeodeticStateC,
    GravityControlsC, GravitySourceC, LvlhFrameC, MassPropertiesC, OrbitalElementsC,
    OrbitalElementsConfigC, PlanetC, PlanetFixedRotationC, RotationalStateC, SolarBetaC,
    SourceInertialPositionC, SunMarker, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_runner::RotationModel;
use jeod_sim::{DerivedStateConfig, GeodeticConfig, GravitySourceEntry, VehicleConfig};
use jeod_sim::{
    DynamicsConfig, EulerSequence, GravityControl, GravityControls, GravityModel, GravitySource,
    PlanetShape, SixDofState, TranslationalState,
};

use common::*;

// ── Scenario I: Derived states (orbital elements, Euler, LVLH, solar beta) ──

#[test]
fn tier3_bevy_derived_states() {
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

    // ── Bevy App ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy_jeod::JeodPlugin);

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
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            OrbitalElementsConfigC {
                gravity_source: planet,
            },
            EulerAnglesConfigC {
                sequence: EulerSequence::ZYX,
            },
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
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            jeod_sim::Vec3Ext::m_at::<jeod_sim::RootInertial>(sun_pos),
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    let mut body = new_sim_body_sixdof(earth_idx, false);
    body.derived = DerivedStateConfig {
        orbital_elements_source: Some(earth_idx),
        euler_sequence: Some(EulerSequence::ZYX),
        lvlh: true,
        solar_beta: true,
        ..Default::default()
    };
    sim.add_body(body);

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (derived states)", &bevy_state, &sim_state);

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
            bevy_euler[i].value,
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

    let earth_shape = jeod_sim::EARTH.shape;

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy_jeod::JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(jeod_sim::FrameTransform::from_matrix(DMat3::IDENTITY)),
            PlanetC(earth_shape),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GeodeticConfigC { planet },
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_trans = read_trans(app.world(), vehicle);
    let bevy_geodetic = app.world().get::<GeodeticStateC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source(),
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: jeod_sim::planet_config::EARTH.omega,
            central: true,
        },
    );

    let body = VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            geodetic: Some(GeodeticConfig {
                source_idx: earth_idx,
                r_eq: earth_shape.r_eq,
                r_pol: earth_shape.r_pol,
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    sim.add_body(body);

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    assert_trans_eq("Bevy vs Sim (geodetic)", &bevy_trans, &sim_body.trans);

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

// ── Scenario M: Eccentric orbit with derived states ──

#[test]
fn tier3_bevy_eccentric_derived_states() {
    println!("Scenario M: Eccentric orbit with derived states");

    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

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
            TranslationalStateC::from(ecc_trans),
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
            OrbitalElementsConfigC {
                gravity_source: planet,
            },
            EulerAnglesConfigC {
                sequence: EulerSequence::XYZ,
            },
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
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            jeod_sim::Vec3Ext::m_at::<jeod_sim::RootInertial>(sun_pos),
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    sim.add_body(VehicleConfig {
        trans: ecc_trans,
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            orbital_elements_source: Some(earth_idx),
            euler_sequence: Some(EulerSequence::XYZ),
            lvlh: true,
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

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
            bevy_euler[i].value,
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

// ── Scenario N: Polar orbit with geodetic ──

#[test]
fn tier3_bevy_polar_geodetic() {
    println!("Scenario N: Polar orbit with geodetic (spherical Earth)");

    let polar_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7668.56),
    };

    // Spherical-Earth shape: use the canonical r_eq for both axes (flat_coeff=0).
    let r_sph = jeod_sim::EARTH.shape.r_eq;
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
    app.add_plugins(bevy_jeod::JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(jeod_sim::FrameTransform::from_matrix(DMat3::IDENTITY)),
            PlanetC(earth_shape),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(polar_trans),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GeodeticConfigC { planet },
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_trans = read_trans(app.world(), vehicle);
    let bevy_geodetic = app.world().get::<GeodeticStateC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source(),
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: jeod_sim::planet_config::EARTH.omega,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: polar_trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            geodetic: Some(GeodeticConfig {
                source_idx: earth_idx,
                r_eq: r_sph,
                r_pol: r_sph,
            }),
            ..Default::default()
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

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

// ── Scenario O: Equatorial orbit with solar beta ──

#[test]
fn tier3_bevy_equatorial_solar_beta() {
    println!("Scenario O: Equatorial orbit with solar beta");

    let sun_pos = DVec3::new(1.496e11, 0.0, 2.5e10);

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
            SolarBetaC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);

    let bevy_state = read_sixdof(app.world(), vehicle);
    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            jeod_sim::Vec3Ext::m_at::<jeod_sim::RootInertial>(sun_pos),
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

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

// ── Euler angle parity ──

fn run_euler_parity(label: &str, trans: TranslationalState, sequence: EulerSequence) {
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);

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
            TranslationalStateC::from(trans),
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
            EulerAnglesConfigC { sequence },
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);
    let bevy_euler = app.world().get::<EulerAnglesC>(vehicle).unwrap().0;

    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans,
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            euler_sequence: Some(sequence),
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);

    let sim_euler = sim_body.euler_angles.expect("euler angles computed");
    for i in 0..3 {
        assert_bits_eq(
            &format!("Bevy vs Sim Euler ({label})"),
            &format!("angle[{i}]"),
            bevy_euler[i].value,
            sim_euler[i],
        );
    }
    println!("  Bevy vs Sim Euler ({label}): bit-identical");
}

#[test]
fn tier3_bevy_euler() {
    run_euler_parity("euler_inc", iss_trans(), EulerSequence::XYZ);
}

#[test]
fn tier3_bevy_euler_ecc() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_euler_parity("euler_ecc", ecc_trans, EulerSequence::XYZ);
}

#[test]
fn tier3_bevy_euler_equ() {
    let equ_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    run_euler_parity("euler_equ", equ_trans, EulerSequence::XYZ);
}

// ── LVLH parity ──

fn run_lvlh_parity(label: &str, trans: TranslationalState) {
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            LvlhFrameC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);
    let bevy_lvlh = app.world().get::<LvlhFrameC>(vehicle).unwrap().0;

    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            lvlh: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    assert_trans_eq(
        &format!("Bevy vs Sim ({label})"),
        &bevy_trans,
        &sim_body.trans,
    );

    let sim_lvlh = sim_body.lvlh_frame.as_ref().expect("LVLH computed");
    assert_lvlh_eq(&format!("Bevy vs Sim LVLH ({label})"), &bevy_lvlh, sim_lvlh);
}

#[test]
fn tier3_bevy_lvlh() {
    run_lvlh_parity("lvlh_inc", iss_trans());
}

#[test]
fn tier3_bevy_lvlh_ecc() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_lvlh_parity("lvlh_ecc", ecc_trans);
}

#[test]
fn tier3_bevy_lvlh_equ() {
    let equ_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    run_lvlh_parity("lvlh_equ", equ_trans);
}

// ── NED/Geodetic parity ──

fn run_ned_parity(label: &str, trans: TranslationalState, r_eq: f64, r_pol: f64) {
    let earth_shape = PlanetShape {
        name: "Earth",
        mu: MU_EARTH,
        r_eq,
        r_pol,
        flat_coeff: if (r_eq - r_pol).abs() < 1.0 {
            0.0
        } else {
            1.0 - r_pol / r_eq
        },
    };

    let mut app = new_bevy_app(DT);
    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
            PlanetFixedRotationC(jeod_sim::FrameTransform::from_matrix(DMat3::IDENTITY)),
            PlanetC(earth_shape),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(trans),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            GeodeticConfigC { planet },
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);
    let bevy_geodetic = app.world().get::<GeodeticStateC>(vehicle).unwrap().0;

    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source(),
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: jeod_sim::planet_config::EARTH.omega,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            geodetic: Some(GeodeticConfig {
                source_idx: earth_idx,
                r_eq,
                r_pol,
            }),
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    assert_trans_eq(
        &format!("Bevy vs Sim ({label})"),
        &bevy_trans,
        &sim_body.trans,
    );
    let sim_geodetic = sim_body.geodetic_state.expect("geodetic computed");
    assert_geodetic_eq(
        &format!("Bevy vs Sim geodetic ({label})"),
        &bevy_geodetic,
        &sim_geodetic,
    );
}

#[test]
fn tier3_bevy_ned_sph_inc() {
    let r_sph = jeod_sim::EARTH.shape.r_eq;
    run_ned_parity("ned_sph_inc", iss_trans(), r_sph, r_sph);
}

#[test]
fn tier3_bevy_ned_sph_polar() {
    let r_sph = jeod_sim::EARTH.shape.r_eq;
    let polar_trans = TranslationalState {
        position: DVec3::new(jeod_sim::EARTH.shape.r_eq + 400_000.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7668.56),
    };
    run_ned_parity("ned_sph_polar", polar_trans, r_sph, r_sph);
}

// ── Orbital elements parity (7 orbit families) ──

fn run_orbelem_parity(label: &str, trans: TranslationalState) {
    let tiny_dt = 1e-9;

    let mut app = new_bevy_app(tiny_dt);
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            OrbitalElementsConfigC {
                gravity_source: planet,
            },
        ))
        .id();

    step_bevy_dt(&mut app, 1, tiny_dt);
    let bevy_oe = app
        .world()
        .get::<OrbitalElementsC>(vehicle)
        .unwrap()
        .0
        .clone();

    let (mut sim, earth_idx) = new_sim_earth(tiny_dt);
    sim.add_body(VehicleConfig {
        trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            orbital_elements_source: Some(earth_idx),
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step().expect("step failed");

    let sim_output = sim.body(0);
    let sim_oe = sim_output.orbital_elements.as_ref().expect("OE computed");
    assert_orbital_elements_eq(&format!("Bevy vs Sim OE ({label})"), &bevy_oe, sim_oe);
}

#[test]
fn tier3_bevy_orbelem_t01() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    run_orbelem_parity("orbelem_t01_circular", trans);
}

#[test]
fn tier3_bevy_orbelem_t10() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_orbelem_parity("orbelem_t10_eccentric", trans);
}

#[test]
fn tier3_bevy_orbelem_t20() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 12_000.0, 0.0),
    };
    run_orbelem_parity("orbelem_t20_hyperbolic", trans);
}

#[test]
fn tier3_bevy_orbelem_t30() {
    let v_esc = (2.0 * MU_EARTH / 6_778_137.0).sqrt();
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_esc * 1.0001, 0.0),
    };
    run_orbelem_parity("orbelem_t30_near_parabolic", trans);
}

#[test]
fn tier3_bevy_orbelem_t40() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, -7668.56, 0.0),
    };
    run_orbelem_parity("orbelem_t40_retrograde", trans);
}

#[test]
fn tier3_bevy_orbelem_t50() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    run_orbelem_parity("orbelem_t50_equatorial", trans);
}

#[test]
fn tier3_bevy_orbelem_t55() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7668.56),
    };
    run_orbelem_parity("orbelem_t55_polar", trans);
}

// ── Orbelem timeseries parity ──

#[test]
fn tier3_bevy_orbelem() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };

    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(ecc_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            OrbitalElementsConfigC {
                gravity_source: planet,
            },
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_oe = app
        .world()
        .get::<OrbitalElementsC>(vehicle)
        .unwrap()
        .0
        .clone();

    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans: ecc_trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            orbital_elements_source: Some(earth_idx),
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_output = sim.body(0);
    let sim_oe = sim_output.orbital_elements.as_ref().expect("OE computed");
    assert_orbital_elements_eq("Bevy vs Sim OE (timeseries)", &bevy_oe, sim_oe);
}

// ── Solar beta parity ──

#[test]
fn tier3_bevy_solar_beta() {
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let inc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 5423.0, 5423.0),
    };

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
            TranslationalStateC::from(inc_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            SolarBetaC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;

    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            jeod_sim::Vec3Ext::m_at::<jeod_sim::RootInertial>(sun_pos),
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);

    sim.add_body(VehicleConfig {
        trans: inc_trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_beta = sim.body(0).solar_beta.expect("solar beta computed");
    assert_bits_eq("Bevy vs Sim", "solar_beta", bevy_beta, sim_beta);
    println!("  Bevy vs Sim solar beta: bit-identical");
}
