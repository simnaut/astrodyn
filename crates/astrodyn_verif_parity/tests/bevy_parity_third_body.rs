//! Bevy-vs-Simulation parity tests: third-body gravity, Mars, Mercury relativistic,
//! Clementine, solar beta with DE421 ephemeris.

mod common;

use astrodyn::{DerivedStateConfig, GravitySourceEntry, SrpModel, VehicleConfig};
use astrodyn::{
    DynamicsConfig, Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel,
    GravitySource, MassProperties, SixDofState, TranslationalState,
};
use astrodyn_bevy::{
    CannonballSrpC, DynamicsConfigC, EphemerisBodyC, GravityControlsC, GravitySourceC,
    MassPropertiesC, MoonMarker, PlanetFixedRotationC, RotationModelC, RotationalStateC,
    SolarBetaC, SourceInertialPositionC, SourceInertialVelocityC, SunMarker, TranslationalStateC,
};
use astrodyn_runner::RotationModel;
use bevy::prelude::*;
use glam::{DMat3, DVec3};

use common::*;

// ── Solar beta with DE421 ephemeris ──

#[test]
fn tier3_bevy_solar_beta_equ() {
    println!("Solar beta: equatorial orbit with DE421 ephemeris");
    let initial_sun_pos = sun_initial_pos();

    let equ_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };

    let eph_bevy = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let mut app = new_bevy_app(DT);
    app.insert_resource(astrodyn_bevy::EphemerisR(eph_bevy));
    let planet = spawn_earth_source(&mut app);

    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState {
                position: initial_sun_pos,
                velocity: DVec3::ZERO,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                initial_sun_pos,
            )),
            EphemerisBodyC {
                target: EphemerisBody::Sun,
                observer: EphemerisBody::Earth,
            },
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(equ_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            SolarBetaC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;
    let bevy_trans = read_trans(app.world(), vehicle);

    let eph_sim = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(initial_sun_pos),
            None,
        ),
    );
    sim.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Earth);
    sim.sun_source = Some(sun_idx);
    sim.ephemeris = Some(eph_sim);

    sim.add_body(VehicleConfig {
        trans: equ_trans,
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
    assert_bits_eq("Bevy vs Sim", "solar_beta_equ", bevy_beta, sim_beta);
    assert_trans_eq(
        "Bevy vs Sim (solar_beta_equ)",
        &bevy_trans,
        &sim.body(0).trans,
    );
    println!("  Solar beta equatorial: bit-identical");
}

#[test]
fn tier3_bevy_solar_beta_obliquity() {
    println!("Solar beta: obliquity-inclined orbit with DE421 ephemeris");
    let initial_sun_pos = sun_initial_pos();

    let obliq_rad = 23.44_f64.to_radians();
    let v_mag = 7668.56;
    let obl_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_mag * obliq_rad.cos(), v_mag * obliq_rad.sin()),
    };

    let eph_bevy = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let mut app = new_bevy_app(DT);
    app.insert_resource(astrodyn_bevy::EphemerisR(eph_bevy));
    let planet = spawn_earth_source(&mut app);

    let _sun = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState {
                position: initial_sun_pos,
                velocity: DVec3::ZERO,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                initial_sun_pos,
            )),
            EphemerisBodyC {
                target: EphemerisBody::Sun,
                observer: EphemerisBody::Earth,
            },
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(obl_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            SolarBetaC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;
    let bevy_trans = read_trans(app.world(), vehicle);

    let eph_sim = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(initial_sun_pos),
            None,
        ),
    );
    sim.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Earth);
    sim.sun_source = Some(sun_idx);
    sim.ephemeris = Some(eph_sim);

    sim.add_body(VehicleConfig {
        trans: obl_trans,
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
    assert_bits_eq("Bevy vs Sim", "solar_beta_obliquity", bevy_beta, sim_beta);
    assert_trans_eq(
        "Bevy vs Sim (solar_beta_obliquity)",
        &bevy_trans,
        &sim.body(0).trans,
    );
    println!("  Solar beta obliquity: bit-identical");
}

// ── 3rd-body parity ──

fn run_3rd_body_parity(label: &str, trans: TranslationalState, include_moon: bool, sixdof: bool) {
    let initial_sun_pos = sun_initial_pos();
    let initial_moon_pos = if include_moon {
        Some(moon_initial_pos())
    } else {
        None
    };

    let eph_bevy = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let mut app = new_bevy_app(DT);
    app.insert_resource(astrodyn_bevy::EphemerisR(eph_bevy));
    let planet = spawn_earth_source(&mut app);

    let sun_entity = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            GravitySourceC(GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            }),
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState {
                position: initial_sun_pos,
                velocity: DVec3::ZERO,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                initial_sun_pos,
            )),
            EphemerisBodyC {
                target: EphemerisBody::Sun,
                observer: EphemerisBody::Earth,
            },
        ))
        .id();

    let moon_entity = if let Some(moon_pos) = initial_moon_pos {
        let mu_moon = astrodyn::MOON.shape.mu;
        Some(
            app.world_mut()
                .spawn((
                    Name::new("Moon"),
                    MoonMarker,
                    GravitySourceC(GravitySource {
                        mu: mu_moon,
                        model: GravityModel::PointMass,
                    }),
                    TranslationalStateC::<astrodyn::Earth>::from(TranslationalState {
                        position: moon_pos,
                        velocity: DVec3::ZERO,
                    }),
                    SourceInertialPositionC(
                        astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(moon_pos),
                    ),
                    EphemerisBodyC {
                        target: EphemerisBody::Moon,
                        observer: EphemerisBody::Earth,
                    },
                ))
                .id(),
        )
    } else {
        None
    };

    let mut sun_ctrl = GravityControl::new_spherical(sun_entity, false);
    sun_ctrl.differential = true;

    let mut controls = vec![GravityControl::new_spherical(planet, false), sun_ctrl];
    if let Some(moon_ent) = moon_entity {
        let mut moon_ctrl = GravityControl::new_spherical(moon_ent, false);
        moon_ctrl.differential = true;
        controls.push(moon_ctrl);
    }

    let config = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: sixdof,
        three_dof: !sixdof,
    };

    let vehicle = if sixdof {
        app.world_mut()
            .spawn((
                TranslationalStateC::<astrodyn::Earth>::from(trans),
                RotationalStateC::from(tumble_rot()),
                MassPropertiesC::from(iss_mass()),
                DynamicsConfigC(config),
                GravityControlsC(GravityControls { controls }),
            ))
            .id()
    } else {
        app.world_mut()
            .spawn((
                TranslationalStateC::<astrodyn::Earth>::from(trans),
                DynamicsConfigC(config),
                GravityControlsC(GravityControls { controls }),
            ))
            .id()
    };

    step_bevy(&mut app, NUM_STEPS);

    // ── Simulation ──
    let eph_sim = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(initial_sun_pos),
            None,
        ),
    );
    sim.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Earth);
    sim.sun_source = Some(sun_idx);

    let moon_idx = if include_moon {
        let mu_moon = astrodyn::MOON.shape.mu;
        let moon_pos = initial_moon_pos.unwrap();
        let idx = sim.add_source(
            "Moon",
            GravitySourceEntry::new(
                GravitySource {
                    mu: mu_moon,
                    model: GravityModel::PointMass,
                },
                astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(moon_pos),
                None,
            ),
        );
        sim.set_source_ephemeris(idx, EphemerisBody::Moon, EphemerisBody::Earth);
        sim.moon_source = Some(idx);
        Some(idx)
    } else {
        None
    };

    sim.ephemeris = Some(eph_sim);

    let mut sim_sun_ctrl = GravityControl::new_spherical(sun_idx, false);
    sim_sun_ctrl.differential = true;

    let mut sim_controls = vec![
        GravityControl::new_spherical(earth_idx, false),
        sim_sun_ctrl,
    ];
    if let Some(m_idx) = moon_idx {
        let mut sim_moon_ctrl = GravityControl::new_spherical(m_idx, false);
        sim_moon_ctrl.differential = true;
        sim_controls.push(sim_moon_ctrl);
    }

    sim.add_body(VehicleConfig {
        trans,
        rot: if sixdof { Some(tumble_rot()) } else { None },
        mass: if sixdof { Some(iss_mass()) } else { None },
        gravity_controls: GravityControls {
            controls: sim_controls,
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    if sixdof {
        let bevy_state = read_sixdof(app.world(), vehicle);
        let sim_state = SixDofState {
            trans: sim.body(0).trans,
            rot: sim.body(0).rot.unwrap(),
        };
        assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
    } else {
        let bevy_trans = read_trans(app.world(), vehicle);
        assert_trans_eq(
            &format!("Bevy vs Sim ({label})"),
            &bevy_trans,
            &sim.body(0).trans,
        );
    }
}

#[test]
fn tier3_bevy_run3b_3rd_body_sun() {
    println!("Dyncomp run3b: ISS + Sun 3rd body, 3-DOF");
    run_3rd_body_parity("run3b", iss_trans(), false, false);
}

#[test]
fn tier3_bevy_run4_3rd_body_sun_moon() {
    println!("Dyncomp run4: ISS + Sun + Moon 3rd body, 3-DOF");
    run_3rd_body_parity("run4", iss_trans(), true, false);
}

#[test]
fn tier3_bevy_run7a_3rd_body_sixdof() {
    println!("Dyncomp run7a: ISS + Sun 3rd body, 6-DOF");
    run_3rd_body_parity("run7a", iss_trans(), false, true);
}

#[test]
fn tier3_bevy_run7b_3rd_body_sun_moon_sixdof() {
    println!("Dyncomp run7b: ISS + Sun + Moon 3rd body, 6-DOF");
    run_3rd_body_parity("run7b", iss_trans(), true, true);
}

#[test]
fn tier3_bevy_run7c_3rd_body_inclined() {
    println!("Dyncomp run7c: inclined orbit + Sun 3rd body, 3-DOF");
    let inclined = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 5423.0, 5423.0),
    };
    run_3rd_body_parity("run7c", inclined, false, false);
}

#[test]
fn tier3_bevy_run7d_3rd_body_polar() {
    println!("Dyncomp run7d: polar orbit + Sun + Moon 3rd body, 3-DOF");
    let polar = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7668.56),
    };
    run_3rd_body_parity("run7d", polar, true, false);
}

// ── Mars dawn parity ──

#[test]
fn tier3_bevy_mars_dawn() {
    println!("Mars dawn: Mars point-mass + MarsIAU rotation + Sun 3rd body");

    let mu_mars: f64 = 4.282_837_4e13;
    let r_mars: f64 = 3_389_500.0 + 500_000.0;
    let v_circ = (mu_mars / r_mars).sqrt();
    let mars_trans = TranslationalState {
        position: DVec3::new(r_mars, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_circ, 0.0),
    };

    let eph_init = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_rel_mars_typed, _) = eph_init
        .get_state_typed(EphemerisBody::Sun, EphemerisBody::Mars, J2000_JD)
        .expect("Sun wrt Mars at J2000");
    let sun_rel_mars = sun_rel_mars_typed.raw_si();

    let eph_bevy = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let mut app = new_bevy_app(DT);
    app.insert_resource(astrodyn_bevy::EphemerisR(eph_bevy));

    let mars_entity = app
        .world_mut()
        .spawn((
            Name::new("Mars"),
            GravitySourceC(GravitySource {
                mu: mu_mars,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
            PlanetFixedRotationC::<astrodyn::Earth>(astrodyn::FrameTransform::from_matrix(
                DMat3::IDENTITY,
            )),
            RotationModelC(RotationModel::MarsIAU),
        ))
        .id();

    let sun_entity = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            GravitySourceC(GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            }),
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState {
                position: sun_rel_mars,
                velocity: DVec3::ZERO,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                sun_rel_mars,
            )),
            EphemerisBodyC {
                target: EphemerisBody::Sun,
                observer: EphemerisBody::Mars,
            },
        ))
        .id();

    let mut sun_ctrl = GravityControl::new_spherical(sun_entity, false);
    sun_ctrl.differential = true;

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(mars_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(mars_entity, false), sun_ctrl],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);

    let eph_sim = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let mars_idx = sim.add_source(
        "Mars",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_mars,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::MarsIAU,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(sun_rel_mars),
            None,
        ),
    );
    sim.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Mars);
    sim.sun_source = Some(sun_idx);
    sim.ephemeris = Some(eph_sim);

    let mut sim_sun_ctrl = GravityControl::new_spherical(sun_idx, false);
    sim_sun_ctrl.differential = true;

    sim.add_body(VehicleConfig {
        trans: mars_trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(mars_idx, false), sim_sun_ctrl],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    assert_trans_eq("Bevy vs Sim (mars_dawn)", &bevy_trans, &sim.body(0).trans);
    println!("  Mars dawn: bit-identical");
}

// ── Mercury relativistic ──

#[test]
fn tier3_bevy_mercury_relativistic() {
    println!("Mercury relativistic: Sun point-mass with PPN correction");

    let mu_sun = astrodyn::SUN.shape.mu;
    let r_perihelion = 4.6e10;
    let v_perihelion = 5.898e4;
    let mercury_trans = TranslationalState {
        position: DVec3::new(r_perihelion, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_perihelion, 0.0),
    };

    let mut app = new_bevy_app(DT);

    let sun_entity = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            GravitySourceC(GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let mut sun_ctrl = GravityControl::new_spherical(sun_entity, false);
    sun_ctrl.relativistic = true;

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(mercury_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![sun_ctrl],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);

    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let mut sun_entry = GravitySourceEntry::new(
        GravitySource {
            mu: mu_sun,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    sun_entry.central = true;
    let sun_idx = sim.add_source("Sun", sun_entry);

    let mut sim_sun_ctrl = GravityControl::new_spherical(sun_idx, false);
    sim_sun_ctrl.relativistic = true;

    sim.add_body(VehicleConfig {
        trans: mercury_trans,
        gravity_controls: GravityControls {
            controls: vec![sim_sun_ctrl],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    assert_trans_eq(
        "Bevy vs Sim (mercury_relativistic)",
        &bevy_trans,
        &sim.body(0).trans,
    );
    println!("  Mercury relativistic: bit-identical");
}

/// Relativistic parity with non-zero source velocity.
///
/// A satellite orbits a Sun that is itself moving at 30 km/s (as if the Sun
/// were in a barycentric frame). Both Bevy and Simulation receive the same
/// non-zero source velocity for the PPN correction. This exercises
/// `SourceInertialVelocityC` and ensures it is wired through both the
/// `gravity_computation_system` and `integration_system`.
#[test]
fn tier3_bevy_relativistic_moving_source() {
    println!("Relativistic moving source: Sun with non-zero velocity");

    let mu_sun = astrodyn::SUN.shape.mu;
    let r_perihelion = 4.6e10;
    let v_perihelion = 5.898e4;
    let mercury_trans = TranslationalState {
        position: DVec3::new(r_perihelion, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_perihelion, 0.0),
    };

    // Non-zero source velocity (Sun moving in barycentric frame)
    let source_velocity = DVec3::new(0.0, 0.0, 3.0e4);

    // ── Bevy ──
    let mut app = new_bevy_app(DT);

    let sun_entity = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            GravitySourceC(GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            SourceInertialVelocityC(astrodyn::Velocity::<astrodyn::RootInertial>::from_raw_si(
                source_velocity,
            )),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let mut sun_ctrl = GravityControl::new_spherical(sun_entity, false);
    sun_ctrl.relativistic = true;

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(mercury_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![sun_ctrl],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Vec3Ext::m_per_s_at::<astrodyn::RootInertial>(source_velocity),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    let mut sim_sun_ctrl = GravityControl::new_spherical(sun_idx, false);
    sim_sun_ctrl.relativistic = true;

    sim.add_body(VehicleConfig {
        trans: mercury_trans,
        gravity_controls: GravityControls {
            controls: vec![sim_sun_ctrl],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    assert_trans_eq(
        "Bevy vs Sim (relativistic_moving_source)",
        &bevy_trans,
        &sim.body(0).trans,
    );

    println!("  Relativistic moving source: bit-identical");
}

// ── Earth-Moon Clementine ──

#[test]
fn tier3_bevy_earth_moon_clem() {
    println!("Earth-Moon Clementine: Earth + Moon + Sun with cannonball SRP");

    let initial_sun_pos = sun_initial_pos();
    let initial_moon_pos = moon_initial_pos();

    let r_perigee = astrodyn::EARTH.shape.r_eq + 400_000.0;
    let v_perigee = 10_500.0;
    let clem_trans = TranslationalState {
        position: DVec3::new(r_perigee, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_perigee, 0.0),
    };
    let mu_moon = astrodyn::MOON.shape.mu;

    let cx_area = 5.0;
    let albedo = 0.3;
    let diffuse = 0.5;
    let mass = 500.0;
    let mass_props = MassProperties::with_inertia(
        mass,
        DMat3::from_diagonal(DVec3::new(200.0, 200.0, 200.0)),
        DVec3::ZERO,
    );

    let eph_bevy = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let mut app = new_bevy_app(DT);
    app.insert_resource(astrodyn_bevy::EphemerisR(eph_bevy));
    let planet = spawn_earth_source(&mut app);

    let sun_entity = app
        .world_mut()
        .spawn((
            Name::new("Sun"),
            SunMarker,
            GravitySourceC(GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            }),
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState {
                position: initial_sun_pos,
                velocity: DVec3::ZERO,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                initial_sun_pos,
            )),
            EphemerisBodyC {
                target: EphemerisBody::Sun,
                observer: EphemerisBody::Earth,
            },
        ))
        .id();

    let moon_entity = app
        .world_mut()
        .spawn((
            Name::new("Moon"),
            MoonMarker,
            GravitySourceC(GravitySource {
                mu: mu_moon,
                model: GravityModel::PointMass,
            }),
            TranslationalStateC::<astrodyn::Earth>::from(TranslationalState {
                position: initial_moon_pos,
                velocity: DVec3::ZERO,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                initial_moon_pos,
            )),
            EphemerisBodyC {
                target: EphemerisBody::Moon,
                observer: EphemerisBody::Earth,
            },
        ))
        .id();

    let mut sun_ctrl = GravityControl::new_spherical(sun_entity, false);
    sun_ctrl.differential = true;
    let mut moon_ctrl = GravityControl::new_spherical(moon_entity, false);
    moon_ctrl.differential = true;

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(clem_trans),
            astrodyn_bevy::MassPropertiesC::from(mass_props),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![
                    GravityControl::new_spherical(planet, false),
                    sun_ctrl,
                    moon_ctrl,
                ],
            }),
            CannonballSrpC {
                cx_area,
                albedo,
                diffuse,
            },
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);

    let eph_sim = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(initial_sun_pos),
            None,
        ),
    );
    sim.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Earth);
    sim.sun_source = Some(sun_idx);

    let moon_sim_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: mu_moon,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(initial_moon_pos),
            None,
        ),
    );
    sim.set_source_ephemeris(moon_sim_idx, EphemerisBody::Moon, EphemerisBody::Earth);
    sim.moon_source = Some(moon_sim_idx);
    sim.ephemeris = Some(eph_sim);

    let mut sim_sun_ctrl = GravityControl::new_spherical(sun_idx, false);
    sim_sun_ctrl.differential = true;
    let mut sim_moon_ctrl = GravityControl::new_spherical(moon_sim_idx, false);
    sim_moon_ctrl.differential = true;

    sim.add_body(VehicleConfig {
        trans: clem_trans,
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth_idx, false),
                sim_sun_ctrl,
                sim_moon_ctrl,
            ],
        },
        srp: Some(SrpModel::Cannonball {
            cx_area,
            albedo,
            diffuse,
        }),
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    assert_trans_eq(
        "Bevy vs Sim (earth_moon_clem)",
        &bevy_trans,
        &sim.body(0).trans,
    );
    println!("  Earth-Moon Clementine SRP: bit-identical");
}
