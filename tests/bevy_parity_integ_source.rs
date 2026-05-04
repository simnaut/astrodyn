//! Tier 3: Bevy non-root integration frame parity (issue #71 item 4).
//!
//! With `IntegSourceC(Some(planet))` on a body, the Bevy adapter
//! integrates the body in that planet's inertial frame: gravity is
//! evaluated at `body_pos + integ_origin`, the differential-gravity
//! correction subtracts the integ frame's own acceleration toward each
//! source, and `TranslationalStateC` carries position/velocity in
//! integ-frame coordinates. This test asserts the resulting body state
//! is bit-identical to `jeod_runner::Simulation` configured with the
//! same `VehicleConfig::integ_source`.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, GravityControlsC, IntegSourceC, JeodPlugin, MassPropertiesC, PlanetBundle,
    RotationalStateC, SolarBetaC, SourceInertialPositionC, SourceInertialVelocityC, SourceMutator,
    SunMarker, TranslationalStateC,
};
use glam::DVec3;
use jeod_runner::Simulation;
use jeod_sim::{
    DerivedStateConfig, DynamicsConfig, GravityControl, GravityControls, GravityModel,
    GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationalState, SixDofState,
    TranslationalState, VehicleConfig, EARTH, MOON,
};

const DT: f64 = 60.0;
const NUM_STEPS: usize = 30;
const MOON_OFFSET: DVec3 = DVec3::new(3.844e8, 0.0, 0.0);

fn lunar_initial_trans() -> TranslationalState {
    // 100 km lunar circular orbit (in moon-centered coordinates).
    // r = 1737.4 km + 100 km = 1837.4 km; v = sqrt(mu_moon / r).
    let r = 1_837_400.0;
    let v = (MOON.shape.mu / r).sqrt();
    TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    }
}

fn initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    }
}

fn vehicle_mass() -> MassProperties {
    MassProperties::with_inertia(
        1_000.0,
        glam::DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
        DVec3::ZERO,
    )
}

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
    }
}

#[test]
fn tier3_bevy_integ_source_lunar_orbit_matches_simulation() {
    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    // Earth (central) + Moon (offset) as gravity sources. Earth sits at
    // root-relative origin; Moon is positioned via SourceMutator after
    // registration so its frame node carries the offset that the body
    // integrates relative to.
    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Moon", &MOON))
        .insert(SourceInertialVelocityC::default())
        .id();

    // Vehicle: integrates in Moon.inertial; gravity from both Earth and
    // Moon (Earth is the differential third body).
    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("Lunar"),
            TranslationalStateC::from(lunar_initial_trans()),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(vehicle_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![
                    {
                        let mut c = GravityControl::new_spherical(_earth, false);
                        c.differential = true;
                        c
                    },
                    GravityControl::new_spherical(moon, false),
                ],
            }),
            IntegSourceC(Some(moon)),
        ))
        .id();

    // Run startup so register_source_frames + register_body_frames fire.
    app.world_mut().run_schedule(Startup);

    // Set the Moon's inertial position via SourceMutator (parity with
    // `jeod_runner::Simulation::set_source_position`).
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator| {
            m.set_source_position(moon, MOON_OFFSET);
        });
    app.world_mut().run_system(sys).unwrap();

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let bevy_state = SixDofState {
        trans: app
            .world()
            .get::<TranslationalStateC>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
        rot: app
            .world()
            .get::<RotationalStateC>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
    };

    // ── jeod_runner ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let _earth_idx = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(
            &MOON,
            jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(MOON_OFFSET),
        ),
    );

    sim.add_body(VehicleConfig {
        trans: lunar_initial_trans(),
        rot: Some(initial_rot()),
        mass: Some(vehicle_mass()),
        gravity_controls: GravityControls {
            controls: vec![
                {
                    let mut c = GravityControl::new_spherical(0_usize, false);
                    c.differential = true;
                    c
                },
                GravityControl::new_spherical(moon_idx, false),
            ],
        },
        integ_source: Some(moon_idx),
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");
    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_bit_identical("Bevy integ_source vs Sim", &bevy_state, &sim_state);
}

#[test]
fn tier3_bevy_integ_source_moving_moon_matches_simulation() {
    // PR #260 round-3 N3 fixup: cover the moving-integ-frame path.
    // The lunar-orbit test above keeps Moon velocity at zero, so the
    // per-stage `integ_origin + integ_origin_vel * stage_dt` and source
    // interpolation in `integration_system::eval_gravity` never fire.
    // This test sets a non-zero Moon velocity (orbital speed at the
    // configured Earth–Moon offset) so each RK sub-stage exercises the
    // moving-frame code path. A regression in either the integ-origin
    // interpolation or the source-position interpolation would diverge
    // from `jeod_runner` here.
    let moon_vel = DVec3::new(0.0, 1024.0, 0.0); // ~lunar orbital speed

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Moon", &MOON))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("Lunar"),
            TranslationalStateC::from(lunar_initial_trans()),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(vehicle_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![
                    {
                        let mut c = GravityControl::new_spherical(_earth, false);
                        c.differential = true;
                        c
                    },
                    GravityControl::new_spherical(moon, false),
                ],
            }),
            IntegSourceC(Some(moon)),
        ))
        .id();
    app.world_mut().run_schedule(Startup);

    // Set Moon's inertial position AND velocity. SourceMutator
    // auto-inserts SourceInertialVelocityC since PlanetBundle didn't.
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator| {
            m.set_source_state(moon, MOON_OFFSET, moon_vel);
        });
    app.world_mut().run_system(sys).unwrap();

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let bevy_state = SixDofState {
        trans: app
            .world()
            .get::<TranslationalStateC>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
        rot: app
            .world()
            .get::<RotationalStateC>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
    };

    // ── jeod_runner ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let _earth_idx = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(
            &MOON,
            jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(MOON_OFFSET),
        ),
    );
    sim.set_source_state(moon_idx, MOON_OFFSET, moon_vel);

    sim.add_body(VehicleConfig {
        trans: lunar_initial_trans(),
        rot: Some(initial_rot()),
        mass: Some(vehicle_mass()),
        gravity_controls: GravityControls {
            controls: vec![
                {
                    let mut c = GravityControl::new_spherical(0_usize, false);
                    c.differential = true;
                    c
                },
                GravityControl::new_spherical(moon_idx, false),
            ],
        },
        integ_source: Some(moon_idx),
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");
    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_bit_identical("Bevy moving integ_source vs Sim", &bevy_state, &sim_state);
}

#[test]
fn tier3_bevy_integ_source_root_matches_legacy_no_op() {
    // Sanity: with `IntegSourceC` absent (the legacy default), behavior
    // is unchanged — bodies integrate in root, integ_origin = 0, and
    // results match the existing `bevy_parity_point_mass` semantics.
    // This test mirrors the simplest existing parity scenario but
    // explicitly with the frame-entity registration components
    // present, confirming the new code path is bit-identical to the
    // pre-#71 behavior.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            bevy_jeod::components::GravitySourceC(GravitySource {
                mu: EARTH.shape.mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::from(TranslationalState::default()),
        ))
        .id();

    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("Vehicle"),
            TranslationalStateC::from(trans),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(MassProperties::with_inertia(
                400_000.0,
                glam::DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
                DVec3::ZERO,
            )),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            }),
        ))
        .id();
    app.world_mut().run_schedule(Startup);

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let bevy_state = SixDofState {
        trans: app
            .world()
            .get::<TranslationalStateC>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
        rot: app
            .world()
            .get::<RotationalStateC>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
    };

    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::central_body(&EARTH);
    earth_entry.source.model = GravityModel::PointMass;
    sim.add_source("Earth", earth_entry);
    sim.add_body(VehicleConfig {
        trans,
        rot: Some(initial_rot()),
        mass: Some(MassProperties::with_inertia(
            400_000.0,
            glam::DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
            DVec3::ZERO,
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(0_usize, false)],
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

    assert_sixdof_bit_identical("Bevy root-integ vs Sim", &bevy_state, &sim_state);
}

/// Regression for PR #296 Thread 3: a non-root-integrated body's
/// `TranslationalStateC` is in `<PlanetInertial<SelfPlanet>>` (Moon-
/// inertial here). `solar_beta_system` mixes that state with the Sun
/// position, which is in the simulation's root-inertial frame — so
/// without the integ-origin shift it would compute solar beta off by
/// the Earth–Moon separation (~3.844e8 m), producing a value many
/// degrees away from the correct geometry. With the shift, the
/// computed beta is bit-identical to `jeod_runner::Simulation`'s
/// post-integration `body.solar_beta` for the same configuration.
///
/// Geometry rationale: `solar_beta = asin(h_hat · sun_hat)` with
/// `h = r × v` and `sun_hat = (sun_pos − r) / |sun_pos − r|`. The
/// bug case computes `h` and `sun_hat` from `r =
/// body_moon_inertial`; the fix case uses `r = body_moon_inertial +
/// moon_offset`. With the orbit confined to the XY plane (`vel.z =
/// 0`) and the Sun on the +X axis both `h_hat` and `sun_hat` end up
/// in the same direction in either case, making bug and fix
/// numerically indistinguishable. To force the geometry apart, this
/// test tilts the lunar orbit (non-zero `vel.z` ⇒ `h_hat` rotates
/// out of the body-frame +Z axis as the body advances along its
/// orbit, with the rotation axis depending on `r`) and places the
/// Sun off the X axis (so `sun_hat` differs between the two `r`
/// values). The result: bug-shape and fix-shape `h_hat · sun_hat`
/// differ by orders of magnitude above f64 round-off.
#[test]
fn tier3_bevy_solar_beta_in_lunar_integ_frame() {
    // Sun off the +X axis (~1 AU in X, ~0.7 AU in Z) so `sun_hat`
    // depends on the X *and* Z components of `r` — making the bug
    // and fix shapes geometrically distinct.
    let sun_pos = DVec3::new(1.496e11, 0.0, 1.0e11);

    // Tilted 100 km lunar circular orbit: redistribute the orbital
    // velocity into Y/Z so `h = r × v = r_x · (0, -v_z, v_y)` has
    // both Y and Z components — the orbit normal is no longer aligned
    // with +Z, and `h_hat · sun_hat` becomes sensitive to whether `r`
    // includes the Earth–Moon offset.
    let r = 1_837_400.0;
    let v = (MOON.shape.mu / r).sqrt();
    let lunar_tilted = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        // 30° tilt: v_y = v cos(30°), v_z = v sin(30°).
        velocity: DVec3::new(0.0, v * 0.866_025_403_784_438_6, v * 0.5),
    };

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Moon", &MOON))
        .insert(SourceInertialVelocityC::default())
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
            Name::new("Lunar"),
            TranslationalStateC::from(lunar_tilted),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(vehicle_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![
                    {
                        let mut c = GravityControl::new_spherical(_earth, false);
                        c.differential = true;
                        c
                    },
                    GravityControl::new_spherical(moon, false),
                ],
            }),
            IntegSourceC(Some(moon)),
            SolarBetaC::default(),
        ))
        .id();
    app.world_mut().run_schedule(Startup);

    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator| {
            m.set_source_position(moon, MOON_OFFSET);
        });
    app.world_mut().run_system(sys).unwrap();

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let bevy_beta = app.world().get::<SolarBetaC>(vehicle).unwrap().0;

    // ── jeod_runner ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let _earth_idx = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(
            &MOON,
            jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(MOON_OFFSET),
        ),
    );
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
        trans: lunar_tilted,
        rot: Some(initial_rot()),
        mass: Some(vehicle_mass()),
        gravity_controls: GravityControls {
            controls: vec![
                {
                    let mut c = GravityControl::new_spherical(0_usize, false);
                    c.differential = true;
                    c
                },
                GravityControl::new_spherical(moon_idx, false),
            ],
        },
        integ_source: Some(moon_idx),
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_beta = sim
        .body(0)
        .solar_beta
        .expect("solar beta computed in lunar integ frame");
    assert_bits_eq(
        "Bevy lunar-integ solar_beta vs Sim",
        "solar_beta",
        bevy_beta,
        sim_beta,
    );
}
