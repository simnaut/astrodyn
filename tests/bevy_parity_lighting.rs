//! Bevy-vs-Simulation parity tests: earth lighting.

mod common;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, EarthLightingConfigC, EarthLightingStateC, GravityControlsC, MoonMarker,
    SunMarker, TranslationalStateC,
};
use glam::DVec3;
use jeod_runner::{DerivedStateConfig, EarthLightingConfig, GravitySourceEntry, VehicleConfig};
use jeod_sim::{GravityControl, GravityControls, GravityModel, GravitySource, TranslationalState};

use common::*;

// ── Scenario S: Earth lighting consistency ──

#[test]
fn tier3_sim_earth_lighting_consistency() {
    use jeod_sim::compute_earth_lighting;
    println!("Scenario S: Earth lighting consistency");

    let pos_veh = DVec3::new(6.778e6, 0.0, 0.0);
    let pos_sun = DVec3::new(1.496e11, 0.0, 0.0);
    let pos_moon = DVec3::new(0.0, 3.844e8, 0.0);

    let state1 = compute_earth_lighting(pos_veh, pos_sun, pos_moon, 6.96e8, 6.371e6, 1.737e6);
    let state2 = compute_earth_lighting(pos_veh, pos_sun, pos_moon, 6.96e8, 6.371e6, 1.737e6);

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

    assert!(state1.sun_earth.visible >= 0.0 && state1.sun_earth.visible <= 1.0);
    assert!(state1.sun_earth.occlusion >= 0.0 && state1.sun_earth.occlusion <= 1.0);
    assert!((state1.sun_earth.visible + state1.sun_earth.occlusion - 1.0).abs() < 1e-12);
    assert!(state1.earth_albedo.lighting >= 0.0);

    assert!(
        state1.sun_earth.visible > 0.99,
        "sunlit vehicle should have visible > 0.99, got {}",
        state1.sun_earth.visible
    );
    println!("  Earth lighting: deterministic, physically consistent");
}

// ── Earth lighting parity tests ──

fn run_earth_lighting_parity(label: &str, veh_pos: DVec3, sun_pos: DVec3, moon_pos: DVec3) {
    let earth_r = jeod_sim::EARTH.shadow_radius;
    let moon_r = 1_737_400.0;
    let sun_r = 6.96e8;

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);
    app.world_mut().spawn((
        Name::new("Sun"),
        SunMarker,
        TranslationalStateC::from(TranslationalState {
            position: sun_pos,
            velocity: DVec3::ZERO,
        }),
    ));
    app.world_mut().spawn((
        Name::new("Moon"),
        MoonMarker,
        TranslationalStateC::from(TranslationalState {
            position: moon_pos,
            velocity: DVec3::ZERO,
        }),
    ));

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(TranslationalState {
                position: veh_pos,
                velocity: DVec3::new(0.0, 7668.56, 0.0),
            }),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            EarthLightingConfigC {
                earth_radius: earth_r,
                moon_radius: moon_r,
                sun_radius: sun_r,
            },
        ))
        .id();

    step_bevy_dt(&mut app, 1, DT);
    let bevy_lighting = app
        .world()
        .get::<EarthLightingStateC>(vehicle)
        .unwrap()
        .0
        .clone();

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
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            moon_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);
    sim.moon_source = Some(moon_idx);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: veh_pos,
            velocity: DVec3::new(0.0, 7668.56, 0.0),
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            earth_lighting: Some(EarthLightingConfig {
                earth_radius: earth_r,
                moon_radius: moon_r,
                sun_radius: sun_r,
            }),
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step();

    let sim_body = sim.body(0);
    let sim_lighting = sim_body
        .earth_lighting
        .as_ref()
        .expect("earth lighting computed");
    assert_earth_lighting_eq(
        &format!("Bevy vs Sim ({label})"),
        &bevy_lighting,
        sim_lighting,
    );
}

#[test]
fn tier3_bevy_earth_lighting_t01() {
    run_earth_lighting_parity(
        "t01_sunlit",
        DVec3::new(6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t02() {
    run_earth_lighting_parity(
        "t02_shadow",
        DVec3::new(-6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t03() {
    run_earth_lighting_parity(
        "t03_terminator",
        DVec3::new(0.0, 6_778_137.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t04() {
    run_earth_lighting_parity(
        "t04_moon_inline",
        DVec3::new(6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(3.844e8, 0.0, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t05() {
    run_earth_lighting_parity(
        "t05_geo_sunlit",
        DVec3::new(42_164_000.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t06() {
    run_earth_lighting_parity(
        "t06_polar",
        DVec3::new(0.0, 0.0, 6_778_137.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t07() {
    run_earth_lighting_parity(
        "t07_offset_sun_moon",
        DVec3::new(6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 1e10, 0.0),
        DVec3::new(3.844e8, 1e7, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t08() {
    run_earth_lighting_parity(
        "t08_deep_shadow",
        DVec3::new(-1e7, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 3.844e8),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t09() {
    run_earth_lighting_parity(
        "t09_moon_near_veh_dir",
        DVec3::new(6_778_137.0, 1e5, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(6_778_137.0 * 50.0, 1e5 * 50.0, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_t10() {
    run_earth_lighting_parity(
        "t10_coplanar_45deg",
        DVec3::new(4_793_000.0, 4_793_000.0, 0.0),
        DVec3::new(1.058e11, 1.058e11, 0.0),
        DVec3::new(-2.718e8, 2.718e8, 0.0),
    );
}

#[test]
fn tier3_bevy_earth_lighting_pipeline() {
    let earth_r = jeod_sim::EARTH.shadow_radius;
    let moon_r = 1_737_400.0;
    let sun_r = 6.96e8;
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let moon_pos = DVec3::new(0.0, 3.844e8, 0.0);

    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);
    app.world_mut().spawn((
        Name::new("Sun"),
        SunMarker,
        TranslationalStateC::from(TranslationalState {
            position: sun_pos,
            velocity: DVec3::ZERO,
        }),
    ));
    app.world_mut().spawn((
        Name::new("Moon"),
        MoonMarker,
        TranslationalStateC::from(TranslationalState {
            position: moon_pos,
            velocity: DVec3::ZERO,
        }),
    ));

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(iss_trans()),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            EarthLightingConfigC {
                earth_radius: earth_r,
                moon_radius: moon_r,
                sun_radius: sun_r,
            },
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);
    let bevy_lighting = app
        .world()
        .get::<EarthLightingStateC>(vehicle)
        .unwrap()
        .0
        .clone();

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
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            moon_pos,
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);
    sim.moon_source = Some(moon_idx);

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        derived: DerivedStateConfig {
            earth_lighting: Some(EarthLightingConfig {
                earth_radius: earth_r,
                moon_radius: moon_r,
                sun_radius: sun_r,
            }),
            ..Default::default()
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);
    assert_trans_eq(
        "Bevy vs Sim (earth lighting pipeline)",
        &bevy_trans,
        &sim_body.trans,
    );
    let sim_lighting = sim_body
        .earth_lighting
        .as_ref()
        .expect("earth lighting computed");
    assert_earth_lighting_eq(
        "Bevy vs Sim earth lighting (pipeline)",
        &bevy_lighting,
        sim_lighting,
    );
}
