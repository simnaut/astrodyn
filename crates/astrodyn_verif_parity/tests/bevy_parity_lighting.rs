//! Bevy-vs-Simulation parity tests: earth lighting.

mod common;

use astrodyn::{DerivedStateConfig, EarthLightingConfig, GravitySourceEntry, VehicleConfig};
use astrodyn::{
    GravityControl, GravityControls, GravityModel, GravitySource, TranslationalState, MOON,
};
use astrodyn_bevy::{
    DynamicsConfigC, EarthLightingConfigC, EarthLightingStateC, GravityControlsC, IntegSourceC,
    MoonMarker, PlanetBundle, SourceMutator, SunMarker, TranslationalStateC,
};
use bevy::prelude::*;
use glam::DVec3;

use common::*;

// ── Scenario S: Earth lighting consistency ──

#[test]
fn bevy_parity_lighting_earth_lighting_consistency() {
    use astrodyn::compute_earth_lighting;
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
    let earth_r = astrodyn::EARTH.shadow_radius;
    let moon_r = 1_737_400.0;
    let sun_r = 6.96e8;

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);
    app.world_mut().spawn((
        Name::new("Sun"),
        SunMarker,
        TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
            position: sun_pos,
            velocity: DVec3::ZERO,
        }),
    ));
    app.world_mut().spawn((
        Name::new("Moon"),
        MoonMarker,
        TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
            position: moon_pos,
            velocity: DVec3::ZERO,
        }),
    ));

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
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
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(sun_pos),
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
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(moon_pos),
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);
    sim.moon_source = Some(moon_idx);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: veh_pos,
            velocity: DVec3::new(0.0, 7668.56, 0.0),
        }),
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
    sim.step().expect("step failed");

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
fn bevy_parity_lighting_earth_lighting_t01() {
    run_earth_lighting_parity(
        "t01_sunlit",
        DVec3::new(6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t02() {
    run_earth_lighting_parity(
        "t02_shadow",
        DVec3::new(-6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t03() {
    run_earth_lighting_parity(
        "t03_terminator",
        DVec3::new(0.0, 6_778_137.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t04() {
    run_earth_lighting_parity(
        "t04_moon_inline",
        DVec3::new(6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(3.844e8, 0.0, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t05() {
    run_earth_lighting_parity(
        "t05_geo_sunlit",
        DVec3::new(42_164_000.0, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t06() {
    run_earth_lighting_parity(
        "t06_polar",
        DVec3::new(0.0, 0.0, 6_778_137.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 3.844e8, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t07() {
    run_earth_lighting_parity(
        "t07_offset_sun_moon",
        DVec3::new(6_778_137.0, 0.0, 0.0),
        DVec3::new(1.496e11, 1e10, 0.0),
        DVec3::new(3.844e8, 1e7, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t08() {
    run_earth_lighting_parity(
        "t08_deep_shadow",
        DVec3::new(-1e7, 0.0, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 3.844e8),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t09() {
    run_earth_lighting_parity(
        "t09_moon_near_veh_dir",
        DVec3::new(6_778_137.0, 1e5, 0.0),
        DVec3::new(1.496e11, 0.0, 0.0),
        DVec3::new(6_778_137.0 * 50.0, 1e5 * 50.0, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_t10() {
    run_earth_lighting_parity(
        "t10_coplanar_45deg",
        DVec3::new(4_793_000.0, 4_793_000.0, 0.0),
        DVec3::new(1.058e11, 1.058e11, 0.0),
        DVec3::new(-2.718e8, 2.718e8, 0.0),
    );
}

#[test]
fn bevy_parity_lighting_earth_lighting_pipeline() {
    let earth_r = astrodyn::EARTH.shadow_radius;
    let moon_r = 1_737_400.0;
    let sun_r = 6.96e8;
    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let moon_pos = DVec3::new(0.0, 3.844e8, 0.0);

    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);
    app.world_mut().spawn((
        Name::new("Sun"),
        SunMarker,
        TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
            position: sun_pos,
            velocity: DVec3::ZERO,
        }),
    ));
    app.world_mut().spawn((
        Name::new("Moon"),
        MoonMarker,
        TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
            position: moon_pos,
            velocity: DVec3::ZERO,
        }),
    ));

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
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
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(sun_pos),
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
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(moon_pos),
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
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_trans = astrodyn::typed_bridge::trans_typed_to_raw(&sim_body.trans);
    assert_trans_eq(
        "Bevy vs Sim (earth lighting pipeline)",
        &bevy_trans,
        &sim_trans,
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

// ── Non-root integ-source earth lighting parity ──
//
// All sibling earth-lighting tests above spawn a body with default
// `IntegSourceC` (= root inertial), so the integ-origin lift in
// `earth_lighting_system` is a numerical no-op for them. This test
// exercises a body whose `IntegSourceC` points at the Moon (a non-root
// gravity source parked at `MOON_OFFSET` via `set_source_state`). The
// body's `TranslationalStateC.position` is therefore moon-relative
// (integ-frame coordinates); without the lift through
// `body_integ_origin_in_root`, the typed earth-lighting kernel would
// receive a position off by `MOON_OFFSET` (~3.84e8 m) — orders of
// magnitude larger than any earth-lighting f64 round-off, so a regression
// that drops the lift fails this test loudly.
//
// The runner-side mirror cannot use a non-root `integ_source` because
// `Simulation::validate` rejects non-root integration combined with
// earth-lighting (`NonRootFrameWithRootDependentFeatures`, validate.rs
// row "earth_lighting_config" — issue #263). Instead the runner
// runs an *equivalent* root-integrated body whose initial position is the
// Bevy body's lifted root-inertial state (= moon-relative + Moon offset),
// with Moon and Sun parked at the same root-inertial positions. If the
// Bevy lift is correct, both runtimes evaluate the kernel with
// bit-identical inputs and produce bit-identical
// `EarthLightingState`.

#[test]
fn bevy_parity_lighting_earth_lighting_non_root_integ_source() {
    let earth_r = astrodyn::EARTH.shadow_radius;
    let moon_r = 1_737_400.0;
    let sun_r = 6.96e8;

    // Moon parked along +x at lunar-distance offset; Sun at 1 AU along
    // +x. The body's moon-relative position places it on the sunward
    // side of Earth-after-lift (root-inertial position
    // = moon_offset + body_moon_rel ≈ (3.844e8 + 6.778e6, 1e5, 0)),
    // i.e. firmly outside Earth's shadow cone. The lift therefore
    // changes both the sun_visible classification and the moon_earth
    // geometry vs. the unlifted (moon-relative) interpretation.
    const MOON_OFFSET: DVec3 = DVec3::new(3.844e8, 0.0, 0.0);
    const SUN_POS: DVec3 = DVec3::new(1.496e11, 0.0, 0.0);
    // Body state in moon-relative integ-frame coords (a low lunar
    // orbit). Non-zero y component picked so the lifted root-inertial
    // position differs from `MOON_OFFSET` along two axes — a regression
    // that drops only the x component of the lift would still fail.
    let body_moon_rel_pos = DVec3::new(6_778_137.0, 1.0e5, 0.0);
    let body_moon_rel_vel = DVec3::new(0.0, 7668.56, 0.0);

    // ── Bevy ──
    let mut app = new_bevy_app(DT);

    // Earth: root-source for gravity controls. (`spawn_earth_source`
    // also inserts a `GravitySourceC` at the root.)
    let earth = spawn_earth_source(&mut app);

    // Sun: dual-role marker + (no-mass) gravity source so we don't
    // need a second sun entity. Position is set later via
    // `set_source_state`.
    let sun_entity = app
        .world_mut()
        .spawn((
            PlanetBundle::<astrodyn::Earth>::point_mass("Sun", &astrodyn::SUN),
            SunMarker,
        ))
        .id();

    // Moon: dual-role marker + non-root gravity source. The body's
    // `IntegSourceC` points at this entity, so the body's frame entity
    // is parented to the Moon's frame entity (non-root integ frame).
    let moon_entity = app
        .world_mut()
        .spawn((
            PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON),
            MoonMarker,
        ))
        .id();

    // Run Startup so register_source_frames_system spawns the source
    // frame entities (required by SourceMutator::set_source_state and
    // by register_body_frames_system below).
    app.world_mut().run_schedule(Startup);

    // Park Moon at MOON_OFFSET, Sun at 1 AU. `set_source_state`
    // updates the source's frame entity (FrameTransC) and its
    // TranslationalStateC, so the earth-lighting system reads the
    // correct moon/sun positions when it runs.
    let setup = app
        .world_mut()
        .register_system(move |mut m: SourceMutator<astrodyn::Earth>| {
            m.set_source_state(moon_entity, MOON_OFFSET, DVec3::ZERO);
            m.set_source_state(sun_entity, SUN_POS, DVec3::ZERO);
        });
    app.world_mut().run_system(setup).unwrap();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: body_moon_rel_pos,
                velocity: body_moon_rel_vel,
            }),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            }),
            EarthLightingConfigC {
                earth_radius: earth_r,
                moon_radius: moon_r,
                sun_radius: sun_r,
            },
            // Body integrates in Moon's inertial frame: TranslationalStateC
            // is moon-relative. earth_lighting_system must lift to root
            // (add MOON_OFFSET) before evaluating the kernel.
            IntegSourceC(Some(moon_entity)),
        ))
        .id();

    step_bevy_dt(&mut app, 1, DT);
    let bevy_lighting = app
        .world()
        .get::<EarthLightingStateC>(vehicle)
        .unwrap()
        .0
        .clone();

    // ── Simulation (root-integrated equivalent) ──
    //
    // The runner cannot integrate a body in a non-root frame and
    // also evaluate earth_lighting (validate.rs rejects this — see
    // module-level comment), so we instead spawn a root-integrated
    // body whose initial state is the Bevy body's *lifted*
    // root-inertial state. If the Bevy lift is correct, the kernel
    // inputs match bit-for-bit and the outputs are bit-identical.
    let lifted_pos = body_moon_rel_pos + MOON_OFFSET;
    let lifted_vel = body_moon_rel_vel; // Moon at rest → no velocity lift

    let (mut sim, earth_idx) = new_sim_earth(DT);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(SUN_POS),
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
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(MOON_OFFSET),
            None,
        ),
    );
    sim.sun_source = Some(sun_idx);
    sim.moon_source = Some(moon_idx);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: lifted_pos,
            velocity: lifted_vel,
        }),
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
    sim.step().expect("step failed");

    let sim_body = sim.body(0);
    let sim_lighting = sim_body
        .earth_lighting
        .as_ref()
        .expect("earth lighting computed");
    assert_earth_lighting_eq(
        "Bevy (non-root integ_source, lifted) vs Sim (root-integrated)",
        &bevy_lighting,
        sim_lighting,
    );
}
