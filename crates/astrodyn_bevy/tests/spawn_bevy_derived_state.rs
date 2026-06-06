//! Regression tests for `VehicleConfig::spawn_bevy`'s derived-state
//! wiring.
//!
//! Pre-fix, [`VehicleBuilder::orbital_elements`] /
//! [`VehicleBuilder::euler_angles`] / [`VehicleBuilder::lvlh`] /
//! [`VehicleBuilder::geodetic`] / [`VehicleBuilder::solar_beta`] /
//! [`VehicleBuilder::earth_lighting`] populated
//! [`VehicleConfig::derived`] (`DerivedStateConfig`), but
//! [`VehicleConfigBevyExt::spawn_bevy`] ignored that field. Mission code
//! that requested a derived state on the builder got nothing on the
//! spawned entity, with no compile-time or runtime diagnostic.
//!
//! These tests pin the post-fix contract: each `derived.*` field in
//! `VehicleConfig` is mirrored onto the spawned entity as the matching
//! `*C` (default-initialized — overwritten in
//! `AstrodynSet::DerivedState`) plus the `*ConfigC` (carrying the source
//! / Euler-sequence / radii copy from the builder) when the field is
//! set, and skipped otherwise. After one `FixedUpdate` tick the matching
//! per-step system populates the `*C` with non-default state, which the
//! tests verify by checking a representative field on each derived
//! state.
//!
//! `lvlh` and `solar_beta` carry no `*ConfigC` partner — their presence
//! on the entity alone gates the per-step system. `solar_beta` and
//! `earth_lighting` additionally require [`SunMarker`] / [`MoonMarker`]
//! entities; the tests below spawn them so the precondition validators
//! in `validate_jeod_invariants` pass.

use std::time::Duration;

use astrodyn::{
    EulerSequence, F64Ext, GravityControl, GravityGradient, JeodQuat, MassProperties, RootInertial,
    RotationalState, TranslationalStateTyped, Vec3Ext, VehicleBuilder, EARTH, MOON, SUN,
};
use astrodyn_bevy::{
    AstrodynPlugin, EarthLightingConfigC, EarthLightingStateC, EulerAnglesC, EulerAnglesConfigC,
    GeodeticConfigC, GeodeticStateC, IntegrationDtR, LvlhFrameC, MoonBundle, OrbitalElementsC,
    OrbitalElementsConfigC, PlanetBundle, SolarBetaC, SunBundle, VehicleConfigBevyExt,
};
use bevy::prelude::*;
use glam::DVec3;
use uom::si::f64::Mass;

const DT: f64 = 60.0;

fn iss_trans() -> TranslationalStateTyped<RootInertial> {
    TranslationalStateTyped::<RootInertial> {
        position: DVec3::new(6_778_137.0, 0.0, 0.0).m_at::<RootInertial>(),
        velocity: DVec3::new(0.0, 7668.56, 0.0).m_per_s_at::<RootInertial>(),
    }
}

fn iss_rot() -> RotationalState {
    let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
    q.normalize();
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    }
}

fn iss_mass() -> MassProperties {
    MassProperties::with_inertia(
        400_000.0,
        glam::DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
        DVec3::ZERO,
    )
}

/// Total ISS mass as a typed [`Mass`] for the 3-DoF builder branch.
/// `three_dof_point_mass` skips inertia tensor / CoM offset, so the
/// scalar lift is the right shape.
fn iss_mass_kg() -> Mass {
    400_000.0.kg()
}

/// Build an Earth-only Bevy app primed with `AstrodynPlugin` and a single
/// Earth `PlanetBundle`. Returns the app + the Earth entity. Callers add
/// further sources (Sun, Moon) before `spawn_bevy` when the derived state
/// under test requires them.
fn app_with_earth() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    (app, earth)
}

fn step(app: &mut App) {
    // One Startup pass to let the registration systems wire frame
    // entities; one FixedUpdate tick to let `AstrodynSet::DerivedState`
    // populate the per-step `*C` components.
    app.world_mut().run_schedule(Startup);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

#[test]
fn spawn_bevy_wires_orbital_elements() {
    let (mut app, earth) = app_with_earth();

    let cfg = VehicleBuilder::new()
        .vehicle_named("spawn-bevy-derived-state-0")
        .with_translational(iss_trans())
        .three_dof_point_mass(iss_mass_kg())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .orbital_elements(0)
        .build();

    let vehicle = {
        let mut commands = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth])
    };
    app.world_mut().flush();

    // Both components must be on the entity straight after spawn — the
    // default `*C` is the placeholder the per-step system overwrites,
    // and the `*ConfigC` carries the resolved gravity-source entity.
    let cfgc = app
        .world()
        .get::<OrbitalElementsConfigC>(vehicle)
        .expect("spawn_bevy must insert OrbitalElementsConfigC when orbital_elements is set");
    assert_eq!(
        cfgc.gravity_source, earth,
        "OrbitalElementsConfigC.gravity_source must resolve to the earth Entity"
    );
    assert!(
        app.world()
            .get::<OrbitalElementsC<astrodyn::Earth>>(vehicle)
            .is_some(),
        "spawn_bevy must insert OrbitalElementsC::<P> when orbital_elements is set"
    );

    step(&mut app);

    let oe = app
        .world()
        .get::<OrbitalElementsC<astrodyn::Earth>>(vehicle)
        .expect("OrbitalElementsC<Earth> must remain after one tick")
        .0
        .clone();
    // After one tick the orbital_elements_system populates the
    // component. r_mag for the ISS initial position is ~6.778 Mm — any
    // non-zero r_mag value confirms the per-step write fired.
    assert!(
        oe.r_mag > 6.0e6 && oe.r_mag < 7.5e6,
        "OrbitalElementsC<Earth>.r_mag = {} m — expected ~6.778e6 \
         after the per-step system populates from the ISS state",
        oe.r_mag,
    );
}

#[test]
fn spawn_bevy_wires_euler_angles() {
    let (mut app, earth) = app_with_earth();

    let cfg = VehicleBuilder::new()
        .vehicle_named("spawn-bevy-derived-state-1")
        .with_translational(iss_trans())
        .sixdof(iss_rot(), iss_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .euler_angles(EulerSequence::ZYX)
        .build();

    let vehicle = {
        let mut commands = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth])
    };
    app.world_mut().flush();

    let cfgc = app
        .world()
        .get::<EulerAnglesConfigC>(vehicle)
        .expect("spawn_bevy must insert EulerAnglesConfigC when euler_angles is set");
    assert!(
        matches!(cfgc.sequence, EulerSequence::ZYX),
        "EulerAnglesConfigC.sequence must preserve the builder selection"
    );
    assert!(
        app.world().get::<EulerAnglesC>(vehicle).is_some(),
        "spawn_bevy must insert EulerAnglesC when euler_angles is set"
    );

    step(&mut app);

    let euler = app
        .world()
        .get::<EulerAnglesC>(vehicle)
        .expect("EulerAnglesC must remain after one tick")
        .0;
    // The ISS attitude `iss_rot()` is non-trivial (mixed-sign axis +
    // off-axis ω); after one tick at least one decomposed angle must
    // be non-zero. Default would leave all three at exactly zero.
    let any_nonzero = euler.iter().any(|a| a.value.abs() > 1e-12);
    assert!(
        any_nonzero,
        "EulerAnglesC[..] = [{:?}, {:?}, {:?}] all near zero — \
         per-step system did not fire on the spawn_bevy entity",
        euler[0].value, euler[1].value, euler[2].value,
    );
}

#[test]
fn spawn_bevy_wires_lvlh() {
    let (mut app, earth) = app_with_earth();

    let cfg = VehicleBuilder::new()
        .vehicle_named("spawn-bevy-derived-state-2")
        .with_translational(iss_trans())
        .three_dof_point_mass(iss_mass_kg())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .lvlh()
        .build();

    let vehicle = {
        let mut commands = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth])
    };
    app.world_mut().flush();

    assert!(
        app.world().get::<LvlhFrameC>(vehicle).is_some(),
        "spawn_bevy must insert LvlhFrameC when lvlh() is set"
    );

    step(&mut app);

    let lvlh = app
        .world()
        .get::<LvlhFrameC>(vehicle)
        .expect("LvlhFrameC must remain after one tick")
        .0;
    // Default LVLH `position` is zero; lvlh_system fills it from the
    // body's planet-inertial position (~6.778e6 m for ISS).
    assert!(
        lvlh.position.length() > 1.0e6,
        "LvlhFrameC.position magnitude = {} m — expected ~6.778e6 \
         after the per-step system populates from ISS state",
        lvlh.position.length(),
    );
}

#[test]
fn spawn_bevy_wires_geodetic() {
    let (mut app, earth) = app_with_earth();

    let cfg = VehicleBuilder::new()
        .vehicle_named("spawn-bevy-derived-state-3")
        .with_translational(iss_trans())
        .three_dof_point_mass(iss_mass_kg())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .geodetic(0, &EARTH)
        .build();

    let vehicle = {
        let mut commands = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth])
    };
    app.world_mut().flush();

    let cfgc = app
        .world()
        .get::<GeodeticConfigC>(vehicle)
        .expect("spawn_bevy must insert GeodeticConfigC when geodetic is set");
    assert_eq!(
        cfgc.planet, earth,
        "GeodeticConfigC.planet must resolve to the earth Entity"
    );
    assert!(
        app.world().get::<GeodeticStateC>(vehicle).is_some(),
        "spawn_bevy must insert GeodeticStateC when geodetic is set"
    );

    step(&mut app);

    let geo = app
        .world()
        .get::<GeodeticStateC>(vehicle)
        .expect("GeodeticStateC must remain after one tick")
        .0;
    // ISS at (6.778e6, 0, 0) projects to ~408 km altitude. Default
    // `altitude` is 0, so a > 100 km reading confirms the per-step
    // system fired.
    assert!(
        geo.altitude > 1.0e5,
        "GeodeticStateC.altitude = {} m — expected ~4e5 after the \
         per-step system populates from ISS state",
        geo.altitude,
    );
}

#[test]
fn spawn_bevy_wires_solar_beta() {
    let (mut app, earth) = app_with_earth();
    // `solar_beta_system` and the validation precondition check both
    // require exactly one `SunMarker` entity carrying
    // `TranslationalStateC<P>`. Park the Sun on the +z axis so the
    // orbit normal (z-hat for a zero-inclination ISS state) is
    // parallel to the Sun line — the resulting beta is `π/2` rad,
    // unambiguously distinct from the default 0.
    app.world_mut().spawn(SunBundle::<astrodyn::Earth>::new(
        astrodyn::TranslationalState {
            position: DVec3::new(0.0, 0.0, 1.496e11),
            velocity: DVec3::ZERO,
        },
    ));

    let cfg = VehicleBuilder::new()
        .vehicle_named("spawn-bevy-derived-state-4")
        .with_translational(iss_trans())
        .three_dof_point_mass(iss_mass_kg())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .solar_beta()
        .build();

    let vehicle = {
        let mut commands = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth])
    };
    app.world_mut().flush();

    assert!(
        app.world().get::<SolarBetaC>(vehicle).is_some(),
        "spawn_bevy must insert SolarBetaC when solar_beta() is set"
    );

    step(&mut app);

    let beta = app
        .world()
        .get::<SolarBetaC>(vehicle)
        .expect("SolarBetaC must remain after one tick")
        .0;
    // Default beta is 0.0 (radians). For the ISS-on-x-axis state with
    // the Sun at +x, the beta angle is non-zero (the body's velocity
    // axis lies at 90° to the Sun line). Anything not exactly the
    // default proves the per-step system populated it.
    assert!(
        beta.is_finite() && beta.abs() > 1e-6,
        "SolarBetaC = {} rad — expected non-zero after the per-step \
         system populates from the ISS-vs-Sun geometry",
        beta,
    );
}

#[test]
fn spawn_bevy_wires_earth_lighting() {
    let (mut app, earth) = app_with_earth();
    app.world_mut().spawn(SunBundle::<astrodyn::Earth>::new(
        astrodyn::TranslationalState {
            position: DVec3::new(1.496e11, 0.0, 0.0),
            velocity: DVec3::ZERO,
        },
    ));
    app.world_mut().spawn(MoonBundle::<astrodyn::Earth>::new(
        astrodyn::TranslationalState {
            position: DVec3::new(0.0, 3.844e8, 0.0),
            velocity: DVec3::ZERO,
        },
    ));

    let cfg = VehicleBuilder::new()
        .vehicle_named("spawn-bevy-derived-state-5")
        .with_translational(iss_trans())
        .three_dof_point_mass(iss_mass_kg())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .earth_lighting(&EARTH, &MOON, &SUN)
        .build();

    let vehicle = {
        let mut commands = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth])
    };
    app.world_mut().flush();

    let cfgc = app
        .world()
        .get::<EarthLightingConfigC>(vehicle)
        .expect("spawn_bevy must insert EarthLightingConfigC when earth_lighting is set");
    // Builder copies r_eq off each `PlanetConfig`. The presets aren't
    // re-exported as plain f64s, so just confirm the value is finite,
    // positive, and on the right order of magnitude (Earth ~6.378e6 m,
    // Sun ~6.96e8 m, Moon ~1.737e6 m).
    assert!(
        cfgc.earth_radius > 6.0e6 && cfgc.earth_radius < 7.0e6,
        "EarthLightingConfigC.earth_radius = {} m — expected ~6.378e6 \
         (the EARTH preset's equatorial radius)",
        cfgc.earth_radius,
    );
    assert!(
        cfgc.moon_radius > 1.0e6 && cfgc.moon_radius < 2.5e6,
        "EarthLightingConfigC.moon_radius = {} m — expected ~1.737e6",
        cfgc.moon_radius,
    );
    assert!(
        cfgc.sun_radius > 5.0e8 && cfgc.sun_radius < 1.0e9,
        "EarthLightingConfigC.sun_radius = {} m — expected ~6.96e8",
        cfgc.sun_radius,
    );
    assert!(
        app.world().get::<EarthLightingStateC>(vehicle).is_some(),
        "spawn_bevy must insert EarthLightingStateC when earth_lighting is set"
    );

    step(&mut app);

    let lighting = app
        .world()
        .get::<EarthLightingStateC>(vehicle)
        .expect("EarthLightingStateC must remain after one tick")
        .0
        .clone();
    // Default Sun-body distance is 0.0; the per-step system writes the
    // Sun-relative geometry. ISS-to-Sun (~1.496e11 m) is the
    // unmistakeable non-default signal.
    assert!(
        lighting.sun_body.distance > 1.0e10,
        "EarthLightingStateC.sun_body.distance = {} m — expected ~1.5e11 \
         after the per-step system populates",
        lighting.sun_body.distance,
    );
}
