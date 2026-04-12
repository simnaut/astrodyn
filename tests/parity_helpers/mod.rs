//! Shared helpers for Bevy-vs-Simulation parity tests.
//!
//! Provides common initial conditions, assertion functions, and setup utilities
//! used across all parity test categories.

// Each integration test file includes this module independently, so not all
// items are used in every compilation unit. Suppress dead_code warnings.
#![allow(dead_code)]

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    GravitySourceC, JeodPlugin, RotationalStateC, SourceInertialPositionC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel,
    GravitySource, GravitySourceEntry, MassProperties, RotationModel, RotationalState, SimBody,
    Simulation, SixDofState, TranslationalState,
};

pub const MU_EARTH: f64 = 3.986_004_415e14;
pub const MU_SUN: f64 = 1.327_124_40e20;
pub const DT: f64 = 10.0;
pub const NUM_STEPS: usize = 100;

// ── Shared initial conditions ──

pub fn iss_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

pub fn tumble_rot() -> RotationalState {
    RotationalState {
        quaternion: jeod_sim::JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5),
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    }
}

pub fn iss_mass() -> MassProperties {
    MassProperties::with_inertia(
        400_000.0,
        DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
        DVec3::ZERO,
    )
}

pub fn earth_source() -> GravitySource {
    GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    }
}

// ── Bevy helpers ──

pub fn new_bevy_app(dt: f64) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(dt));
    app.add_plugins(JeodPlugin);
    app
}

pub fn step_bevy(app: &mut App, n: usize) {
    step_bevy_dt(app, n, DT);
}

pub fn step_bevy_dt(app: &mut App, n: usize, dt: f64) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

pub fn read_sixdof(world: &World, entity: Entity) -> SixDofState {
    SixDofState {
        trans: world.get::<TranslationalStateC>(entity).unwrap().0,
        rot: world.get::<RotationalStateC>(entity).unwrap().0,
    }
}

pub fn read_trans(world: &World, entity: Entity) -> TranslationalState {
    world.get::<TranslationalStateC>(entity).unwrap().0
}

/// Assert two f64 values are bit-identical.
pub fn assert_bits_eq(label: &str, component: &str, a: f64, b: f64) {
    assert!(
        a.to_bits() == b.to_bits(),
        "{label} {component} not bit-identical:\n  \
         A: {a} (bits={:#018x})\n  \
         B: {b} (bits={:#018x})",
        a.to_bits(),
        b.to_bits(),
    );
}

pub fn assert_sixdof_eq(label: &str, a: &SixDofState, b: &SixDofState) {
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

pub fn assert_trans_eq(label: &str, a: &TranslationalState, b: &TranslationalState) {
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

pub fn new_sim_earth(dt: f64) -> (Simulation, usize) {
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: earth_source(),
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    (sim, earth_idx)
}

pub fn spawn_earth_source(app: &mut App) -> Entity {
    app.world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id()
}

pub fn assert_geodetic_eq(label: &str, a: &jeod_sim::GeodeticState, b: &jeod_sim::GeodeticState) {
    assert_bits_eq(label, "latitude", a.latitude, b.latitude);
    assert_bits_eq(label, "longitude", a.longitude, b.longitude);
    assert_bits_eq(label, "altitude", a.altitude, b.altitude);
    println!("  {label}: bit-identical (lat, lon, alt)");
}

pub fn new_sim_body_sixdof(earth_idx: usize, gradient: bool) -> SimBody {
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

// ── Derived state assertion helpers ──

pub fn assert_orbital_elements_eq(
    label: &str,
    a: &jeod_sim::OrbitalElements,
    b: &jeod_sim::OrbitalElements,
) {
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

pub fn assert_lvlh_eq(label: &str, a: &jeod_sim::LvlhFrame, b: &jeod_sim::LvlhFrame) {
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

// ── Lighting assertion helpers ──

pub fn assert_lighting_body_eq(
    label: &str,
    prefix: &str,
    a: &jeod_sim::LightingBody,
    b: &jeod_sim::LightingBody,
) {
    assert_bits_eq(label, &format!("{prefix}.radius"), a.radius, b.radius);
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("{prefix}.position[{i}]"),
            a.position[i],
            b.position[i],
        );
    }
    assert_bits_eq(label, &format!("{prefix}.distance"), a.distance, b.distance);
    assert_bits_eq(
        label,
        &format!("{prefix}.half_angle"),
        a.half_angle,
        b.half_angle,
    );
}

pub fn assert_lighting_params_eq(
    label: &str,
    prefix: &str,
    a: &jeod_sim::LightingParams,
    b: &jeod_sim::LightingParams,
) {
    assert_bits_eq(
        label,
        &format!("{prefix}.obs_angle"),
        a.obs_angle,
        b.obs_angle,
    );
    assert_bits_eq(label, &format!("{prefix}.phase"), a.phase, b.phase);
    assert_bits_eq(
        label,
        &format!("{prefix}.occlusion"),
        a.occlusion,
        b.occlusion,
    );
    assert_bits_eq(label, &format!("{prefix}.visible"), a.visible, b.visible);
    assert_bits_eq(label, &format!("{prefix}.lighting"), a.lighting, b.lighting);
}

pub fn assert_earth_lighting_eq(
    label: &str,
    a: &jeod_sim::EarthLightingState,
    b: &jeod_sim::EarthLightingState,
) {
    // Body geometry (position, distance, half-angle)
    assert_lighting_body_eq(label, "sun_body", &a.sun_body, &b.sun_body);
    assert_lighting_body_eq(label, "earth_body", &a.earth_body, &b.earth_body);
    assert_lighting_body_eq(label, "moon_body", &a.moon_body, &b.moon_body);
    // Eclipse/visibility parameters
    assert_lighting_params_eq(label, "sun_earth", &a.sun_earth, &b.sun_earth);
    assert_lighting_params_eq(label, "moon_earth", &a.moon_earth, &b.moon_earth);
    assert_lighting_params_eq(label, "earth_albedo", &a.earth_albedo, &b.earth_albedo);
    println!("  {label}: bit-identical (all earth lighting fields)");
}

// ── DE421 helpers ──

pub fn bsp_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/de421.bsp")
}

pub const J2000_JD: f64 = 2_451_545.0;

/// Cached DE421 initial positions at J2000 to avoid redundant BSP loads.
static SUN_POS_J2000: std::sync::OnceLock<DVec3> = std::sync::OnceLock::new();
static MOON_POS_J2000: std::sync::OnceLock<DVec3> = std::sync::OnceLock::new();

/// Helper: load initial Sun position from DE421 at J2000 (cached).
pub fn sun_initial_pos() -> DVec3 {
    *SUN_POS_J2000.get_or_init(|| {
        let eph = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
        let (pos, _vel) = eph
            .get_earth_centered_state(EphemerisBody::Sun, J2000_JD)
            .expect("Sun state at J2000");
        pos
    })
}

/// Helper: load initial Moon position from DE421 at J2000 (cached).
pub fn moon_initial_pos() -> DVec3 {
    *MOON_POS_J2000.get_or_init(|| {
        let eph = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
        let (pos, _vel) = eph
            .get_earth_centered_state(EphemerisBody::Moon, J2000_JD)
            .expect("Moon state at J2000");
        pos
    })
}
