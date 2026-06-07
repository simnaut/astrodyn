// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Shared helpers for Bevy-vs-Simulation parity tests.
//!
//! Provides common initial conditions, assertion functions, and setup
//! utilities used across all `bevy_parity_*.rs` test categories. Consumers
//! `mod common;` then `use common::*;`.
//!
//! Renamed from `tests/parity_helpers/mod.rs` to `tests/common/mod.rs` in
//! Phase 11 of #101 — the contents are pure test infrastructure (Bevy
//! `App` setup, bit-identical assertion macros, DE421 ephemeris caching);
//! they are deliberately not promoted to `astrodyn::recipes::helpers`,
//! which is reserved for cross-crate-reusable propagation utilities.

// Each integration test file includes this module independently, so not all
// items are used in every compilation unit. Suppress dead_code warnings.
#![allow(dead_code)]

use std::time::Duration;

use astrodyn::{
    AngularVelocity, BodyAttitude, BodyFrame, Ephemeris, EphemerisBody, GravityControl,
    GravityControls, GravityGradient, GravityModel, GravitySource, InertiaTensor,
    MassPropertiesTyped, Position, RootInertial, RotationalStateTyped, SelfRef, SimulationTime,
    SixDofState, StructuralFrame, TranslationalState, TranslationalStateTyped, Velocity,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_bevy::{
    AstrodynPlugin, GravitySourceC, IntegrationDtR, RotationalStateC, SimulationTimeR,
    SourceInertialPositionC, TranslationalStateC,
};
use astrodyn_runner::Simulation;
use bevy::prelude::*;
use glam::{DMat3, DVec3};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

/// Earth gravitational parameter (m^3/s^2) — JEOD `earth_GGM05C.cc` via presets.
pub const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;
/// Sun gravitational parameter (m^3/s^2) — JEOD `sun_spherical.cc` via presets.
pub const MU_SUN: f64 = astrodyn::SUN.shape.mu;
pub const DT: f64 = 10.0;
pub const NUM_STEPS: usize = 100;

// ── Shared initial conditions ──

pub fn iss_trans() -> TranslationalStateTyped<RootInertial> {
    TranslationalStateTyped::<RootInertial> {
        position: Position::<RootInertial>::from_raw_si(DVec3::new(6_778_137.0, 0.0, 0.0)),
        velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 7668.56, 0.0)),
    }
}

pub fn tumble_rot() -> RotationalStateTyped<SelfRef> {
    // Deliberately non-trivial tumble (axis ≠ basis, ω with mixed
    // signs) so attitude propagation exercises off-diagonal RNP terms.
    // The quaternion is normalized at construction because
    // `RotationalStateC` (and its `BodyAttitude` witness) require a
    // unit-norm quaternion at the ECS surface. The integrator's own
    // renormalize-after-step path is still exercised by the
    // `rotational::tests::*` in astrodyn_dynamics.
    let mut q = astrodyn::JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
    q.normalize();
    RotationalStateTyped::<SelfRef>::new(
        BodyAttitude::<SelfRef>::from_jeod_quat(q),
        AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(DVec3::new(0.001, -0.0005, 0.001)),
    )
}

pub fn iss_mass() -> MassPropertiesTyped<SelfRef> {
    MassPropertiesTyped::<SelfRef>::with_inertia(
        Mass::new::<kilogram>(400_000.0),
        InertiaTensor::<BodyFrame<SelfRef>>::from_dmat3_unchecked(DMat3::from_diagonal(
            DVec3::new(1.02e8, 0.91e8, 1.64e8),
        )),
        Position::<StructuralFrame<SelfRef>>::zero(),
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
    // `Time<Fixed>` and `IntegrationDtR` are inserted in lockstep:
    // `Time<Fixed>` drives `FixedUpdate` cadence; `IntegrationDtR`
    // carries the bit-exact f64 the pipeline systems consume as
    // physics `dt` (see `astrodyn_bevy::IntegrationDtR` doc).
    app.insert_resource(Time::<Fixed>::from_seconds(dt));
    app.insert_resource(IntegrationDtR(dt));
    app.add_plugins(AstrodynPlugin);
    app
}

pub fn step_bevy(app: &mut App, n: usize) {
    step_bevy_dt(app, n, DT);
}

pub fn step_bevy_dt(app: &mut App, n: usize, dt: f64) {
    // Keep `IntegrationDtR` in sync with the caller's `dt` so a test
    // that varies `dt` between calls drives the pipeline with the
    // exact f64 it passed in (mirrors `AstrodynAppExt::step_fixed_dt`).
    app.insert_resource(IntegrationDtR(dt));
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

pub fn read_sixdof(world: &World, entity: Entity) -> SixDofState {
    SixDofState {
        trans: astrodyn::typed_bridge::trans_typed_to_raw(
            &world
                .get::<TranslationalStateC<astrodyn::Earth>>(entity)
                .unwrap()
                .0,
        ),
        rot: astrodyn::typed_bridge::rot_typed_to_raw(
            &world.get::<RotationalStateC>(entity).unwrap().0,
        ),
    }
}

pub fn read_trans(world: &World, entity: Entity) -> TranslationalState {
    astrodyn::typed_bridge::trans_typed_to_raw(
        &world
            .get::<TranslationalStateC<astrodyn::Earth>>(entity)
            .unwrap()
            .0,
    )
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
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    (sim, earth_idx)
}

pub fn spawn_earth_source(app: &mut App) -> Entity {
    app.world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::FrameUid::of::<
                astrodyn::PlanetInertial<astrodyn::Earth>,
            >()),
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id()
}

pub fn assert_geodetic_eq(label: &str, a: &astrodyn::GeodeticState, b: &astrodyn::GeodeticState) {
    assert_bits_eq(label, "latitude", a.latitude, b.latitude);
    assert_bits_eq(label, "longitude", a.longitude, b.longitude);
    assert_bits_eq(label, "altitude", a.altitude, b.altitude);
    println!("  {label}: bit-identical (lat, lon, alt)");
}

pub fn new_sim_body_sixdof(earth_idx: usize, gradient: bool) -> VehicleConfig {
    let gradient_mode = if gradient {
        GravityGradient::Compute
    } else {
        GravityGradient::Skip
    };
    VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, gradient_mode)],
        },
        compute_gravity_gradient: gradient,
        ..VehicleConfig::named("mod-0")
    }
}

// ── Derived state assertion helpers ──

pub fn assert_orbital_elements_eq<P: astrodyn::Planet>(
    label: &str,
    a: &astrodyn::OrbitalElements<P>,
    b: &astrodyn::OrbitalElements<P>,
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

pub fn assert_lvlh_eq(label: &str, a: &astrodyn::LvlhFrame, b: &astrodyn::LvlhFrame) {
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
    a: &astrodyn::LightingBody,
    b: &astrodyn::LightingBody,
) {
    assert_bits_eq(label, &format!("{prefix}.radius"), a.radius, b.radius);
    let a_pos = a.position.raw_si();
    let b_pos = b.position.raw_si();
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("{prefix}.position[{i}]"),
            a_pos[i],
            b_pos[i],
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
    a: &astrodyn::LightingParams,
    b: &astrodyn::LightingParams,
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
    a: &astrodyn::EarthLightingState,
    b: &astrodyn::EarthLightingState,
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
    astrodyn::ephemeris_assets::de421_path()
}

pub const J2000_JD: f64 = 2_451_545.0;

/// Cached DE421 ephemeris to avoid redundant BSP loads.
static DE421_EPHEMERIS: std::sync::OnceLock<Ephemeris> = std::sync::OnceLock::new();

fn de421_ephemeris() -> &'static Ephemeris {
    DE421_EPHEMERIS.get_or_init(|| Ephemeris::from_bsp(&bsp_path()).expect("load DE421"))
}

/// Helper: load initial Sun position from DE421 at J2000 (cached ephemeris).
pub fn sun_initial_pos() -> DVec3 {
    let (pos, _vel) = de421_ephemeris()
        .get_earth_centered_state_typed(EphemerisBody::Sun, J2000_JD)
        .expect("Sun state at J2000");
    pos.raw_si()
}

/// Helper: load initial Moon position from DE421 at J2000 (cached ephemeris).
pub fn moon_initial_pos() -> DVec3 {
    let (pos, _vel) = de421_ephemeris()
        .get_earth_centered_state_typed(EphemerisBody::Moon, J2000_JD)
        .expect("Moon state at J2000");
    pos.raw_si()
}

// ── Time-pipeline parity helpers ─────────────────────────────────────────
//
// Shared between `bevy_parity_timescale.rs` and `bevy_parity_time_docker.rs`
// (and any future body-less time-pipeline parity wrappers). The two helpers
// snapshot the Bevy-side `SimulationTimeR` and assert bit-identity against
// the runner-side `Simulation.time` across every `SimulationTime` field
// the Bevy pipeline carries — including the optional MET / UDE scales and
// the EOP-table presence flag. After the #577 unification both runtimes
// drive the same `astrodyn::SimulationTime`, so the parity surface covers
// everything the production resource carries.

/// Snapshot the Bevy app's `SimulationTimeR` resource into a fresh
/// `SimulationTime` clone. Cloning avoids holding a long-lived `Res`
/// across the next mutable world access in a parity loop body.
pub fn bevy_sim_time(app: &App) -> SimulationTime {
    app.world().resource::<SimulationTimeR>().0.clone()
}

/// Assert every load-bearing `SimulationTime` field matches bit-for-bit
/// between the runner-side `Simulation.time` and the Bevy-side
/// `SimulationTimeR.0`. `gmst_radians` follows `gmst_seconds` through
/// `recompute_derived` so both are checked independently;
/// `leap_second_table` is `Copy`-by-value and seeded from the same
/// `default_leap_second_table()` on both runtimes, so only its derived
/// scalars (`tai_seconds`, `utc_seconds`, …) need per-tick assertion.
///
/// `t` and `label` flow into the panic message so a failure pinpoints
/// the tick and SIM-case (e.g. `"SIM_4_common tick 5 t=300.000s"`).
pub fn assert_simulation_time_bits_eq(
    t: f64,
    label: &str,
    runner: &SimulationTime,
    bevy: &SimulationTime,
) {
    fn bits_eq(t: f64, label: &str, field: &str, r: f64, b: f64) {
        assert!(
            r.to_bits() == b.to_bits(),
            "{label} at t={t:.6}s diverged on {field}:\n  \
             runner: {r} (bits={:#018x})\n  \
             bevy:   {b} (bits={:#018x})",
            r.to_bits(),
            b.to_bits(),
        );
    }
    bits_eq(
        t,
        label,
        "tai_seconds",
        runner.tai_seconds,
        bevy.tai_seconds,
    );
    bits_eq(t, label, "tai_tjt", runner.tai_tjt, bevy.tai_tjt);
    bits_eq(
        t,
        label,
        "tai_tjt_at_epoch",
        runner.tai_tjt_at_epoch,
        bevy.tai_tjt_at_epoch,
    );
    bits_eq(
        t,
        label,
        "utc_seconds",
        runner.utc_seconds,
        bevy.utc_seconds,
    );
    bits_eq(
        t,
        label,
        "ut1_seconds",
        runner.ut1_seconds,
        bevy.ut1_seconds,
    );
    bits_eq(t, label, "tt_seconds", runner.tt_seconds, bevy.tt_seconds);
    bits_eq(
        t,
        label,
        "tdb_seconds",
        runner.tdb_seconds,
        bevy.tdb_seconds,
    );
    bits_eq(
        t,
        label,
        "gmst_seconds",
        runner.gmst_seconds,
        bevy.gmst_seconds,
    );
    bits_eq(
        t,
        label,
        "gmst_radians",
        runner.gmst_radians,
        bevy.gmst_radians,
    );
    bits_eq(
        t,
        label,
        "gps_seconds",
        runner.gps_seconds,
        bevy.gps_seconds,
    );
    bits_eq(t, label, "simtime", runner.simtime, bevy.simtime);
    bits_eq(
        t,
        label,
        "ut1_tai_offset",
        runner.ut1_tai_offset,
        bevy.ut1_tai_offset,
    );
    bits_eq(
        t,
        label,
        "scale_factor",
        runner.scale_factor(),
        bevy.scale_factor(),
    );

    // MET: presence + seconds parity. Differing presence between the two
    // runtimes is a setup bug — surface it directly rather than letting
    // a `None`-vs-`Some` slip through with a 0.0 default.
    assert_eq!(
        runner.met.is_some(),
        bevy.met.is_some(),
        "{label} at t={t:.6}s diverged on met presence: runner={:?} bevy={:?}",
        runner.met.is_some(),
        bevy.met.is_some(),
    );
    if let (Some(rmet), Some(bmet)) = (runner.met.as_ref(), bevy.met.as_ref()) {
        bits_eq(t, label, "met.seconds", rmet.seconds, bmet.seconds);
    }

    // UDE: vec length + per-slot seconds parity. Same setup-bug rationale.
    assert_eq!(
        runner.ude.len(),
        bevy.ude.len(),
        "{label} at t={t:.6}s diverged on ude.len(): runner={} bevy={}",
        runner.ude.len(),
        bevy.ude.len(),
    );
    for (i, (r, b)) in runner.ude.iter().zip(bevy.ude.iter()).enumerate() {
        bits_eq(t, label, &format!("ude[{i}].seconds"), r.seconds, b.seconds);
    }

    // EOP-table presence (only the flag — the table itself is large and
    // both runtimes share the same `default_eop_table()` by construction
    // when EOP is in use, so an `is_some()` mismatch is the setup signal).
    assert_eq!(
        runner.has_eop_table(),
        bevy.has_eop_table(),
        "{label} at t={t:.6}s diverged on has_eop_table(): runner={} bevy={}",
        runner.has_eop_table(),
        bevy.has_eop_table(),
    );
}
