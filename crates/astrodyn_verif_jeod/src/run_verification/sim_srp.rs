// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! `VerificationCase` constructors for the SIM_3_ORBIT solar-radiation
//! pressure verification family.
//!
//! Two variants share the 6-flat-plate vehicle and conical Earth
//! shadow model, differing only in the JEOD reference sim and the
//! consequent thermal integrator order:
//!
//! - [`srp_orbit_trajectory`] (default `SIM_3_ORBIT`): JEOD updates the
//!   Sun every integration step (1 s). The recipe wires the
//!   simulation's auto-ephemeris path
//!   (`SimulationBuilder::ephemeris` + `set_source_ephemeris`) so the
//!   Sun source is refreshed from DE421 before every internal step,
//!   matching JEOD's update frequency without needing a per-step hook.
//! - [`srp_1st_order_trajectory`] (`SIM_3_ORBIT_1st_ORDER`): JEOD
//!   updates Sun at record boundaries; uses the per-record `pre_step`
//!   factory to push the Sun position forward by one record before
//!   each `step_until`.

use astrodyn::Vec3Ext;
use std::path::PathBuf;

use crate::verification::{
    CsvReference, InitialConditions, PreStepClosure, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, AtmosphereConfig, AtmosphereModel, DragConfig, Ephemeris,
    EphemerisBody, ExponentialAtmosphere, FlatPlate, FlatPlateParams, FlatPlateState,
    FlatPlateThermal, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, JeodQuat, MassProperties, RotationModel, RotationalState, ShadowBody,
    SimulationBuilder, SimulationTime, SrpModel, ThermalIntegrationOrder, TranslationalState,
    VehicleConfig, EARTH,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

const SIM_3_ORBIT: &str = "models/interactions/radiation_pressure/verif/SIM_3_ORBIT";
const SIM_3_ORBIT_1ST: &str = "models/interactions/radiation_pressure/verif/SIM_3_ORBIT_1st_ORDER";

const SRP_MASS: f64 = 300.0;
const INITIAL_PLATE_TEMP_K: f64 = 270.0;

/// Sun source index inside the SRP scenario. The builder
/// `debug_assert!`s this against the actual returned index so a future
/// reorder can't silently desync `srp_pre_step` from the registry.
const SRP_SUN_IDX: usize = 1;

fn bsp_path() -> PathBuf {
    let p = astrodyn::ephemeris_assets::de421_path();
    assert!(p.exists(), "DE421 ephemeris not found at {}", p.display());
    p
}

/// 6 flat plates arranged as a 4 m × 4 m × 15 m box (the JEOD
/// SIM_3_ORBIT vehicle).
fn srp_plates() -> Vec<(
    FlatPlate<astrodyn::SelfRef>,
    FlatPlateParams,
    FlatPlateThermal,
)> {
    let params = FlatPlateParams {
        albedo: 0.5,
        diffuse: 0.5,
    };
    let thermal = FlatPlateThermal {
        emissivity: 0.5,
        heat_capacity_per_area: 50.0,
        thermal_power_dump: 0.0,
    };
    vec![
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::X,
                position: DVec3::new(2.0, 0.0, 0.0)
                    .m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::Y,
                position: DVec3::new(0.0, -2.0, 0.0)
                    .m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::X,
                position: DVec3::new(-2.0, 0.0, 0.0)
                    .m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::Y,
                position: DVec3::new(0.0, 2.0, 0.0)
                    .m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: DVec3::Z,
                position: DVec3::new(0.0, 0.0, 7.5)
                    .m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: -DVec3::Z,
                position: DVec3::new(0.0, 0.0, -7.5)
                    .m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
            },
            params,
            thermal,
        ),
    ]
}

/// Translation `tdb_jd = (tai_tjt_at_epoch + sim_days) + 40000 + 2_400_000.5`.
/// Used at scenario construction (sim_time = 0) and inside the per-step
/// pre-step hook so the Sun source moves the same way the bespoke test
/// did.
fn srp_sun_position(sim_time_s: f64, epoch_tai_tjt: f64, ephemeris: &Ephemeris) -> DVec3 {
    let sim_days = sim_time_s / 86_400.0;
    let tdb_jd = (epoch_tai_tjt + sim_days) + 40_000.0 + 2_400_000.5;
    let (sun_pos, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, tdb_jd)
        .expect("Sun position query");
    sun_pos.raw_si()
}

fn earth_point_mass(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
    }
}

fn sun_zero_mu(initial_pos: DVec3) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            // mu = 0: Sun is referenced only for SRP direction, never
            // for gravitational perturbation.
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: initial_pos.m_at::<astrodyn::RootInertial>(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
    }
}

fn srp_time(modified_data_dir: &str) -> SimulationTime {
    let cfg = crate::time_config::load_time_config(
        &crate::jeod_inputs::path(modified_data_dir).join("Modified_data/date_and_time.py"),
    );
    SimulationTime::new(cfg.tai_tjt(), default_leap_second_table())
}

/// Selects how the Sun source position is refreshed during stepping.
#[derive(Clone, Copy)]
enum SunUpdate {
    /// Per-record `pre_step` hook (matches `SIM_3_ORBIT_1st_ORDER`).
    PreStepHook,
    /// Per-internal-step auto-ephemeris via
    /// `SimulationBuilder::ephemeris` plus `set_source_ephemeris`.
    /// Matches `SIM_3_ORBIT`'s 1 s Sun refresh.
    AutoEphemeris,
}

fn build_srp(
    init: &InitialConditions,
    sim_subdir: &'static str,
    integration_order: ThermalIntegrationOrder,
    sun_update: SunUpdate,
) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(sim_subdir);

    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;

    let time = srp_time(sim_subdir);
    let epoch_tai_tjt = time.tai_tjt_at_epoch;
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let initial_sun = srp_sun_position(0.0, epoch_tai_tjt, &ephemeris);

    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", earth_point_mass(earth_mu));
    let sun = sb.add_source("Sun", sun_zero_mu(initial_sun));
    debug_assert_eq!(
        sun, SRP_SUN_IDX,
        "Sun source index drift: srp_pre_step assumes Sun is at SRP_SUN_IDX={SRP_SUN_IDX}, \
         but add_source returned {sun}."
    );
    sb = sb.sun(sun);

    if let SunUpdate::AutoEphemeris = sun_update {
        // Attach DE421 + register the Sun source to it. The simulation
        // will refresh the Sun position from DE421 every internal step,
        // matching JEOD's per-step Sun update without needing a hook.
        sb.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Earth);
        sb = sb.ephemeris(ephemeris);
    }

    let plates = srp_plates();
    let num_plates = plates.len();

    sb.add_body(VehicleConfig {
        trans: astrodyn::TranslationalStateTyped::<astrodyn::RootInertial>::from_untyped_unchecked(
            &TranslationalState {
                position: init.position,
                velocity: init.velocity,
            },
        ),
        mass: Some(
            MassProperties::with_inertia(
                SRP_MASS,
                DMat3::from_diagonal(DVec3::splat(1.0)),
                DVec3::ZERO,
            )
            .into(),
        ),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        srp: Some(SrpModel::FlatPlate(FlatPlateState {
            plates,
            temperatures: vec![INITIAL_PLATE_TEMP_K; num_plates],
            t_pow4_cached: vec![INITIAL_PLATE_TEMP_K.powi(4); num_plates],
            integration_order,
            ..Default::default()
        })),
        shadow_body: Some(ShadowBody {
            source_idx: earth,
            radius: EARTH.shadow_radius,
        }),
        ..Default::default()
    });
    sb
}

fn build_srp_1st_order(init: &InitialConditions) -> SimulationBuilder {
    build_srp(
        init,
        SIM_3_ORBIT_1ST,
        ThermalIntegrationOrder::DerivativeFirstOrder,
        SunUpdate::PreStepHook,
    )
}

fn build_srp_orbit(init: &InitialConditions) -> SimulationBuilder {
    build_srp(
        init,
        SIM_3_ORBIT,
        ThermalIntegrationOrder::default(),
        SunUpdate::AutoEphemeris,
    )
}

/// Pre-step factory: capture DE421 + the epoch-anchored TAI TJT once,
/// then update the Sun source position before each `step_until` call so
/// the SRP direction matches the upcoming step's Sun geometry.
fn srp_pre_step_for(
    modified_data_dir: &'static str,
) -> impl Fn(&InitialConditions) -> PreStepClosure {
    move |_init| {
        let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
        let epoch_tai_tjt = srp_time(modified_data_dir).tai_tjt_at_epoch;
        Box::new(move |sim, time_s: f64| {
            let sun_pos = srp_sun_position(time_s, epoch_tai_tjt, &ephemeris);
            sim.set_source_position(SRP_SUN_IDX, sun_pos);
        })
    }
}

fn srp_1st_order_pre_step(init: &InitialConditions) -> PreStepClosure {
    srp_pre_step_for(SIM_3_ORBIT_1ST)(init)
}

/// SIM_3_ORBIT_1st_ORDER — same SRP physics with first-order thermal
/// integration order (the JEOD reference uses ER7_Utils first-order on
/// plate temperature as a derivative-class job).
pub fn srp_1st_order_trajectory() -> VerificationCase {
    VerificationCase {
        name: "tier3_srp_1st_order_trajectory",
        scenario: build_srp_1st_order,
        reference: CsvReference::Srp("srp_1st_order_radiation_srp_orbit.csv"),
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            position_m: [7.709e1, 8.021e1, 3.481e1],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(srp_1st_order_pre_step),
    }
}

/// SIM_3_ORBIT — flat-plate SRP + conical Earth shadow, GEO orbit,
/// ~23 days. Sun position auto-refreshed from DE421 every internal
/// step (matches JEOD's 1 s Sun cadence) via the simulation's
/// auto-ephemeris path; no `pre_step` hook needed.
pub fn srp_orbit_trajectory() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_srp_flat_plate",
        scenario: build_srp_orbit,
        reference: CsvReference::Srp("srp_orbit_radiation_srp_orbit.csv"),
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            // Inherited verbatim from the bespoke assertion. Auto-ephemeris
            // queries DE421 directly each step (vs the bespoke 100 s
            // interpolation table); errors come in at or below the
            // bespoke baseline.
            position_m: [0.034, 0.040, 0.016],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── Bevy ↔ runner parity recipes (no JEOD CSV) ─────────────────────────────
//
// These nine recipes migrate the hand-rolled `bevy_parity_srp.rs` test
// suite into the `VerificationCase` shape so the parity trait can drive
// both runtimes from a single factory. Each scenario uses a synthetic
// Sun at `(1.496e11, 0, 0)` (no DE421, no `pre_step`) and a point-mass
// Earth; the parity assertion is bit-identity per checkpoint.
//
// Cadence: dt=1.0 s sampling against `srp_basic_srp_basic.csv` (200
// records at 1 s) via `CsvReference::TimesOnly`. The CSV's body is
// never read — the parity trait only needs the time column for
// scheduling. The resulting 200 s × 1 s schedule strictly dominates the
// pre-migration 100 × 10 s schedule (more checkpoints, same physics).

const PARITY_SUN_POS: DVec3 = DVec3::new(1.496e11, 0.0, 0.0);
const PARITY_DT: f64 = 1.0;
const PARITY_TIMES_CSV: &str = "srp_basic_srp_basic.csv";

fn parity_iss_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

fn parity_tumble_rot() -> RotationalState {
    let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
    q.normalize();
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    }
}

fn parity_iss_mass() -> MassProperties {
    MassProperties::with_inertia(
        400_000.0,
        DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
        DVec3::ZERO,
    )
}

fn parity_earth_source(mu: f64, central: bool) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central,
    }
}

fn parity_sun_source() -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            // mu=0: Sun is referenced only for SRP / shadow direction.
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(PARITY_SUN_POS),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
    }
}

/// Single-plate convenience used by the basic / shadow / derivative
/// flavors. Mirrors `bevy_parity_srp.rs::make_single_plate`.
fn parity_single_plate(
    albedo: f64,
    diffuse: f64,
    emissivity: f64,
) -> Vec<(
    FlatPlate<astrodyn::SelfRef>,
    FlatPlateParams,
    FlatPlateThermal,
)> {
    use astrodyn::Vec3Ext;
    vec![(
        FlatPlate {
            area: 100.0,
            normal: DVec3::X,
            position: DVec3::ZERO.m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
        },
        FlatPlateParams { albedo, diffuse },
        FlatPlateThermal {
            emissivity,
            heat_capacity_per_area: 50.0,
            thermal_power_dump: 0.0,
        },
    )]
}

fn parity_flat_plate_state(
    plates: Vec<(
        FlatPlate<astrodyn::SelfRef>,
        FlatPlateParams,
        FlatPlateThermal,
    )>,
    order: ThermalIntegrationOrder,
) -> FlatPlateState<astrodyn::SelfRef> {
    let n = plates.len();
    FlatPlateState {
        plates,
        temperatures: vec![INITIAL_PLATE_TEMP_K; n],
        t_pow4_cached: vec![INITIAL_PLATE_TEMP_K.powi(4); n],
        integration_order: order,
        ..Default::default()
    }
}

/// Earth + synthetic Sun source skeleton shared by every parity recipe.
/// Returns `(builder, earth_idx)`; the caller then registers its body
/// and (optionally) adds atmosphere.
fn parity_skeleton() -> (SimulationBuilder, usize) {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, PARITY_DT);
    let earth = sb.add_source("Earth", parity_earth_source(EARTH.shape.mu, true));
    let sun = sb.add_source("Sun", parity_sun_source());
    sb = sb.sun(sun);
    (sb, earth)
}

fn parity_zero_tols() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

// ── Scenario E: full stack — drag + 1 SRP plate + gravity-torque, 6-DOF ──

fn build_parity_full_stack_sixdof(_init: &InitialConditions) -> SimulationBuilder {
    use astrodyn::Vec3Ext;
    // The full-stack scenario adds atmosphere, and the Bevy
    // `atmosphere_update_system` requires the planet source to carry a
    // `PlanetFixedRotationC` (or fall back to spherical via
    // `planet_entity: None`). The shared `parity_skeleton` adds Earth
    // with `RotationModel::None` and `t_inertial_pfix: None`, so the
    // bridge wouldn't install `PlanetFixedRotationC`. Build a
    // skeleton with Earth carrying an identity inertial→pfix transform
    // so both runtimes resolve the atmosphere stage with the same
    // (identity) rotation; `planet_omega` stays 0, matching the
    // hand-rolled test the recipe replaces.
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, PARITY_DT);
    let mut earth_entry = parity_earth_source(EARTH.shape.mu, true);
    earth_entry.t_inertial_pfix = Some(DMat3::IDENTITY);
    let earth = sb.add_source("Earth", earth_entry);
    let sun = sb.add_source("Sun", parity_sun_source());
    sb = sb.sun(sun);
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Exponential(ExponentialAtmosphere::default()),
            r_eq: astrodyn::planet_config::EARTH.shape.r_eq,
            r_pol: astrodyn::planet_config::EARTH.shape.r_pol,
            planet_omega: astrodyn::planet_config::EARTH.omega,
        },
        earth,
    );

    let plate = vec![(
        FlatPlate {
            area: 100.0,
            normal: DVec3::X,
            position: DVec3::ZERO.m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
        },
        FlatPlateParams {
            albedo: 0.0,
            diffuse: 0.0,
        },
        // Emissivity must be > 0 (JEOD_INV: IN.33). Both runtimes use
        // the same value, so parity holds.
        FlatPlateThermal {
            emissivity: 1.0,
            heat_capacity_per_area: 50.0,
            thermal_power_dump: 0.0,
        },
    )];

    sb.add_body(VehicleConfig {
        trans: parity_iss_trans().into(),
        rot: Some(parity_tumble_rot().into()),
        mass: Some(parity_iss_mass().into()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, true)],
        },
        compute_gravity_gradient: true,
        drag: Some(DragConfig {
            cd: 2.2,
            area: 1000.0,
            constant_density: None,
        }),
        srp: Some(SrpModel::FlatPlate(parity_flat_plate_state(
            plate,
            ThermalIntegrationOrder::default(),
        ))),
        ..Default::default()
    });
    sb
}

/// Full-stack parity: drag + 1 SRP plate + gravity-torque, ISS 6-DOF.
pub fn full_stack_sixdof() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_full_stack_sixdof",
        scenario: build_parity_full_stack_sixdof,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

// ── Scenario H: 6-plate flat-plate SRP with shadow, 3-DOF ──

fn build_parity_flat_plate_with_shadow(_init: &InitialConditions) -> SimulationBuilder {
    let (mut sb, earth) = parity_skeleton();
    let plates = srp_plates(); // same 6-plate cube as SIM_3_ORBIT

    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::new(4.2e7, 0.0, 0.0),
            velocity: DVec3::new(0.0, 3074.0, 0.0),
        }
        .into(),
        mass: Some(
            MassProperties::with_inertia(
                300.0,
                DMat3::from_diagonal(DVec3::splat(1.0)),
                DVec3::ZERO,
            )
            .into(),
        ),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        srp: Some(SrpModel::FlatPlate(parity_flat_plate_state(
            plates,
            ThermalIntegrationOrder::default(),
        ))),
        shadow_body: Some(ShadowBody {
            source_idx: earth,
            radius: EARTH.shadow_radius,
        }),
        ..Default::default()
    });
    sb
}

/// 6-plate flat-plate SRP with Earth shadow (3-DOF).
pub fn flat_plate_with_shadow() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_flat_plate_srp_with_shadow",
        scenario: build_parity_flat_plate_with_shadow,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

// ── Scenario family: shadow / basic / derivative single-plate 6-DOF ──

fn build_parity_single_plate_sixdof(
    plates: Vec<(
        FlatPlate<astrodyn::SelfRef>,
        FlatPlateParams,
        FlatPlateThermal,
    )>,
    order: ThermalIntegrationOrder,
    with_shadow: bool,
    t_struct_body: DMat3,
) -> SimulationBuilder {
    let (mut sb, earth) = parity_skeleton();
    let mut cfg = VehicleConfig {
        trans: parity_iss_trans().into(),
        rot: Some(parity_tumble_rot().into()),
        mass: Some(parity_iss_mass().into()),
        t_struct_body,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        srp: Some(SrpModel::FlatPlate(parity_flat_plate_state(plates, order))),
        ..Default::default()
    };
    if with_shadow {
        cfg.shadow_body = Some(ShadowBody {
            source_idx: earth,
            // Matches the hand-rolled shadow tests' literal radius.
            radius: 6_371_000.0,
        });
    }
    sb.add_body(cfg);
    sb
}

fn build_shadow_2a_annular(_init: &InitialConditions) -> SimulationBuilder {
    // Emissivity > 0 (JEOD_INV: IN.33). Tests shadow geometry, not thermal.
    build_parity_single_plate_sixdof(
        parity_single_plate(0.0, 0.0, 0.5),
        ThermalIntegrationOrder::default(),
        true,
        DMat3::IDENTITY,
    )
}

/// Shadow 2a annular flavor — 6-DOF, ε=0.5 single plate, Earth shadow.
pub fn shadow_2a_annular() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_shadow_2a_annular",
        scenario: build_shadow_2a_annular,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

fn build_shadow_2a_cooling(_init: &InitialConditions) -> SimulationBuilder {
    build_parity_single_plate_sixdof(
        parity_single_plate(0.0, 0.0, 0.9),
        ThermalIntegrationOrder::default(),
        true,
        DMat3::IDENTITY,
    )
}

/// Shadow 2a cooling flavor — 6-DOF, ε=0.9 single plate, Earth shadow.
pub fn shadow_2a_cooling() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_shadow_2a_cooling",
        scenario: build_shadow_2a_cooling,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

fn build_srp_basic_default(_init: &InitialConditions) -> SimulationBuilder {
    build_parity_single_plate_sixdof(
        parity_single_plate(0.3, 0.3, 0.5),
        ThermalIntegrationOrder::default(),
        false,
        DMat3::IDENTITY,
    )
}

/// SRP basic default — 6-DOF, single plate (albedo=0.3, diffuse=0.3).
pub fn srp_basic_default() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_basic_default",
        scenario: build_srp_basic_default,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

fn build_srp_basic_varied_cr(_init: &InitialConditions) -> SimulationBuilder {
    build_parity_single_plate_sixdof(
        parity_single_plate(0.8, 0.1, 0.5),
        ThermalIntegrationOrder::default(),
        false,
        DMat3::IDENTITY,
    )
}

/// SRP basic varied-Cr — 6-DOF, single plate (albedo=0.8, diffuse=0.1).
pub fn srp_basic_varied_cr() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_basic_varied_cr",
        scenario: build_srp_basic_varied_cr,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

fn build_srp_derivative_first_order(_init: &InitialConditions) -> SimulationBuilder {
    build_parity_single_plate_sixdof(
        parity_single_plate(0.3, 0.3, 0.5),
        ThermalIntegrationOrder::DerivativeFirstOrder,
        false,
        DMat3::IDENTITY,
    )
}

/// Derivative-class SRP, first-order thermal integration.
pub fn srp_derivative_first_order() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_derivative_first_order",
        scenario: build_srp_derivative_first_order,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

fn build_srp_derivative_rk4(_init: &InitialConditions) -> SimulationBuilder {
    build_parity_single_plate_sixdof(
        parity_single_plate(0.3, 0.3, 0.5),
        ThermalIntegrationOrder::DerivativeRk4,
        false,
        DMat3::IDENTITY,
    )
}

/// Derivative-class SRP, RK4 thermal integration.
pub fn srp_derivative_rk4() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_derivative_rk4",
        scenario: build_srp_derivative_rk4,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}

fn build_srp_derivative_rk4_rotated_struct(_init: &InitialConditions) -> SimulationBuilder {
    use astrodyn::Vec3Ext;
    // 90° rotation about body-Z: structural X→body Y, Y→-X. A
    // structural-frame SRP torque with X component becomes a
    // body-frame torque along Y. The hand-rolled regression test
    // caught a bug where SRP torque was added to constant_torque
    // without the t_struct_body rotation; this recipe is the parity
    // counterpart.
    let t_struct_body = DMat3::from_cols(
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );
    // Plate offset along structural +Y (15 m) with normal +X — produces
    // a non-zero structural-frame torque via r × F.
    let offset_plate = vec![(
        FlatPlate {
            area: 10.0,
            normal: DVec3::X,
            position: DVec3::new(0.0, 15.0, 0.0)
                .m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
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
    build_parity_single_plate_sixdof(
        offset_plate,
        ThermalIntegrationOrder::DerivativeRk4,
        false,
        t_struct_body,
    )
}

/// Derivative RK4 with non-identity `t_struct_body` and offset plate —
/// regression coverage for the structural↔body torque rotation in the
/// coupled RK4 stage closure.
pub fn srp_derivative_rk4_rotated_struct() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_derivative_rk4_rotated_struct",
        scenario: build_srp_derivative_rk4_rotated_struct,
        reference: CsvReference::TimesOnly(PARITY_TIMES_CSV),
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tols(),
        extras: None,
        pre_step: None,
    }
}
