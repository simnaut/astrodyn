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
    default_leap_second_table, Ephemeris, EphemerisBody, FlatPlate, FlatPlateParams,
    FlatPlateState, FlatPlateThermal, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, MassProperties, RotationModel, ShadowBody, SimulationBuilder,
    SimulationTime, SrpModel, ThermalIntegrationOrder, TranslationalState, VehicleConfig, EARTH,
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
