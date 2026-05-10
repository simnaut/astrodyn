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
    FlatPlateState, FlatPlateThermal, GravityControl, GravityControls, GravityModel, GravityRole,
    GravitySource, GravitySourceEntry, MassProperties, RotationModel, ShadowBody,
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
        marker_only: false,
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
        marker_only: false,
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
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        // allowed: typed↔raw kernel-boundary lift on scenario mass
        // construction (named-method opt-in; see #397).
        mass: Some(super::typed_helpers::mass_typed(
            &MassProperties::with_inertia(
                SRP_MASS,
                DMat3::from_diagonal(DVec3::splat(1.0)),
                DVec3::ZERO,
            ),
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
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

// ════════════════════════════════════════════════════════════════════
// Bevy-mechanism stress-scenario recipes (#395 sub-task B)
//
// These nine factories drive the same scenarios `bevy_parity_srp.rs`
// hand-rolled prior to #395. They share fixed ICs (ISS-like
// translational + tumbling rotational state from
// `astrodyn_verif_parity::common`) and a synthetic Sun position
// (`PARITY_SUN_POS`) — no DE421 ephemeris hook, no JEOD CSV
// reference, only `CsvReference::TimesOnly` for the parity-trait
// 10 s checkpoint cadence (parity is a runner ↔ bevy bit-identity
// check, not tolerance-bounded against JEOD-logged columns).
//
// Tolerances are all-zero for every metric group so any accidental
// invocation of `VerificationCaseExt::run_and_assert` on these
// scenarios opts out of the assertion via the documented "all-zero
// skips the metric group" rule, leaving the parity trait as the only
// path that actually compares state.
// ════════════════════════════════════════════════════════════════════

/// Synthetic Sun position used by every Bevy-mechanism SRP recipe
/// (1 AU along inertial +X). No ephemeris hook, no time variation —
/// the parity-only scenarios deliberately freeze the Sun so the
/// runner ↔ bevy comparison stays bit-identical without any
/// per-record state injection.
const PARITY_SUN_POS: DVec3 = DVec3::new(1.496e11, 0.0, 0.0);

/// `DT` shared with `astrodyn_verif_parity::common::DT`. The
/// synthetic-times cadence pins this; `SimulationBuilder::new`
/// receives the same value so runner ↔ bevy step in lockstep.
const PARITY_DT: f64 = 10.0;

/// Number of `PARITY_DT`-sized ticks driven by the parity loop.
/// Mirrors `astrodyn_verif_parity::common::NUM_STEPS = 100`.
const PARITY_NUM_STEPS: usize = 100;

/// Bevy-parity ISS-like initial position. Mirrors
/// `astrodyn_verif_parity::common::iss_trans()`.
fn parity_iss_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

/// Bevy-parity ISS-like tumble rotation. Mirrors
/// `astrodyn_verif_parity::common::tumble_rot()`.
fn parity_tumble_rot() -> astrodyn::RotationalState {
    let mut q = astrodyn::JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
    q.normalize();
    astrodyn::RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    }
}

/// Bevy-parity ISS-like mass: 400 t with diagonal inertia mirroring
/// `astrodyn_verif_parity::common::iss_mass()`.
fn parity_iss_mass() -> MassProperties {
    MassProperties::with_inertia(
        400_000.0,
        DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
        DVec3::ZERO,
    )
}

/// Earth point-mass source matching `common::earth_source()` plus the
/// `central=true` flag the runner-side `add_source` requires for the
/// integration-frame planet.
fn parity_earth_central() -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu: astrodyn::EARTH.shape.mu,
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
        marker_only: false,
    }
}

/// Sun source with `mu = 0` (referenced only for SRP direction, never
/// for gravitational perturbation) seeded at [`PARITY_SUN_POS`]. The
/// `marker_only = true` flag tells `populate_app` to spawn this Sun
/// as a `SunMarker`-only entity (no `GravitySourceC`,
/// `SourceInertialPositionC`, or frame-tree entity), matching the
/// hand-rolled `bevy_parity_srp.rs` Sun-as-marker setup so
/// runner ↔ bevy bit-identity holds.
fn parity_sun_source() -> GravitySourceEntry {
    let mut entry = sun_zero_mu(PARITY_SUN_POS);
    entry.marker_only = true;
    entry
}

/// Single-plate vehicle config: 100 m² plate with normal +X at the
/// CoM. Matches `bevy_parity_srp::make_single_plate(albedo, diffuse,
/// emissivity)` — used by the `shadow_2a_*`, `srp_basic_*`,
/// `srp_derivative_*` scenarios.
fn single_plate(
    albedo: f64,
    diffuse: f64,
    emissivity: f64,
) -> Vec<(
    FlatPlate<astrodyn::SelfRef>,
    FlatPlateParams,
    FlatPlateThermal,
)> {
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

/// 6-plate 4 m × 4 m × 15 m box used by `flat_plate_with_shadow`.
/// Distinct from [`srp_plates`] (which targets the SIM_3_ORBIT
/// vehicle's albedo/diffuse/emissivity = 0.5/0.5/0.5) — this box uses
/// (0.5, 0.5, 0.5) too but is wired to a parity-only scenario, so the
/// existing helper is reused verbatim.
fn parity_six_plate_box() -> Vec<(
    FlatPlate<astrodyn::SelfRef>,
    FlatPlateParams,
    FlatPlateThermal,
)> {
    srp_plates()
}

/// Pack a `FlatPlateState` for the recipe builder — initial
/// temperature 270 K shared across every Bevy-parity SRP scenario.
fn parity_flat_plate_state(
    plates: Vec<(
        FlatPlate<astrodyn::SelfRef>,
        FlatPlateParams,
        FlatPlateThermal,
    )>,
    integration_order: ThermalIntegrationOrder,
) -> FlatPlateState<astrodyn::SelfRef> {
    let n = plates.len();
    FlatPlateState {
        plates,
        temperatures: vec![INITIAL_PLATE_TEMP_K; n],
        t_pow4_cached: vec![INITIAL_PLATE_TEMP_K.powi(4); n],
        integration_order,
        ..Default::default()
    }
}

/// Skeleton scenario builder: Earth (central) + Sun (mu=0) + a
/// 6-DOF ISS-like vehicle. Returns the builder for the recipe to
/// finalise (drag, SRP, shadow, t_struct_body, gradient toggles).
/// `<earth_idx>` and `<sun_idx>` are 0 and 1 by registration order.
fn parity_skeleton() -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, PARITY_DT);
    let _earth = sb.add_source("Earth", parity_earth_central());
    let sun = sb.add_source("Sun", parity_sun_source());
    sb = sb.sun(sun);
    sb
}

/// Empty-tolerance group for parity-only recipes — every metric
/// opts out via the documented "all-zero skips" rule.
fn parity_zero_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

/// Common 6-DOF body skeleton: ISS-like trans/rot/mass with
/// spherical-Earth gravity controls and the gravity-gradient toggle
/// passed in by the caller.
fn parity_body_sixdof(earth_idx: usize, gradient: bool) -> VehicleConfig {
    let role = if gradient {
        GravityRole::ThirdBody
    } else {
        GravityRole::Central
    };
    VehicleConfig {
        trans: super::typed_helpers::trans_typed(&parity_iss_trans()),
        rot: Some(super::typed_helpers::rot_typed(&parity_tumble_rot())),
        mass: Some(super::typed_helpers::mass_typed(&parity_iss_mass())),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, role)],
        },
        compute_gravity_gradient: gradient,
        ..Default::default()
    }
}

// ── 1. full_stack_sixdof — drag + flat-plate SRP + gravity torque ──

fn build_full_stack_sixdof(_init: &InitialConditions) -> SimulationBuilder {
    let exp_atmos = astrodyn::ExponentialAtmosphere::default();
    let drag_config = astrodyn::DragConfig {
        cd: 2.2,
        area: 1000.0,
        constant_density: None,
    };
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
        FlatPlateThermal {
            emissivity: 1.0,
            heat_capacity_per_area: 50.0,
            thermal_power_dump: 0.0,
        },
    )];
    // Construct the skeleton manually — atmospheric drag requires a
    // rotating central body so the bridge can wire `PlanetFixedRotationC`
    // (spherical-fallback atmosphere is rejected by populate_app's
    // fence per the "fail loudly on misconfig" rule). Use Earth's full
    // `central_body` config (point-mass gravity + EarthRNP rotation)
    // for the central source. Sun follows the standard
    // `parity_sun_source` marker-only pattern.
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, PARITY_DT);
    let earth_idx = sb.add_source(
        "Earth",
        GravitySourceEntry::central_body(&astrodyn::planet_config::EARTH),
    );
    let sun = sb.add_source("Sun", parity_sun_source());
    sb = sb.sun(sun);
    sb = sb.atmosphere(
        astrodyn::AtmosphereConfig {
            model: astrodyn::AtmosphereModel::Exponential(exp_atmos),
            r_eq: astrodyn::EARTH.shape.r_eq,
            r_pol: astrodyn::EARTH.shape.r_pol,
            planet_omega: astrodyn::EARTH.omega,
        },
        earth_idx,
    );
    let mut body = parity_body_sixdof(earth_idx, true);
    body.drag = Some(drag_config);
    body.srp = Some(SrpModel::FlatPlate(parity_flat_plate_state(
        plate,
        ThermalIntegrationOrder::default(),
    )));
    sb.add_body(body);
    sb
}

/// Full stack — drag + flat-plate SRP + gravity-torque, 6-DOF ISS.
/// Mirrors `bevy_parity_srp::tier3_bevy_full_stack_sixdof`.
pub fn full_stack_sixdof() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_full_stack_sixdof",
        scenario: build_full_stack_sixdof,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── 2. flat_plate_with_shadow — 6-plate box + cylindrical shadow ──

fn build_flat_plate_with_shadow(_init: &InitialConditions) -> SimulationBuilder {
    let mut sb = parity_skeleton();
    let mut body = VehicleConfig {
        trans: super::typed_helpers::trans_typed(&parity_iss_trans()),
        // 3-DOF (translational only) per the hand-rolled
        // `tier3_bevy_flat_plate_srp_with_shadow` (rotational_dynamics
        // = false, three_dof = true). The mass ICs differ from the
        // shared ISS mass — the test uses a 300 kg vehicle to
        // emphasise SRP-driven trajectory deflection.
        rot: None,
        mass: Some(super::typed_helpers::mass_typed(
            &MassProperties::with_inertia(
                300.0,
                DMat3::from_diagonal(DVec3::splat(1.0)),
                DVec3::ZERO,
            ),
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(0, GravityRole::Central)],
        },
        ..Default::default()
    };
    // Override translational ICs to match the hand-rolled scenario
    // (GEO-radius circular orbit, not LEO-ISS).
    body.trans = super::typed_helpers::trans_typed(&TranslationalState {
        position: DVec3::new(4.2e7, 0.0, 0.0),
        velocity: DVec3::new(0.0, 3074.0, 0.0),
    });
    body.srp = Some(SrpModel::FlatPlate(parity_flat_plate_state(
        parity_six_plate_box(),
        ThermalIntegrationOrder::default(),
    )));
    body.shadow_body = Some(ShadowBody {
        source_idx: 0,
        radius: astrodyn::EARTH.shadow_radius,
    });
    sb.add_body(body);
    sb
}

/// 6-plate box + cylindrical Earth shadow, GEO 3-DOF.
/// Mirrors `bevy_parity_srp::tier3_bevy_flat_plate_srp_with_shadow`.
pub fn flat_plate_with_shadow() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_flat_plate_srp_with_shadow",
        scenario: build_flat_plate_with_shadow,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── 3. shadow_2a_annular — single-plate 6-DOF + Earth shadow ──

fn build_shadow_2a(albedo: f64, diffuse: f64, emissivity: f64) -> SimulationBuilder {
    let mut sb = parity_skeleton();
    let mut body = parity_body_sixdof(0, false);
    body.srp = Some(SrpModel::FlatPlate(parity_flat_plate_state(
        single_plate(albedo, diffuse, emissivity),
        ThermalIntegrationOrder::default(),
    )));
    body.shadow_body = Some(ShadowBody {
        source_idx: 0,
        // Hand-rolled `run_shadow_parity` uses a hand-typed
        // 6_371_000 m radius (not `EARTH.shadow_radius`) — preserve
        // the literal so the recipes drive the same kernel inputs.
        radius: 6_371_000.0,
    });
    sb.add_body(body);
    sb
}

fn build_shadow_2a_annular(_init: &InitialConditions) -> SimulationBuilder {
    build_shadow_2a(0.0, 0.0, 0.5)
}

fn build_shadow_2a_cooling(_init: &InitialConditions) -> SimulationBuilder {
    build_shadow_2a(0.0, 0.0, 0.9)
}

/// Shadow 2a annular: single-plate (albedo=0, diffuse=0,
/// emissivity=0.5) + Earth shadow. Mirrors
/// `bevy_parity_srp::tier3_bevy_shadow_2a_annular`.
pub fn shadow_2a_annular() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_shadow_2a_annular",
        scenario: build_shadow_2a_annular,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// Shadow 2a cooling: single-plate (albedo=0, diffuse=0,
/// emissivity=0.9) + Earth shadow. Mirrors
/// `bevy_parity_srp::tier3_bevy_shadow_2a_cooling`.
pub fn shadow_2a_cooling() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_shadow_2a_cooling",
        scenario: build_shadow_2a_cooling,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── 4. srp_basic — single-plate 6-DOF, no shadow, no drag ──

fn build_srp_basic(albedo: f64, diffuse: f64, emissivity: f64) -> SimulationBuilder {
    let mut sb = parity_skeleton();
    let mut body = parity_body_sixdof(0, false);
    body.srp = Some(SrpModel::FlatPlate(parity_flat_plate_state(
        single_plate(albedo, diffuse, emissivity),
        ThermalIntegrationOrder::default(),
    )));
    sb.add_body(body);
    sb
}

fn build_srp_basic_default(_init: &InitialConditions) -> SimulationBuilder {
    build_srp_basic(0.3, 0.3, 0.5)
}

fn build_srp_basic_varied_cr(_init: &InitialConditions) -> SimulationBuilder {
    build_srp_basic(0.8, 0.1, 0.5)
}

/// SRP basic, default Cr (albedo=0.3, diffuse=0.3, emissivity=0.5).
/// Mirrors `bevy_parity_srp::tier3_bevy_srp_basic_default`.
pub fn srp_basic_default() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_basic_default",
        scenario: build_srp_basic_default,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// SRP basic, varied Cr (albedo=0.8, diffuse=0.1, emissivity=0.5).
/// Mirrors `bevy_parity_srp::tier3_bevy_srp_basic_varied_cr`.
pub fn srp_basic_varied_cr() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_basic_varied_cr",
        scenario: build_srp_basic_varied_cr,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── 5. srp_derivative_* — DerivativeFirstOrder / DerivativeRk4 thermal ──

fn build_srp_deriv(integration_order: ThermalIntegrationOrder) -> SimulationBuilder {
    let mut sb = parity_skeleton();
    let mut body = parity_body_sixdof(0, false);
    body.srp = Some(SrpModel::FlatPlate(parity_flat_plate_state(
        single_plate(0.3, 0.3, 0.5),
        integration_order,
    )));
    sb.add_body(body);
    sb
}

fn build_srp_derivative_first_order(_init: &InitialConditions) -> SimulationBuilder {
    build_srp_deriv(ThermalIntegrationOrder::DerivativeFirstOrder)
}

fn build_srp_derivative_rk4(_init: &InitialConditions) -> SimulationBuilder {
    build_srp_deriv(ThermalIntegrationOrder::DerivativeRk4)
}

/// SRP derivative-class thermal at first-order. Mirrors
/// `bevy_parity_srp::tier3_bevy_srp_derivative_first_order`.
pub fn srp_derivative_first_order() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_derivative_first_order",
        scenario: build_srp_derivative_first_order,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// SRP derivative-class thermal at RK4 order. Mirrors
/// `bevy_parity_srp::tier3_bevy_srp_derivative_rk4`.
pub fn srp_derivative_rk4() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_derivative_rk4",
        scenario: build_srp_derivative_rk4,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── 6. srp_derivative_rk4_rotated_struct — non-identity t_struct_body ──

fn build_srp_derivative_rk4_rotated_struct(_init: &InitialConditions) -> SimulationBuilder {
    // 90° about body-Z: structural X → body Y, Y → -X.
    let t_struct_body = DMat3::from_cols(
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );
    // Plate offset structural +Y (15 m from CoM), normal +X — produces
    // a non-zero structural-frame torque on SRP.
    let plate = vec![(
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
    let mut sb = parity_skeleton();
    let mut body = parity_body_sixdof(0, false);
    body.t_struct_body = t_struct_body;
    body.srp = Some(SrpModel::FlatPlate(parity_flat_plate_state(
        plate,
        ThermalIntegrationOrder::DerivativeRk4,
    )));
    sb.add_body(body);
    sb
}

/// SRP derivative-class thermal at RK4 order with non-identity
/// `t_struct_body`. Regression test for the structural-vs-body frame
/// torque-summation bug (#114). Mirrors
/// `bevy_parity_srp::tier3_bevy_srp_derivative_rk4_with_rotated_struct_frame`.
pub fn srp_derivative_rk4_rotated_struct() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_srp_derivative_rk4_with_rotated_struct_frame",
        scenario: build_srp_derivative_rk4_rotated_struct,
        reference: CsvReference::SyntheticTimes {
            dt: PARITY_DT,
            num_steps: PARITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: parity_zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}
