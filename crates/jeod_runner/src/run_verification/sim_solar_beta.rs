//! `VerificationCase` constructors for SIM_SolarBeta.
//!
//! Cross-validates ISS LEO trajectory against JEOD's RUN_2 point-mass
//! reference (8h, point-mass tolerance) with the Sun source registered
//! and updated from DE421 ephemeris each step via the `pre_step` hook
//! (#156). Sun mu is 0 because the reference is RUN_2 (Earth-only
//! gravity), so the Sun position update doesn't perturb the trajectory
//! — it feeds `body.solar_beta` (`DerivedStateConfig::solar_beta = true`)
//! at every step but the recipe doesn't currently assert against an
//! external beta reference.
//!
//! ## Scope of the assertion
//!
//! `run_and_assert` currently validates only position/velocity vs the
//! RUN_2 CSV; with Sun mu=0, the trajectory is independent of the
//! per-step Sun update, so the migrated recipe doesn't exercise the
//! solar_beta computation directly. Comparing `body.solar_beta` against
//! JEOD's logged `SIM_SolarBeta` beta column requires extending the
//! recipe framework with `ExtrasComparator::SolarBeta` — tracked as
//! #169 (a focused follow-up that also enables migrating the four
//! `tier3_sim_solar_beta_edge` cases).
//!
//! 3rd-body gravity validation (Sun + Moon as gravitating bodies) lives
//! separately under `tier3_sim_dyncomp_run4` and `tier3_sim_torque_simple`.

use std::path::PathBuf;

use glam::DVec3;
use jeod_sim::recipes::verification::{
    CsvReference, InitialConditions, PreStepClosure, Tolerances, VerificationCase,
};
use jeod_sim::{
    coefficients, default_leap_second_table, DerivedStateConfig, Ephemeris, EphemerisBody,
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry,
    RotationModel, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use uom::si::f64::Time;
use uom::si::time::second;

/// SIM_dyncomp directory inside the JEOD checkout — solar_beta drives
/// itself off SIM_dyncomp's RUN_2 reference trajectory.
const SIM_DYNCOMP_DIR: &str = "verif/SIM_dyncomp";

/// J2000.0 epoch Julian date (JD 2_451_545.0), used as the baseline for
/// ephemeris-query Julian dates computed from simulation time-since-epoch
/// (`jd0 + seconds / 86_400.0`). Not converted between time scales here —
/// the simulation starts at `SimulationTime::at_j2000`, so adding
/// elapsed-seconds-divided-by-day gives a JD adequate for DE421 queries
/// at point-mass-tolerance precision (the residual time-scale offset
/// is small relative to the trajectory drift this case asserts on).
const J2000_JD: f64 = 2_451_545.0;

/// The Sun source's index inside the scenario, exposed as a named
/// constant so `solar_beta_pre_step`'s `set_source_position` can't
/// silently drift if `build_solar_beta_run2`'s source-add order
/// changes. The scenario builder `debug_assert!`s this against the
/// actual returned index at construction time.
const SUN_SOURCE_IDX: usize = 1;

fn jeod_root() -> PathBuf {
    let r = jeod_test_data::jeod_path();
    assert!(
        r.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        r.display()
    );
    r
}

fn bsp_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(p.exists(), "DE421 ephemeris not found at {}", p.display());
    p
}

fn load_mu_earth() -> f64 {
    let jeod = jeod_root();
    coefficients::load_mu_from_jeod_cc(
        &jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu from GGM05C")
}

fn earth_point_mass(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
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
            // mu=0 because RUN_2 is Earth-only gravity. Sun is used
            // solely for solar beta direction.
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: initial_pos,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
    }
}

fn build_solar_beta_run2(init: &InitialConditions) -> SimulationBuilder {
    let jeod = jeod_root();
    let mu_earth = load_mu_earth();
    let dt =
        jeod_test_data::s_define::load_dynamics_dt(&jeod.join(SIM_DYNCOMP_DIR).join("S_define"));

    // Load DE421 once for the t=0 sun position. The pre_step factory
    // below loads its own ephemeris instance for per-step queries.
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, J2000_JD)
        .expect("Sun position at J2000");

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", earth_point_mass(mu_earth));
    let sun = sb.add_source("Sun", sun_zero_mu(sun_t0.raw_si()));
    debug_assert_eq!(
        sun, SUN_SOURCE_IDX,
        "Sun source index drift: solar_beta_pre_step assumes Sun is at \
         SUN_SOURCE_IDX={SUN_SOURCE_IDX}, but add_source returned {sun}. \
         Either preserve the Earth-then-Sun add order or update SUN_SOURCE_IDX."
    );
    sb = sb.sun(sun);
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

/// Pre-step factory: capture a DE421 `Ephemeris` once, then update the
/// Sun source's position before each `sim.step_until` call. Uses
/// [`SUN_SOURCE_IDX`] to address the source — the scenario builder's
/// `debug_assert_eq!` keeps the two in sync if either side is edited.
fn solar_beta_pre_step(_init: &InitialConditions) -> PreStepClosure {
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    Box::new(move |sim, time_s: f64| {
        let tdb_jd = J2000_JD + time_s / 86_400.0;
        let (sun_pos_typed, _) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Sun, tdb_jd)
            .expect("Sun position query");
        sim.set_source_position(SUN_SOURCE_IDX, sun_pos_typed.raw_si());
    })
}

/// SIM_SolarBeta — RUN_2 trajectory + DE421 sun direction, exercising
/// the solar beta wiring at point-mass tolerance (8 hours).
///
/// Tolerances inherit from the underlying `run2_3dof` (same trajectory,
/// same Earth gravity, same dt). The Sun position update doesn't
/// perturb the trajectory because Sun mu is 0; it only feeds the solar
/// beta computation. **Note**: external validation of `body.solar_beta`
/// against JEOD's `SIM_SolarBeta` reference column is tracked in #169
/// (requires `ExtrasComparator::SolarBeta` framework addition).
pub fn solar_beta_run2() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_solar_beta",
        scenario: build_solar_beta_run2,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run2_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(solar_beta_pre_step),
    }
}
