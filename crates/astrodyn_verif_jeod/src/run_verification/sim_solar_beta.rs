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

use astrodyn::Vec3Ext;
use std::path::PathBuf;

use crate::verification::{
    CsvReference, ExtrasComparator, InitialConditions, PreStepClosure, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, DerivedStateConfig, Ephemeris, EphemerisBody, GravityControl,
    GravityControls, GravityModel, GravitySource, GravitySourceEntry, RotationModel,
    SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::DVec3;
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

fn bsp_path() -> PathBuf {
    let p = astrodyn::ephemeris_assets::de421_path();
    assert!(p.exists(), "DE421 ephemeris not found at {}", p.display());
    p
}

fn load_mu_earth() -> f64 {
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    astrodyn::gravity_fixtures::load_ggm05c().mu
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
            // mu=0 because RUN_2 is Earth-only gravity. Sun is used
            // solely for solar beta direction.
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

fn build_solar_beta_run2(init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let dt = crate::s_define::load_dynamics_dt(
        &crate::jeod_inputs::path(SIM_DYNCOMP_DIR).join("S_define"),
    );

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
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
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

// ── SIM_SolarBeta edge cases (#169) ───────────────────────────────────────
//
// Both edge cases run from JEOD's SIM_SolarBeta epoch (1991-01-01 UTC) and
// validate `body.solar_beta` against the matching reference CSV column via
// `ExtrasComparator::SolarBeta`. The migrated scenarios use
// `SimulationBuilder::ephemeris` + `set_source_ephemeris` + `.sun(idx)` so
// the simulation auto-updates the Sun source position each step from the
// attached DE421 ephemeris — no `pre_step` hook needed for these.

const SIM_SOLAR_BETA_DIR: &str = "models/dynamics/derived_state/verif/SIM_SolarBeta";

// SIM_SolarBeta epoch is 1991-01-01 00:00:00 UTC. The two derived forms
// (TT JD passed to ephemeris queries as a TDB approximation, TAI TJT for
// SimulationTime::new) are computed from one canonical UTC anchor + the
// 1991 TAI-UTC offset so a future change in either applies to both
// consumers automatically. The TT-TAI offset is shared with the
// `astrodyn_time` epoch module so the time arithmetic here cannot drift from
// the canonical definition.
const SIM_SOLAR_BETA_EPOCH_UTC_JD: f64 = 2_448_257.5;
/// TAI-UTC offset at 1991-01-01 (seconds).
const SIM_SOLAR_BETA_TAI_UTC_S: f64 = 26.0;

/// SIM_SolarBeta epoch in **TT JD**, passed to
/// [`Ephemeris::get_earth_centered_state_typed`] as a TDB approximation.
///
/// `Ephemeris::get_earth_centered_state_typed` documents its argument as
/// TDB JD. We pass TT JD here because:
/// * The TT−TDB periodic offset at 1991-01-01 00:00:00 UTC is ≈ 1.6 ms
///   (peak ~1.6 ms, varies by season). Sun's apparent motion against the
///   celestial sphere is ~30 km/s, so a 1.6 ms offset shifts the
///   Earth-centered Sun position by ≲ 50 m — well under this test's beta
///   tolerance (≈ 50 m / 1.5e11 m → 3.3e-10 rad, vs the asserted
///   1.892e-5 rad).
/// * Tier 3 baselines are frozen with this TT-as-TDB value; switching to
///   true TDB via [`SimulationTime::tdb_julian_date`] would shift the
///   baselines by the magnitudes above. A future PR can make that
///   physical-correctness improvement and refreeze if desired; this
///   constant is kept stable for now.
///
/// Computed as JD(TT) = JD(UTC) + (TAI-UTC + TT-TAI) / 86 400, where
/// TT-TAI is the canonical [`astrodyn::TAI_TT_OFFSET`] (= 32.184 s).
const SIM_SOLAR_BETA_EPOCH_TT_JD: f64 =
    SIM_SOLAR_BETA_EPOCH_UTC_JD + (SIM_SOLAR_BETA_TAI_UTC_S + astrodyn::TAI_TT_OFFSET) / 86_400.0;

/// SIM_SolarBeta epoch as TAI TJT (the form `SimulationTime::new` consumes).
/// = MJD(TAI) − 40 000 = (JD(UTC) + TAI-UTC/86 400 − 2 400 000.5) − 40 000.
const SIM_SOLAR_BETA_EPOCH_TAI_TJT: f64 =
    SIM_SOLAR_BETA_EPOCH_UTC_JD + SIM_SOLAR_BETA_TAI_UTC_S / 86_400.0 - 2_400_000.5 - 40_000.0;

fn sim_solar_beta_time() -> SimulationTime {
    let mut time = SimulationTime::new(SIM_SOLAR_BETA_EPOCH_TAI_TJT, default_leap_second_table());
    let time_cfg = crate::time_config::load_time_config(
        &crate::jeod_inputs::path(SIM_SOLAR_BETA_DIR).join("Modified_data/date_and_time.py"),
    );
    if let Some(ut1_tai) = time_cfg.ut1_tai_offset() {
        time.set_ut1_tai_offset(ut1_tai);
    }
    time
}

fn sim_solar_beta_dt() -> f64 {
    crate::s_define::load_dynamics_dt(
        &crate::jeod_inputs::path(SIM_SOLAR_BETA_DIR).join("S_define"),
    )
}

fn build_solar_beta_equ(init: &InitialConditions) -> SimulationBuilder {
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let mu_earth = astrodyn::gravity_fixtures::load_ggm05c().mu;

    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, SIM_SOLAR_BETA_EPOCH_TT_JD)
        .expect("Sun position at SIM_SolarBeta epoch");

    let mut sb = SimulationBuilder::new(sim_solar_beta_time(), sim_solar_beta_dt());
    sb = sb.ephemeris(ephemeris);
    let earth = sb.add_source("Earth", earth_point_mass(mu_earth));
    let sun = sb.add_source("Sun", sun_zero_mu(sun_t0.raw_si()));
    sb.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Earth);
    sb = sb.sun(sun);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
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

/// SIM_SolarBeta RUN_incl_0 — equatorial orbit (i=0), point-mass gravity.
/// Validates `body.solar_beta` against JEOD's logged beta column via
/// `ExtrasComparator::SolarBeta`. Sun position is auto-updated by the
/// simulation each step (no `pre_step` needed) because the source is
/// attached to the ephemeris via `set_source_ephemeris`.
pub fn solar_beta_equ() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_solar_beta_equ",
        scenario: build_solar_beta_equ,
        reference: CsvReference::SolarBeta("solarbeta_incl_0_solarbeta.csv"),
        duration: Time::new::<second>(0.0), // run full CSV
        tolerances: Tolerances {
            // The original bespoke test asserted only beta tolerance;
            // position/velocity/quat/ang_vel are skipped (all zero per
            // recipe semantics).
            position_m: [0.0; 3],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            // 1.892e-5 rad inherited from the bespoke assertion.
            extras: &[("beta", 1.892e-5)],
        },
        extras: Some(ExtrasComparator::SolarBeta),
        pre_step: None,
    }
}

fn build_solar_beta_obliquity(init: &InitialConditions) -> SimulationBuilder {
    // Earth GGM05C SH from the committed fixture (Wave 1 of #232).
    let sh_data = astrodyn::gravity_fixtures::load_ggm05c();
    let mu_earth = sh_data.mu;

    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, SIM_SOLAR_BETA_EPOCH_TT_JD)
        .expect("Sun position at SIM_SolarBeta epoch");

    let time = sim_solar_beta_time();
    // For SH gravity, initialize t_inertial_pfix to the correct RNP
    // rotation at the epoch (not IDENTITY). IDENTITY is only valid near
    // J2000; at 1991 the precession/nutation offset is significant.
    let initial_rotation =
        astrodyn::compute_t_parent_this_from_tjt_with_polar(time.gmst_seconds, time.tt_tjt(), None);

    let mut sb = SimulationBuilder::new(time, sim_solar_beta_dt());
    sb = sb.ephemeris(ephemeris);

    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(initial_rotation),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
    let sun = sb.add_source("Sun", sun_zero_mu(sun_t0.raw_si()));
    sb.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Earth);
    sb = sb.sun(sun);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(earth, 8, 8, false)],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

/// SIM_SolarBeta RUN_incl_23_4 — Earth-obliquity inclination (23.44°),
/// 8×8 spherical harmonics gravity. Captures J2 RAAN drift that
/// changes the orbital plane orientation vs Sun, directly affecting
/// solar beta.
pub fn solar_beta_obliquity() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_solar_beta_obliquity",
        scenario: build_solar_beta_obliquity,
        reference: CsvReference::SolarBeta("solarbeta_incl_23_4_solarbeta.csv"),
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            position_m: [0.0; 3],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            // 3.446e-5 rad inherited from the bespoke assertion.
            extras: &[("beta", 3.446e-5)],
        },
        extras: Some(ExtrasComparator::SolarBeta),
        pre_step: None,
    }
}
