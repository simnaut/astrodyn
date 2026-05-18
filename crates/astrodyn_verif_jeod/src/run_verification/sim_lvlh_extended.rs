//! `VerificationCase` constructors for the SIM_LVLH analytical-extended
//! family (`tier3_sim_lvlh_extended`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "verif step counts bounded by Tier 3 propagation span (<< usize / f64 mantissa)"
)]
//!
//! These cases have no JEOD reference CSV — they exercise analytical
//! properties of the LVLH frame (orbit-normal sign on retrograde flips,
//! `|ω_LVLH| = |h|/r²` at perigee vs apogee, return-to-orientation after
//! one period). The recipes share a single point-mass Earth +
//! `lvlh: true` derived-state shape; only the per-case initial state and
//! step size differ, so each recipe wires its own `SimulationBuilder`
//! and pairs it with [`CsvReference::SyntheticTimes`] for the parity
//! trait to drive a lockstep `runner ↔ bevy` bit-identity assertion.
//!
//! The matching analytical assertions live in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_lvlh_extended.rs`; each
//! tier3 test pulls the recipe's scenario factory, builds the
//! `Simulation`, and asserts the closed-form LVLH property after
//! propagation. Splitting the scenario into a recipe is what makes the
//! parity wrapper possible — the bridge needs an adapter-neutral
//! `SimulationBuilder` to materialize, and a hand-rolled tier3 test
//! that constructs a `Simulation` directly has no bridge entry point.

use super::fixtures::load_mu_earth;
use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    default_leap_second_table, DerivedStateConfig, GravityControl, GravityControls,
    GravityGradient, GravityModel, GravitySource, GravitySourceEntry, RotationModel,
    SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::DVec3;
use uom::si::f64::Time;
use uom::si::time::second;

/// Reference orbit radius for the circular (`prograde`/`retrograde`/
/// `periodicity`) recipes — 400 km altitude above a spherical Earth at
/// the equatorial radius. Matches the shared constant the analytical
/// tests use so the parity-wrapper and the tier3 assertions drive the
/// identical initial state.
const R_LEO_M: f64 = 6_778_137.0;

/// Step size shared by every analytical recipe except `periodicity`,
/// which needs an exact-integer-period grid and overrides it locally.
const DT_S: f64 = 10.0;

/// Propagation horizon for `prograde_circular` and `retrograde_circular`.
/// The tier3 test reads back the LVLH frame after a single step, so a
/// modest cadence (10 records) is enough to drive the parity trait's
/// lockstep comparison.
const SHORT_NUM_STEPS: usize = 10;

/// Shared Earth source + LVLH-only body, parameterised by the body's
/// translational state. Mirrors the inline `make_earth_lvlh_sim` helper
/// the tier3 file used pre-recipe; refactoring it into a `SimulationBuilder`
/// is the only structural change needed to bridge the scenario onto Bevy.
fn build_lvlh_extended(mu_earth: f64, dt: f64, body: TranslationalState) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
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
        },
    );
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&body),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            lvlh: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

/// Prograde equatorial circular LEO: position +X, velocity +Y. Orbit
/// normal is +Z, so the LVLH Y-axis (row 1 of `T_parent_this`) points
/// −Ẑ. Used as the prograde half of the sign-flip comparison.
fn build_lvlh_prograde(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let v = (mu_earth / R_LEO_M).sqrt();
    build_lvlh_extended(
        mu_earth,
        DT_S,
        TranslationalState {
            position: DVec3::new(R_LEO_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    )
}

/// Retrograde equatorial circular LEO: position +X, velocity −Y. Orbit
/// normal flips to −Z, so the LVLH Y-axis points +Ẑ — the closed-form
/// sign flip the analytical test asserts.
fn build_lvlh_retrograde(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let v = (mu_earth / R_LEO_M).sqrt();
    build_lvlh_extended(
        mu_earth,
        DT_S,
        TranslationalState {
            position: DVec3::new(R_LEO_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, -v, 0.0),
        },
    )
}

/// Eccentric equatorial orbit with perigee at 6 778 137 m (≈400 km
/// altitude) and apogee at 20 000 km from Earth's centre. Starting at
/// perigee gives |ω_LVLH| = |h|/r² the analytical test compares against
/// the closed-form value at perigee and apogee.
pub const ECCENTRIC_R_PERIGEE_M: f64 = R_LEO_M;
/// See [`ECCENTRIC_R_PERIGEE_M`].
pub const ECCENTRIC_R_APOGEE_M: f64 = 20_000_000.0;

/// Number of `DT_S`-sized intervals needed for the parity trait's
/// checkpoint cadence to span at least one full orbital period of the
/// eccentric case. The ceiling matters: with truncation the final
/// checkpoint `N * DT_S` lands strictly before `period` (by up to one
/// step), so the analytical scan for per-revolution radius extrema
/// could miss apogee whenever apogee falls inside the dropped tail.
/// Rounding up guarantees the cadence covers the entire orbit at the
/// cost of one extra checkpoint past `period`.
fn eccentric_num_steps() -> usize {
    let mu_earth = load_mu_earth();
    let a = 0.5 * (ECCENTRIC_R_PERIGEE_M + ECCENTRIC_R_APOGEE_M);
    let period = 2.0 * std::f64::consts::PI * (a * a * a / mu_earth).sqrt();
    (period / DT_S).ceil() as usize
}

fn build_lvlh_eccentric(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let a = 0.5 * (ECCENTRIC_R_PERIGEE_M + ECCENTRIC_R_APOGEE_M);
    let e = (ECCENTRIC_R_APOGEE_M - ECCENTRIC_R_PERIGEE_M)
        / (ECCENTRIC_R_APOGEE_M + ECCENTRIC_R_PERIGEE_M);
    let v_p = (mu_earth * (1.0 + e) / (a * (1.0 - e))).sqrt();
    build_lvlh_extended(
        mu_earth,
        DT_S,
        TranslationalState {
            position: DVec3::new(ECCENTRIC_R_PERIGEE_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, v_p, 0.0),
        },
    )
}

/// Number of integration ticks per orbital period for the periodicity
/// case. Picked so `dt = period / PERIODICITY_NUM_STEPS` lands the
/// `step_until` stop time exactly on the period boundary (any non-
/// integer ratio dominates the LVLH-frame diff with grid-rounding
/// noise instead of physics error).
pub const PERIODICITY_NUM_STEPS: usize = 560;

/// Closed-form orbital period for the circular LEO used by the
/// `periodicity` recipe. Exposed because both the recipe and the
/// analytical test need the same value: the recipe to compute its `dt`
/// and to size the `SyntheticTimes` cadence, the test to drive
/// `step_until` to exactly one period.
pub fn periodicity_period_s() -> f64 {
    let mu_earth = load_mu_earth();
    2.0 * std::f64::consts::PI * (R_LEO_M * R_LEO_M * R_LEO_M / mu_earth).sqrt()
}

/// Step size derived from [`periodicity_period_s`] and
/// [`PERIODICITY_NUM_STEPS`]; shared between the recipe and the tier3
/// test so both drive identical integration ticks.
pub fn periodicity_dt_s() -> f64 {
    periodicity_period_s() / PERIODICITY_NUM_STEPS as f64
}

fn build_lvlh_periodicity(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let v = (mu_earth / R_LEO_M).sqrt();
    build_lvlh_extended(
        mu_earth,
        periodicity_dt_s(),
        TranslationalState {
            position: DVec3::new(R_LEO_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    )
}

fn analytical_tolerances() -> Tolerances {
    // Parity-only recipes use all-zero tolerances so the runner-vs-JEOD
    // `run_and_assert` opts out of every metric group (per the
    // CsvReference::SyntheticTimes contract); the parity trait still
    // asserts `runner ↔ bevy` bit-identity at every record.
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

/// Prograde equatorial circular LEO recipe — companion to
/// [`retrograde_circular`] for the LVLH-Y-axis sign-flip analytical
/// check.
pub fn prograde_circular() -> VerificationCase {
    VerificationCase {
        name: "tier3_lvlh_prograde_circular",
        scenario: build_lvlh_prograde,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: SHORT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// Retrograde equatorial circular LEO recipe — the orbit-normal sign
/// flips relative to [`prograde_circular`], so the LVLH Y-axis flips
/// sign.
pub fn retrograde_circular() -> VerificationCase {
    VerificationCase {
        name: "tier3_lvlh_retrograde_circular",
        scenario: build_lvlh_retrograde,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: SHORT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// Eccentric equatorial orbit (perigee 400 km, apogee 20 000 km from
/// Earth centre) used to verify `|ω_LVLH| = |h|/r²` at the radius
/// extrema. Cadence covers at least one full period (rounded up from
/// `period / DT_S`) so the analytical test sees both perigee and
/// apogee even when the orbital period is not an exact multiple of
/// `DT_S`.
pub fn eccentric() -> VerificationCase {
    VerificationCase {
        name: "tier3_lvlh_eccentric",
        scenario: build_lvlh_eccentric,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: eccentric_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// Circular LEO recipe whose `dt` divides the orbital period exactly,
/// so `step_until(period)` lands on the period boundary without
/// grid-rounding noise. The analytical test asserts the LVLH frame
/// returns to its initial orientation after one period.
pub fn periodicity() -> VerificationCase {
    VerificationCase {
        name: "tier3_lvlh_periodicity",
        scenario: build_lvlh_periodicity,
        reference: CsvReference::SyntheticTimes {
            dt: periodicity_dt_s(),
            num_steps: PERIODICITY_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}
