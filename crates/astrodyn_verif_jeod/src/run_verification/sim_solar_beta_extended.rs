//! `VerificationCase` constructors for the SIM_SolarBeta analytical-extended
//! family (`tier3_sim_solar_beta_extended`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "verif step counts bounded by Tier 3 propagation span (<< usize / f64 mantissa)"
)]
//!
//! These cases have no JEOD reference CSV — they exercise closed-form
//! identities of the solar-beta angle (β = asin(ĥ · ŝ)) by parking a
//! fake Sun source at a chosen position and propagating a vehicle with
//! `solar_beta: true`. Each recipe shares the same Earth-point-mass +
//! mu=0 Sun shape; only the per-case initial state, Sun position, and
//! step count differ, so the per-case factories all delegate to a
//! shared `build_solar_beta_extended` constructor and pair the
//! resulting [`SimulationBuilder`] with [`CsvReference::SyntheticTimes`]
//! for the parity trait's lockstep `runner ↔ bevy` bit-identity
//! assertion.
//!
//! The matching analytical assertions live in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_solar_beta_extended.rs`;
//! each tier3 test pulls one or more recipes' scenario factories, builds
//! the `Simulation`, propagates, and asserts the closed-form solar-beta
//! property. Splitting the scenario into a recipe is what makes the
//! parity wrapper possible — the bridge needs an adapter-neutral
//! `SimulationBuilder` to materialize, and a hand-rolled tier3 test
//! that constructs a `Simulation` directly has no bridge entry point.

use super::fixtures::load_mu_earth;
use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::Vec3Ext;
use astrodyn::{
    default_leap_second_table, DerivedStateConfig, GravityControl, GravityControls,
    GravityGradient, GravityModel, GravitySource, GravitySourceEntry, RotationModel,
    SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::DVec3;
use uom::si::f64::Time;
use uom::si::time::second;

/// Sun parked at 1 AU. The body-relative Sun direction stays parallel
/// to this vector to within `r_orbit / AU` (≈ 5e-5 rad for LEO), which
/// is well below every test's tolerance.
const SUN_DISTANCE_M: f64 = 1.495_978_707e11;

/// Step size shared by every recipe. Matches the value the pre-recipe
/// tier3 file used so the SyntheticTimes cadence drives identical
/// integration ticks.
const DT_S: f64 = 10.0;

/// Reference radius for the LEO recipes (`equatorial_at_equinox`,
/// `polar_*`, `iss_*`) — 400 km altitude above a spherical Earth at
/// the equatorial radius. Matches the constant the analytical tests
/// use so the recipe and the closed-form assertion drive the identical
/// initial state.
const R_LEO_M: f64 = 6_778_137.0;

/// Reference radius (7000 km — ~622 km altitude above the equatorial
/// Earth radius, above the LEO cluster used for the ISS-snapshot
/// recipes) for the `sun_in_orbital_plane`,
/// `sun_perpendicular_to_plane`, and `bounded` recipes. The β formula
/// is radius-independent, so this is just the constant the pre-recipe
/// tier3 file used for these specific cases; holding it fixed here
/// keeps the recipe and the analytical assertion driving identical
/// initial states.
const R_MID_M: f64 = 7_000_000.0;

/// Shared scenario builder for every recipe. Parameterised by:
///   * `mu_earth` — Earth's gravitational parameter (point-mass);
///   * `dt` — integrator timestep (always `DT_S` for this family);
///   * `sun_position` — the fake Sun's inertial position (mu=0);
///   * `body` — the vehicle's initial translational state.
///
/// Each recipe wraps this with its case-specific values and pairs the
/// returned builder with the matching `SyntheticTimes` cadence so the
/// parity trait can drive `runner ↔ bevy` bit-identity at every step.
fn build_solar_beta_extended(
    mu_earth: f64,
    dt: f64,
    sun_position: DVec3,
    body: TranslationalState,
) -> SimulationBuilder {
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
    let sun = sb.add_source(
        "Sun",
        GravitySourceEntry {
            // mu=0 — the Sun's gravity does not perturb the trajectory.
            // Only the kinematic position feeds the solar-beta direction
            // computation.
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: sun_position.m_at::<astrodyn::RootInertial>(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    sb = sb.sun(sun);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&body),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..VehicleConfig::named("sim-solar-beta-extended-0")
    });
    sb
}

/// Analytical recipes opt out of every runner-vs-JEOD tolerance group
/// because they pair with [`CsvReference::SyntheticTimes`] and assert
/// in-test against closed-form values rather than logged JEOD columns.
/// The parity trait still asserts `runner ↔ bevy` bit-identity at every
/// synthetic record.
fn analytical_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

/// Closed-form circular-orbit period for radius `r` and gravitational
/// parameter `mu`. Used to size SyntheticTimes cadences for the
/// full-period scans.
fn period_s(mu_earth: f64, r: f64) -> f64 {
    2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt()
}

// ── Equatorial-at-equinox (full-period scan) ─────────────────────────

fn equatorial_at_equinox_num_steps() -> usize {
    // `.ceil()` so the cadence actually covers a full orbital period —
    // bare `as usize` would truncate and stop a few seconds short.
    (period_s(load_mu_earth(), R_LEO_M) / DT_S).ceil() as usize
}

fn build_equatorial_at_equinox(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let v = (mu_earth / R_LEO_M).sqrt();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(SUN_DISTANCE_M, 0.0, 0.0),
        TranslationalState {
            position: DVec3::new(R_LEO_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    )
}

/// Equatorial circular LEO with Sun in the equatorial plane (+X). The
/// orbit normal is +Z so ĥ · ŝ = 0 over the entire orbit — β stays
/// within floating-point noise of 0. Cadence covers one full period so
/// the analytical test can scan `max |β|`.
pub fn equatorial_at_equinox() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_equatorial_at_equinox",
        scenario: build_equatorial_at_equinox,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: equatorial_at_equinox_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Polar-orbit snapshot family (Sun along +X / +Y / +Z) ─────────────
//
// All three share the same polar-orbit body state (position +X, velocity
// +Z → h = +Y) and propagate a single step so the derived state is
// populated for the closed-form check. Splitting per Sun position keeps
// each recipe's `scenario` factory deterministic without a runtime
// switch argument.

const POLAR_SNAPSHOT_NUM_STEPS: usize = 1;

fn polar_body_state(mu_earth: f64) -> TranslationalState {
    let v = (mu_earth / R_LEO_M).sqrt();
    TranslationalState {
        position: DVec3::new(R_LEO_M, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, v),
    }
}

fn build_polar_sun_x(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(SUN_DISTANCE_M, 0.0, 0.0),
        polar_body_state(mu_earth),
    )
}

fn build_polar_sun_y(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(0.0, SUN_DISTANCE_M, 0.0),
        polar_body_state(mu_earth),
    )
}

fn build_polar_sun_z(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(0.0, 0.0, SUN_DISTANCE_M),
        polar_body_state(mu_earth),
    )
}

/// Polar orbit with Sun along +X (equinox-like): ĥ · ŝ = 0 → β = 0.
pub fn polar_sun_x() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_polar_sun_x",
        scenario: build_polar_sun_x,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: POLAR_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// Polar orbit (h = +Y) with Sun along +Y → ĥ · ŝ = ±1 → |β| = 90°.
pub fn polar_sun_y() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_polar_sun_y",
        scenario: build_polar_sun_y,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: POLAR_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// Polar orbit (h = +Y) with Sun along +Z: Sun lies in the orbital
/// plane (xz), so ĥ · ŝ = 0 → β = 0.
pub fn polar_sun_z() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_polar_sun_z",
        scenario: build_polar_sun_z,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: POLAR_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── ISS-inclination snapshot family (Sun along +X / +Z / −Y) ─────────
//
// Position +X, velocity = (0, v cos i, v sin i) at i = 51.6° → orbit
// normal direction (0, −sin i, cos i). Each recipe pairs with a Sun
// position chosen to exercise one of the closed-form identities:
//   * +X: ĥ · x̂ = 0 → β = 0;
//   * +Z: ĥ · ẑ = cos i → β = π/2 − i;
//   * −Y: ĥ · (−ŷ) = sin i → β = i.

/// ISS inclination (51.6°) as a radian constant. Mirrors the
/// expression used by [`f64::to_radians`] (`x * (PI / 180)`) — not
/// the alternate associativity `(x * PI) / 180` — so the body-state
/// inclination this constant fixes matches `51.6_f64.to_radians()`
/// at the last bit. The tier3 closed-form assertion encodes 51.6°
/// independently via `to_radians`; this convention keeps the two
/// evaluation paths bit-identical even though they are textually
/// separate.
const ISS_INCLINATION_RAD: f64 = 51.6 * (std::f64::consts::PI / 180.0);
/// Number of cycle steps for the ISS-snapshot recipes — same single-
/// step rationale as [`POLAR_SNAPSHOT_NUM_STEPS`].
const ISS_SNAPSHOT_NUM_STEPS: usize = 1;

fn iss_body_state(mu_earth: f64) -> TranslationalState {
    let v = (mu_earth / R_LEO_M).sqrt();
    let inc = ISS_INCLINATION_RAD;
    TranslationalState {
        position: DVec3::new(R_LEO_M, 0.0, 0.0),
        velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
    }
}

fn build_iss_sun_x(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(SUN_DISTANCE_M, 0.0, 0.0),
        iss_body_state(mu_earth),
    )
}

fn build_iss_sun_z(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(0.0, 0.0, SUN_DISTANCE_M),
        iss_body_state(mu_earth),
    )
}

fn build_iss_sun_neg_y(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(0.0, -SUN_DISTANCE_M, 0.0),
        iss_body_state(mu_earth),
    )
}

/// ISS orbit (i = 51.6°), Sun in the equatorial plane (+X). β ≈ 0.
pub fn iss_sun_x() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_iss_sun_x",
        scenario: build_iss_sun_x,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: ISS_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// ISS orbit (i = 51.6°), Sun along +Z. β = π/2 − i (peak β for this
/// inclination).
pub fn iss_sun_z() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_iss_sun_z",
        scenario: build_iss_sun_z,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: ISS_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// ISS orbit (i = 51.6°), Sun along −Y. β = i (β reaches the orbit
/// inclination at this Sun direction).
pub fn iss_sun_neg_y() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_iss_sun_neg_y",
        scenario: build_iss_sun_neg_y,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: ISS_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Single-snapshot edge geometries (in-plane Sun, perpendicular Sun) ─

const SINGLE_SNAPSHOT_NUM_STEPS: usize = 1;

fn build_sun_in_orbital_plane(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let v = (mu_earth / R_MID_M).sqrt();
    let inc = 30.0_f64.to_radians();
    // Position +X (ascending node), velocity tipped into the plane.
    // Orbit normal direction is (0, −sin i, cos i); Sun along +X
    // satisfies +X · (orbit normal) = 0, so Sun lies in the plane.
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(SUN_DISTANCE_M, 0.0, 0.0),
        TranslationalState {
            position: DVec3::new(R_MID_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
        },
    )
}

/// 30°-inclination orbit with Sun at the ascending node (+X). Sun is
/// in the orbital plane → β = 0.
pub fn sun_in_orbital_plane() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_sun_in_orbital_plane",
        scenario: build_sun_in_orbital_plane,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: SINGLE_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

fn build_sun_perpendicular_to_plane(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let v = (mu_earth / R_MID_M).sqrt();
    // Equatorial orbit: orbit normal = +Z. Sun along +Z → ĥ · ŝ = ±1.
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        DVec3::new(0.0, 0.0, SUN_DISTANCE_M),
        TranslationalState {
            position: DVec3::new(R_MID_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    )
}

/// Equatorial orbit with Sun along the orbit normal (+Z). |β| = π/2.
pub fn sun_perpendicular_to_plane() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_sun_perpendicular_to_plane",
        scenario: build_sun_perpendicular_to_plane,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: SINGLE_SNAPSHOT_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Bounded scan (multi-period, |β| ≤ π/2 at every record) ───────────

/// Cadence for the bounded scan: at least three orbital periods at
/// `DT_S` (`.ceil()` so bare-truncation never stops the scan short of
/// the third period boundary).
fn bounded_num_steps() -> usize {
    (3.0 * period_s(load_mu_earth(), R_MID_M) / DT_S).ceil() as usize
}

fn build_bounded(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let v = (mu_earth / R_MID_M).sqrt();
    let inc = 45.0_f64.to_radians();
    build_solar_beta_extended(
        mu_earth,
        DT_S,
        // Arbitrary Sun direction with components on all three axes
        // — exercises the generic β formula rather than an axis-aligned
        // limit.
        DVec3::new(
            0.7 * SUN_DISTANCE_M,
            0.5 * SUN_DISTANCE_M,
            0.2 * SUN_DISTANCE_M,
        ),
        TranslationalState {
            position: DVec3::new(R_MID_M, 0.0, 0.0),
            velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
        },
    )
}

/// 45°-inclination orbit with an off-axis Sun, propagated for three
/// periods. The analytical test asserts |β| ≤ π/2 at every record and
/// also confirms `max |β|` is large enough to actually exercise the
/// bound (rejects degenerate geometries).
pub fn bounded() -> VerificationCase {
    VerificationCase {
        name: "tier3_solar_beta_bounded",
        scenario: build_bounded,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: bounded_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}
