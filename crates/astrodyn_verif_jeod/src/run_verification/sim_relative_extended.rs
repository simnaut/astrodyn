//! `VerificationCase` constructors for the extended relative-dynamics
//! analytical family (`tier3_sim_relative_extended`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "orbit-period step counts bounded by Tier 3 propagation span (<< usize / f64 mantissa)"
)]
//!
//! These cases have no JEOD reference CSV — they exercise closed-form
//! identities of [`astrodyn::compute_relative_state`] and
//! [`astrodyn::compute_lvlh_relative_state_typed`] by spawning two
//! orbital bodies around a point-mass Earth and propagating through
//! `Simulation::step()` for a multi-period scan. Each recipe shares the
//! same Earth-point-mass + LVLH-derived-state scaffolding; only the
//! per-case body pair (positions, velocities, and which derived states
//! are enabled) and the propagation cadence differ, so the per-case
//! factories all delegate to a shared `build_relative_extended`
//! constructor and pair the resulting [`SimulationBuilder`] with
//! [`CsvReference::SyntheticTimes`] for the parity trait's lockstep
//! `runner ↔ bevy` bit-identity assertion.
//!
//! The matching analytical assertions live in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_relative_extended.rs`;
//! each tier3 test pulls one or more recipes' scenario factories, builds
//! the `Simulation`, propagates, and asserts the closed-form relative-
//! state property. Splitting the scenario into a recipe is what makes
//! the parity wrapper possible — the bridge needs an adapter-neutral
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

/// Reference radius for the LEO recipes — 400 km altitude above the
/// equatorial Earth radius. Matches the constant the pre-recipe tier3
/// file used so the recipe and the closed-form assertion drive the
/// identical initial state.
const R_LEO_M: f64 = 6_778_137.0;

/// Default integrator step size shared by all but the inclination
/// recipe. Matches the value the pre-recipe tier3 file used so the
/// SyntheticTimes cadence drives identical integration ticks across
/// runner and bevy.
const DT_DEFAULT_S: f64 = 10.0;

/// Tighter step size for the inclined-orbit recipe. The pre-recipe
/// tier3 file used 5 s here (versus 10 s for the others) so the
/// cross-track amplitude assertion (`< 1 m` against the analytical
/// `r * sin(i)`) stays well above the RK4 truncation floor; keep that
/// value to preserve the same numerical regime under parity.
const DT_INCLINATION_S: f64 = 5.0;

/// Closed-form circular-orbit period for radius `r` and gravitational
/// parameter `mu`. Used to size SyntheticTimes cadences for the
/// multi-period scans.
fn period_s(mu_earth: f64, r: f64) -> f64 {
    2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt()
}

/// Shared scenario builder for every recipe. Parameterised by:
///   * `mu_earth` — Earth's gravitational parameter (point-mass);
///   * `dt` — integrator timestep (one of `DT_DEFAULT_S` or
///     `DT_INCLINATION_S` in this family);
///   * `trans_chief` / `trans_deputy` — the two bodies' initial
///     translational states;
///   * `lvlh_chief` — whether to enable the LVLH derived state on the
///     chief (body 0). Only the co-orbiting recipe sets this; the
///     others read raw inertial positions for their analytical
///     assertions and leave the chief's LVLH unrequested. The deputy
///     never carries an LVLH request because the LVLH frame in this
///     family is always the chief's: `compute_lvlh_relative_state_typed`
///     takes the chief as the reference body, and no test reads the
///     deputy's own LVLH derived field.
///
/// Both bodies are spawned point-mass-only (no rotation, no mass) under
/// the same Earth source so the integration frame is
/// `PlanetInertial<Earth>` and parity holds without rotational-state
/// columns to compare.
fn build_relative_extended(
    mu_earth: f64,
    dt: f64,
    trans_chief: TranslationalState,
    trans_deputy: TranslationalState,
    lvlh_chief: bool,
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
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&trans_chief),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            lvlh: lvlh_chief,
            ..Default::default()
        },
        ..VehicleConfig::named("sim-relative-extended-1")
    });
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&trans_deputy),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig::default(),
        ..VehicleConfig::named("sim-relative-extended-0")
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

// ── Two co-orbiting vehicles (3 periods, LVLH derived state on) ──────

fn two_coorbiting_vehicles_num_steps() -> usize {
    // `.ceil()` so the cadence actually covers three full orbital
    // periods — bare `as usize` would truncate and stop a few seconds
    // short.
    (3.0 * period_s(load_mu_earth(), R_LEO_M) / DT_DEFAULT_S).ceil() as usize
}

fn build_two_coorbiting_vehicles(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let r = R_LEO_M;
    let v = (mu_earth / r).sqrt();

    // Chief at (r, 0, 0), velocity +y.
    let trans_chief = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    };
    // Deputy 100 m "ahead" — offset by a tiny true anomaly Δν so that
    // Δs ≈ r * Δν = 100 m, Δν = 100/r rad.
    let dnu = 100.0 / r;
    let trans_deputy = TranslationalState {
        position: DVec3::new(r * dnu.cos(), r * dnu.sin(), 0.0),
        velocity: DVec3::new(-v * dnu.sin(), v * dnu.cos(), 0.0),
    };
    build_relative_extended(mu_earth, DT_DEFAULT_S, trans_chief, trans_deputy, true)
}

/// Two vehicles on the same 400 km circular equatorial orbit with a
/// small along-track Δν offset. In the chief's LVLH frame the deputy
/// stays at an almost-constant along-track offset over 3 orbits — the
/// analytical sibling asserts both the inertial separation bound and
/// the LVLH along-track / out-of-plane components.
pub fn two_coorbiting_vehicles() -> VerificationCase {
    VerificationCase {
        name: "tier3_relative_two_coorbiting_vehicles",
        scenario: build_two_coorbiting_vehicles,
        reference: CsvReference::SyntheticTimes {
            dt: DT_DEFAULT_S,
            num_steps: two_coorbiting_vehicles_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Hohmann-shaped transfer geometry (one deputy period) ─────────────

fn hohmann_transfer_geometry_num_steps() -> usize {
    let mu_earth = load_mu_earth();
    let r_chief = R_LEO_M;
    let r_apo = 1.05 * r_chief;
    let a_d = 0.5 * (r_chief + r_apo);
    // Deputy semi-major axis sets the propagation horizon: one full
    // deputy period brings it back to periapsis. `.ceil()` so the
    // cadence covers the full period — bare truncation would stop a
    // few seconds short and miss the apoapsis flyby.
    (period_s(mu_earth, a_d) / DT_DEFAULT_S).ceil() as usize
}

fn build_hohmann_transfer_geometry(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let r_chief = R_LEO_M;
    let v_chief = (mu_earth / r_chief).sqrt();

    // Chief: circular at r_chief.
    let trans_chief = TranslationalState {
        position: DVec3::new(r_chief, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_chief, 0.0),
    };
    // Deputy: periapsis at (r_chief, 0, 0), same direction of motion,
    // apoapsis at r_apo = 1.05 * r_chief.
    let r_apo = 1.05 * r_chief;
    let a_d = 0.5 * (r_chief + r_apo);
    let e_d = (r_apo - r_chief) / (r_apo + r_chief);
    let v_peri = (mu_earth * (1.0 + e_d) / (a_d * (1.0 - e_d))).sqrt();
    let trans_deputy = TranslationalState {
        position: DVec3::new(r_chief, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_peri, 0.0),
    };
    build_relative_extended(mu_earth, DT_DEFAULT_S, trans_chief, trans_deputy, false)
}

/// Chief in circular orbit at 400 km; deputy in a coplanar ellipse
/// whose periapsis coincides with the chief's orbit. The analytical
/// sibling asserts the inertial separation oscillates between the
/// closed-form geometric bounds over one deputy period.
pub fn hohmann_transfer_geometry() -> VerificationCase {
    VerificationCase {
        name: "tier3_relative_hohmann_transfer_geometry",
        scenario: build_hohmann_transfer_geometry,
        reference: CsvReference::SyntheticTimes {
            dt: DT_DEFAULT_S,
            num_steps: hohmann_transfer_geometry_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Same-orbit 90° phase difference (2 periods) ──────────────────────

fn same_orbit_phase_difference_num_steps() -> usize {
    // `.ceil()` so the cadence covers two full periods — same
    // rationale as the co-orbiting recipe.
    (2.0 * period_s(load_mu_earth(), R_LEO_M) / DT_DEFAULT_S).ceil() as usize
}

fn build_same_orbit_phase_difference(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let r = R_LEO_M;
    let v = (mu_earth / r).sqrt();

    // Body A: at (r, 0, 0), velocity (0, v, 0).
    let trans_a = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    };
    // Body B: at (0, r, 0) [90° ahead], velocity (-v, 0, 0).
    let trans_b = TranslationalState {
        position: DVec3::new(0.0, r, 0.0),
        velocity: DVec3::new(-v, 0.0, 0.0),
    };
    build_relative_extended(mu_earth, DT_DEFAULT_S, trans_a, trans_b, false)
}

/// Two vehicles in the same circular orbit, 90° apart in true anomaly.
/// The analytical sibling asserts the chord length stays at
/// `r * sqrt(2)` over two orbits (RK4 truncation floor is sub-mm at
/// 10 s).
pub fn same_orbit_phase_difference() -> VerificationCase {
    VerificationCase {
        name: "tier3_relative_same_orbit_phase_difference",
        scenario: build_same_orbit_phase_difference,
        reference: CsvReference::SyntheticTimes {
            dt: DT_DEFAULT_S,
            num_steps: same_orbit_phase_difference_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Different inclinations (2 periods, 5 s cadence) ──────────────────

fn different_inclinations_num_steps() -> usize {
    // Tighter `DT_INCLINATION_S` keeps the cross-track amplitude
    // assertion (< 1 m against the analytical `r * sin(i)`) above the
    // RK4 truncation floor. `.ceil()` so the cadence covers two
    // periods at the tighter step.
    (2.0 * period_s(load_mu_earth(), R_LEO_M) / DT_INCLINATION_S).ceil() as usize
}

fn build_different_inclinations(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let r = R_LEO_M;
    let v = (mu_earth / r).sqrt();
    let inc = 1.0_f64.to_radians();

    // Chief: equatorial.
    let trans_chief = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    };
    // Deputy: inclined +1° about +X, same initial position; rotation
    // sends (0, v, 0) → (0, v cos i, v sin i).
    let trans_deputy = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
    };
    build_relative_extended(mu_earth, DT_INCLINATION_S, trans_chief, trans_deputy, false)
}

/// Chief on equatorial circular orbit; deputy on the same circular
/// orbit inclined by +1°. The analytical sibling asserts the
/// cross-track separation amplitude matches `r * sin(i)` within 1 m
/// over two orbital periods.
pub fn different_inclinations() -> VerificationCase {
    VerificationCase {
        name: "tier3_relative_different_inclinations",
        scenario: build_different_inclinations,
        reference: CsvReference::SyntheticTimes {
            dt: DT_INCLINATION_S,
            num_steps: different_inclinations_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Round-trip frames (one chief period) ─────────────────────────────

fn round_trip_frames_num_steps() -> usize {
    // `.ceil()` to fully cover one orbital period at the chief radius.
    (period_s(load_mu_earth(), R_LEO_M) / DT_DEFAULT_S).ceil() as usize
}

fn build_round_trip_frames(_init: &InitialConditions) -> SimulationBuilder {
    let mu_earth = load_mu_earth();
    let r = R_LEO_M;
    let v = (mu_earth / r).sqrt();

    // Body 0 at (r, 0, 0); Body 1 in a different coplanar circular
    // orbit 500 km further out.
    let trans_a = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    };
    let r2 = r + 500_000.0;
    let v2 = (mu_earth / r2).sqrt();
    let trans_b = TranslationalState {
        position: DVec3::new(r2, 0.0, 0.0),
        velocity: DVec3::new(0.0, v2, 0.0),
    };
    build_relative_extended(mu_earth, DT_DEFAULT_S, trans_a, trans_b, false)
}

/// Two vehicles on coplanar circular orbits 500 km apart. The
/// analytical sibling asserts `r_AB = -r_BA` and `v_AB = -v_BA`
/// (relative-state operator symmetry when neither body carries a
/// rotational state) at every checkpoint over one orbital period.
pub fn round_trip_frames() -> VerificationCase {
    VerificationCase {
        name: "tier3_relative_round_trip_frames",
        scenario: build_round_trip_frames,
        reference: CsvReference::SyntheticTimes {
            dt: DT_DEFAULT_S,
            num_steps: round_trip_frames_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}
