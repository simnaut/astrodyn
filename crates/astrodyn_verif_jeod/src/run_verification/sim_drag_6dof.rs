//! `VerificationCase` constructors for the 6-DOF drag analytical family
//! (`tier3_sim_drag_6dof`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "verif step counts bounded by Tier 3 propagation span (<< usize / f64 mantissa)"
)]
//!
//! These cases have no JEOD reference CSV — they exercise closed-form
//! identities of the ballistic-drag model (constant `Cd·A` with a
//! constant-density override) by propagating a spinning body in a
//! 400 km circular orbit through `Simulation::step()` for one orbital
//! period. Each recipe shares the same Earth-point-mass +
//! exponential-atmosphere + constant-density scaffolding; only the
//! initial attitude and angular velocity differ, so the per-case
//! factories all delegate to a shared `build_drag_6dof` constructor and
//! pair the resulting [`SimulationBuilder`] with
//! [`CsvReference::SyntheticTimes`] for the parity trait's lockstep
//! `runner ↔ bevy` bit-identity assertion.
//!
//! The matching analytical assertions live in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_drag_6dof.rs`; each tier3
//! test pulls one or more recipes' scenario factories, builds the
//! `Simulation`, propagates, and asserts the closed-form drag property
//! (monotonic specific-orbital-energy loss; ballistic-drag attitude
//! invariance). Splitting the scenario into a recipe is what makes the
//! parity wrapper possible — the bridge needs an adapter-neutral
//! `SimulationBuilder` to materialize, and a hand-rolled tier3 test
//! that constructs a `Simulation` directly has no bridge entry point.

use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    default_leap_second_table, AtmosphereConfig, AtmosphereModel, DragConfig,
    ExponentialAtmosphere, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationModel, RotationalState,
    SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

/// Earth gravitational parameter (m³/s²) — JEOD `earth_GGM05C.cc`.
/// Inlined from `astrodyn::EARTH.shape` (rather than reaching into the
/// gravity-fixture decode the way `sim_relative_extended` does) so that
/// the recipe-driven `Simulation` reproduces the bit-pattern of the
/// pre-recipe inline `make_6dof_drag_sim` constructor exactly. The two
/// `mu` sources agree numerically today, but pinning the recipe to the
/// same constant the test always used keeps the analytical assertions
/// driving from a single value.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Earth mean equatorial radius (m) — JEOD `earth.cc`.
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq;

/// Orbit radius for every recipe — 400 km altitude above the equatorial
/// Earth radius. Matches the constant the pre-recipe tier3 file used so
/// the recipe and the closed-form assertion drive the identical initial
/// state.
const R_ORBIT_M: f64 = R_EARTH + 400_000.0;

/// Integrator step size shared by every recipe. Matches the value the
/// pre-recipe tier3 file used so the SyntheticTimes cadence drives
/// identical integration ticks across runner and bevy.
const DT_S: f64 = 10.0;

/// Vehicle mass (kg) — bespoke 1 t test geometry.
const MASS_KG: f64 = 1000.0;

/// Drag coefficient — bespoke ballistic-drag geometry shared across
/// every recipe.
const CD: f64 = 2.2;

/// Drag reference area (m²) — bespoke geometry.
const AREA_M2: f64 = 10.0;

/// Constant atmospheric density override (kg/m³) — bypasses the
/// exponential atmosphere model and drives the assertion content.
const DENSITY: f64 = 1e-12;

/// Number of integration steps to cover approximately one
/// circular-orbit period at `R_ORBIT_M`, truncated to an integer number
/// of `DT_S` ticks so the simulated duration is
/// `floor(period / DT_S) * DT_S` (always ≤ one period). The pre-recipe
/// tier3 file computed this with bare `as usize` truncation; reproduce
/// that convention here for bit-identical step counts.
fn num_steps_one_period() -> usize {
    let period = 2.0 * std::f64::consts::PI * (R_ORBIT_M.powi(3) / MU_EARTH).sqrt();
    (period / DT_S) as usize
}

/// Shared scenario builder for every recipe. Parameterised by the body's
/// initial translational state, rotational state, and mass properties.
/// All other knobs (Earth source, atmosphere config, drag config,
/// integrator dt) are recipe-wide constants.
///
/// Mirrors `make_6dof_drag_sim` in the pre-recipe tier3 file
/// field-for-field. The atmosphere config is required by validation
/// even when `constant_density` overrides the atmospheric density.
fn build_drag_6dof(
    trans: TranslationalState,
    rot: RotationalState,
    mass_props: MassProperties,
) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, DT_S);
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            // The atmosphere kernel treats `t_inertial_pfix: None` as
            // "position is already in planet-fixed coordinates" (it
            // skips the inertial→pfix rotation and feeds the position
            // straight into the geodetic conversion). For an
            // Earth-inertial body that semantics is wrong — but with
            // `RotationModel::None` + `planet_omega: 0.0` the only
            // numerical effect of supplying `Some(IDENTITY)` instead
            // of `None` is to make the kernel run `IDENTITY * position`
            // (a no-op) before the geodetic conversion, with no
            // co-rotation wind added either way. The runner therefore
            // stays bit-identical to the previous `None`-based
            // configuration on this scenario, while the Bevy atmosphere
            // stage gets the `PlanetFixedRotationC` it requires on the
            // planet entity for the bridge to lift across.
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Exponential(ExponentialAtmosphere {
                rho_0: 1e-12,
                h_0: 400_000.0,
                scale_height: 50_000.0,
            }),
            r_eq: R_EARTH,
            // Match the pre-recipe inline expression for `r_pol`
            // exactly: `R_EARTH * (1.0 - 1.0 / 298.257_223_563)`. This
            // is also what `astrodyn::EARTH.shape.r_pol` evaluates to
            // (the `EARTH_R_POL` constant in `body_constants.rs` is
            // `EARTH_R_EQ * (1.0 - EARTH_FLAT_COEFF)` with
            // `EARTH_FLAT_COEFF = 1.0 / 298.257_223_563`), so using the
            // preset keeps the value bit-identical and avoids
            // hard-coding the flattening literal here.
            r_pol: astrodyn::EARTH.shape.r_pol,
            planet_omega: 0.0,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&trans),
        rot: Some(super::typed_helpers::rot_typed(&rot)),
        mass: Some(super::typed_helpers::mass_typed(&mass_props)),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        drag: Some(DragConfig {
            cd: CD,
            area: AREA_M2,
            constant_density: Some(DENSITY),
        }),
        ..Default::default()
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

/// Initial circular-orbit translational state at `R_ORBIT_M` with the
/// body at (r, 0, 0) and velocity along +Y at the local circular speed.
/// Shared across every recipe in the family.
fn circular_orbit_trans() -> TranslationalState {
    let v = (MU_EARTH / R_ORBIT_M).sqrt();
    TranslationalState {
        position: DVec3::new(R_ORBIT_M, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    }
}

/// Uniform-sphere mass properties for the bespoke 1 t / 1 m-radius body:
/// `I = (2/5) m r² = 0.4 * MASS_KG * 1.0` on each diagonal, CoM at the
/// origin.
fn uniform_sphere_mass_props() -> MassProperties {
    let i_val = 0.4 * MASS_KG * 1.0;
    let inertia = DMat3::from_diagonal(DVec3::splat(i_val));
    MassProperties::with_inertia(MASS_KG, inertia, DVec3::ZERO)
}

// ── Drag with rotation: energy-loss assertion ─────────────────────────

fn build_drag_with_rotation_energy_loss(_init: &InitialConditions) -> SimulationBuilder {
    let eigen_angle = 30.0_f64.to_radians();
    let eigen_axis = DVec3::new(1.0, 1.0, 1.0).normalize();
    let quat = JeodQuat::left_quat_from_eigen_rotation(eigen_angle, eigen_axis);
    let ang_vel = DVec3::new(0.01, -0.005, 0.003);
    build_drag_6dof(
        circular_orbit_trans(),
        RotationalState {
            quaternion: quat,
            ang_vel_body: ang_vel,
        },
        uniform_sphere_mass_props(),
    )
}

/// 6-DOF body on a 400 km equatorial circular orbit with a non-trivial
/// initial attitude (30° about the (1,1,1) axis) and small spin. The
/// analytical sibling asserts the specific orbital energy decreases
/// monotonically (ballistic drag removes energy) and that the angular
/// velocity magnitude is conserved (ballistic drag produces no torque,
/// and the gravity gradient is disabled).
pub fn drag_with_rotation_energy_loss() -> VerificationCase {
    VerificationCase {
        name: "tier3_drag_with_rotation_energy_loss",
        scenario: build_drag_with_rotation_energy_loss,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: num_steps_one_period(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Ballistic-drag attitude invariance: identity-attitude leg ─────────

fn build_drag_attitude_invariance_identity(_init: &InitialConditions) -> SimulationBuilder {
    build_drag_6dof(
        circular_orbit_trans(),
        RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        },
        uniform_sphere_mass_props(),
    )
}

/// Identity-attitude leg of the ballistic-drag attitude-invariance
/// pair. Same translational state, mass, and drag config as the
/// rotated leg; the analytical sibling propagates both for one orbital
/// period and asserts their translational trajectories agree to within
/// numerical precision.
pub fn drag_attitude_invariance_identity() -> VerificationCase {
    VerificationCase {
        name: "tier3_drag_attitude_invariance_identity",
        scenario: build_drag_attitude_invariance_identity,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: num_steps_one_period(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Ballistic-drag attitude invariance: rotated-attitude leg ──────────

fn build_drag_attitude_invariance_rotated(_init: &InitialConditions) -> SimulationBuilder {
    let quat =
        JeodQuat::left_quat_from_eigen_rotation(45.0_f64.to_radians(), DVec3::new(0.0, 0.0, 1.0));
    build_drag_6dof(
        circular_orbit_trans(),
        RotationalState {
            quaternion: quat,
            ang_vel_body: DVec3::new(0.0, 0.0, 0.05),
        },
        uniform_sphere_mass_props(),
    )
}

/// Rotated-attitude leg of the ballistic-drag attitude-invariance pair
/// (45° about +Z, spinning about +Z at 0.05 rad/s). Same translational
/// state, mass, and drag config as the identity-attitude leg.
pub fn drag_attitude_invariance_rotated() -> VerificationCase {
    VerificationCase {
        name: "tier3_drag_attitude_invariance_rotated",
        scenario: build_drag_attitude_invariance_rotated,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: num_steps_one_period(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}
