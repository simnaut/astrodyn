// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! `VerificationCase` constructors for the analytical SIM_dyncomp
//! physics-combinations family (`tier3_sim_dyncomp_combinations`).
//!
//! The numbered SIM_dyncomp RUN_* scenarios already have Docker-backed
//! cross-validation tests (`tier3_sim_dyncomp_run2`..`run10`). The
//! recipes here cover the closed-form / conservation-law checks the
//! `tier3_sim_dyncomp_combinations.rs` sibling exercises against the
//! same physics combinations, without needing a JEOD reference CSV:
//!
//! - Keplerian energy + angular-momentum conservation (RUN_2 family,
//!   point-mass gravity);
//! - Third-body torque on a LEO orbit (RUN_7A/7B analog, Sun + Moon
//!   point-mass perturbers off the orbital plane);
//! - Monotonic semi-major-axis decay under drag (RUN_6/RUN_7C/7D analog,
//!   point-mass Earth + exponential atmosphere with a constant-density
//!   override);
//! - Torque-free rigid-body rotation conserving the inertial angular-
//!   momentum vector (RUN_8A analog);
//! - F = m·a impulse response on a 3-DOF orbiting body
//!   (RUN_9C/9D analog, external inertial force applied via
//!   [`VehicleConfig::external_force`]);
//! - τ = I·α impulse response on a 6-DOF body (RUN_9A/9B/9C analog,
//!   external body-frame torque applied via
//!   [`VehicleConfig::external_torque`]);
//! - Major-axis spin stability (intermediate-axis theorem, RUN_8B
//!   neighborhood).
//!
//! Each recipe pairs with [`CsvReference::SyntheticTimes`] so the
//! parity trait can assert `runner ↔ bevy` bit-identity at every
//! synthetic record while the matching tier3 file asserts the closed-
//! form analytical identity on the runner. Shared constructors
//! (`build_kepler_sim`, `build_kepler_6dof_sim`, `build_drag_sim`)
//! mirror the pre-recipe `make_kepler_sim`-style helpers field-for-
//! field so the recipe path reproduces the inline tier3 numerics
//! exactly.
//!
//! Every Earth gravity source is configured with
//! `t_inertial_pfix: Some(DMat3::IDENTITY)` rather than `None`. With
//! `RotationModel::None` + `planet_omega: 0.0` the runner side is
//! bit-identical to the previous `None`-based configuration (the
//! kernel runs `IDENTITY * position` as a no-op before the geodetic
//! conversion, with no co-rotation wind), while the Bevy adapter gets
//! the `PlanetFixedRotationC` it requires on the planet entity for
//! the bridge to lift the scenario across. See `sim_drag_6dof.rs` for
//! the same rationale on a drag-bearing recipe.

use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    default_leap_second_table, AtmosphereConfig, AtmosphereModel, DragConfig,
    ExponentialAtmosphere, Force, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RootInertial, RotationModel,
    RotationalState, SimulationBuilder, SimulationTime, Torque, TranslationalState, VehicleConfig,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

/// Earth gravitational parameter (m³/s²) — JEOD `earth_GGM05C.cc`.
/// Inlined from `astrodyn::EARTH.shape` so the recipe-driven
/// `Simulation` reproduces the pre-recipe inline `make_kepler_sim` /
/// `make_kepler_6dof_sim` numerics exactly; the inline tier3 file
/// reads the constant from the same place.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Earth mean equatorial radius (m) — JEOD `earth.cc`. Drives the
/// 400 km LEO orbit radius shared by every recipe in this file.
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq;

/// Sun gravitational parameter (m³/s²) — JEOD `sun_spherical.cc`.
/// Used only by the third-body recipe.
const MU_SUN: f64 = astrodyn::SUN.shape.mu;

/// Moon gravitational parameter (m³/s²) — JEOD `moon_GRAIL150.cc`.
/// Used only by the third-body recipe.
const MU_MOON: f64 = astrodyn::MOON.shape.mu;

/// Typical Earth–Sun distance (m, ~1 AU). Matches the value the pre-
/// recipe inline tier3 file used to place the Sun off the orbital
/// plane.
const R_EARTH_SUN: f64 = 1.495_978_707e11;

/// Typical Earth–Moon distance (m). Matches the pre-recipe value.
const R_EARTH_MOON: f64 = 3.844_0e8;

/// 400 km altitude (ISS-like) shared by every recipe. The orbit radius
/// `R_EARTH + 400 km` and circular-orbit speed `sqrt(mu / r)` are the
/// only initial state the recipes share — recipes differ on attitude,
/// inertia, and external loads.
const ALT_M: f64 = 400_000.0;

/// Vehicle mass (kg) shared by tests 1, 2, 3, 5, and 7. Tests 4 and 6
/// override with their own asymmetric inertia geometry; both still use
/// `mass = 1000.0 kg` so the recipe-wide constant stays meaningful.
const MASS_KG: f64 = 1000.0;

/// Recipes opt out of every runner-vs-JEOD tolerance group: the tier3
/// file asserts closed-form analytical identities directly on the
/// runner-side result, and the parity trait asserts `runner ↔ bevy`
/// bit-identity at every synthetic record without consulting these
/// tolerances.
fn analytical_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

/// Initial 400 km circular orbit, body at (r, 0, 0) with velocity
/// along +Y at the local circular speed. Shared by every recipe.
fn iss_circular_state() -> (DVec3, DVec3) {
    let r = R_EARTH + ALT_M;
    let v = (MU_EARTH / r).sqrt();
    (DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, v, 0.0))
}

/// Closed-form circular-orbit period at the recipe's 400 km radius.
/// Used to size SyntheticTimes cadences for the Kepler and drag scans.
fn orbital_period_s() -> f64 {
    2.0 * std::f64::consts::PI * ((R_EARTH + ALT_M).powi(3) / MU_EARTH).sqrt()
}

/// Number of integration ticks per orbital period at the given `dt`.
/// Matches the pre-recipe tier3 file's integer-truncated count exactly
/// so SyntheticTimes cadence reproduces the inline `step_n` loop count.
fn steps_per_orbit(dt: f64) -> usize {
    (orbital_period_s() / dt) as usize
}

/// Add a point-mass Earth gravity source with `t_inertial_pfix:
/// Some(IDENTITY)`. The identity transform is a no-op on the runner
/// side (and keeps the Earth-pfix frame present so the Bevy adapter's
/// atmosphere / planet-fixed lookup succeeds); see the module docstring
/// for the parity rationale.
fn add_earth_point_mass(sb: &mut SimulationBuilder) -> usize {
    sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    )
}

/// Shared 3-DOF Kepler scenario constructor. Used by the point-mass
/// conservation recipe and (with a non-zero `external_force`) by the
/// F = m·a impulse-response recipe. Mirrors the pre-recipe inline
/// `make_kepler_sim` helper field-for-field; the additional
/// `external_force` parameter funnels the impulse case's load through
/// `VehicleConfig` so the runner-side `SimBody.external_force` and the
/// Bevy-side `ExternalForceC` start at identical values without a
/// `pre_step` hook (the impulse window covers the full propagation
/// horizon, so no mid-run sign change is needed).
fn build_kepler_sim(
    pos: DVec3,
    vel: DVec3,
    mass: f64,
    dt: f64,
    external_force: DVec3,
) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = add_earth_point_mass(&mut sb);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: None,
        mass: Some(super::typed_helpers::mass_typed(&MassProperties::new(mass))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        external_force: Force::<RootInertial>::from_raw_si(external_force),
        ..Default::default()
    });
    sb
}

/// Shared 6-DOF Kepler scenario constructor. Used by tests 4, 6, and 7.
/// Mirrors the inline 6-DOF setup field-for-field, with
/// `external_force` and `external_torque` parameters routed through
/// `VehicleConfig` for the impulse-response cases.
fn build_kepler_6dof_sim(
    pos: DVec3,
    vel: DVec3,
    mass_props: MassProperties,
    ang_vel_body: DVec3,
    dt: f64,
    external_force: DVec3,
    external_torque: DVec3,
) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = add_earth_point_mass(&mut sb);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: Some(super::typed_helpers::rot_typed(&RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body,
        })),
        mass: Some(super::typed_helpers::mass_typed(&mass_props)),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        compute_gravity_gradient: false,
        external_force: Force::<RootInertial>::from_raw_si(external_force),
        external_torque: Torque::<astrodyn::BodyFrame<astrodyn::SelfRef>>::from_raw_si(
            external_torque,
        ),
        ..Default::default()
    });
    sb
}

// ─── Test 1: Point-mass Kepler conservation ───

const TEST1_DT_S: f64 = 10.0;
/// Three full orbits at `TEST1_DT_S = 10 s`. Pre-recipe
/// `tier3_dyncomp_point_mass_3dof_conservation` used the same integer-
/// truncated `(3 * period / dt) as usize` count.
fn test1_num_steps() -> usize {
    3 * steps_per_orbit(TEST1_DT_S)
}

fn build_point_mass_3dof_conservation(_init: &InitialConditions) -> SimulationBuilder {
    let (pos, vel) = iss_circular_state();
    build_kepler_sim(pos, vel, MASS_KG, TEST1_DT_S, DVec3::ZERO)
}

/// 3-DOF point-mass orbit propagated for 3 orbits. The tier3 sibling
/// asserts the specific orbital energy and angular momentum stay
/// conserved (any drift is RK4 truncation error).
pub fn point_mass_3dof_conservation() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_point_mass_3dof_conservation",
        scenario: build_point_mass_3dof_conservation,
        reference: CsvReference::SyntheticTimes {
            dt: TEST1_DT_S,
            num_steps: test1_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ─── Test 2: Point-mass Earth + third body produces non-conservation of h ───

const TEST2_DT_S: f64 = 10.0;
/// One full orbital period at `TEST2_DT_S = 10 s`. Pre-recipe
/// `tier3_dyncomp_point_mass_plus_thirdbody_conservation` used the
/// same integer-truncated `(period / dt) as usize` count.
fn test2_num_steps() -> usize {
    steps_per_orbit(TEST2_DT_S)
}

/// Sun position ≈ 1 AU at the obliquity-of-the-ecliptic angle (23.4°)
/// out of the equatorial plane. Pre-recipe placed the Sun here so its
/// differential acceleration produces a torque on the orbit about an
/// in-plane axis (nodal regression / inclination wobble); preserving
/// the same components keeps the third-body torque signature
/// identical.
fn sun_inertial_position() -> DVec3 {
    DVec3::new(R_EARTH_SUN * 0.9175, 0.0, R_EARTH_SUN * 0.3977)
}

/// Moon position ≈ 384 400 km out of the orbital (X-Y) plane along the
/// +Y / +Z combination the pre-recipe inline test used (so the +Y
/// orbital velocity sees a non-zero out-of-plane perturber from the
/// first step).
fn moon_inertial_position() -> DVec3 {
    DVec3::new(0.0, R_EARTH_MOON * 0.9063, R_EARTH_MOON * 0.4226)
}

fn build_point_mass_plus_thirdbody_conservation(_init: &InitialConditions) -> SimulationBuilder {
    let (pos, vel) = iss_circular_state();
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, TEST2_DT_S);
    let earth = add_earth_point_mass(&mut sb);
    let sun = sb.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                sun_inertial_position(),
            ),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    let moon = sb.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_MOON,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                moon_inertial_position(),
            ),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: None,
        mass: Some(super::typed_helpers::mass_typed(&MassProperties::new(
            MASS_KG,
        ))),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth, GravityGradient::Skip),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        ..Default::default()
    });
    sb
}

/// 3-DOF point-mass orbit with Sun + Moon third bodies. The tier3
/// sibling asserts orbital energy stays bounded over one orbit and
/// the angular-momentum vector has measurably tilted (third-body
/// torque > numerical noise).
pub fn point_mass_plus_thirdbody_conservation() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_point_mass_plus_thirdbody_conservation",
        scenario: build_point_mass_plus_thirdbody_conservation,
        reference: CsvReference::SyntheticTimes {
            dt: TEST2_DT_S,
            num_steps: test2_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ─── Test 3: drag leads to monotonic decay of SMA ───

const TEST3_DT_S: f64 = 10.0;
/// Total propagation horizon for the drag-decay scan: 5 orbits at
/// `TEST3_DT_S = 10 s`. The tier3 sibling samples SMA at each
/// orbital-period boundary, so the SyntheticTimes cadence must run
/// long enough to reach the fifth sample.
fn test3_num_steps() -> usize {
    5 * steps_per_orbit(TEST3_DT_S)
}

/// Constant atmospheric density (kg/m³) used for the drag-decay scan.
/// Boosted well above the realistic 400 km value so SMA decay is
/// dominant over RK4 truncation error within five orbits. Same value
/// the pre-recipe inline test used.
const TEST3_DENSITY: f64 = 1e-11;

fn build_drag_point_mass_monotonic_decay(_init: &InitialConditions) -> SimulationBuilder {
    let (pos, vel) = iss_circular_state();
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, TEST3_DT_S);
    let earth = add_earth_point_mass(&mut sb);
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Exponential(ExponentialAtmosphere {
                rho_0: TEST3_DENSITY,
                h_0: ALT_M,
                scale_height: 50_000.0,
            }),
            r_eq: R_EARTH,
            // Match the pre-recipe inline expression
            // `R_EARTH * (1.0 - 1.0 / 298.257_223_563)` exactly. The
            // preset `astrodyn::EARTH.shape.r_pol` evaluates to the
            // same value (the constant in `body_constants.rs` is
            // `EARTH_R_EQ * (1.0 - EARTH_FLAT_COEFF)` with
            // `EARTH_FLAT_COEFF = 1.0 / 298.257_223_563`), so the
            // preset keeps the value bit-identical without hard-coding
            // the flattening literal here.
            r_pol: astrodyn::EARTH.shape.r_pol,
            planet_omega: 0.0,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: Some(super::typed_helpers::rot_typed(&RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        })),
        mass: Some(super::typed_helpers::mass_typed(&MassProperties::new(
            MASS_KG,
        ))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        drag: Some(DragConfig {
            cd: 2.2,
            area: 20.0,
            constant_density: Some(TEST3_DENSITY),
        }),
        ..Default::default()
    });
    sb
}

/// 3-DOF (+identity rotation for `DragConfig` requirements) point-
/// mass orbit with constant-density drag. The tier3 sibling samples
/// the semi-major axis at each orbital period and asserts strict
/// monotonic decrease across five orbits.
pub fn drag_point_mass_monotonic_decay() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_drag_point_mass_monotonic_decay",
        scenario: build_drag_point_mass_monotonic_decay,
        reference: CsvReference::SyntheticTimes {
            dt: TEST3_DT_S,
            num_steps: test3_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ─── Test 4: torque-free rigid-body rotation conserves inertial H ───

const TEST4_DT_S: f64 = 0.5;
/// 60 s torque-free rotation propagation horizon. Pre-recipe
/// `tier3_dyncomp_6dof_rigid_body_invariance` used the same integer-
/// truncated `(60.0 / dt) as usize` step count.
fn test4_num_steps() -> usize {
    (60.0 / TEST4_DT_S) as usize
}

/// Asymmetric diagonal inertia for the torque-free recipe:
/// `I_x = 1000`, `I_y = I_z = 2500`. Off-axis initial omega exercises
/// all three Euler equations.
fn test4_inertia() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(1000.0, 0.0, 0.0),
        DVec3::new(0.0, 2500.0, 0.0),
        DVec3::new(0.0, 0.0, 2500.0),
    )
}

/// Initial body-frame angular velocity for the torque-free recipe
/// — tipped off the major axis to exercise Euler's coupling.
fn test4_omega0() -> DVec3 {
    DVec3::new(0.1, 0.02, 0.0)
}

fn build_rigid_body_invariance_6dof(_init: &InitialConditions) -> SimulationBuilder {
    let (pos, vel) = iss_circular_state();
    let mass_props = MassProperties::with_inertia(MASS_KG, test4_inertia(), DVec3::ZERO);
    build_kepler_6dof_sim(
        pos,
        vel,
        mass_props,
        test4_omega0(),
        TEST4_DT_S,
        DVec3::ZERO,
        DVec3::ZERO,
    )
}

/// 6-DOF asymmetric rigid body in a 400 km LEO orbit, no applied
/// torque (gravity gradient off). The tier3 sibling asserts the
/// inertial-frame angular-momentum vector is conserved while the
/// body-frame omega evolves under Euler's equations.
pub fn rigid_body_invariance_6dof() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_6dof_rigid_body_invariance",
        scenario: build_rigid_body_invariance_6dof,
        reference: CsvReference::SyntheticTimes {
            dt: TEST4_DT_S,
            num_steps: test4_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ─── Test 5: external force delta-v ───

const TEST5_DT_S: f64 = 1.0;
/// External-force window duration (s). The force is constant over the
/// whole propagation horizon, so the SyntheticTimes count covers the
/// full window. Pre-recipe `tier3_dyncomp_external_force_impulse_response`
/// computed the count as `(force_duration / dt) as usize`.
const TEST5_FORCE_DURATION_S: f64 = 10.0;
fn test5_num_steps() -> usize {
    (TEST5_FORCE_DURATION_S / TEST5_DT_S) as usize
}

/// Inertial-frame external force applied throughout the impulse window
/// — pre-recipe used a pure +X 50 N vector to make the closed-form
/// `dv = F·t/m` direction check trivial.
///
/// Exposed `pub` so the matching tier3 sibling can read it back when
/// computing the expected `F·t/m` impulse without having to also
/// duplicate the literal at the call site.
pub const EXTERNAL_FORCE_IMPULSE_INERTIAL_N: DVec3 = DVec3::new(50.0, 0.0, 0.0);

/// Vehicle mass used by the F = m·a impulse-response recipe. Exposed
/// so the tier3 sibling can read it back when computing the expected
/// `F·t/m` impulse magnitude.
pub const EXTERNAL_FORCE_IMPULSE_MASS_KG: f64 = MASS_KG;

/// Force-window duration used by the F = m·a impulse-response recipe.
/// Equals the SyntheticTimes `num_steps * dt`. Exposed so the tier3
/// sibling can compute the closed-form `F·t/m` impulse without
/// dividing back through the SyntheticTimes variant.
pub const EXTERNAL_FORCE_IMPULSE_DURATION_S: f64 = TEST5_FORCE_DURATION_S;

fn build_external_force_impulse_response(_init: &InitialConditions) -> SimulationBuilder {
    let (pos, vel) = iss_circular_state();
    build_kepler_sim(
        pos,
        vel,
        MASS_KG,
        TEST5_DT_S,
        EXTERNAL_FORCE_IMPULSE_INERTIAL_N,
    )
}

/// Inertial-frame force applied to a 3-DOF orbiting body. The tier3
/// sibling builds this recipe alongside a no-force reference (the
/// [`external_force_impulse_kepler_reference`] recipe sized to match),
/// propagates both for the force window, and asserts the difference
/// of their velocity vectors matches the closed-form `F·t/m` impulse
/// (`< 1e-4` relative).
pub fn external_force_impulse_response() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_external_force_impulse_response",
        scenario: build_external_force_impulse_response,
        reference: CsvReference::SyntheticTimes {
            dt: TEST5_DT_S,
            num_steps: test5_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// No-force reference sibling for [`external_force_impulse_response`].
/// Same translational state, mass, and integrator timestep — only the
/// external force is zero. The impulse test subtracts this run's final
/// velocity from the forced run's final velocity to isolate the force
/// contribution from the gravity-induced delta-v over the same window.
pub fn external_force_impulse_kepler_reference() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_external_force_impulse_kepler_reference",
        scenario: |_init| {
            let (pos, vel) = iss_circular_state();
            build_kepler_sim(pos, vel, MASS_KG, TEST5_DT_S, DVec3::ZERO)
        },
        reference: CsvReference::SyntheticTimes {
            dt: TEST5_DT_S,
            num_steps: test5_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ─── Test 6: external torque delta-omega ───

const TEST6_DT_S: f64 = 1.0;
/// Torque window duration (s). Same shape as the force-impulse case:
/// torque is constant over the whole propagation horizon.
const TEST6_TORQUE_DURATION_S: f64 = 10.0;
fn test6_num_steps() -> usize {
    (TEST6_TORQUE_DURATION_S / TEST6_DT_S) as usize
}

/// Asymmetric diagonal inertia for the torque-impulse recipe:
/// `I_x = 1000`, `I_y = I_z = 2500`. Torque applied along body +X so
/// the closed-form `omega_x = τ·t/I_x` is the only non-zero component.
fn test6_inertia() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(EXTERNAL_TORQUE_IMPULSE_INERTIA_X_KGM2, 0.0, 0.0),
        DVec3::new(0.0, 2500.0, 0.0),
        DVec3::new(0.0, 0.0, 2500.0),
    )
}

/// Body-frame torque vector applied throughout the torque-impulse
/// window — pre-recipe used a pure +X 10 N·m vector.
///
/// Exposed `pub` so the matching tier3 sibling can read it back when
/// computing the expected `τ·t/I_x` delta-omega without duplicating
/// the literal.
pub const EXTERNAL_TORQUE_IMPULSE_BODY_NM: DVec3 = DVec3::new(10.0, 0.0, 0.0);

/// `I_x` principal moment of inertia for the torque-impulse recipe.
/// Exposed so the tier3 sibling can compute the expected
/// `omega_x = τ·t/I_x` without re-deriving the inertia from the
/// typed-bridge accessors.
pub const EXTERNAL_TORQUE_IMPULSE_INERTIA_X_KGM2: f64 = 1000.0;

/// Torque-window duration used by the τ = I·α impulse-response recipe.
/// Equals the SyntheticTimes `num_steps * dt`.
pub const EXTERNAL_TORQUE_IMPULSE_DURATION_S: f64 = TEST6_TORQUE_DURATION_S;

fn build_external_torque_impulse_response(_init: &InitialConditions) -> SimulationBuilder {
    let (pos, vel) = iss_circular_state();
    let mass_props = MassProperties::with_inertia(MASS_KG, test6_inertia(), DVec3::ZERO);
    build_kepler_6dof_sim(
        pos,
        vel,
        mass_props,
        DVec3::ZERO,
        TEST6_DT_S,
        DVec3::ZERO,
        EXTERNAL_TORQUE_IMPULSE_BODY_NM,
    )
}

/// Body-frame torque applied to a 6-DOF body initially at rest
/// (rotationally). The tier3 sibling asserts the final body-frame
/// angular velocity matches the closed-form `omega_x = τ·t/I_x`
/// and the perpendicular components stay below numerical noise.
pub fn external_torque_impulse_response() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_external_torque_impulse_response",
        scenario: build_external_torque_impulse_response,
        reference: CsvReference::SyntheticTimes {
            dt: TEST6_DT_S,
            num_steps: test6_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ─── Test 7: major-axis spin stability (intermediate-axis theorem) ───

const TEST7_DT_S: f64 = 0.1;
/// 60 s spin propagation (`600` ticks at `dt = 0.1 s`).
fn test7_num_steps() -> usize {
    (60.0 / TEST7_DT_S) as usize
}

/// Asymmetric diagonal inertia with the largest moment along +Z.
/// `I_z = 2500` is the major axis; `I_x = 500`, `I_y = 1000`. Pre-
/// recipe `tier3_dyncomp_attitude_stability_major_axis` used the same
/// ordering so the intermediate-axis-theorem prediction (stable spin
/// about +Z) is the regime the assertion checks.
fn test7_inertia() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(500.0, 0.0, 0.0),
        DVec3::new(0.0, 1000.0, 0.0),
        DVec3::new(0.0, 0.0, 2500.0),
    )
}

/// Initial body-frame omega: 1 rad/s spin about +Z with 1% perpendicular
/// perturbations on +X and +Y. The pre-recipe inline test used the same
/// triple so the bound on `|omega_perp|` stays comparable.
fn test7_omega0() -> DVec3 {
    DVec3::new(0.01, 0.01, 1.0)
}

fn build_attitude_stability_major_axis(_init: &InitialConditions) -> SimulationBuilder {
    let (pos, vel) = iss_circular_state();
    let mass_props = MassProperties::with_inertia(MASS_KG, test7_inertia(), DVec3::ZERO);
    build_kepler_6dof_sim(
        pos,
        vel,
        mass_props,
        test7_omega0(),
        TEST7_DT_S,
        DVec3::ZERO,
        DVec3::ZERO,
    )
}

/// 6-DOF body spinning about its major principal axis (+Z) with a
/// small perpendicular perturbation. The tier3 sibling asserts the
/// perpendicular components of body-frame omega stay bounded
/// (intermediate-axis theorem — stable major-axis rotation).
pub fn attitude_stability_major_axis() -> VerificationCase {
    VerificationCase {
        name: "tier3_dyncomp_attitude_stability_major_axis",
        scenario: build_attitude_stability_major_axis,
        reference: CsvReference::SyntheticTimes {
            dt: TEST7_DT_S,
            num_steps: test7_num_steps(),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}
