//! NESC Lunar Check Case 8 (NRHO) — initial conditions, scenario builder,
//! and reference-trajectory loader.
//!
//! Source: <https://nescacademy.nasa.gov/flightsim/2023/cc08>.
//! Body model: <https://nescacademy.nasa.gov/flightsim/2023/bodies#apollo-model>.
//!
//! - Central body: **Moon** (8×8 GRAIL gravity).
//! - Third bodies: Earth + Sun (gravity perturbations only).
//! - Ephemeris: DE440 (per case spec; see [`crate`] readme for the
//!   procurement workflow until the BSP is committed).
//! - Vehicle: Apollo body model (16 642 kg + full inertia tensor + CoM offset).
//! - Duration: 7 days at 60 s checkpoint cadence.
//! - Forces: gravity only (no SRP, no drag, no gravity-gradient torque).
//! - Frame: Moon-Centered Inertial (MCI).

use astrodyn::recipes::{epoch, moon, vehicle};
use astrodyn::{
    EphemerisBody, GravityControl, JeodQuat, RootInertial, RotationalState, SimulationBuilder,
    TranslationalStateTyped, Vec3Ext, VehicleBuilder, EARTH, SUN,
};
use glam::DVec3;

/// CC8 epoch as published by NESC (UTC).
pub const EPOCH_UTC: &str = "2026-01-28T06:42:03.51Z";

/// CC8 initial position in Moon-Centered Inertial frame, in metres.
///
/// The values are the **full-precision** ICs that 6 of 8 NESC participating
/// simulations (sim_01, sim_02, sim_04, sim_05, sim_07, sim_08) all reported
/// at their t=0 row. They round to the spec-published display values
/// `(-5_838_140.151, 2_538_924.866, 1_055_566.197)` but the propagation
/// must start from the unrounded values to land on the same trajectory.
pub const INITIAL_POSITION_MCI_M: [f64; 3] = [
    -5_838_140.150_980_704,
    2_538_924.866_208_574,
    1_055_566.196_900_589,
];

/// CC8 initial velocity in Moon-Centered Inertial frame, in metres/second.
///
/// Full-precision sim_01 t=0 values; rounds to spec `(-685.576, 751.298, -621.914)`.
pub const INITIAL_VELOCITY_MCI_MPS: [f64; 3] = [
    -685.576_059_820_879_5,
    751.297_639_180_154_4,
    -621.913_991_301_903_1,
];

/// CC8 initial attitude quaternion `(W, X, Y, Z)` — body-from-inertial,
/// scalar-first / left-transformation. Matches [`JeodQuat::new`] argument
/// order verbatim; do **not** reorder for `glam::DQuat::new` (which is
/// `(X, Y, Z, W)`).
///
/// Full-precision sim_01 t=0 values; rounds to spec
/// `(0.6461, 0.3344, 0.6855, 0.0282)`.
pub const INITIAL_QUATERNION_WXYZ: [f64; 4] = [
    0.646_080_387_557_766_1,
    0.334_383_101_748_747_3,
    0.685_546_322_693_561_1,
    0.028_183_568_252_256_43,
];

/// CC8 initial body-frame angular velocity in degrees/second.
///
/// Convert to rad/s before feeding into [`RotationalState::ang_vel_body`].
pub const INITIAL_BODY_RATE_DEG_PER_S: [f64; 3] = [0.001, 0.0, 0.0];

/// CC8 propagation duration in seconds (7 days).
pub const DURATION_S: f64 = 604_800.0;

/// CC8 reference checkpoint cadence in seconds.
pub const CHECKPOINT_CADENCE_S: f64 = 60.0;

/// Reference trajectory CSV filename under `test_data/`.
///
/// Produced by the `extract_nesc` regen binary; see this crate's
/// `README.md` for the procurement workflow.
pub const REFERENCE_CSV: &str = "cc8_nrho_reference.csv";

/// Compose the CC8 scenario into a [`SimulationBuilder`].
///
/// Both [`astrodyn_runner::Simulation`] and the Bevy adapter consume the
/// same builder shape. The runner-side test calls `cc8_builder().build()`;
/// the Bevy parity wrapper calls `cc8_builder().populate_app::<Moon>(&mut app)`.
///
/// # Force model
///
/// - Moon central, 8×8 GRAIL truncation (the GRAIL150 fixture is
///   evaluated up to degree=order=8 at the call site).
/// - Earth + Sun third bodies, positions driven by DE440 each step.
/// - No SRP, no drag, no gravity-gradient torque.
///
/// # Vehicle
///
/// 6-DoF Apollo body (`recipes::vehicle::nesc_apollo_lm()`):
/// 16 642 kg + full inertia tensor (with off-diagonal terms) + CoM offset.
/// Translational and rotational ICs match the NESC publication verbatim.
///
/// # Panics
///
/// Panics if `recipes::ephemeris::de440()` fails to load — until the
/// DE440 BSP is committed under `crates/astrodyn_ephemeris/assets/`,
/// this builder is unusable. See the crate `README.md` for the
/// procurement workflow.
pub fn cc8_builder() -> SimulationBuilder {
    // 1. Time. UTC string + leap-second + TAI-UTC handled inside the recipe.
    let time = epoch::at_iso(EPOCH_UTC);
    let mut sb = SimulationBuilder::new(time, 1.0);

    // 2. Moon (central) — GRAIL150 fixture + libration rotation. The
    // GravityControl below truncates the SH evaluation to 8×8 per the
    // case spec.
    let moon_idx = sb.add_source("Moon", moon::grail150_with_libration());

    // 3. Earth + Sun third bodies. One call per source — the recipe
    // wires both the source entry and the per-step ephemeris update.
    let earth_idx = sb.add_third_body_with_ephemeris(
        "Earth",
        &EARTH,
        EphemerisBody::Earth,
        EphemerisBody::Moon,
    );
    let sun_idx =
        sb.add_third_body_with_ephemeris("Sun", &SUN, EphemerisBody::Sun, EphemerisBody::Moon);
    // `moon::grail150_with_libration()` registers the Moon source with
    // `RotationModel::MoonDE421`, which queries a BPC kernel each step
    // for the libration matrix. `de440_with_moon_pa()` loads the BPC
    // alongside the DE440 SPK so the per-step rotation update has data
    // to read; bare `de440()` is SPK-only and would panic at the first
    // step.
    sb = sb.ephemeris(
        astrodyn::recipes::ephemeris::de440_with_moon_pa().expect("DE440 + Moon BPC ephemeris"),
    );

    // 4. Translational IC (Moon-centered inertial).
    let trans = TranslationalStateTyped::<RootInertial> {
        position: DVec3::from(INITIAL_POSITION_MCI_M).m_at::<RootInertial>(),
        velocity: DVec3::from(INITIAL_VELOCITY_MCI_MPS).m_per_s_at::<RootInertial>(),
    };

    // 5. Rotational IC. NESC publishes (W, X, Y, Z) — scalar-first,
    // matches JeodQuat directly. Body rate in deg/s → rad/s.
    let [qw, qx, qy, qz] = INITIAL_QUATERNION_WXYZ;
    let [wx_dps, wy_dps, wz_dps] = INITIAL_BODY_RATE_DEG_PER_S;
    let rot = RotationalState {
        quaternion: JeodQuat::new(qw, qx, qy, qz),
        ang_vel_body: DVec3::new(wx_dps, wy_dps, wz_dps).map(f64::to_radians),
    };

    // 6. Body. `nesc_apollo_lm()` returns the published mass + inertia
    // tensor (incl. off-diagonal) + CoM offset.
    let vehicle_cfg = VehicleBuilder::new()
        .with_translational(trans)
        .sixdof(rot, vehicle::nesc_apollo_lm())
        .rk4()
        .gravity(GravityControl::new_nonspherical(moon_idx, 8, 8, false))
        .gravity(GravityControl::new_third_body(earth_idx))
        .gravity(GravityControl::new_third_body(sun_idx))
        .build();
    sb.add_body(vehicle_cfg);

    sb
}

/// Load the committed CC8 reference trajectory.
///
/// Reads CSV columns `time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, qw, qx, qy, qz, wx, wy, wz`
/// (units: s, m, m/s, dimensionless quaternion components, rad/s).
///
/// # Panics
///
/// Panics with a clear message if `test_data/cc8_nrho_reference.csv`
/// is missing. Per the *No Half-Baked Implementations* rule, the test
/// asserts (panics) when required data is absent — it never skips
/// gracefully. See this crate's `README.md` for the regen workflow.
pub fn cc8_reference() -> Vec<crate::StateLog> {
    let path = crate::test_data_path(REFERENCE_CSV);
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read CC8 reference CSV from {}: {e}.\n\
             Generate it via:\n  cargo run -p astrodyn_verif_nesc --bin extract_nesc \
             -- --nesc-home <PATH>\n\
             See crates/astrodyn_verif_nesc/README.md for the canonical \
             release pin and the source URL.",
            path.display()
        )
    });
    parse_cc8_csv(&content)
}

fn parse_cc8_csv(content: &str) -> Vec<crate::StateLog> {
    let mut out = Vec::new();
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 14,
            "cc8 reference: line {} has {} fields (need >= 14: time, pos[3], vel[3], quat[4], ang_vel[3])",
            lineno + 1,
            f.len()
        );
        // Skip the header row by detecting a non-numeric first field
        // ("time" instead of a t value). Robust against any number of
        // leading comment / blank lines.
        let Ok(time) = f[0].trim().parse::<f64>() else {
            continue;
        };
        let p = |idx: usize| -> f64 {
            f[idx].trim().parse().unwrap_or_else(|e| {
                panic!("cc8 reference line {}: column {idx} parse: {e}", lineno + 1)
            })
        };
        out.push(crate::StateLog {
            time,
            position: Some(DVec3::new(p(1), p(2), p(3))),
            velocity: Some(DVec3::new(p(4), p(5), p(6))),
            // Canonical column layout: qw, qx, qy, qz at columns 7, 8, 9, 10.
            // glam::DQuat::from_xyzw expects (x, y, z, w) — translate.
            quaternion: Some(glam::DQuat::from_xyzw(p(8), p(9), p(10), p(7))),
            ang_vel: Some(DVec3::new(p(11), p(12), p(13))),
            ..Default::default()
        });
    }
    out
}
