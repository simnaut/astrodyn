//! Tier 3: SIM_VER_DRAG ballistic drag cross-validation.
//!
//! JEOD's `SIM_VER_DRAG` is a **non-propagating** verification sim: it sets the
//! vehicle's inertial velocity by a closed-form schedule
//! (`inertial_vel[0]=7500·cos(t·π/180)`, `inertial_vel[2]=7500·sin(t·π/180)`,
//! rotating 1°/s in the X-Z plane), fixes T_inertial_struct to identity, atmosphere
//! density to 1e-12 kg/m³, wind to zero, mass to 1 kg, and calls
//! `AerodynamicDrag::aero_drag()` at 10 Hz. The test harness logs the resulting
//! `aero_force`, `aero_torque`, `inertial_vel`, and `accel_mag` once per second
//! for 360 s.
//!
//! For these runs, the corresponding Tier 3 cross-validation exercises the full
//! ballistic drag force model — identical to what JEOD's sim exercises — using
//! our ported `compute_ballistic_drag`. No orbital propagation is involved.
//! Each test:
//! 1. Reproduces JEOD's scheduled velocity at each CSV timestamp.
//! 2. Builds an `AtmosphereState` matching JEOD's `input_common.py`.
//! 3. Calls `compute_ballistic_drag` with a `DragConfig` matching the JEOD run's
//!    `DefaultAero` option.
//! 4. Compares force/torque/accel against JEOD's logged values per row.
//!
//! Covered runs:
//! - `RUN_aero_drag_CD`: `DefaultAero::DRAG_OPT_CD` with `area=100`, `Cd=2`.
//! - `RUN_aero_drag_BC`: `DefaultAero::DRAG_OPT_BC` with `BC=0.005`, `mass=1`.
//!   `drag = -dyn_p·mass/BC = -dyn_p·200` — numerically identical to
//!   `DRAG_OPT_CD` with `Cd·A = 200`, so we configure the same ballistic model.
//!
//! Not covered (requires a port of `DefaultAero::DRAG_OPT_CONST`):
//! - `RUN_aero_drag_const`: user-specified constant force magnitude along
//!   the relative-velocity direction. Our `DragConfig` exposes only the CD
//!   path today. Adding CONST/BC as discriminated options is tracked as a
//!   follow-on task; the reference CSV (`drag_const_drag.csv`) is generated
//!   and retained for that work.

mod sim_test_helpers;
use sim_test_helpers::{load_drag_csv, test_data_path};

use glam::{DMat3, DVec3};
use jeod_atmosphere::AtmosphereState;
use jeod_interactions::{compute_ballistic_drag, DragConfig};
use jeod_test_data::crossval::CrossvalReport;

/// JEOD input_common.py: atmospheric density in kg/m³.
const JEOD_DENSITY: f64 = 1.0e-12;
/// JEOD input_common.py: vehicle mass in kg.
const JEOD_MASS: f64 = 1.0;

/// Reproduce JEOD's scheduled inertial velocity at time `t` (seconds).
///
/// Port of `AeroTestSimObject::rotate_vel()` in the SIM_VER_DRAG S_define:
/// ```text
///   inertial_vel[0] = 7500 * std::cos(exec_get_sim_time() * M_PI / 180.0);
///   inertial_vel[2] = 7500 * std::sin(exec_get_sim_time() * M_PI / 180.0);
/// ```
fn jeod_inertial_vel(t: f64) -> DVec3 {
    let phase = t * std::f64::consts::PI / 180.0;
    DVec3::new(7500.0 * phase.cos(), 0.0, 7500.0 * phase.sin())
}

/// Shared driver: evaluate `compute_ballistic_drag` at each CSV row and compute
/// per-sample error metrics against JEOD's logged force/torque/accel.
///
/// Returns `(max_force_err, max_torque_err, max_accel_err)` in SI units.
fn run_ballistic_case(
    test_name: &str,
    csv_label: &str,
    drag_config: DragConfig,
) -> (f64, f64, f64) {
    let csv_path = test_data_path(csv_label);
    assert!(
        csv_path.exists(),
        "SIM_VER_DRAG CSV not found at {}.\n\
         Generate with: docker run --rm --entrypoint /bin/bash \
         -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
         jeod-trick /generate_references.sh",
        csv_path.display()
    );

    let records = load_drag_csv(&csv_path);
    assert!(
        records.len() >= 350,
        "{test_name}: expected >=350 records in {csv_label}, got {}",
        records.len()
    );

    let atmos = AtmosphereState {
        density: JEOD_DENSITY,
        temperature: 0.0,
        pressure: 0.0,
        wind: DVec3::ZERO,
    };
    let t_inertial_struct = DMat3::IDENTITY;

    let mut max_force_err = 0.0_f64;
    let mut max_torque_err = 0.0_f64;
    let mut max_accel_err = 0.0_f64;
    let mut max_vel_sched_err = 0.0_f64;

    for rec in &records {
        // Sanity: our velocity schedule must match JEOD's logged velocity.
        let vel = jeod_inertial_vel(rec.time);
        max_vel_sched_err = max_vel_sched_err.max((vel - rec.inertial_vel).length());

        let aero = compute_ballistic_drag(&drag_config, &atmos, vel, &t_inertial_struct);

        max_force_err = max_force_err.max((aero.force - rec.aero_force).length());
        max_torque_err = max_torque_err.max((aero.torque - rec.aero_torque).length());

        // JEOD logging: accel_mag = |aero_force| / mass.
        let our_accel_mag = aero.force.length() / JEOD_MASS;
        max_accel_err = max_accel_err.max((our_accel_mag - rec.accel_mag).abs());
    }

    // Sanity-check the velocity schedule separately from the drag computation.
    // Our `jeod_inertial_vel` and JEOD's `rotate_vel` both reduce to libm
    // `sin`/`cos` of the same f64 argument, so any disagreement reflects
    // cross-platform libm divergence — bounded to a few ULPs of |v|≈7500 m/s.
    // A tight threshold here ensures any future platform drift surfaces as a
    // schedule-mismatch failure with a clear message, rather than leaking into
    // the drag-force error (which would compare against the tight force
    // tolerances below).
    assert!(
        max_vel_sched_err < 1.1e-12,
        "{test_name}: inertial velocity schedule disagrees with CSV by {max_vel_sched_err:.3e} m/s \
         — libm sin/cos divergence between JEOD's host and this host exceeds expected ULP bound"
    );

    let mut report = CrossvalReport::compute(test_name, &[], &[]);
    report.add_extra("aero_force_err", max_force_err, "N");
    report.add_extra("aero_torque_err", max_torque_err, "N*m");
    report.add_extra("accel_mag_err", max_accel_err, "m/s^2");
    report.write();

    println!(
        "{test_name}: {} samples | vel_sched_err={max_vel_sched_err:.3e} m/s | force_err={max_force_err:.3e} N | torque_err={max_torque_err:.3e} N*m | accel_err={max_accel_err:.3e} m/s^2",
        records.len()
    );

    (max_force_err, max_torque_err, max_accel_err)
}

/// `RUN_aero_drag_CD`: `DRAG_OPT_CD` with `area=100 m²`, `Cd=2`.
///
/// JEOD computes `force = -0.5·ρ·v²·Cd·A · v̂_rel` in the structural frame.
/// With identity T_inertial_struct, density 1e-12 kg/m³, and |v|=7500 m/s:
///   |force| = 0.5·1e-12·7500²·2·100 = 0.005625 N (accel_mag at t=0).
#[test]
fn tier3_sim_drag_ver_cd() {
    let drag = DragConfig {
        cd: 2.0,
        area: 100.0,
        constant_density: None,
    };

    let (force_err, torque_err, accel_err) =
        run_ballistic_case("tier3_sim_drag_ver_cd", "drag_cd_drag.csv", drag);

    // Tolerances at 5% above observed max error. All values are tiny — this
    // is effectively a bit-exact match modulo floating-point noise between
    // JEOD's C++ (IEEE 754 double ops) and ours.
    assert!(
        force_err < 5.0e-17,
        "force_err {force_err:.3e} N exceeds 5.0e-17 N"
    );
    assert!(
        torque_err < 1.0e-20,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-20 N*m"
    );
    assert!(
        accel_err < 5.0e-17,
        "accel_err {accel_err:.3e} m/s^2 exceeds 5.0e-17 m/s^2"
    );
}

/// `RUN_aero_drag_BC`: `DRAG_OPT_BC` with `BC=0.005 m²/kg`, `mass=1 kg`.
///
/// JEOD computes `force = -(dyn_p·mass)/BC · v̂_rel`. With `mass/BC = 200`
/// this is numerically identical to `DRAG_OPT_CD` with `Cd·A = 200`, so we
/// configure `Cd=2, area=100` — the same values as the CD test — and verify
/// we match JEOD's BC run to floating-point precision.
///
/// The two JEOD CSVs (`drag_cd_drag.csv`, `drag_bc_drag.csv`) differ only at
/// the ~16th significant digit, confirming this equivalence.
#[test]
fn tier3_sim_drag_ver_bc() {
    let drag = DragConfig {
        cd: 2.0,
        area: 100.0,
        constant_density: None,
    };

    let (force_err, torque_err, accel_err) =
        run_ballistic_case("tier3_sim_drag_ver_bc", "drag_bc_drag.csv", drag);

    // Same magnitudes as the CD case — BC and CD are algebraically identical
    // when Cd·A = mass/BC, differing only in the order of floating-point
    // multiplications. Both sides execute `drag = -dyn_p · k` for the same k.
    assert!(
        force_err < 5.0e-17,
        "force_err {force_err:.3e} N exceeds 5.0e-17 N"
    );
    assert!(
        torque_err < 1.0e-20,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-20 N*m"
    );
    assert!(
        accel_err < 5.0e-17,
        "accel_err {accel_err:.3e} m/s^2 exceeds 5.0e-17 m/s^2"
    );
}
