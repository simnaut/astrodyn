//! Tier 3: SIM_VER_DRAG flat-plate drag cross-validation.
//!
//! JEOD's `SIM_VER_DRAG` is a **non-propagating** verification sim (see
//! `tier3_sim_drag_ver.rs` for background). This test file exercises the
//! flat-plate path: `use_default_behavior = False` + a user-supplied
//! `AeroSurface` of `FlatPlateAeroFacet` instances. JEOD evaluates
//! `FlatPlateAeroFacet::aerodrag_force()` per facet and sums force/torque.
//!
//! We reproduce JEOD's scheduled velocity at each CSV timestamp and call our
//! ported `compute_flat_plate_aero` with the same facet geometry, gas state,
//! and temperature configuration. Total force and torque are then compared
//! against JEOD's logged values row-by-row.
//!
//! Covered runs (match JEOD `SIM_VER_DRAG` `Modified_data/` configurations):
//! - `RUN_one_plate_accel_spec_max_coef`   — `Specular`        (two plates at origin, ε=1.0)
//! - `RUN_one_plate_accel_diff_max_coef`   — `Diffuse`         (two plates at origin, ε=0.0)
//! - `RUN_one_plate_accel_mixed_eps05_max_coef` — `Mixed{ε=0.5}` (two plates at origin)
//! - `RUN_one_plate_accel_calc_coef_eps00` — `CalcCoef{ε=0.0}` (two plates at origin)
//! - `RUN_one_plate_accel_calc_coef_eps05` — `CalcCoef{ε=0.5}` (two plates at origin)
//! - `RUN_one_plate_accel_calc_coef_eps1`  — `CalcCoef{ε=1.0}` (two plates at origin)
//! - `RUN_orbiter`                         — `CalcCoef{ε=0.0}` (6-plate shuttle orbiter)
//!
//! Not covered (requires JEOD's `calculate_drag_coef=False` pre-set coefficient
//! path, which our flat-plate port currently does not expose):
//! - `RUN_one_plate_torque` — uses hardcoded `drag_coef_norm/tang/spec/diff=5.0`
//!   instead of computing coefficients from speed ratio and incidence. Our
//!   `compute_single_facet` always computes coefficients. The reference CSV
//!   (`drag_one_plate_torque_drag.csv`) is still generated for a future port
//!   of the pre-set-coefficients path.

mod sim_test_helpers;
use sim_test_helpers::{load_drag_csv, test_data_path};

use glam::DVec3;
use jeod_interactions::{compute_flat_plate_aero, AeroCoeffMethod, AeroFacet, AeroGasParams};
use jeod_test_data::crossval::CrossvalReport;

// ── JEOD SIM_VER_DRAG configuration (Modified_data/input_common.py) ──

/// Gas constant R in N·m/(kg·K) — JEOD `aero_drag.param.gas_const`.
const JEOD_GAS_CONST: f64 = 287.0;
/// Freestream temperature in K — JEOD `aero_drag.param.temp_free_stream`.
const JEOD_TEMP_FREE_STREAM: f64 = 1487.0;
/// Atmospheric density in kg/m³ — JEOD `aero_test.atmos_state.density`.
const JEOD_DENSITY: f64 = 1.0e-12;
/// Vehicle mass in kg — JEOD `aero_test.mass` for one-plate runs.
const JEOD_MASS_ONE_PLATE: f64 = 1.0;
/// Vehicle mass in kg — JEOD `aero_test.mass` for the orbiter run.
const JEOD_MASS_ORBITER: f64 = 91589.71;
/// Shuttle orbiter CoG (m) in the structural frame — JEOD `aero_test.center_grav`.
/// From `Modified_data/shuttle_plate_orbiter.py` via `RUN_orbiter/input.py`:
///   center_grav = [1098.0 * 0.0254, 0.0, 372.0 * 0.0254]
const JEOD_CENTER_GRAV_ORBITER: DVec3 = DVec3::new(1098.0 * 0.0254, 0.0, 372.0 * 0.0254);
/// Flat-plate surface temperature (K) — JEOD `FlatPlate::temperature` for every
/// facet in SIM_VER_DRAG `Modified_data/*_plate_*.py`.
const JEOD_PLATE_TEMP: f64 = 70.0;

/// Reproduce JEOD's scheduled inertial velocity at time `t` (seconds). See
/// `tier3_sim_drag_ver.rs::jeod_inertial_vel` for reference.
fn jeod_inertial_vel(t: f64) -> DVec3 {
    let phase = t * std::f64::consts::PI / 180.0;
    DVec3::new(7500.0 * phase.cos(), 0.0, 7500.0 * phase.sin())
}

/// Build the two-plate surface from `Modified_data/two_sided_plate.py`.
///
/// Two facets with opposite normals (±X), both at `[0, 0, z_pos]`, area 1 m²,
/// plate temperature 70 K. The SIM_VER_DRAG `two_sided_plate(z_pos=0.0, …)`
/// helper is used by all `RUN_one_plate_accel_*` runs.
fn two_sided_plate_facets(method: AeroCoeffMethod, z_pos: f64) -> Vec<AeroFacet> {
    let cp = DVec3::new(0.0, 0.0, z_pos);
    vec![
        AeroFacet {
            area: 1.0,
            normal: DVec3::new(1.0, 0.0, 0.0),
            center_pressure: cp,
            coeff_method: method,
            temperature: JEOD_PLATE_TEMP,
        },
        AeroFacet {
            area: 1.0,
            normal: DVec3::new(-1.0, 0.0, 0.0),
            center_pressure: cp,
            coeff_method: method,
            temperature: JEOD_PLATE_TEMP,
        },
    ]
}

/// Build the six-plate shuttle orbiter surface from
/// `Modified_data/shuttle_plate_orbiter.py` (`RUN_orbiter`).
///
/// All positions are converted from inches to metres (× 0.0254). All facets
/// share `temperature=70 K`, material `flat_plate_material`, and
/// `coef_method=CalcCoef` with `epsilon=0.0`.
fn orbiter_facets() -> Vec<AeroFacet> {
    let method = AeroCoeffMethod::calc_coef(0.0);
    let in_to_m = 0.0254;
    let make = |pos_in: [f64; 3], area: f64, normal: DVec3| AeroFacet {
        area,
        normal,
        center_pressure: DVec3::new(
            pos_in[0] * in_to_m,
            pos_in[1] * in_to_m,
            pos_in[2] * in_to_m,
        ),
        coeff_method: method,
        temperature: JEOD_PLATE_TEMP,
    };
    vec![
        make([1255.0, 0.0, 383.4], 119.4454385, DVec3::new(1.0, 0.0, 0.0)),
        make([1069.0, 0.0, 396.5], 229.91644, DVec3::new(0.0, 1.0, 0.0)),
        make([1059.6, 0.0, 332.0], 454.4538, DVec3::new(0.0, 0.0, 1.0)),
        make(
            [1255.0, 0.0, 383.4],
            119.4454385,
            DVec3::new(-1.0, 0.0, 0.0),
        ),
        make([1069.0, 0.0, 396.5], 229.91644, DVec3::new(0.0, -1.0, 0.0)),
        make([1059.6, 0.0, 332.0], 454.4538, DVec3::new(0.0, 0.0, -1.0)),
    ]
}

struct FlatPlateCase {
    test_name: &'static str,
    csv_label: &'static str,
    facets: Vec<AeroFacet>,
    center_grav: DVec3,
    mass: f64,
}

/// Evaluate `compute_flat_plate_aero` at each CSV row; return per-test error
/// metrics against JEOD's logged force/torque/accel.
///
/// Returns `(max_force_err, max_torque_err, max_accel_err)` in SI units.
fn run_flat_plate_case(case: &FlatPlateCase) -> (f64, f64, f64) {
    let csv_path = test_data_path(case.csv_label);
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
        "{}: expected >=350 records in {}, got {}",
        case.test_name,
        case.csv_label,
        records.len()
    );

    let gas = AeroGasParams {
        gas_const: JEOD_GAS_CONST,
        temp_free_stream: JEOD_TEMP_FREE_STREAM,
    };

    let mut max_force_err = 0.0_f64;
    let mut max_torque_err = 0.0_f64;
    let mut max_accel_err = 0.0_f64;
    let mut max_vel_sched_err = 0.0_f64;

    for rec in &records {
        let vel = jeod_inertial_vel(rec.time);
        max_vel_sched_err = max_vel_sched_err.max((vel - rec.inertial_vel).length());

        // T_inertial_struct = I in SIM_VER_DRAG, so structural velocity equals
        // inertial velocity and dynamic pressure uses |vel|.
        let rel_vel_mag = vel.length();
        let dynamic_pressure = 0.5 * JEOD_DENSITY * rel_vel_mag * rel_vel_mag;

        let result =
            compute_flat_plate_aero(&case.facets, vel, dynamic_pressure, &gas, case.center_grav);

        max_force_err = max_force_err.max((result.force - rec.aero_force).length());
        max_torque_err = max_torque_err.max((result.torque - rec.aero_torque).length());

        let our_accel_mag = result.force.length() / case.mass;
        max_accel_err = max_accel_err.max((our_accel_mag - rec.accel_mag).abs());
    }

    assert!(
        max_vel_sched_err < 1e-9,
        "{}: inertial velocity schedule disagrees with CSV by {max_vel_sched_err:.3e} m/s",
        case.test_name
    );

    let mut report = CrossvalReport::compute(case.test_name, &[], &[]);
    report.add_extra("aero_force_err", max_force_err, "N");
    report.add_extra("aero_torque_err", max_torque_err, "N*m");
    report.add_extra("accel_mag_err", max_accel_err, "m/s^2");
    report.write();

    println!(
        "{}: {} samples | force_err={max_force_err:.3e} N | torque_err={max_torque_err:.3e} N*m | accel_err={max_accel_err:.3e} m/s^2",
        case.test_name,
        records.len()
    );

    (max_force_err, max_torque_err, max_accel_err)
}

// ── Specular (ε=1.0, `coef_method=Specular`) ──

#[test]
fn tier3_sim_drag_ver_flatplate_specular() {
    let case = FlatPlateCase {
        test_name: "tier3_sim_drag_ver_flatplate_specular",
        csv_label: "drag_one_plate_spec_drag.csv",
        facets: two_sided_plate_facets(AeroCoeffMethod::Specular, 0.0),
        center_grav: DVec3::ZERO,
        mass: JEOD_MASS_ONE_PLATE,
    };
    let (force_err, torque_err, accel_err) = run_flat_plate_case(&case);

    // Tolerances at 5% above observed max error. A one-time Docker regen will
    // populate the CSV; these start at bit-level-equivalence-of-JEOD bounds
    // and can be tightened once reference data is available.
    assert!(
        force_err < 1.0e-13,
        "force_err {force_err:.3e} N exceeds 1.0e-13 N"
    );
    assert!(
        torque_err < 1.0e-14,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-14 N*m"
    );
    assert!(
        accel_err < 1.0e-13,
        "accel_err {accel_err:.3e} m/s^2 exceeds 1.0e-13 m/s^2"
    );
}

// ── Diffuse (ε=0.0, `coef_method=Diffuse`) ──

#[test]
fn tier3_sim_drag_ver_flatplate_diffuse() {
    let case = FlatPlateCase {
        test_name: "tier3_sim_drag_ver_flatplate_diffuse",
        csv_label: "drag_one_plate_diff_drag.csv",
        facets: two_sided_plate_facets(AeroCoeffMethod::Diffuse, 0.0),
        center_grav: DVec3::ZERO,
        mass: JEOD_MASS_ONE_PLATE,
    };
    let (force_err, torque_err, accel_err) = run_flat_plate_case(&case);

    assert!(
        force_err < 1.0e-13,
        "force_err {force_err:.3e} N exceeds 1.0e-13 N"
    );
    assert!(
        torque_err < 1.0e-14,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-14 N*m"
    );
    assert!(
        accel_err < 1.0e-13,
        "accel_err {accel_err:.3e} m/s^2 exceeds 1.0e-13 m/s^2"
    );
}

// ── Mixed (ε=0.5, `coef_method=Mixed`) ──

#[test]
fn tier3_sim_drag_ver_flatplate_mixed() {
    let case = FlatPlateCase {
        test_name: "tier3_sim_drag_ver_flatplate_mixed",
        csv_label: "drag_one_plate_mixed_drag.csv",
        facets: two_sided_plate_facets(AeroCoeffMethod::mixed(0.5), 0.0),
        center_grav: DVec3::ZERO,
        mass: JEOD_MASS_ONE_PLATE,
    };
    let (force_err, torque_err, accel_err) = run_flat_plate_case(&case);

    assert!(
        force_err < 1.0e-13,
        "force_err {force_err:.3e} N exceeds 1.0e-13 N"
    );
    assert!(
        torque_err < 1.0e-14,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-14 N*m"
    );
    assert!(
        accel_err < 1.0e-13,
        "accel_err {accel_err:.3e} m/s^2 exceeds 1.0e-13 m/s^2"
    );
}

// ── CalcCoef at ε=0.0 (pure-diffuse CalcCoef path) ──

#[test]
fn tier3_sim_drag_ver_flatplate_calc_eps00() {
    let case = FlatPlateCase {
        test_name: "tier3_sim_drag_ver_flatplate_calc_eps00",
        csv_label: "drag_one_plate_calc_eps00_drag.csv",
        facets: two_sided_plate_facets(AeroCoeffMethod::calc_coef(0.0), 0.0),
        center_grav: DVec3::ZERO,
        mass: JEOD_MASS_ONE_PLATE,
    };
    let (force_err, torque_err, accel_err) = run_flat_plate_case(&case);

    // CalcCoef uses an erf() approximation (A&S 7.1.26, ≤ 1.5e-7 max error).
    // Tolerances inflated from the pure-specular/diffuse bounds to accommodate
    // that approximation vs. JEOD's libc `erf()`.
    assert!(
        force_err < 1.0e-8,
        "force_err {force_err:.3e} N exceeds 1.0e-8 N"
    );
    assert!(
        torque_err < 1.0e-9,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-9 N*m"
    );
    assert!(
        accel_err < 1.0e-8,
        "accel_err {accel_err:.3e} m/s^2 exceeds 1.0e-8 m/s^2"
    );
}

// ── CalcCoef at ε=0.5 (mixed via full coefficient method) ──

#[test]
fn tier3_sim_drag_ver_flatplate_calc_eps05() {
    let case = FlatPlateCase {
        test_name: "tier3_sim_drag_ver_flatplate_calc_eps05",
        csv_label: "drag_one_plate_calc_eps05_drag.csv",
        facets: two_sided_plate_facets(AeroCoeffMethod::calc_coef(0.5), 0.0),
        center_grav: DVec3::ZERO,
        mass: JEOD_MASS_ONE_PLATE,
    };
    let (force_err, torque_err, accel_err) = run_flat_plate_case(&case);

    assert!(
        force_err < 1.0e-8,
        "force_err {force_err:.3e} N exceeds 1.0e-8 N"
    );
    assert!(
        torque_err < 1.0e-9,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-9 N*m"
    );
    assert!(
        accel_err < 1.0e-8,
        "accel_err {accel_err:.3e} m/s^2 exceeds 1.0e-8 m/s^2"
    );
}

// ── CalcCoef at ε=1.0 (pure-specular CalcCoef path) ──

#[test]
fn tier3_sim_drag_ver_flatplate_calc_eps1() {
    let case = FlatPlateCase {
        test_name: "tier3_sim_drag_ver_flatplate_calc_eps1",
        csv_label: "drag_one_plate_calc_eps1_drag.csv",
        facets: two_sided_plate_facets(AeroCoeffMethod::calc_coef(1.0), 0.0),
        center_grav: DVec3::ZERO,
        mass: JEOD_MASS_ONE_PLATE,
    };
    let (force_err, torque_err, accel_err) = run_flat_plate_case(&case);

    assert!(
        force_err < 1.0e-8,
        "force_err {force_err:.3e} N exceeds 1.0e-8 N"
    );
    assert!(
        torque_err < 1.0e-9,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-9 N*m"
    );
    assert!(
        accel_err < 1.0e-8,
        "accel_err {accel_err:.3e} m/s^2 exceeds 1.0e-8 m/s^2"
    );
}

// ── Shuttle orbiter (6 plates, CalcCoef ε=0.0, non-zero CoG and offsets) ──

#[test]
fn tier3_sim_drag_ver_flatplate_orbiter() {
    let case = FlatPlateCase {
        test_name: "tier3_sim_drag_ver_flatplate_orbiter",
        csv_label: "drag_orbiter_drag.csv",
        facets: orbiter_facets(),
        center_grav: JEOD_CENTER_GRAV_ORBITER,
        mass: JEOD_MASS_ORBITER,
    };
    let (force_err, torque_err, accel_err) = run_flat_plate_case(&case);

    // Orbiter: forces scale as area·dyn_p (~hundreds of m² × 2.8e-5 N/m²),
    // torques scale by plate offset (m) from CoG. Tolerances track CalcCoef's
    // erf() approximation error across 6 plates. Each plate: normal force
    // bounded by ~2.8e-2 N (area 454 m²); with 1.5e-7 relative erf() error
    // per plate, max per-plate force error ~4e-9 N × 6 plates ~ 2.5e-8 N.
    assert!(
        force_err < 1.0e-7,
        "force_err {force_err:.3e} N exceeds 1.0e-7 N"
    );
    assert!(
        torque_err < 1.0e-7,
        "torque_err {torque_err:.3e} N*m exceeds 1.0e-7 N*m"
    );
    assert!(
        accel_err < 1.0e-11,
        "accel_err {accel_err:.3e} m/s^2 exceeds 1.0e-11 m/s^2"
    );
}
