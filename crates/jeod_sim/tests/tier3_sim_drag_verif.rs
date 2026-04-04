//! Tier 3: SIM_VER_DRAG cross-validation (aerodynamics/verif/SIM_VER_DRAG)
//!
//! Validates aerodynamic drag force computation against JEOD in isolation
//! (no orbit propagation). Three drag modes:
//!   RUN_aero_drag_const: Constant drag force magnitude = 0.05 N
//!   RUN_aero_drag_CD:    Cd=2, A=100 m²
//!   RUN_aero_drag_BC:    BC=0.005, mass=1.0 kg → mass/BC = 200 (equivalent to Cd*A = 200)
//!
//! Atmospheric conditions: density=1e-12 kg/m³, T=1487 K, zero wind.
//! Vehicle rotation: identity (body frame = inertial frame).
//! Velocity from CSV (time-varying as drag decelerates the vehicle).

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_atmosphere::AtmosphereState;
use jeod_interactions::{compute_ballistic_drag, DragConfig};
use jeod_test_data::crossval::crossval_report;

const DRAG_DENSITY: f64 = 1e-12; // kg/m³

fn run_drag_comparison(csv_filename: &str, label: &str, config: DragConfig, test_name: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_VER_DRAG CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_drag_csv(&csv_path);
    assert!(
        !records.is_empty(),
        "{label}: no records found in {csv_filename}"
    );

    println!(
        "Tier 3 (Simulation): SIM_VER_DRAG {label}, {} points",
        records.len()
    );

    let atmos = AtmosphereState {
        density: DRAG_DENSITY,
        temperature: 1487.0,
        pressure: 0.0,
        wind: DVec3::ZERO,
    };
    let t_inertial_struct = DMat3::IDENTITY;

    let mut max_force_err = 0.0_f64;
    let mut max_force_rel_err = 0.0_f64;

    for record in &records {
        // Compute drag using our code with JEOD's velocity from CSV
        let result =
            compute_ballistic_drag(&config, &atmos, record.inertial_vel, &t_inertial_struct);

        let force_err = (result.force - record.aero_force).length();
        max_force_err = max_force_err.max(force_err);

        let jeod_mag = record.aero_force.length();
        if jeod_mag > 1e-20 {
            let rel_err = force_err / jeod_mag;
            max_force_rel_err = max_force_rel_err.max(rel_err);
        }

        if (record.time % 60.0).abs() < 0.5 {
            println!(
                "  t={:5.0}s: our=[{:.6e}, {:.6e}, {:.6e}] jeod=[{:.6e}, {:.6e}, {:.6e}] err={:.3e} N",
                record.time,
                result.force.x, result.force.y, result.force.z,
                record.aero_force.x, record.aero_force.y, record.aero_force.z,
                force_err,
            );
        }
    }

    println!("  Max force error:     {:.6e} N", max_force_err);
    println!("  Max force rel error: {:.6e}", max_force_rel_err);

    crossval_report(
        test_name,
        &[
            ("force", max_force_err, 1e-3, "N"),
            ("force_rel", max_force_rel_err, 1e-10, ""),
        ],
    );

    // Drag force should match JEOD to high precision (same formula, same inputs)
    assert!(
        max_force_err < 1e-3,
        "{label}: force error {max_force_err:.3e} N exceeds 1e-3 N"
    );
    assert!(
        max_force_rel_err < 1e-10,
        "{label}: relative force error {max_force_rel_err:.3e} exceeds 1e-10"
    );
}

#[test]
fn tier3_drag_const_force() {
    // DRAG_OPT_CONST: JEOD uses drag=0.05 as a CONSTANT FORCE MAGNITUDE (N),
    // not a coefficient. The formula is: force = rel_vel_hat * drag.
    // Our compute_ballistic_drag doesn't support this mode — it always computes
    // F = -0.5*ρ*v²*Cd*A. Validate the reference data instead.
    let csv_path = test_data_path("drag_const_drag.csv");
    assert!(csv_path.exists(), "CSV not found at {}", csv_path.display());
    let records = load_drag_csv(&csv_path);
    assert!(!records.is_empty());

    println!(
        "Tier 3 (Simulation): SIM_VER_DRAG const force mode, {} points",
        records.len()
    );

    // JEOD DRAG_OPT_CONST: force magnitude = drag = 0.05 N (constant)
    // Acceleration = force / mass = 0.05 / 1.0 = 0.05 m/s² (constant)
    let mut max_accel_err = 0.0_f64;
    for record in &records {
        let force_mag = record.aero_force.length();
        let accel_err = (record.accel_mag - 0.05).abs();
        max_accel_err = max_accel_err.max(accel_err);

        // Force magnitude should be ~0.05 N (constant) but direction follows velocity
        // As velocity decreases from drag, force magnitude stays at 0.05 N
        assert!(
            (force_mag - 0.05).abs() < 0.001,
            "Constant drag force {force_mag:.6} N deviates from 0.05 N at t={}",
            record.time
        );
    }
    // Acceleration should be consistent: force/mass = 0.05/1.0 = 0.05 m/s²
    assert!(
        max_accel_err < 1e-6,
        "Acceleration error {max_accel_err:.3e} m/s² exceeds 1e-6"
    );
    println!(
        "  DRAG_OPT_CONST: force=0.05 N (constant), max accel_err={:.3e}",
        max_accel_err
    );
    println!(
        "  Note: DRAG_OPT_CONST mode not implemented in our code — JEOD sets force \
         magnitude directly, bypassing F=0.5*ρ*v²*Cd*A. Validated as reference data."
    );

    crossval_report(
        "tier3_drag_const_force",
        &[("accel", max_accel_err, 1e-6, "m/s2")],
    );
}

#[test]
fn tier3_drag_variable_cd() {
    // DRAG_OPT_CD: drag = -0.5*ρ*v² * area * Cd
    // Cd=2, area=100 m² → Cd*A = 200
    run_drag_comparison(
        "drag_cd_drag.csv",
        "RUN_aero_drag_CD (Cd=2, A=100)",
        DragConfig {
            cd: 2.0,
            area: 100.0,
            constant_density: Some(DRAG_DENSITY),
        },
        "tier3_drag_variable_cd",
    );
}

#[test]
fn tier3_drag_ballistic_coeff() {
    // DRAG_OPT_BC: drag = -(0.5*ρ*v² * mass) / BC
    // BC=0.005, mass=1.0 kg → mass/BC = 200 → same as Cd*A=200
    run_drag_comparison(
        "drag_bc_drag.csv",
        "RUN_aero_drag_BC (BC=0.005, m=1kg → Cd*A=200)",
        DragConfig {
            cd: 200.0,
            area: 1.0,
            constant_density: Some(DRAG_DENSITY),
        },
        "tier3_drag_ballistic_coeff",
    );
}
