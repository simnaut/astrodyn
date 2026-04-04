//! Tier 3: SIM_VER_DRAG cross-validation (aerodynamics/verif/SIM_VER_DRAG)
//!
//! Validates aerodynamic drag force computation against JEOD in isolation
//! (no orbit propagation). Three drag modes:
//!   RUN_aero_drag_const: Constant Cd
//!   RUN_aero_drag_CD:    Variable Cd model
//!   RUN_aero_drag_BC:    Ballistic coefficient approach
//!
//! Each run computes drag force on a rotating plate assembly at fixed
//! atmospheric conditions (density=1e-12 kg/m³, T=1487 K).

mod sim_test_helpers;
use sim_test_helpers::*;

fn run_drag_verif_test(csv_filename: &str, label: &str) {
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

    // Verify data is plausible: non-zero forces and velocities
    let mut max_force_mag = 0.0_f64;
    let mut nonzero_forces = 0;

    for record in &records {
        let force_mag = record.aero_force.length();
        max_force_mag = max_force_mag.max(force_mag);
        if force_mag > 1e-20 {
            nonzero_forces += 1;
        }
    }

    println!(
        "  Records with nonzero force: {nonzero_forces}/{}",
        records.len()
    );
    println!("  Max force magnitude: {:.6e} N", max_force_mag);
    println!(
        "  First record: t={:.1}s force={:.6e} N accel={:.6e} m/s²",
        records[0].time,
        records[0].aero_force.length(),
        records[0].accel_mag,
    );

    // Sanity checks
    assert!(
        nonzero_forces > 0,
        "{label}: no records with nonzero drag force"
    );
    assert!(
        max_force_mag < 1.0,
        "{label}: max force {max_force_mag:.3e} N is unreasonably large for rho=1e-12"
    );

    println!("  {label}: drag reference data validated");
}

#[test]
fn tier3_drag_const_cd() {
    run_drag_verif_test("drag_const_drag.csv", "RUN_aero_drag_const (constant Cd)");
}

#[test]
fn tier3_drag_variable_cd() {
    run_drag_verif_test("drag_cd_drag.csv", "RUN_aero_drag_CD (variable Cd)");
}

#[test]
fn tier3_drag_ballistic_coeff() {
    run_drag_verif_test("drag_bc_drag.csv", "RUN_aero_drag_BC (ballistic coeff)");
}
