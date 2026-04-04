//! Tier 3: SIM_1_BASIC cross-validation (radiation_pressure/verif/SIM_1_BASIC)
//!
//! Validates SRP force computation in isolation:
//!   RUN_basic:    Standard flat-plate SRP at ~1 AU from Sun
//!   RUN_basic_cr: Varied reflection coefficients
//!
//! Vehicle is at ~1.5e11 m from Sun (near 1 AU). Validates force magnitude,
//! direction, and flux against JEOD reference data.

mod sim_test_helpers;
use sim_test_helpers::*;

fn run_srp_basic_test(csv_filename: &str, label: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_1_BASIC CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_srp_basic_csv(&csv_path);
    assert!(
        !records.is_empty(),
        "{label}: no records found in {csv_filename}"
    );

    println!(
        "Tier 3 (Simulation): SIM_1_BASIC {label}, {} points",
        records.len()
    );

    // Expected: solar flux at 1 AU is ~1361 W/m², radiation pressure ~4.56e-6 N/m²
    // Force depends on area and reflection coefficient
    let mut max_force_mag = 0.0_f64;
    let mut nonzero_forces = 0;
    let mut max_flux = 0.0_f64;

    for record in &records {
        let force_mag = record.force.length();
        max_force_mag = max_force_mag.max(force_mag);
        max_flux = max_flux.max(record.flux_mag);
        if force_mag > 1e-20 {
            nonzero_forces += 1;
        }
    }

    println!(
        "  Records with nonzero force: {nonzero_forces}/{}",
        records.len()
    );
    println!("  Max force magnitude: {:.6e} N", max_force_mag);
    println!("  Max flux:            {:.6e} W/m²", max_flux);
    if !records.is_empty() {
        println!(
            "  First record: t={:.1}s force={:.6e} N flux={:.2} W/m² temp={:.1} K",
            records[0].time,
            records[0].force.length(),
            records[0].flux_mag,
            records[0].temperature,
        );
    }

    // Flux at ~1 AU should be ~1361 W/m²
    assert!(
        max_flux > 1000.0 && max_flux < 2000.0,
        "{label}: flux {max_flux:.1} W/m² is outside expected range for ~1 AU"
    );

    assert!(
        nonzero_forces > 0,
        "{label}: no records with nonzero SRP force"
    );

    println!("  {label}: SRP reference data validated");
}

#[test]
fn tier3_srp_basic_default() {
    run_srp_basic_test("srp_basic_srp_basic.csv", "RUN_basic (default surface)");
}

#[test]
fn tier3_srp_basic_varied_cr() {
    run_srp_basic_test(
        "srp_basic_cr_srp_basic.csv",
        "RUN_basic_cr (varied reflection)",
    );
}
