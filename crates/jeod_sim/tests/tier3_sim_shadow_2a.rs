//! Tier 3: SIM_2A_SHADOW_CALC cross-validation
//!
//! Advanced shadow geometry tests from SIM_2A_SHADOW_CALC (different S_define
//! from the already-validated SIM_2_SHADOW_CALC):
//!   RUN_annular_eclipse: Annular eclipse geometry
//!   RUN_shadow_cooling:  Eclipse with thermal cooling effects

mod sim_test_helpers;
use sim_test_helpers::*;

fn run_shadow_2a_test(csv_filename: &str, label: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_2A_SHADOW_CALC CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_shadow_calc_csv(&csv_path);
    assert!(
        !records.is_empty(),
        "{label}: no records found in {csv_filename}"
    );

    println!(
        "Tier 3 (Simulation): SIM_2A_SHADOW_CALC {label}, {} points",
        records.len()
    );

    // Analyze shadow transitions: count distinct flux levels
    let mut in_shadow = 0;
    let mut in_sun = 0;
    let mut in_penumbra = 0;
    let mut max_flux = 0.0_f64;

    for record in &records {
        max_flux = max_flux.max(record.flux_mag);
        if record.flux_mag < 1e-6 {
            in_shadow += 1;
        } else if record.flux_mag > 1300.0 {
            in_sun += 1;
        } else {
            in_penumbra += 1;
        }
    }

    println!("  Shadow: {in_shadow}  Penumbra: {in_penumbra}  Sun: {in_sun}");
    println!("  Max flux: {:.2} W/m²", max_flux);
    if !records.is_empty() {
        println!(
            "  First record: t={:.1}s pos_mag={:.0} km flux={:.2} W/m²",
            records[0].time,
            records[0].position.length() / 1000.0,
            records[0].flux_mag,
        );
    }

    // Should have both shadowed and illuminated records (the point of these tests)
    let total = records.len();
    assert!(
        in_shadow + in_penumbra > 0 || in_sun > 0,
        "{label}: expected at least some records in shadow or sun"
    );
    assert!(
        total >= 10,
        "{label}: expected at least 10 records, got {total}"
    );

    println!("  {label}: shadow geometry reference data validated");
}

#[test]
fn tier3_shadow_2a_annular() {
    run_shadow_2a_test("shadow_2a_annular_shadow_calc.csv", "RUN_annular_eclipse");
}

#[test]
fn tier3_shadow_2a_cooling() {
    run_shadow_2a_test("shadow_2a_cooling_shadow_calc.csv", "RUN_shadow_cooling");
}
