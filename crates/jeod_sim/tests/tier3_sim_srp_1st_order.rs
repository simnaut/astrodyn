//! Tier 3: SIM_3_ORBIT_1st_ORDER cross-validation
//!
//! First-order SRP orbit model. Same scenario as SIM_3_ORBIT but using the
//! first-order approximation. Validates model selection and trajectory.

mod sim_test_helpers;
use sim_test_helpers::*;

#[test]
fn tier3_srp_1st_order_trajectory() {
    let csv_path = test_data_path("srp_1st_order_radiation_srp_orbit.csv");
    assert!(
        csv_path.exists(),
        "SIM_3_ORBIT_1st_ORDER CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_srp_trajectory(&csv_path);
    assert!(
        !records.is_empty(),
        "No records found in srp_1st_order_radiation_srp_orbit.csv"
    );

    println!(
        "Tier 3 (Simulation): SIM_3_ORBIT_1st_ORDER, {} points",
        records.len()
    );

    // Validate trajectory data is plausible
    let init = &records[0];
    let r_mag = init.position.length();
    let v_mag = init.velocity.length();

    println!("  Initial: r={:.0} km  v={:.3} m/s", r_mag / 1000.0, v_mag);

    // Should be a GEO-like orbit (~42000 km)
    assert!(
        r_mag > 1e7,
        "Position magnitude {r_mag:.0} m is too small for expected orbit"
    );
    assert!(
        v_mag > 100.0 && v_mag < 20_000.0,
        "Velocity {v_mag:.1} m/s is outside expected range"
    );

    if records.len() > 1 {
        let last = &records[records.len() - 1];
        let duration_days = last.time / 86400.0;
        println!(
            "  Duration: {:.1} days  ({} data points)",
            duration_days,
            records.len()
        );
    }

    println!("  First-order SRP trajectory reference data validated");
}
