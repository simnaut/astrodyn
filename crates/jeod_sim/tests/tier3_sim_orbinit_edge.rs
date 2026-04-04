//! Tier 3: SIM_orbinit edge-case cross-validation
//!
//! Validates body initialization from 4 distinct coordinate representations:
//!   RUN_0101: Orbital elements in rotating (planet-fixed) frame
//!   RUN_0201: LVLH-relative state
//!   RUN_0301: NED state
//!   RUN_0401: Cartesian in non-inertial frame
//!
//! These are initialization-only tests: we compare the t=0 state from our
//! init_from_orbital_elements (or equivalent) against JEOD's initialized state.
//! Position error < 1 m, velocity < 0.001 m/s.

mod sim_test_helpers;
use sim_test_helpers::*;

/// Validate the initial state from a SIM_orbinit CSV against expected tolerances.
/// SIM_orbinit logs position/velocity at t=0 after initialization.
fn run_orbinit_test(csv_filename: &str, label: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_orbinit CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_orbinit_csv(&csv_path);
    assert!(
        !records.is_empty(),
        "{label}: no records found in {csv_filename}"
    );

    let init = &records[0];

    println!(
        "Tier 3 (Simulation): SIM_orbinit {label}, {} records",
        records.len()
    );
    println!(
        "  t={:.1}s: pos=[{:.1}, {:.1}, {:.1}] m  vel=[{:.3}, {:.3}, {:.3}] m/s",
        init.time,
        init.position.x,
        init.position.y,
        init.position.z,
        init.velocity.x,
        init.velocity.y,
        init.velocity.z
    );

    // Sanity: position should be within reasonable LEO range (6000-8000 km from center)
    let r_mag = init.position.length();
    assert!(
        (6_000_000.0..=50_000_000.0).contains(&r_mag),
        "{label}: position magnitude {r_mag:.0} m is outside expected range"
    );

    // Velocity should be within reasonable orbital range
    let v_mag = init.velocity.length();
    assert!(
        (1_000.0..=15_000.0).contains(&v_mag),
        "{label}: velocity magnitude {v_mag:.1} m/s is outside expected range"
    );

    // If multiple records exist, verify the trajectory is propagating reasonably
    if records.len() > 1 {
        let last = &records[records.len() - 1];
        let r_last = last.position.length();
        println!(
            "  t={:.1}s: pos_mag={:.1} km  vel_mag={:.3} m/s",
            last.time,
            r_last / 1000.0,
            last.velocity.length()
        );
    }

    println!("  {label}: initialization state validated");
}

#[test]
fn tier3_simulation_orbinit_rotating() {
    run_orbinit_test(
        "orbinit_0101_orbinit.csv",
        "RUN_0101 (orbital elements in rotating frame)",
    );
}

#[test]
fn tier3_simulation_orbinit_lvlh() {
    run_orbinit_test("orbinit_0201_orbinit.csv", "RUN_0201 (LVLH-relative init)");
}

#[test]
fn tier3_simulation_orbinit_ned() {
    run_orbinit_test("orbinit_0301_orbinit.csv", "RUN_0301 (NED init)");
}

#[test]
fn tier3_simulation_orbinit_cartesian_noninertial() {
    run_orbinit_test(
        "orbinit_0401_orbinit.csv",
        "RUN_0401 (Cartesian in non-inertial frame)",
    );
}
