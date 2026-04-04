//! Tier 3: SIM_orbinit cross-validation
//!
//! Validates body initialization from 4 distinct coordinate representations
//! by checking cross-consistency between methods and against JEOD output.
//!
//!   RUN_0101: Orbital elements in inertial frame (STS-114)
//!   RUN_0201: Orbital elements in planet-fixed frame (ISS)
//!   RUN_0301: Orbital elements in planet-fixed frame (STS-114)
//!   RUN_0401: Cartesian state in inertial frame (STS-114)
//!
//! All produce the same orbit (STS-114 / ISS at similar epoch).
//! Cross-consistency: RUN_0101, RUN_0301, RUN_0401 (same vehicle, STS-114)
//! should agree to < 1 m position, < 0.001 m/s velocity.
//! RUN_0201 (ISS) differs slightly due to different state vector source.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;

#[test]
fn tier3_simulation_orbinit_cross_consistency() {
    // Load all 4 initialization results
    let runs: Vec<(&str, &str)> = vec![
        ("orbinit_0101_orbinit.csv", "RUN_0101 (STS-114 inertial OE)"),
        ("orbinit_0201_orbinit.csv", "RUN_0201 (ISS pfix OE)"),
        ("orbinit_0301_orbinit.csv", "RUN_0301 (STS-114 pfix OE)"),
        (
            "orbinit_0401_orbinit.csv",
            "RUN_0401 (STS-114 inertial cart)",
        ),
    ];

    let mut states: Vec<(DVec3, DVec3, &str)> = Vec::new();

    for (filename, label) in &runs {
        let csv_path = test_data_path(filename);
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
            "{label}: no records found in {filename}"
        );

        let init = &records[0];
        let r_mag = init.position.length();
        let v_mag = init.velocity.length();

        println!(
            "  {label}: r={:.3} km  v={:.6} km/s  pos=[{:.1}, {:.1}, {:.1}] m",
            r_mag / 1000.0,
            v_mag / 1000.0,
            init.position.x,
            init.position.y,
            init.position.z,
        );

        // Sanity: LEO orbit
        assert!(
            (6_000_000.0..=8_000_000.0).contains(&r_mag),
            "{label}: r={r_mag:.0} m outside LEO range"
        );
        assert!(
            (6_000.0..=8_000.0).contains(&v_mag),
            "{label}: v={v_mag:.1} m/s outside LEO range"
        );

        states.push((init.position, init.velocity, label));
    }

    println!();

    // Cross-consistency: STS-114 runs (0101, 0301, 0401) should agree closely.
    // They use the same vehicle state initialized through different methods.
    let sts_indices = [0, 2, 3]; // RUN_0101, RUN_0301, RUN_0401
    for i in 0..sts_indices.len() {
        for j in (i + 1)..sts_indices.len() {
            let (pos_a, vel_a, label_a) = states[sts_indices[i]];
            let (pos_b, vel_b, label_b) = states[sts_indices[j]];
            let pos_err = (pos_a - pos_b).length();
            let vel_err = (vel_a - vel_b).length();
            println!(
                "  {label_a} vs {label_b}: pos_err={:.6} m  vel_err={:.6e} m/s",
                pos_err, vel_err,
            );
            assert!(
                pos_err < 1.0,
                "STS-114 cross-consistency: position error {pos_err:.3} m exceeds 1.0 m \
                 between {label_a} and {label_b}"
            );
            assert!(
                vel_err < 0.001,
                "STS-114 cross-consistency: velocity error {vel_err:.3e} m/s exceeds 0.001 m/s \
                 between {label_a} and {label_b}"
            );
        }
    }

    // ISS vs STS-114: different vehicles at similar epoch, expect ~200 m difference
    let (pos_iss, _, _) = states[1]; // RUN_0201
    let (pos_sts, _, _) = states[0]; // RUN_0101
    let cross_vehicle_err = (pos_iss - pos_sts).length();
    println!(
        "\n  ISS vs STS-114: pos_diff={:.1} m (expected: different vehicles)",
        cross_vehicle_err,
    );
    assert!(
        cross_vehicle_err < 1000.0,
        "ISS vs STS-114 position difference {cross_vehicle_err:.0} m exceeds 1 km \
         (should be similar orbits)"
    );

    println!("\n  All initialization methods produce consistent LEO states");
}
