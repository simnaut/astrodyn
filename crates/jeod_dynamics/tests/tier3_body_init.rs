//! Tier 3: Cross-validate body initialization round-trip using JEOD
//! SIM_OrbElem trajectory data.
//!
//! At each timestep in the orbelem CSV, extract Cartesian state, compute
//! orbital elements via `from_cartesian()`, reconstruct via
//! `init_from_orbital_elements()`, and verify the round-trip to machine
//! precision. This validates our element ↔ Cartesian pipeline against
//! JEOD-propagated states across a full orbit.
//!
//! Requires Docker-generated CSV (see test_data/README.md).

use glam::DVec3;
use jeod_dynamics::init_from_orbital_elements;
use std::path::Path;

const MU_EARTH: f64 = 3.986_004_418e14;

#[test]
fn tier3_body_init_round_trip_over_trajectory() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/orbelem_ecc_orbelem.csv");

    assert!(
        csv_path.exists(),
        "SIM_OrbElem CSV not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let content = std::fs::read_to_string(&csv_path).unwrap();
    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    let mut count = 0usize;

    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 21 {
            continue;
        }
        let p = |col: usize| -> f64 {
            let raw = fields[col].trim();
            raw.parse().unwrap_or_else(|e| {
                panic!(
                    "Failed to parse CSV field at {}:{}:{} (raw='{}'): {}",
                    csv_path.display(),
                    i + 1,
                    col + 1,
                    raw,
                    e
                )
            })
        };

        // Columns: 0=time, 15-17=position, 18-20=velocity
        let time = p(0);
        let position = DVec3::new(p(15), p(16), p(17));
        let velocity = DVec3::new(p(18), p(19), p(20));

        let oe = match jeod_math::OrbitalElements::from_cartesian(MU_EARTH, position, velocity) {
            Ok(e) => e,
            Err(e) => panic!("from_cartesian failed at t={time:.0}s: {e}"),
        };

        let reconstructed = init_from_orbital_elements(
            oe.semi_major_axis,
            oe.e_mag,
            oe.inclination,
            oe.long_asc_node,
            oe.arg_periapsis,
            oe.true_anom,
            MU_EARTH,
        );

        let pos_err = (reconstructed.position - position).length();
        let vel_err = (reconstructed.velocity - velocity).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
        count += 1;

        assert!(
            pos_err < 1e-6,
            "t={time:.0}s: position round-trip error {pos_err:.2e} m exceeds 1e-6 m"
        );
        assert!(
            vel_err < 1e-6,
            "t={time:.0}s: velocity round-trip error {vel_err:.2e} m/s exceeds 1e-6 m/s \
             (position error {pos_err:.2e} m)"
        );
    }

    eprintln!("Tier 3: body init round-trip over {count} JEOD trajectory points");
    eprintln!("  Max position error: {max_pos_err:.2e} m");
    eprintln!("  Max velocity error: {max_vel_err:.2e} m/s");
}
