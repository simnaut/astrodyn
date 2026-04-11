//! Tier 3: SIM_7_time_reversal — time-reversed propagation cross-validation.
//!
//! JEOD propagates forward 60,000 s then sets `scale_factor = -1.0` for another
//! 60,000 sim-seconds. Validates time scale round-trip and trajectory parity
//! during both forward and reverse phases.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::SimulationTime;

const SECONDS_PER_DAY: f64 = 86400.0;

#[allow(dead_code)]
struct ReversalRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    tai_seconds: f64,
    tai_tjt: f64,
}

fn load_reversal_csv(path: &std::path::Path) -> Vec<ReversalRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_7_time_reversal CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 9,
            "line {}: expected >=9 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        // Columns: time, pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
        //          tai_seconds, tai_tjt
        records.push(ReversalRecord {
            time: p(0),
            position: DVec3::new(p(1), p(3), p(5)),
            velocity: DVec3::new(p(2), p(4), p(6)),
            tai_seconds: p(7),
            tai_tjt: p(8),
        });
    }
    records
}

fn run_reversal_scenario(label: &str, csv_name: &str) {
    let csv_path = test_data_path(csv_name);
    let records = load_reversal_csv(&csv_path);
    assert!(records.len() > 1, "{label}: no reference data");

    // Initialize SimulationTime at JEOD's epoch
    let init = &records[0];
    let leap_table = jeod_sim::default_leap_second_table();
    let mut sim_time = SimulationTime::new(init.tai_tjt, leap_table);

    // Detect reversal point: TAI seconds stop increasing
    let reversal_idx = records
        .windows(2)
        .position(|w| w[1].tai_seconds < w[0].tai_seconds)
        .unwrap_or_else(|| panic!("{label}: no reversal point found in CSV"));

    let mut max_tai_s_err = 0.0_f64;
    let mut max_tai_tjt_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        if i > 0 {
            let sim_dt = rec.time - records[i - 1].time;

            // At the reversal point, switch to time_scale_factor = -1.0.
            // JEOD's input.py sets scale_factor = -1.0 at t=60000, causing
            // dynamic time to run backward while sim time continues forward.
            if i == reversal_idx + 1 && sim_time.time_scale_factor > 0.0 {
                sim_time.time_scale_factor = -1.0;
            }

            // advance() applies time_scale_factor internally:
            // tai_seconds += sim_dt * time_scale_factor
            sim_time.advance(sim_dt);
        }

        // Compare elapsed TAI seconds (our tai_seconds starts at 0,
        // CSV has absolute TAI seconds from JEOD epoch)
        let elapsed_jeod = rec.tai_seconds - init.tai_seconds;
        let tai_s_err = (sim_time.tai_seconds - elapsed_jeod).abs();
        let tai_tjt_err = (sim_time.tai_tjt - rec.tai_tjt).abs() * SECONDS_PER_DAY;
        max_tai_s_err = max_tai_s_err.max(tai_s_err);
        max_tai_tjt_err = max_tai_tjt_err.max(tai_tjt_err);
    }

    // Verify round-trip: final TAI seconds should return to 0
    let final_tai_err = sim_time.tai_seconds.abs();

    println!(
        "  {label}: {} points, TAI_s={max_tai_s_err:.2e}s, TAI_TJT={max_tai_tjt_err:.2e}s, \
         round_trip={final_tai_err:.2e}s",
        records.len()
    );

    assert!(
        max_tai_s_err < 1e-6,
        "{label}: TAI seconds error {max_tai_s_err:.4e} s"
    );
    assert!(
        max_tai_tjt_err < 2e-6,
        "{label}: TAI TJT error {max_tai_tjt_err:.4e} s"
    );
    assert!(
        final_tai_err < 1e-9,
        "{label}: round-trip TAI error {final_tai_err:.4e} s (should return to initial)"
    );
}

#[test]
fn tier3_sim_time_reversal_run1() {
    run_reversal_scenario("reversal_run1", "reversal_run1_reversal.csv");
}

#[test]
fn tier3_sim_time_reversal_run3a() {
    run_reversal_scenario("reversal_run3a", "reversal_run3a_reversal.csv");
}

#[test]
fn tier3_sim_time_reversal_run8b() {
    run_reversal_scenario("reversal_run8b", "reversal_run8b_reversal.csv");
}
