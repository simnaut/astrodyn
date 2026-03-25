//! Tier 3: Cross-validate LVLH frame computation against JEOD SIM_LVLH RUN_inc.
//!
//! At each timestep, reads vehicle position and velocity from the JEOD CSV,
//! computes `compute_lvlh_frame()`, and compares the resulting T_parent_this
//! matrix against JEOD's logged values.
//!
//! Requires Docker-generated CSV (see test_data/README.md).

use glam::{DMat3, DVec3};
use jeod_math::compute_lvlh_frame;
use std::path::Path;

/// Parsed record from the SIM_LVLH CSV.
#[derive(Debug)]
struct LvlhRecord {
    time: f64,
    /// T_parent_this for vehA: 3x3 rotation matrix (inertial -> LVLH).
    t_parent_this: DMat3,
    /// Angular velocity magnitude for vehA.
    ang_vel_mag: f64,
    /// vehA inertial position (m).
    position: DVec3,
    /// vehA inertial velocity (m/s).
    velocity: DVec3,
}

fn load_lvlh_csv(path: &Path) -> Vec<LvlhRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_LVLH CSV from {}: {e}",
            path.display()
        )
    });

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 17 {
            continue;
        }

        let line_no = i + 1;
        let parse = |col: usize| -> f64 {
            fields[col].trim().parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "Failed to parse LVLH CSV at line {line_no}, col {col}: {:?} ({e})",
                    fields[col]
                )
            })
        };

        // CSV columns for vehA (vehB follows at cols 17-32):
        // 0: time
        // 1-9: T_parent_this[0][0..2], T[1][0..2], T[2][0..2] (row-major, contiguous)
        // 10: ang_vel_mag
        // 11: position[0], 12: velocity[0]
        // 13: position[1], 14: velocity[1]
        // 15: position[2], 16: velocity[2]

        // JEOD stores T in row-major: T[row][col].
        // glam DMat3 is column-major, so DMat3::from_cols takes columns.
        // Column j = (T[0][j], T[1][j], T[2][j])
        let t_parent_this = DMat3::from_cols(
            DVec3::new(parse(1), parse(4), parse(7)),
            DVec3::new(parse(2), parse(5), parse(8)),
            DVec3::new(parse(3), parse(6), parse(9)),
        );

        records.push(LvlhRecord {
            time: parse(0),
            t_parent_this,
            ang_vel_mag: parse(10),
            position: DVec3::new(parse(11), parse(13), parse(15)),
            velocity: DVec3::new(parse(12), parse(14), parse(16)),
        });
    }
    records
}

/// Max absolute element-wise difference between two 3x3 matrices.
fn max_mat_diff(a: &DMat3, b: &DMat3) -> f64 {
    let mut max_d = 0.0_f64;
    for c in 0..3 {
        for r in 0..3 {
            let d = (a.col(c)[r] - b.col(c)[r]).abs();
            if d > max_d {
                max_d = d;
            }
        }
    }
    max_d
}

#[test]
fn tier3_lvlh_frame_vs_jeod_sim_lvlh() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/lvlh_inc_lvlh.csv");

    assert!(
        csv_path.exists(),
        "SIM_LVLH RUN_inc CSV not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let records = load_lvlh_csv(&csv_path);
    assert!(
        records.len() > 10,
        "Expected more than 10 records in LVLH CSV, got {}",
        records.len()
    );

    eprintln!(
        "Tier 3: SIM_LVLH RUN_inc cross-validation ({} timesteps)",
        records.len()
    );

    let mut max_mat_error = 0.0_f64;
    let mut max_angvel_error = 0.0_f64;

    for (idx, rec) in records.iter().enumerate() {
        let lvlh = compute_lvlh_frame(rec.position, rec.velocity);

        let mat_err = max_mat_diff(&lvlh.t_parent_this, &rec.t_parent_this);
        let angvel_err = (lvlh.ang_vel_this.length() - rec.ang_vel_mag).abs();

        max_mat_error = max_mat_error.max(mat_err);
        max_angvel_error = max_angvel_error.max(angvel_err);

        assert!(
            mat_err < 1e-10,
            "t={:.1}s: T_parent_this max element error {mat_err:.6e} exceeds 1e-10",
            rec.time
        );
        assert!(
            angvel_err < 1e-12,
            "t={:.1}s: ang_vel magnitude error {angvel_err:.6e} rad/s exceeds 1e-12",
            rec.time
        );

        // Log every 10th record
        if idx % 10 == 0 {
            eprintln!(
                "  t={:>8.1}s: mat_err={:.3e}, angvel_err={:.3e} rad/s",
                rec.time, mat_err, angvel_err
            );
        }
    }

    eprintln!("\n  === Max errors across {} timesteps ===", records.len());
    eprintln!("  T_parent_this element: {max_mat_error:.6e}");
    eprintln!("  ang_vel magnitude:     {max_angvel_error:.6e} rad/s");
}
