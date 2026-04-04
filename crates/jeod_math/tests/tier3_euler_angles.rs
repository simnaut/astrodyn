//! Tier 3: Cross-validate Euler angle extraction against JEOD SIM_Euler RUN_inc.
//!
//! At each timestep, reads the rotation matrix T_parent_this from the JEOD CSV,
//! then calls `compute_euler_angles_from_matrix()` for the inertial RPY (XYZ)
//! sequence. Compares against JEOD's logged ref_body_angles and body_ref_angles.
//!
//! The Euler sim logs 6 sequences: rpy (inertial), and 5 LVLH-relative sequences
//! (pyr_lvlh, rpy_lvlh, ypr_lvlh, ryp_lvlh, yrp_lvlh). This test focuses on
//! the inertial RPY sequence since the LVLH sequences require composing with the
//! LVLH frame rotation first.
//!
//! Requires Docker-generated CSV (see test_data/README.md).

use glam::{DMat3, DVec3};
use jeod_math::{compute_euler_angles_from_matrix, EulerSequence};
use jeod_test_data::crossval::crossval_report;
use std::path::Path;

/// Number of angle fields per sequence: ref_body_angles[3] + body_ref_angles[3] = 6.
const FIELDS_PER_SEQ: usize = 6;
/// Number of sequences logged.
const NUM_SEQUENCES: usize = 6;
/// Parsed record from the SIM_Euler CSV.
#[derive(Debug)]
#[allow(dead_code)]
struct EulerRecord {
    time: f64,
    /// ref_body_angles[0..3] for each sequence (RPY, PYR_LVLH, RPY_LVLH, YPR_LVLH, RYP_LVLH, YRP_LVLH).
    ref_body_angles: [[f64; 3]; NUM_SEQUENCES],
    /// body_ref_angles[0..3] for each sequence.
    body_ref_angles: [[f64; 3]; NUM_SEQUENCES],
    /// Inertial position (m).
    position: DVec3,
    /// Inertial velocity (m/s).
    velocity: DVec3,
    /// Rotation matrix T_parent_this (inertial -> body).
    t_parent_this: DMat3,
}

fn load_euler_csv(path: &Path) -> Vec<EulerRecord> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read SIM_Euler CSV from {}: {e}", path.display()));

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        // Need: 1 (time) + 36 (angles) + 3 (pos) + 3 (vel) + 9 (T) + 4 (Q) = 56 minimum
        if fields.len() < 56 {
            continue;
        }

        let line_no = i + 1;
        let parse = |col: usize| -> f64 {
            fields[col].trim().parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "Failed to parse Euler CSV at line {line_no}, col {col}: {:?} ({e})",
                    fields[col]
                )
            })
        };

        // CSV columns:
        // 0: time
        // 1..36: For each of 6 sequences: ref_body_angles[0..2], body_ref_angles[0..2]
        // 37-42: position[0], velocity[0], position[1], velocity[1], position[2], velocity[2]
        // 43-51: T_parent_this (9 elements, row-major)
        // 52-54: Q_parent_this.vector[0,1,2]
        // 55: Q_parent_this.scalar

        let mut ref_body_angles = [[0.0_f64; 3]; NUM_SEQUENCES];
        let mut body_ref_angles = [[0.0_f64; 3]; NUM_SEQUENCES];

        for seq in 0..NUM_SEQUENCES {
            let base = 1 + seq * FIELDS_PER_SEQ;
            ref_body_angles[seq] = [parse(base), parse(base + 1), parse(base + 2)];
            body_ref_angles[seq] = [parse(base + 3), parse(base + 4), parse(base + 5)];
        }

        // Position/velocity are interleaved: pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]
        let pv_base = 37;
        let t_base = 43;

        // JEOD stores T in row-major: T[row][col].
        // glam DMat3 column-major: column j = (T[0][j], T[1][j], T[2][j]).
        let t_parent_this = DMat3::from_cols(
            DVec3::new(parse(t_base), parse(t_base + 3), parse(t_base + 6)),
            DVec3::new(parse(t_base + 1), parse(t_base + 4), parse(t_base + 7)),
            DVec3::new(parse(t_base + 2), parse(t_base + 5), parse(t_base + 8)),
        );

        records.push(EulerRecord {
            time: parse(0),
            ref_body_angles,
            body_ref_angles,
            position: DVec3::new(parse(pv_base), parse(pv_base + 2), parse(pv_base + 4)),
            velocity: DVec3::new(parse(pv_base + 1), parse(pv_base + 3), parse(pv_base + 5)),
            t_parent_this,
        });
    }
    records
}

/// Compute angular difference accounting for wraparound at 2*pi.
fn angle_diff(a: f64, b: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    let mut d = (a - b) % tau;
    if d > std::f64::consts::PI {
        d -= tau;
    }
    if d < -std::f64::consts::PI {
        d += tau;
    }
    d.abs()
}

/// Check if a rotation matrix is near gimbal lock for XYZ sequence.
/// Gimbal lock occurs when theta (the Y-axis rotation) is near +/- pi/2.
fn near_gimbal_lock_xyz(t: &DMat3) -> bool {
    // For XYZ, theta = asin(T[2][0]) => gimbal lock when |T[2][0]| ~ 1
    let t20 = t.col(0)[2];
    t20.abs() > 0.999
}

#[test]
fn tier3_euler_angles_vs_jeod_sim_euler() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/euler_inc_euler.csv");

    assert!(
        csv_path.exists(),
        "SIM_Euler RUN_inc CSV not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let records = load_euler_csv(&csv_path);
    assert!(
        records.len() > 10,
        "Expected more than 10 records in Euler CSV, got {}",
        records.len()
    );

    eprintln!(
        "Tier 3: SIM_Euler RUN_inc cross-validation ({} timesteps)",
        records.len()
    );

    // Focus on the inertial RPY (XYZ) sequence: index 0 in the CSV.
    // Sequence mapping: rpy = Roll-Pitch-Yaw = XYZ
    let seq_idx = 0;
    let euler_seq = EulerSequence::XYZ;

    let mut max_angle_err = [0.0_f64; 3];
    let mut skipped_gimbal = 0_usize;

    for (idx, rec) in records.iter().enumerate() {
        // Extract Euler angles from the rotation matrix using our implementation
        let angles = compute_euler_angles_from_matrix(&rec.t_parent_this, euler_seq);

        // JEOD logs ref_body_angles (parent-to-body decomposition)
        let jeod_angles = rec.ref_body_angles[seq_idx];

        // Skip gimbal lock regions where Euler angle extraction is numerically unstable
        if near_gimbal_lock_xyz(&rec.t_parent_this) {
            skipped_gimbal += 1;
            if idx % 10 == 0 {
                eprintln!(
                    "  t={:>8.1}s: SKIPPED (near gimbal lock, T[2][0]={:.6})",
                    rec.time,
                    rec.t_parent_this.col(0)[2]
                );
            }
            continue;
        }

        for k in 0..3 {
            let err = angle_diff(angles[k], jeod_angles[k]);
            max_angle_err[k] = max_angle_err[k].max(err);

            assert!(
                err < 1e-6,
                "t={:.1}s: RPY angle[{k}] error {err:.6e} rad exceeds 1e-6 rad \
                 (ours={:.10}, JEOD={:.10})",
                rec.time,
                angles[k],
                jeod_angles[k]
            );
        }

        // Log every 10th record
        if idx % 10 == 0 {
            eprintln!(
                "  t={:>8.1}s: err=[{:.3e}, {:.3e}, {:.3e}] rad, \
                 ours=[{:.6}, {:.6}, {:.6}], JEOD=[{:.6}, {:.6}, {:.6}]",
                rec.time,
                angle_diff(angles[0], jeod_angles[0]),
                angle_diff(angles[1], jeod_angles[1]),
                angle_diff(angles[2], jeod_angles[2]),
                angles[0],
                angles[1],
                angles[2],
                jeod_angles[0],
                jeod_angles[1],
                jeod_angles[2],
            );
        }
    }

    eprintln!(
        "\n  === Max errors across {} timesteps (RPY/XYZ inertial) ===",
        records.len()
    );
    eprintln!("  angle[0] (roll):  {:.6e} rad", max_angle_err[0]);
    eprintln!("  angle[1] (pitch): {:.6e} rad", max_angle_err[1]);
    eprintln!("  angle[2] (yaw):   {:.6e} rad", max_angle_err[2]);
    if skipped_gimbal > 0 {
        eprintln!("  Gimbal-lock regions skipped: {skipped_gimbal}");
    }

    crossval_report(
        "tier3_euler_angles_vs_jeod_sim_euler",
        &[
            ("euler_roll", max_angle_err[0], "rad"),
            ("euler_pitch", max_angle_err[1], "rad"),
            ("euler_yaw", max_angle_err[2], "rad"),
        ],
    );
}
