//! Shared helpers for jeod_sim Tier 3 tests.
//!
//! Provides CSV parsing for SIM_dyncomp trajectory data, quaternion error
//! computation, and test data path resolution.

#![allow(dead_code)]

use glam::DVec3;
use jeod_sim::JeodQuat;
use std::path::Path;

pub const MU_EARTH: f64 = 3.986_004_415e14;
pub const DT: f64 = 0.03125; // 32 Hz, matches JEOD SIM_dyncomp

/// Earth rotation rate (JEOD RNPJ2000 default).
pub const OMEGA_EARTH: f64 = 7.292_115_146_706_388e-5;

#[derive(Debug)]
pub struct TransRecord {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

#[derive(Debug)]
pub struct SixDofRecord {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub quaternion: JeodQuat,
    pub ang_vel: DVec3,
}

pub fn load_trans_trajectory(path: &Path) -> Vec<TransRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD trajectory CSV from {}: {e}\n\
             Generate with: docker build -f trick/Dockerfile -t jeod-trick .. && \
             docker run --rm -v $(pwd)/test_data:/output jeod-trick",
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
            f.len() >= 17,
            "line {}: expected >=17 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |s: &str| -> f64 { s.trim().parse().unwrap() };
        records.push(TransRecord {
            time: p(f[0]),
            position: DVec3::new(p(f[1]), p(f[8]), p(f[15])),
            velocity: DVec3::new(p(f[2]), p(f[9]), p(f[16])),
        });
    }
    records
}

pub fn load_sixdof_trajectory(path: &Path) -> Vec<SixDofRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD trajectory CSV from {}: {e}\n\
             Generate with: docker build -f trick/Dockerfile -t jeod-trick .. && \
             docker run --rm -v $(pwd)/test_data:/output jeod-trick",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(f.len() >= 23, "line {}: expected >=23 columns", i + 1);
        let p = |s: &str| -> f64 { s.trim().parse().unwrap() };
        records.push(SixDofRecord {
            time: p(f[0]),
            position: DVec3::new(p(f[1]), p(f[8]), p(f[15])),
            velocity: DVec3::new(p(f[2]), p(f[9]), p(f[16])),
            ang_vel: DVec3::new(p(f[3]), p(f[10]), p(f[17])),
            quaternion: JeodQuat::new(p(f[22]), p(f[7]), p(f[14]), p(f[21])),
        });
    }
    records
}

pub fn quaternion_angle_error(q1: &JeodQuat, q2: &JeodQuat) -> f64 {
    let dot = (q1.scalar() * q2.scalar()
        + q1.vector().x * q2.vector().x
        + q1.vector().y * q2.vector().y
        + q1.vector().z * q2.vector().z)
        .abs();
    2.0 * dot.min(1.0).acos()
}

pub fn test_data_path(filename: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(filename)
}
