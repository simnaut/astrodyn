//! Shared helpers for jeod_sim Tier 3 tests.
//!
//! Provides CSV parsing for SIM_dyncomp and derived-state trajectory data,
//! quaternion error computation, and test data path resolution.

#![allow(dead_code)]

use glam::{DMat3, DVec3};
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
    let content = read_csv(path, "SIM_dyncomp");
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
    let content = read_csv(path, "SIM_dyncomp");
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

// ── Derived-state CSV loaders ──

#[derive(Debug)]
pub struct OrbElemRecord {
    pub time: f64,
    pub semi_major_axis: f64,
    pub e_mag: f64,
    pub inclination: f64,
    pub arg_periapsis: f64,
    pub long_asc_node: f64,
    pub true_anom: f64,
    pub mean_anom: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

pub fn load_orbelem_csv(path: &Path) -> Vec<OrbElemRecord> {
    let content = read_csv(path, "SIM_OrbElem");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 21,
            "line {}: expected >=21 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(OrbElemRecord {
            time: p(0),
            semi_major_axis: p(1),
            e_mag: p(3),
            inclination: p(4),
            arg_periapsis: p(5),
            long_asc_node: p(6),
            true_anom: p(9),
            mean_anom: p(10),
            position: DVec3::new(p(15), p(16), p(17)),
            velocity: DVec3::new(p(18), p(19), p(20)),
        });
    }
    records
}

#[derive(Debug)]
pub struct LvlhRecord {
    pub time: f64,
    pub t_parent_this: DMat3,
    pub ang_vel_mag: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

pub fn load_lvlh_csv(path: &Path) -> Vec<LvlhRecord> {
    let content = read_csv(path, "SIM_LVLH");
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
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        // JEOD row-major T[row][col] → glam column-major
        let t_parent_this = DMat3::from_cols(
            DVec3::new(p(1), p(4), p(7)),
            DVec3::new(p(2), p(5), p(8)),
            DVec3::new(p(3), p(6), p(9)),
        );
        records.push(LvlhRecord {
            time: p(0),
            t_parent_this,
            ang_vel_mag: p(10),
            position: DVec3::new(p(11), p(13), p(15)),
            velocity: DVec3::new(p(12), p(14), p(16)),
        });
    }
    records
}

#[derive(Debug)]
pub struct NedRecord {
    pub time: f64,
    pub ellip_altitude: f64,
    pub ellip_latitude: f64,
    pub ellip_longitude: f64,
    pub sphere_altitude: f64,
    pub sphere_latitude: f64,
    pub sphere_longitude: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

pub fn load_ned_csv(path: &Path) -> Vec<NedRecord> {
    let content = read_csv(path, "SIM_NED");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 16,
            "line {}: expected >=16 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        // CSV columns: 0=time, 1-3=cart_coords, 4=ellip_alt, 5=sphere_alt,
        // 6=ellip_lat, 7=sphere_lat, 8=ellip_lon, 9=sphere_lon,
        // 10-15=pos/vel interleaved
        records.push(NedRecord {
            time: p(0),
            ellip_altitude: p(4),
            ellip_latitude: p(6),
            ellip_longitude: p(8),
            sphere_altitude: p(5),
            sphere_latitude: p(7),
            sphere_longitude: p(9),
            position: DVec3::new(p(10), p(12), p(14)),
            velocity: DVec3::new(p(11), p(13), p(15)),
        });
    }
    records
}

#[derive(Debug)]
pub struct SrpRecord {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

pub fn load_srp_trajectory(path: &Path) -> Vec<SrpRecord> {
    let content = read_csv(path, "SIM_3_ORBIT");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 7,
            "line {}: expected >=7 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(SrpRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
        });
    }
    records
}

/// Compute angular difference accounting for wraparound at 2π.
pub fn angle_diff(a: f64, b: f64) -> f64 {
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

/// Max absolute element-wise difference between two 3×3 matrices.
pub fn max_mat_diff(a: &DMat3, b: &DMat3) -> f64 {
    let mut max_d = 0.0_f64;
    for c in 0..3 {
        for r in 0..3 {
            let d = (a.col(c)[r] - b.col(c)[r]).abs();
            max_d = max_d.max(d);
        }
    }
    max_d
}

fn read_csv(path: &Path, sim_name: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {sim_name} CSV from {}: {e}\n\
             Generate with: docker run --rm -v $(pwd)/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            path.display()
        )
    })
}
