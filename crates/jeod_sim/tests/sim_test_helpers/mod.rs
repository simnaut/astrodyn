//! Shared helpers for jeod_sim Tier 3 tests.
//!
//! Provides CSV parsing for SIM_dyncomp and derived-state trajectory data,
//! quaternion error computation, and test data path resolution.

#![allow(dead_code)]

use glam::{DMat3, DQuat, DVec3};
use jeod_sim::JeodQuat;
use std::path::Path;

#[allow(unused_imports)] // Not all test binaries use dyncomp CSV loading
pub use jeod_test_data::dyncomp_csv::{load_dyncomp_csv, DyncompRecord};

pub const MU_EARTH: f64 = 3.986_004_415e14;
pub const DT: f64 = 0.03125; // 32 Hz, matches JEOD SIM_dyncomp

/// Earth rotation rate (JEOD RNPJ2000 default).
pub const OMEGA_EARTH: f64 = 7.292_115_146_706_388e-5;

pub fn quaternion_angle_error(q1: &JeodQuat, q2: &JeodQuat) -> f64 {
    let dot = (q1.scalar() * q2.scalar()
        + q1.vector().x * q2.vector().x
        + q1.vector().y * q2.vector().y
        + q1.vector().z * q2.vector().z)
        .abs();
    2.0 * dot.min(1.0).acos()
}

/// Angle error between two glam `DQuat` values (radians).
pub fn dquat_angle_error(a: DQuat, b: DQuat) -> f64 {
    let dot = a.dot(b).abs();
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

// ── SIM_torque_compare_simple CSV loader (26 columns) ──

#[derive(Debug)]
pub struct TorqueSimpleRecord {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub ang_vel: DVec3,
    pub t_parent_this: DMat3,
    pub quaternion: JeodQuat,
    pub gravity_torque: DVec3,
}

pub fn load_torque_simple_csv(path: &Path) -> Vec<TorqueSimpleRecord> {
    let content = read_csv(path, "SIM_torque_compare_simple");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 26,
            "line {}: expected >=26 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        // Cols 10-18: T_parent_this row-major T[row][col]
        // glam DMat3::from_cols is column-major: col0=(T00,T10,T20), etc.
        let t_parent_this = DMat3::from_cols(
            DVec3::new(p(10), p(13), p(16)),
            DVec3::new(p(11), p(14), p(17)),
            DVec3::new(p(12), p(15), p(18)),
        );
        // Cols 19-21: Q.vector[0..2], Col 22: Q.scalar (JEOD scalar-first)
        let quaternion = JeodQuat::new(p(22), p(19), p(20), p(21));
        records.push(TorqueSimpleRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
            ang_vel: DVec3::new(p(7), p(8), p(9)),
            t_parent_this,
            quaternion,
            gravity_torque: DVec3::new(p(23), p(24), p(25)),
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

// ── SIM_VER_DRAG CSV loader (11 columns) ──

#[derive(Debug)]
pub struct DragRecord {
    pub time: f64,
    pub aero_force: DVec3,
    pub aero_torque: DVec3,
    pub inertial_vel: DVec3,
    pub accel_mag: f64,
}

pub fn load_drag_csv(path: &Path) -> Vec<DragRecord> {
    let content = read_csv(path, "SIM_VER_DRAG");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 11,
            "line {}: expected >=11 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(DragRecord {
            time: p(0),
            aero_force: DVec3::new(p(1), p(2), p(3)),
            aero_torque: DVec3::new(p(4), p(5), p(6)),
            inertial_vel: DVec3::new(p(7), p(8), p(9)),
            accel_mag: p(10),
        });
    }
    records
}

// ── SIM_1_BASIC CSV loader (9 columns) ──

#[derive(Debug)]
pub struct SrpBasicRecord {
    pub time: f64,
    pub force: DVec3,
    pub torque: DVec3,
    pub flux_mag: f64,
    pub temperature: f64,
}

pub fn load_srp_basic_csv(path: &Path) -> Vec<SrpBasicRecord> {
    let content = read_csv(path, "SIM_1_BASIC");
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
        records.push(SrpBasicRecord {
            time: p(0),
            force: DVec3::new(p(1), p(2), p(3)),
            torque: DVec3::new(p(4), p(5), p(6)),
            flux_mag: p(7),
            temperature: p(8),
        });
    }
    records
}

// ── SIM_2A_SHADOW_CALC CSV loader (11 columns) ──

#[derive(Debug)]
pub struct ShadowCalcRecord {
    pub time: f64,
    pub position: DVec3,
    pub flux_mag: f64,
    pub force: DVec3,
    pub torque: DVec3,
}

pub fn load_shadow_calc_csv(path: &Path) -> Vec<ShadowCalcRecord> {
    let content = read_csv(path, "SIM_2A_SHADOW_CALC");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 11,
            "line {}: expected >=11 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(ShadowCalcRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            flux_mag: p(4),
            force: DVec3::new(p(5), p(6), p(7)),
            torque: DVec3::new(p(8), p(9), p(10)),
        });
    }
    records
}

// ── SIM_orbinit CSV loader (7 columns) ──

#[derive(Debug)]
pub struct OrbInitRecord {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

pub fn load_orbinit_csv(path: &Path) -> Vec<OrbInitRecord> {
    let content = read_csv(path, "SIM_orbinit");
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
        records.push(OrbInitRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
        });
    }
    records
}

// ── SIM_GJ_test CSV loader (7 columns: time + pos[3] + vel[3]) ──

pub fn load_gj_csv(path: &Path) -> Vec<OrbInitRecord> {
    let content = read_csv(path, "SIM_GJ_test");
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
        records.push(OrbInitRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
        });
    }
    records
}

// ── Euler CSV loader (56 columns: time + 36 angles + 6 pos/vel + 9 T + 4 quat) ──

#[derive(Debug)]
pub struct EulerRecord {
    pub time: f64,
    /// 6 sequences x 2 forms (ref_body, body_ref) x 3 angles = 36 values.
    /// Layout: [seq0_ref_body[3], seq0_body_ref[3], seq1_ref_body[3], ...]
    pub angles: [f64; 36],
    pub position: DVec3,
    pub velocity: DVec3,
    pub t_parent_this: DMat3,
    pub quaternion: JeodQuat,
}

pub fn load_euler_csv(path: &Path) -> Vec<EulerRecord> {
    let content = read_csv(path, "SIM_Euler");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 56,
            "line {}: expected >=56 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        let mut angles = [0.0_f64; 36];
        for (j, angle) in angles.iter_mut().enumerate() {
            *angle = p(1 + j);
        }
        // Cols 37-42: position[3], velocity[3]
        let position = DVec3::new(p(37), p(38), p(39));
        let velocity = DVec3::new(p(40), p(41), p(42));
        // Cols 43-51: T_parent_this row-major T[row][col]
        let t_parent_this = DMat3::from_cols(
            DVec3::new(p(43), p(46), p(49)),
            DVec3::new(p(44), p(47), p(50)),
            DVec3::new(p(45), p(48), p(51)),
        );
        // Cols 52-54: Q.vector[0..2], Col 55: Q.scalar
        let quaternion = JeodQuat::new(p(55), p(52), p(53), p(54));
        records.push(EulerRecord {
            time: p(0),
            angles,
            position,
            velocity,
            t_parent_this,
            quaternion,
        });
    }
    records
}

// ── SolarBeta CSV loader (8 columns: time + beta + 3×(pos,vel) interleaved) ──

#[derive(Debug)]
pub struct SolarBetaRecord {
    pub time: f64,
    pub solar_beta: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

pub fn load_solar_beta_csv(path: &Path) -> Vec<SolarBetaRecord> {
    let content = read_csv(path, "SIM_SolarBeta");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 8,
            "line {}: expected >=8 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        // SOLARBETA_SNIPPET: pos/vel interleaved (pos[0],vel[0],pos[1],vel[1],pos[2],vel[2])
        records.push(SolarBetaRecord {
            time: p(0),
            solar_beta: p(1),
            position: DVec3::new(p(2), p(4), p(6)),
            velocity: DVec3::new(p(3), p(5), p(7)),
        });
    }
    records
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
