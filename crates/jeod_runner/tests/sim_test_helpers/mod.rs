//! Shared helpers for jeod_sim Tier 3 tests.
//!
//! Phase 8 of #101 hoisted the propagation utilities (state-error
//! metrics, energy conservation, periapsis detection, integrator
//! agreement, attach/detach scheduling, force/torque profiles, custom
//! CSV reader, parametric orbinit cases) into
//! [`jeod_sim::recipes::helpers`]. Phase 7 owns the typed CSV loaders
//! in `jeod_sim::recipes::verification::csv_loader`. What remains in
//! this file is:
//!
//! - Tests-only glue that depends on `jeod_test_data` types
//!   ([`mass_props_from_init`]).
//! - Schema-specific CSV loaders not yet absorbed by either Phase
//!   (`load_orbelem_csv`, `load_lvlh_csv`, `load_ned_csv`,
//!   `load_srp_trajectory`, …) — these stay until Phase 7's loader
//!   catalogue lands or until Phase 10's cleanup.
//! - The `test_data_path` resolver and `OMEGA_EARTH` re-export.
//!
//! Most tests will eventually import directly from
//! `jeod_sim::recipes::helpers::*`; for backward compatibility this
//! module re-exports the hoisted helpers under their original names so
//! the migration can proceed file-by-file without flag-day churn.

#![allow(dead_code, unused_imports)]

use glam::{DMat3, DVec3};
use jeod_sim::{JeodQuat, MassProperties};
use std::path::Path;

#[allow(unused_imports)] // Not all test binaries use dyncomp CSV loading
pub use jeod_test_data::dyncomp_csv::{load_dyncomp_csv, DyncompRecord};

// Re-exports from the recipes layer (Phase 8 hoist).
pub use jeod_sim::recipes::helpers::state_helpers::{
    angle_diff, dquat_angle_error, max_mat_diff, state_from_elements,
};

/// Earth rotation rate (JEOD RNPJ2000 default), sourced from
/// `jeod_sim::planet_config::EARTH.omega`.
pub const OMEGA_EARTH: f64 = jeod_sim::planet_config::EARTH.omega;

/// Build `MassProperties` from parsed JEOD mass initialization data.
///
/// Converts the row-major `[[f64; 3]; 3]` inertia tensor to glam `DMat3`
/// (column-major) and passes through mass and CoM position.
///
/// Stays here (rather than `recipes::helpers`) because it depends on
/// `jeod_test_data::mass_data::MassInitData`, which is test-data
/// plumbing — not a recipe.
pub fn mass_props_from_init(init: &jeod_test_data::mass_data::MassInitData) -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(init.inertia[0][0], init.inertia[1][0], init.inertia[2][0]),
        DVec3::new(init.inertia[0][1], init.inertia[1][1], init.inertia[2][1]),
        DVec3::new(init.inertia[0][2], init.inertia[1][2], init.inertia[2][2]),
    );
    MassProperties::with_inertia(init.mass, inertia, DVec3::from_slice(&init.position))
}

/// Quaternion angular error (back-compat alias for callers that
/// haven't migrated to `recipes::helpers::state_helpers::jeodquat_angle_error`).
pub fn quaternion_angle_error(q1: &JeodQuat, q2: &JeodQuat) -> f64 {
    jeod_sim::recipes::helpers::state_helpers::jeodquat_angle_error(q1, q2)
}

pub fn test_data_path(filename: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(filename)
}

// ── Derived-state CSV loaders ──
//
// `load_orbelem_csv`, `load_lvlh_csv`, `load_ned_csv`, `load_euler_csv`,
// `load_atmos_traj_csv`, and `load_aero_traj_csv` previously lived here
// but moved to `jeod_test_data::tier3_csv` when Phase 7 introduced
// `VerificationCaseExt::run_and_assert`. Tests that still parse these
// CSV layouts (orbelem_comprehensive, etc.) should import directly
// from `jeod_test_data::tier3_csv`. Loaders kept here are used only by
// tests that haven't migrated to `run_and_assert` yet.

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

// Phase 8 #110: `state_from_elements`, `angle_diff`, and `max_mat_diff`
// moved to `jeod_sim::recipes::helpers::state_helpers` and are re-
// exported from this module's preamble.

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
