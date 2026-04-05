//! Unified parser for the 80-column SIM_dyncomp CSV format.
//!
//! JEOD's `log_state_ASCII` logs three reference frames (composite_body,
//! core_body, structure) plus frame derivatives. This module parses all
//! columns once into [`DyncompRecord`], which any test can use.
//!
//! Column layout (0-indexed):
//! - 0: time
//! - 1..22: composite_body frame state (position, velocity, ang_vel, T, Q)
//! - 23..44: core_body frame state
//! - 45..66: structure frame state
//! - 67..79: derivs (non_grav_accel, trans_accel, rot_accel, Qdot)
//!
//! Only non_grav_accel, trans_accel, and rot_accel are parsed from the
//! derivs block. Qdot_parent_this (columns 70/74/78 vector + 79 scalar)
//! is present in the CSV but not extracted — it is not needed for
//! cross-validation.

use glam::{DMat3, DQuat, DVec3};
use std::path::Path;

/// State of a single JEOD reference frame at one timestep.
#[derive(Debug, Clone)]
pub struct FrameState {
    pub position: DVec3,
    pub velocity: DVec3,
    pub ang_vel: DVec3,
    pub t_parent_this: DMat3,
    pub quaternion: DQuat,
}

/// Frame derivatives at one timestep.
#[derive(Debug, Clone)]
pub struct FrameDerivs {
    pub non_grav_accel: DVec3,
    pub trans_accel: DVec3,
    pub rot_accel: DVec3,
}

/// One row from the 80-column SIM_dyncomp state CSV.
#[derive(Debug, Clone)]
pub struct DyncompRecord {
    pub time: f64,
    pub composite_body: FrameState,
    pub core_body: FrameState,
    pub structure: FrameState,
    pub derivs: Option<FrameDerivs>,
}

/// Parse a single frame state from 22 CSV columns starting at `base` (0-indexed).
///
/// Per-axis layout (stride 7):
///   +0: position\[i\], +1: velocity\[i\], +2: ang_vel\[i\],
///   +3..+5: T_parent_this\[i\]\[0..2\], +6: Q.vector\[i\]
/// After 3 axes (offset 21): Q.scalar
fn parse_frame(f: &[&str], base: usize, p: &dyn Fn(&str) -> f64) -> FrameState {
    let position = DVec3::new(p(f[base]), p(f[base + 7]), p(f[base + 14]));
    let velocity = DVec3::new(p(f[base + 1]), p(f[base + 8]), p(f[base + 15]));
    let ang_vel = DVec3::new(p(f[base + 2]), p(f[base + 9]), p(f[base + 16]));

    // T_parent_this: JEOD stores row-major T[row][col], glam uses column-major
    let t_parent_this = DMat3::from_cols(
        DVec3::new(p(f[base + 3]), p(f[base + 10]), p(f[base + 17])),
        DVec3::new(p(f[base + 4]), p(f[base + 11]), p(f[base + 18])),
        DVec3::new(p(f[base + 5]), p(f[base + 12]), p(f[base + 19])),
    );

    // Q_parent_this: scalar-first JEOD convention → glam DQuat(x,y,z,w)
    let q_scalar = p(f[base + 21]);
    let q_vec = DVec3::new(p(f[base + 6]), p(f[base + 13]), p(f[base + 20]));
    let quaternion = DQuat::from_xyzw(q_vec.x, q_vec.y, q_vec.z, q_scalar);

    FrameState {
        position,
        velocity,
        ang_vel,
        t_parent_this,
        quaternion,
    }
}

/// Parse frame derivatives from CSV columns 67..79 (0-indexed).
///
/// Per-axis layout (stride 4, starting at 67):
///   +0: non_grav_accel\[i\], +1: trans_accel\[i\], +2: rot_accel\[i\], +3: Qdot.vector\[i\]
fn parse_derivs(f: &[&str], p: &dyn Fn(&str) -> f64) -> FrameDerivs {
    FrameDerivs {
        non_grav_accel: DVec3::new(p(f[67]), p(f[71]), p(f[75])),
        trans_accel: DVec3::new(p(f[68]), p(f[72]), p(f[76])),
        rot_accel: DVec3::new(p(f[69]), p(f[73]), p(f[77])),
    }
}

/// Load a SIM_dyncomp state CSV (80-column format).
///
/// Parses all three frames and derivatives. Gracefully handles CSVs with
/// fewer than 80 columns by omitting optional sections.
pub fn load_dyncomp_csv(path: &Path) -> Vec<DyncompRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_dyncomp CSV from {}: {e}",
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
            f.len() >= 23,
            "line {}: expected >=23 columns (composite_body frame), got {}",
            i + 1,
            f.len()
        );

        let line_no = i + 1;
        let p = |s: &str| -> f64 {
            s.trim().parse().unwrap_or_else(|e| {
                panic!(
                    "{}: line {line_no}: failed to parse {s:?} as f64: {e}",
                    path.display()
                )
            })
        };

        let composite_body = parse_frame(&f, 1, &p);

        let core_body = if f.len() >= 45 {
            parse_frame(&f, 23, &p)
        } else {
            composite_body.clone()
        };

        let structure = if f.len() >= 67 {
            parse_frame(&f, 45, &p)
        } else {
            composite_body.clone()
        };

        // Derivs block: highest index is f[77] (rot_accel[2]), so >= 78.
        let derivs = if f.len() >= 78 {
            Some(parse_derivs(&f, &p))
        } else {
            None
        };

        records.push(DyncompRecord {
            time: p(f[0]),
            composite_body,
            core_body,
            structure,
            derivs,
        });
    }
    records
}
