//! Centralized CSV loaders for Tier 3 verification trajectories.
//!
//! Phase 7 of #101 consolidates the per-test CSV parsers that previously lived
//! in `astrodyn_runner/tests/sim_test_helpers/mod.rs` into a single typed loader
//! exposed to the verification machinery.
//!
//! Each `load_*` function reads one of JEOD's `log_*_ASCII.csv` formats and
//! returns a `Vec` of typed records. Records keep their natural per-row shape
//! (position/velocity, derived elements, etc.); higher-level conversions to
//! [`crossval::StateLog`](super::crossval::StateLog) are done by callers when
//! they want the unified comparison report.

use crate::crossval::StateLog;
use glam::{DMat3, DVec3};
use std::path::{Path, PathBuf};

pub use super::dyncomp_csv::{load_dyncomp_csv, DyncompRecord, FrameDerivs, FrameState};

/// Resolve a fixture under `crates/astrodyn_verif_jeod/test_data/`.
///
/// This is the home for Tier 3 reference CSVs, Apollo `.out` files,
/// `baselines.{json,md}`, the JEOD-source mirror under `jeod_inputs/`,
/// and the JEOD-derived Tier 2 reference data under `body_init/` and
/// `jeod_validation/`. Cross-cutting fixtures (gravity coefficients,
/// leap-second table, ephemerides, planet seeds) are *not* under this
/// path — they live with their owner crates and have dedicated
/// resolvers in their owner crates (e.g. `astrodyn_gravity::fixtures`,
/// `astrodyn_planet::geodetic_verif`).
pub fn test_data_path(filename: &str) -> PathBuf {
    workspace_root()
        .join("crates/astrodyn_verif_jeod/test_data")
        .join(filename)
}

/// Walk up from `CARGO_MANIFEST_DIR` to the workspace root (the directory
/// containing `Cargo.lock`). Falls back to a best-effort `../..` if no
/// `Cargo.lock` is reachable.
pub fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            return dir;
        }
        if !dir.pop() {
            return PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .map(PathBuf::from)
                .expect("CARGO_MANIFEST_DIR has at least two ancestors");
        }
    }
}

fn read_csv(path: &Path, sim_name: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {sim_name} CSV from {}: {e}\n\
             Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            path.display()
        )
    })
}

// ── SIM_OrbElem CSV (21+ columns) ──────────────────────────────────────────

/// One row from the SIM_OrbElem 21-column reference CSV.
#[derive(Debug)]
pub struct OrbElemRecord {
    /// Sample time in seconds since simulation t=0.
    pub time: f64,
    /// Semi-major axis in metres.
    pub semi_major_axis: f64,
    /// Eccentricity magnitude.
    pub e_mag: f64,
    /// Inclination in radians.
    pub inclination: f64,
    /// Argument of periapsis in radians.
    pub arg_periapsis: f64,
    /// Longitude of ascending node in radians.
    pub long_asc_node: f64,
    /// True anomaly in radians.
    pub true_anom: f64,
    /// Mean anomaly in radians.
    pub mean_anom: f64,
    /// Cartesian position in metres.
    pub position: DVec3,
    /// Cartesian velocity in m/s.
    pub velocity: DVec3,
}

/// Parse a SIM_OrbElem CSV at `path` into [`OrbElemRecord`] rows.
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

// ── SIM_LVLH CSV (17+ columns) ─────────────────────────────────────────────

/// One row from a SIM_LVLH 17-column reference CSV.
#[derive(Debug)]
pub struct LvlhRecord {
    /// Sample time in seconds since simulation t=0.
    pub time: f64,
    /// LVLH parent-to-this rotation matrix.
    pub t_parent_this: DMat3,
    /// Magnitude of the LVLH angular velocity in rad/s.
    pub ang_vel_mag: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
}

/// Parse a SIM_LVLH CSV at `path` into [`LvlhRecord`] rows.
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

// ── SIM_NED CSV (16+ columns) ──────────────────────────────────────────────

/// One row from a SIM_NED 16-column reference CSV.
#[derive(Debug)]
pub struct NedRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// NED frame origin (the body's sub-point) in planet-fixed coordinates,
    /// metres — JEOD `ned_state.cart_coords`.
    pub cart_coords: DVec3,
    /// Geodetic altitude in metres.
    pub ellip_altitude: f64,
    /// Geodetic latitude in radians.
    pub ellip_latitude: f64,
    /// Geodetic longitude in radians.
    pub ellip_longitude: f64,
    /// Spherical altitude in metres.
    pub sphere_altitude: f64,
    /// Geocentric latitude in radians.
    pub sphere_latitude: f64,
    /// Spherical longitude in radians.
    pub sphere_longitude: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
}

/// Parse a SIM_NED CSV at `path` into [`NedRecord`] rows.
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
        records.push(NedRecord {
            time: p(0),
            cart_coords: DVec3::new(p(1), p(2), p(3)),
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

// ── SrpRecord (7 columns: time + pos[3] + vel[3]) ──────────────────────────

/// One row from a SIM_3_ORBIT SRP-trajectory CSV (7 columns).
#[derive(Debug)]
pub struct SrpRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
}

/// Parse a SIM_3_ORBIT SRP-trajectory CSV at `path`.
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

// ── SIM_1_BASIC SRP CSV (9 columns) ────────────────────────────────────────

/// One row from a SIM_1_BASIC SRP CSV (9 columns).
#[derive(Debug)]
pub struct SrpBasicRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// SRP force in newtons.
    pub force: DVec3,
    /// SRP torque in N·m.
    pub torque: DVec3,
    /// Solar-flux magnitude in W/m².
    pub flux_mag: f64,
    /// Surface temperature in K.
    pub temperature: f64,
}

/// Parse a SIM_1_BASIC SRP CSV at `path`.
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

// ── SIM_VER_DRAG CSV (11 columns) ──────────────────────────────────────────

/// One row from a SIM_VER_DRAG CSV (11 columns).
#[derive(Debug)]
pub struct DragRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Aerodynamic force in newtons.
    pub aero_force: DVec3,
    /// Aerodynamic torque in N·m.
    pub aero_torque: DVec3,
    /// Inertial velocity in m/s.
    pub inertial_vel: DVec3,
    /// Aerodynamic-acceleration magnitude in m/s².
    pub accel_mag: f64,
}

/// Parse a SIM_VER_DRAG CSV at `path`.
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

// ── SIM_Euler CSV (56 columns: time + 36 angles + 6 pos/vel + 9 T + 4 quat) ─

/// One row from a SIM_Euler CSV (56 columns).
#[derive(Debug)]
pub struct EulerRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// 36 Euler-angle samples (12 sequences × 3 angles).
    pub angles: [f64; 36],
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
    /// Parent-to-this rotation matrix.
    pub t_parent_this: DMat3,
    /// JEOD scalar-first: `[q0_scalar, q1, q2, q3]`.
    pub quaternion: [f64; 4],
}

/// Parse a SIM_Euler CSV at `path` into [`EulerRecord`] rows.
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
        let position = DVec3::new(p(37), p(38), p(39));
        let velocity = DVec3::new(p(40), p(41), p(42));
        let t_parent_this = DMat3::from_cols(
            DVec3::new(p(43), p(46), p(49)),
            DVec3::new(p(44), p(47), p(50)),
            DVec3::new(p(45), p(48), p(51)),
        );
        // Cols 52-54: Q.vector[0..2], Col 55: Q.scalar (JEOD scalar-first)
        let quaternion = [p(55), p(52), p(53), p(54)];
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

// ── SIM_SolarBeta CSV (8 columns) ──────────────────────────────────────────

/// One row from a SIM_SolarBeta CSV (8 columns).
#[derive(Debug)]
pub struct SolarBetaRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Solar beta angle in radians.
    pub solar_beta: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
}

/// Parse a SIM_SolarBeta CSV at `path`.
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
        records.push(SolarBetaRecord {
            time: p(0),
            solar_beta: p(1),
            position: DVec3::new(p(2), p(4), p(6)),
            velocity: DVec3::new(p(3), p(5), p(7)),
        });
    }
    records
}

// ── SIM_2A_SHADOW_CALC CSV (11 columns) ────────────────────────────────────

/// One row from a SIM_2A_SHADOW_CALC CSV (11 columns).
#[derive(Debug)]
pub struct ShadowCalcRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Solar-flux magnitude in W/m².
    pub flux_mag: f64,
    /// SRP force in newtons.
    pub force: DVec3,
    /// SRP torque in N·m.
    pub torque: DVec3,
}

/// Parse a SIM_2A_SHADOW_CALC CSV at `path`.
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

// ── SIM_torque_compare_simple CSV (26 columns) ─────────────────────────────

/// One row from a SIM_torque_compare_simple CSV (26 columns).
#[derive(Debug)]
pub struct TorqueSimpleRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
    /// Body angular velocity in rad/s.
    pub ang_vel: DVec3,
    /// Parent-to-this rotation matrix.
    pub t_parent_this: DMat3,
    /// JEOD scalar-first: `[q0_scalar, q1, q2, q3]`.
    pub quaternion: [f64; 4],
    /// Gravity-gradient torque in N·m.
    pub gravity_torque: DVec3,
}

/// Parse a SIM_torque_compare_simple CSV at `path`.
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
        let t_parent_this = DMat3::from_cols(
            DVec3::new(p(10), p(13), p(16)),
            DVec3::new(p(11), p(14), p(17)),
            DVec3::new(p(12), p(15), p(18)),
        );
        let quaternion = [p(22), p(19), p(20), p(21)];
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

// ── SIM_dyncomp atmosphere trajectory (9 columns) ──────────────────────────

/// One row from a SIM_dyncomp atmosphere-trajectory CSV (9 columns).
#[derive(Debug)]
pub struct AtmosTrajRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
    /// Atmospheric density in kg/m³.
    pub density: f64,
    /// Atmospheric temperature in K.
    pub temperature: f64,
}

/// Parse a SIM_dyncomp atmosphere-trajectory CSV at `path`.
pub fn load_atmos_traj_csv(path: &Path) -> Vec<AtmosTrajRecord> {
    let content = read_csv(path, "SIM_dyncomp (atmos_traj)");
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
        records.push(AtmosTrajRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
            density: p(7),
            temperature: p(8),
        });
    }
    records
}

// ── SIM_dyncomp aero trajectory (14 columns) ───────────────────────────────

/// One row from a SIM_dyncomp aero-trajectory CSV (14 columns).
#[derive(Debug)]
pub struct AeroTrajRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
    /// Aerodynamic force in newtons.
    pub aero_force: DVec3,
    /// Aerodynamic torque in N·m.
    pub aero_torque: DVec3,
    /// Atmospheric density in kg/m³.
    pub density: f64,
}

/// Parse a SIM_dyncomp aero-trajectory CSV at `path`.
pub fn load_aero_traj_csv(path: &Path) -> Vec<AeroTrajRecord> {
    let content = read_csv(path, "SIM_dyncomp (aero_traj)");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 14,
            "line {}: expected >=14 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(AeroTrajRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
            aero_force: DVec3::new(p(7), p(8), p(9)),
            aero_torque: DVec3::new(p(10), p(11), p(12)),
            density: p(13),
        });
    }
    records
}

// ── SIM_orbinit CSV (7 columns: time + pos[3] + vel[3]) ───────────────────

/// One row from a SIM_orbinit CSV (7 columns: time + pos + vel).
#[derive(Debug)]
pub struct OrbInitRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
}

/// Parse a SIM_orbinit CSV at `path`.
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

/// One row from a full-state SIM_orbinit CSV (14 columns: time +
/// `pos[3]` + `vel[3]` + quaternion `vector[3]` + quaternion scalar +
/// `ang_vel[3]`). Used by the rotational-init RUNs (RUN_1230 / RUN_2100)
/// where the comparison covers attitude and rate, not just position /
/// velocity.
#[derive(Debug)]
pub struct OrbInitFullStateRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
    /// Composite-body attitude quaternion `[q0, q1, q2, q3]`, JEOD
    /// scalar-first left-transformation (parent→body) convention.
    pub quaternion: [f64; 4],
    /// Body-frame angular velocity in rad/s
    /// (`composite_body.state.rot.ang_vel_this`).
    pub ang_vel_body: DVec3,
}

/// Parse a full-state SIM_orbinit CSV at `path`.
///
/// Column layout (matching `ORBINIT_ROT_SNIPPET` in
/// `trick/generate_references.sh`): `time, pos[0..2], vel[0..2],
/// Q_parent_this.vector[0..2], Q_parent_this.scalar, ang_vel_this[0..2]`.
/// The scalar component lands at column 10; the returned `quaternion`
/// reorders it to JEOD's scalar-first `[q0, q1, q2, q3]` layout.
pub fn load_orbinit_full_state_csv(path: &Path) -> Vec<OrbInitFullStateRecord> {
    let content = read_csv(path, "SIM_orbinit (full state)");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 14,
            "line {}: full-state SIM_orbinit CSV expected >=14 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(OrbInitFullStateRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
            // Columns 7,8,9 = vector[0..2]; column 10 = scalar.
            quaternion: [p(10), p(7), p(8), p(9)],
            ang_vel_body: DVec3::new(p(11), p(12), p(13)),
        });
    }
    records
}

/// Same shape as [`OrbInitRecord`] (time + pos + vel). Used by SIM_GJ_test.
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

// ── SIM_Relative CSV (57 columns) ──────────────────────────────────────────

/// One row from the SIM_Relative reference CSV (57-column layout).
///
/// Columns 0–25 are vehicle A's state with position / velocity stored
/// **interleaved per axis** (`px0, vx0, py0, vy0, pz0, vz0, q0, q1,
/// q2, q3, …, ω`); columns 26–50 mirror the layout for vehicle B;
/// columns 51–56 carry JEOD's own `compute_relative_state` output for
/// vehicle A relative to vehicle B (`rel_pos[3]` then `rel_vel[3]`,
/// not interleaved).
///
/// The quaternion columns are emitted by JEOD multiple times (the
/// SIM_Relative log records the same `Q_parent_this` against four
/// different rotation matrices; only the first scalar+vector pair is
/// the body quaternion). This loader captures the first scalar+vector
/// at columns 7–10 / 32–35 — the same window the bespoke
/// `tier3_sim_relative.rs` parser used.
#[derive(Debug)]
#[allow(dead_code)]
pub struct RelativeRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Vehicle A inertial position.
    pub veh_a_pos: DVec3,
    /// Vehicle A inertial velocity.
    pub veh_a_vel: DVec3,
    /// Vehicle A scalar-first JEOD quaternion `[q0, q1, q2, q3]`.
    pub veh_a_quat: [f64; 4],
    /// Vehicle A body-frame angular velocity.
    pub veh_a_ang_vel: DVec3,
    /// Vehicle B inertial position.
    pub veh_b_pos: DVec3,
    /// Vehicle B inertial velocity.
    pub veh_b_vel: DVec3,
    /// Vehicle B scalar-first JEOD quaternion.
    pub veh_b_quat: [f64; 4],
    /// Vehicle B body-frame angular velocity.
    pub veh_b_ang_vel: DVec3,
    /// JEOD-logged relative position (vehicle A minus vehicle B,
    /// rotated into vehicle B's body frame when both have rotational
    /// state, else in inertial — matches our
    /// `compute_relative_state::<SelfRef, SelfRef>` output).
    pub jeod_rel_pos: DVec3,
    /// JEOD-logged relative velocity (same frame as `jeod_rel_pos`).
    pub jeod_rel_vel: DVec3,
}

/// Parse a SIM_Relative CSV at `path` into [`RelativeRecord`] rows.
pub fn load_relative_csv(path: &Path) -> Vec<RelativeRecord> {
    let content = read_csv(path, "SIM_Relative");
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 57,
            "line {}: expected >=57 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(RelativeRecord {
            time: p(0),
            // vehA: interleaved pos/vel per axis at columns 1..7.
            veh_a_pos: DVec3::new(p(1), p(3), p(5)),
            veh_a_vel: DVec3::new(p(2), p(4), p(6)),
            veh_a_quat: [p(7), p(8), p(9), p(10)],
            veh_a_ang_vel: DVec3::new(p(23), p(24), p(25)),
            // vehB: interleaved pos/vel at columns 26..32.
            veh_b_pos: DVec3::new(p(26), p(28), p(30)),
            veh_b_vel: DVec3::new(p(27), p(29), p(31)),
            veh_b_quat: [p(32), p(33), p(34), p(35)],
            veh_b_ang_vel: DVec3::new(p(48), p(49), p(50)),
            // JEOD-logged relative state (cols 51..57): rel_pos[3] then rel_vel[3].
            jeod_rel_pos: DVec3::new(p(51), p(52), p(53)),
            jeod_rel_vel: DVec3::new(p(54), p(55), p(56)),
        });
    }
    records
}

// ── SIM_tide_verif CSV (8 columns: time + pos[3] + vel[3] + dC20) ──────────

/// One row from a SIM_tide_verif CSV (8 columns).
#[derive(Debug)]
pub struct TideRecord {
    /// Sample time in seconds.
    pub time: f64,
    /// Body position in metres.
    pub position: DVec3,
    /// Body velocity in m/s.
    pub velocity: DVec3,
    /// Tidal ΔC20 correction logged by JEOD's SIM_tide_verif.
    pub delta_c20: f64,
}

/// Parse a SIM_tide_verif CSV at `path`.
pub fn load_tide_csv(path: &Path) -> Vec<TideRecord> {
    let content = read_csv(path, "SIM_tide_verif");
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
        records.push(TideRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
            delta_c20: p(7),
        });
    }
    records
}

// ── SIM_ref_attach CSV (14 columns) ────────────────────────────────────────

/// One row from a SIM_ref_attach `ref_attach_*_ref_attach_state.csv`
/// reference (14 columns: time, `pos[3]`, `vel[3]`, `q_scalar`,
/// `q_vec[3]`, `ang_vel[3]`).
#[derive(Debug, Clone)]
pub struct RefAttachRecord {
    /// Sample time in seconds since simulation t=0.
    pub time: f64,
    /// Composite-body inertial position in metres.
    pub position: DVec3,
    /// Composite-body inertial velocity in m/s.
    pub velocity: DVec3,
    /// Composite-body left-quaternion scalar component (JEOD's `[q0,
    /// q1, q2, q3]` layout, scalar-first). Kept for follow-up attitude
    /// validation; the SIM_ref_attach scenarios have no rotational
    /// dynamics pre-attach and the post-attach attitude is derived from
    /// the parent frame composed with the captured `t_pframe_struct`,
    /// so attitude drift already manifests as position / velocity error
    /// through the rigid-body composition.
    #[allow(dead_code)]
    pub quat_scalar: f64,
    /// Composite-body left-quaternion vector component (`[q1, q2, q3]`).
    #[allow(dead_code)]
    pub quat_vec: DVec3,
    /// Body-frame angular velocity (rad/s).
    #[allow(dead_code)]
    pub ang_vel_body: DVec3,
}

/// Parse a SIM_ref_attach `ref_attach_*_ref_attach_state.csv` at `path`.
///
/// `keep_only_integer_seconds_for_dt` filters out fractional-second
/// rows: SIM_ref_attach's `IntegLoop` runs at `DYNAMICS = 1.0` s but
/// Trick's logger samples at 0.5 s, so the half-second rows simply
/// repeat the previous integer-second integrator output. Comparing
/// against those would mix an integer-second state (our `step_until`
/// integration cadence) with a half-second CSV index that doesn't
/// correspond to an integration step — keep only rows where
/// `time / dt` is integer (within an epsilon).
pub fn load_ref_attach_csv(
    path: &Path,
    keep_only_integer_seconds_for_dt: f64,
) -> Vec<RefAttachRecord> {
    let content = read_csv(path, "SIM_ref_attach");
    assert!(
        keep_only_integer_seconds_for_dt > 0.0,
        "load_ref_attach_csv: filter dt must be positive, got {keep_only_integer_seconds_for_dt}"
    );
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if i == 0 || trimmed.is_empty() {
            continue;
        }
        let f: Vec<&str> = trimmed.split(',').collect();
        assert!(
            f.len() == 14,
            "line {}: SIM_ref_attach CSV expected 14 columns, got {} ({:?})",
            i + 1,
            f.len(),
            trimmed,
        );
        let p = |idx: usize| -> f64 {
            f[idx].trim().parse().unwrap_or_else(|e| {
                panic!(
                    "line {}: SIM_ref_attach CSV column {idx} parse failed for {:?}: {e}",
                    i + 1,
                    f[idx]
                )
            })
        };
        let time = p(0);
        // Drop fractional-second rows: the integrator runs at integer
        // seconds and the half-second rows hold the previous tick's
        // state. `time / dt - (time / dt).round()` is the f64-rounding
        // distance from the nearest integer multiple; 1e-9 absorbs CSV
        // formatting jitter.
        let n = time / keep_only_integer_seconds_for_dt;
        if (n - n.round()).abs() > 1e-9 {
            continue;
        }
        records.push(RefAttachRecord {
            time,
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
            quat_scalar: p(7),
            quat_vec: DVec3::new(p(8), p(9), p(10)),
            ang_vel_body: DVec3::new(p(11), p(12), p(13)),
        });
    }
    assert!(
        !records.is_empty(),
        "SIM_ref_attach CSV at {} contained no integer-second data rows",
        path.display(),
    );
    records
}

// ── Dyncomp helpers ────────────────────────────────────────────────────────

/// Convert a [`DyncompRecord`] into a [`StateLog`] using its `composite_body`
/// frame (3-DOF subset: position + velocity only, plus translational
/// acceleration when present).
pub fn dyncomp_to_state_log_3dof(r: &DyncompRecord) -> StateLog {
    StateLog {
        time: r.time,
        position: Some(r.composite_body.position),
        velocity: Some(r.composite_body.velocity),
        acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
        ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
        ..Default::default()
    }
}

/// Convert a [`DyncompRecord`] into a 6-DOF [`StateLog`] using its
/// `composite_body` frame.
pub fn dyncomp_to_state_log_6dof(r: &DyncompRecord) -> StateLog {
    StateLog {
        time: r.time,
        position: Some(r.composite_body.position),
        velocity: Some(r.composite_body.velocity),
        acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
        quaternion: Some(r.composite_body.quaternion),
        ang_vel: Some(r.composite_body.ang_vel),
        ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
    }
}
