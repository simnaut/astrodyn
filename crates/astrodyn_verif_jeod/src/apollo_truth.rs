//! Loader for `test_data/apollo_attach_truth.csv` — high-cadence (1 ms)
//! ground-truth recorder added to `APOLLO_SNIPPET` in
//! `trick/generate_references.sh` (commit 09c4327).
//!
//! The CSV records `composite_body` inertial state plus `mass.composite_properties`
//! for both `cm_dyn` and `lm_dyn` across the full 12 s SIM_Apollo run, at 1 ms
//! cadence. It is consumed by:
//!
//! - `crates/astrodyn_dynamics/tests/attach_with_jeod_truth.rs` — single-sample
//!   replay of `combine_states_at_attach` against JEOD inputs at t = 5.999.
//! - `crates/astrodyn_runner/tests/tier3_sim_apollo_trajectory.rs` — full-12 s
//!   diagnostic that compares our LM detached-subtree state against JEOD's
//!   recorded `lm_dyn.composite_body` at every event boundary.
//!
//! The CSV is gitignored (~18 MB). When missing, regenerate via
//! `cargo xtask regenerate-tier3 --force`. Loaders return `Err` rather than
//! panic so the consuming test can choose its own gating policy (typically
//! `#[ignore]` plus a manual run).
//!
//! ## Column layout
//!
//! Per `APOLLO_SNIPPET` in `trick/generate_references.sh` (search for
//! `attach_truth`):
//!
//! ```text
//! col   0: time
//! per vehicle (cm_dyn @ cols 1..36, lm_dyn @ cols 36..71,
//!              s3_dyn @ cols 71..106 — added in the #248 follow-up):
//!   +0..+5   : pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]
//!   +6       : q.scalar
//!   +7..+9   : q.vec[0..2]
//!   +10..+12 : ang_vel[0..2]
//!   +13      : mass
//!   +14..+16 : composite CoM struct[0..2]
//!   +17..+25 : inertia row-major (i*3+j)
//!   +26..+34 : T_parent_this row-major
//! ```
//!
//! Older CSVs (regenerated before the s3 columns were added) have only 71
//! numeric columns. The loader treats `s3` as `Option<VehState>` so it can
//! read both layouts; consumers that need s3 must check the field and
//! either fall back gracefully or panic with a clear regen instruction.

use std::path::{Path, PathBuf};

use astrodyn_math::JeodQuat;
use glam::{DMat3, DVec3};

/// Per-vehicle composite-body state as recorded by JEOD's truth recorder.
///
/// Position, velocity, and angular velocity are in Earth.inertial; the
/// quaternion is JEOD's `q_parent_this` (inertial → body, scalar-first).
/// `cm_struct` is the composite CoM in the vehicle's structural frame and
/// `t_struct_to_body` is `MassProperties::t_parent_this` (the structure →
/// composite-body rotation).
#[derive(Clone, Debug)]
pub struct VehState {
    /// RootInertial position of the composite CoM (m).
    pub position: DVec3,
    /// RootInertial velocity of the composite CoM (m/s).
    pub velocity: DVec3,
    /// RootInertial → body rotation, scalar-first JEOD convention.
    pub quaternion: JeodQuat,
    /// Body-frame angular velocity (rad/s).
    pub ang_vel_body: DVec3,
    /// Composite mass (kg).
    pub mass: f64,
    /// Composite CoM in the vehicle's structural frame (m).
    pub cm_struct: DVec3,
    /// Composite inertia tensor about the composite CoM, in body axes (kg·m²).
    pub inertia: DMat3,
    /// Structure → composite-body rotation (`MassProperties::t_parent_this`).
    pub t_struct_to_body: DMat3,
}

/// One row of `apollo_attach_truth.csv`. `cm` and `lm` are always
/// present; `s3` is `Some` only for CSVs regenerated after the #248
/// follow-up that added s3_dyn to the recorder. Older CSVs (71 numeric
/// columns) leave `s3 = None`; newer ones (≥ 106 columns) populate it.
#[derive(Clone, Debug)]
pub struct ApolloTruthRow {
    /// Simulation time (s) since the run started.
    pub time: f64,
    /// `cm_dyn` composite-body state.
    pub cm: VehState,
    /// `lm_dyn` composite-body state.
    pub lm: VehState,
    /// `s3_dyn` composite-body state — `Some` only when the CSV was
    /// regenerated with the extended recorder.
    pub s3: Option<VehState>,
}

/// Errors returned by [`load_apollo_attach_truth`].
#[derive(Debug, thiserror::Error)]
pub enum ApolloTruthError {
    /// The CSV is gitignored and was not regenerated locally.
    #[error(
        "{path} not found — regenerate via `cargo xtask regenerate-tier3 --force` \
         (the truth recorder is added to APOLLO_SNIPPET in trick/generate_references.sh)"
    )]
    Missing {
        /// Absolute path the loader looked for.
        path: PathBuf,
    },
    /// I/O error while reading the file.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Absolute path the loader looked for.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Successfully read but no parseable rows were found.
    #[error("{path} contained no parseable rows (expected ≥ 71 numeric columns per row)")]
    NoRows {
        /// Absolute path the loader looked for.
        path: PathBuf,
    },
    /// A row has an unexpected number of columns (must be 71 for the
    /// original cm+lm layout or 106 for the cm+lm+s3 extended layout).
    #[error("{path}:{line_no}: expected 71 or 106 columns, got {got}: {line:?}")]
    UnexpectedColumns {
        /// Absolute path the loader looked for.
        path: PathBuf,
        /// 1-indexed source line number (header counts as line 1).
        line_no: usize,
        /// Number of comma-separated fields actually seen.
        got: usize,
        /// The raw line text (truncated for the error message).
        line: String,
    },
    /// A column failed to parse as a `f64`.
    #[error("{path}:{line_no}: failed to parse column {col} ({field:?}) as f64: {source}")]
    ParseFloat {
        /// Absolute path the loader looked for.
        path: PathBuf,
        /// 1-indexed source line number.
        line_no: usize,
        /// 0-indexed column.
        col: usize,
        /// The raw text that failed to parse.
        field: String,
        /// Underlying parse error.
        #[source]
        source: std::num::ParseFloatError,
    },
}

/// Default location of `apollo_attach_truth.csv` in the workspace.
///
/// Resolves to `<workspace>/test_data/apollo_attach_truth.csv` from the
/// caller's `CARGO_MANIFEST_DIR`. Caller's manifest dir must sit two levels
/// below the workspace root (i.e. `crates/<name>/Cargo.toml`). For tests
/// living elsewhere, pass an explicit path to [`load_apollo_attach_truth_at`].
pub fn default_csv_path(manifest_dir: &str) -> PathBuf {
    PathBuf::from(manifest_dir).join("../../test_data/apollo_attach_truth.csv")
}

/// Load the truth CSV from the workspace's standard location, resolving via
/// the caller's `CARGO_MANIFEST_DIR`. See [`load_apollo_attach_truth_at`] for
/// loading from an explicit path.
pub fn load_apollo_attach_truth(
    manifest_dir: &str,
) -> Result<Vec<ApolloTruthRow>, ApolloTruthError> {
    load_apollo_attach_truth_at(&default_csv_path(manifest_dir))
}

/// Load the truth CSV from an explicit path. Returns `Err` if missing,
/// unreadable, or empty.
pub fn load_apollo_attach_truth_at(path: &Path) -> Result<Vec<ApolloTruthRow>, ApolloTruthError> {
    if !path.exists() {
        return Err(ApolloTruthError::Missing {
            path: path.to_path_buf(),
        });
    }
    let content = std::fs::read_to_string(path).map_err(|source| ApolloTruthError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    // Parse positionally and fail loudly on any column-count or parse
    // error. This is verification data — silently dropping malformed
    // rows shifts column indices and produces subtly-wrong test
    // results. Two valid widths are accepted:
    //   71 cols  — original cm + lm layout (time + 35 + 35).
    //   106 cols — cm + lm + s3 layout added with #248 follow-up.
    // Anything else is an error.
    for (row_idx, line) in content.lines().skip(1).enumerate() {
        let line_no = row_idx + 2; // 1-indexed; +2 accounts for skipped header.
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let n = fields.len();
        if n != 71 && n != 106 {
            return Err(ApolloTruthError::UnexpectedColumns {
                path: path.to_path_buf(),
                line_no,
                got: n,
                line: line.chars().take(200).collect(),
            });
        }
        let mut v = Vec::<f64>::with_capacity(n);
        for (col, field) in fields.iter().enumerate() {
            v.push(
                field
                    .parse::<f64>()
                    .map_err(|source| ApolloTruthError::ParseFloat {
                        path: path.to_path_buf(),
                        line_no,
                        col,
                        field: (*field).to_string(),
                        source,
                    })?,
            );
        }
        let s3 = if n >= 106 {
            Some(parse_veh(&v, 71))
        } else {
            None
        };
        out.push(ApolloTruthRow {
            time: v[0],
            cm: parse_veh(&v, 1),
            lm: parse_veh(&v, 36),
            s3,
        });
    }
    if out.is_empty() {
        return Err(ApolloTruthError::NoRows {
            path: path.to_path_buf(),
        });
    }
    Ok(out)
}

/// Return the row whose `time` is closest to `target_t`. Panics if `rows`
/// is empty (callers should hold a non-empty `Vec` from
/// [`load_apollo_attach_truth_at`]).
pub fn nearest_truth_at(rows: &[ApolloTruthRow], target_t: f64) -> &ApolloTruthRow {
    rows.iter()
        .min_by(|a, b| {
            (a.time - target_t)
                .abs()
                .partial_cmp(&(b.time - target_t).abs())
                .unwrap()
        })
        .expect("nearest_truth_at: empty truth rows — load_apollo_attach_truth returned no data")
}

fn parse_veh(v: &[f64], base: usize) -> VehState {
    VehState {
        position: DVec3::new(v[base], v[base + 2], v[base + 4]),
        velocity: DVec3::new(v[base + 1], v[base + 3], v[base + 5]),
        quaternion: JeodQuat::new(v[base + 6], v[base + 7], v[base + 8], v[base + 9]),
        ang_vel_body: DVec3::new(v[base + 10], v[base + 11], v[base + 12]),
        mass: v[base + 13],
        cm_struct: DVec3::new(v[base + 14], v[base + 15], v[base + 16]),
        inertia: dmat3_from_row_major(&v[base + 17..base + 26]),
        t_struct_to_body: dmat3_from_row_major(&v[base + 26..base + 35]),
    }
}

fn dmat3_from_row_major(row_major: &[f64]) -> DMat3 {
    // row_major[i*3+j] = M[i][j]. glam DMat3 is column-major.
    DMat3::from_cols(
        DVec3::new(row_major[0], row_major[3], row_major[6]),
        DVec3::new(row_major[1], row_major[4], row_major[7]),
        DVec3::new(row_major[2], row_major[5], row_major[8]),
    )
}
