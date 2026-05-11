//! Regenerate committed NESC reference CSVs from the NESC Academy site
//! (or from a locally-mirrored NESC artifact tree).
//!
//! Mirrors the `extract_*` pattern in `astrodyn_gravity` and
//! `astrodyn_verif_jeod` but works against the public NESC GN&C 2023
//! check-case data hosted under
//! `https://nescacademy.nasa.gov/flightsim/2023/scn_<N>/`.
//!
//! ## Workflow
//!
//! 1. (Optional) Pre-stage `Lunar_08_sim_<NN>.csv` files locally and
//!    point at them with `--nesc-home <DIR>` (the binary expects the
//!    NESC site's `scn_8/` subdirectory layout under that root).
//! 2. Otherwise, the binary downloads the upstream sim CSVs over HTTPS.
//! 3. The binary parses each sim CSV by **header name**, extracts the
//!    14 columns the runner-side test consumes, converts angular rates
//!    from deg/s to rad/s, and emits a canonical
//!    `cc8_nrho_reference.csv` under
//!    `crates/astrodyn_verif_nesc/test_data/`.
//!
//! Today's reference is **sim_01** by convention (one of 8 participating
//! NESC propagators). Six of eight sims report the same IC at their t=0
//! row to ≥ 9 decimal places, so the choice among the in-family sims is
//! immaterial at the IC. A consensus-of-8 methodology (median +
//! inter-sim spread tolerances) is tracked as a follow-up — see the
//! crate `README.md`.
//!
//! ## Run
//!
//! ```bash
//! # Fetch from NESC over HTTPS:
//! cargo run -p astrodyn_verif_nesc --bin extract_nesc
//!
//! # Or use a local mirror:
//! cargo run -p astrodyn_verif_nesc --bin extract_nesc -- --nesc-home /path/to/nesc
//! ```

use std::path::PathBuf;
use std::process::Command;

const NESC_BASE_URL: &str = "https://nescacademy.nasa.gov/flightsim/2023/scn_8";
const SIM_INDEX: &str = "01"; // sim_01 by convention; see module docs.

fn main() {
    let mut nesc_home: Option<PathBuf> = std::env::var_os("NESC_HOME").map(PathBuf::from);

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--nesc-home" {
            let val = args.next().expect("--nesc-home requires a path argument");
            nesc_home = Some(PathBuf::from(val));
        } else {
            eprintln!("extract_nesc: unrecognized argument: {arg}");
            std::process::exit(2);
        }
    }

    // Resolve workspace root + output path.
    let workspace_root = workspace_root();
    let out_dir = workspace_root.join("crates/astrodyn_verif_nesc/test_data");
    let out_path = out_dir.join("cc8_nrho_reference.csv");
    std::fs::create_dir_all(&out_dir).expect("create test_data/");

    // Source the raw sim CSV either from a local mirror or by HTTPS download.
    let raw = match nesc_home {
        Some(ref dir) => {
            let local = dir
                .join("scn_8")
                .join(format!("Lunar_08_sim_{SIM_INDEX}.csv"));
            eprintln!("extract_nesc: reading {}", local.display());
            std::fs::read_to_string(&local).unwrap_or_else(|e| {
                panic!(
                    "extract_nesc: failed to read {}: {e}.\n\
                     Expected the NESC site's `scn_8/` subdirectory under --nesc-home.",
                    local.display()
                )
            })
        }
        None => {
            let url = format!("{NESC_BASE_URL}/Lunar_08_sim_{SIM_INDEX}.csv");
            eprintln!("extract_nesc: downloading {url}");
            curl_get(&url)
        }
    };

    let canonical = transform_to_canonical(&raw);

    std::fs::write(&out_path, canonical).expect("write canonical CSV");
    eprintln!("extract_nesc: wrote {}", out_path.display());
}

/// Walk up from `CARGO_MANIFEST_DIR` to the workspace root.
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            return dir;
        }
        if !dir.pop() {
            panic!("extract_nesc: could not locate workspace root from CARGO_MANIFEST_DIR");
        }
    }
}

/// Shell out to `curl` to fetch the upstream artifact.
///
/// The crate intentionally has no `reqwest`/HTTP-client dependency —
/// `extract_nesc` is a developer tool that runs once per regen, and a
/// shelled `curl` keeps the dep graph clean.
fn curl_get(url: &str) -> String {
    let output = Command::new("curl")
        .args(["-fsSL", "--max-time", "300", url])
        .output()
        .unwrap_or_else(|e| panic!("extract_nesc: failed to invoke `curl`: {e}"));
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "extract_nesc: curl returned {} fetching {url}\n  stderr: {stderr}",
            output.status
        );
    }
    String::from_utf8(output.stdout).expect("upstream CSV is UTF-8")
}

/// Parse the NESC sim CSV by header name, project to our 14 canonical
/// columns, and return the canonical CSV text (with a small header
/// banner documenting source + column layout).
fn transform_to_canonical(raw: &str) -> String {
    // Split header from body.
    let mut lines = raw.lines();
    let header = lines.next().expect("nesc csv: empty input");
    let cols: Vec<&str> = header.split(',').collect();
    let idx = |name: &str| -> usize {
        cols.iter()
            .position(|c| c.trim() == name)
            .unwrap_or_else(|| panic!("nesc csv: missing required column {name}"))
    };
    // Required columns (deg/s on the angular-rate side; we convert to rad/s on emit).
    let i_t = idx("elapsedTime_s");
    let i_px = idx("miPosition_m_X");
    let i_py = idx("miPosition_m_Y");
    let i_pz = idx("miPosition_m_Z");
    let i_vx = idx("miVelocity_m_s_X");
    let i_vy = idx("miVelocity_m_s_Y");
    let i_vz = idx("miVelocity_m_s_Z");
    let i_qw = idx("quaternionWrtMi_W");
    let i_qx = idx("quaternionWrtMi_X");
    let i_qy = idx("quaternionWrtMi_Y");
    let i_qz = idx("quaternionWrtMi_Z");
    let i_wx = idx("bodyAngularRateWrtMi_deg_s_Roll");
    let i_wy = idx("bodyAngularRateWrtMi_deg_s_Pitch");
    let i_wz = idx("bodyAngularRateWrtMi_deg_s_Yaw");

    let d2r = std::f64::consts::PI / 180.0;
    let mut out = String::with_capacity(raw.len() / 4);
    out.push_str(
        "# CC8 NRHO canonical reference trajectory.\n\
         # Source: NESC sim_01 (Lunar_08_sim_01.csv) — one of 8 participating high-fidelity propagators.\n\
         # Columns: time(s), pos[3](m, MCI), vel[3](m/s, MCI), quat[4](W,X,Y,Z body-from-MCI), ang_vel[3](rad/s, body-frame)\n",
    );
    out.push_str("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,qw,qx,qy,qz,wx,wy,wz\n");

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let pf = |i: usize| -> f64 {
            f[i].trim().parse::<f64>().unwrap_or_else(|e| {
                panic!("nesc csv parse failure at column {i} value {:?}: {e}", f[i])
            })
        };
        let t = pf(i_t);
        let px = pf(i_px);
        let py = pf(i_py);
        let pz = pf(i_pz);
        let vx = pf(i_vx);
        let vy = pf(i_vy);
        let vz = pf(i_vz);
        let qw = pf(i_qw);
        let qx = pf(i_qx);
        let qy = pf(i_qy);
        let qz = pf(i_qz);
        let wx = pf(i_wx) * d2r;
        let wy = pf(i_wy) * d2r;
        let wz = pf(i_wz) * d2r;
        // 17 sig figs preserve f64 round-trip.
        let line_out = format!(
            "{t},{px:.17},{py:.17},{pz:.17},{vx:.17},{vy:.17},{vz:.17},\
             {qw:.17},{qx:.17},{qy:.17},{qz:.17},{wx:.17},{wy:.17},{wz:.17}\n"
        );
        out.push_str(&line_out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_handles_trivial_header() {
        let raw = "elapsedTime_s,miPosition_m_X,miPosition_m_Y,miPosition_m_Z,\
                   miVelocity_m_s_X,miVelocity_m_s_Y,miVelocity_m_s_Z,\
                   quaternionWrtMi_W,quaternionWrtMi_X,quaternionWrtMi_Y,quaternionWrtMi_Z,\
                   bodyAngularRateWrtMi_deg_s_Roll,bodyAngularRateWrtMi_deg_s_Pitch,bodyAngularRateWrtMi_deg_s_Yaw\n\
                   0,1,2,3,4,5,6,0.7,0.0,0.0,0.7,180,0,0\n";
        let out = transform_to_canonical(raw);
        // Header rows + two body lines (assert reasonable structure).
        assert!(out.contains("time,pos_x"));
        assert!(out.contains("0,1.0000000"));
        // 180 deg/s = π rad/s
        assert!(out.contains(&format!("{:.17}", std::f64::consts::PI)));
    }

    /// Robustness: extra trailing columns (sensor / DEM data some sims
    /// emit) must not break the projection — the header lookup is
    /// position-independent.
    #[test]
    fn transform_ignores_extra_trailing_columns() {
        let raw = "elapsedTime_s,miPosition_m_X,miPosition_m_Y,miPosition_m_Z,\
                   miVelocity_m_s_X,miVelocity_m_s_Y,miVelocity_m_s_Z,\
                   quaternionWrtMi_W,quaternionWrtMi_X,quaternionWrtMi_Y,quaternionWrtMi_Z,\
                   bodyAngularRateWrtMi_deg_s_Roll,bodyAngularRateWrtMi_deg_s_Pitch,bodyAngularRateWrtMi_deg_s_Yaw,\
                   miSensedPositionOfSensor_m_X,miSensedPositionOfSensor_m_Y,pamLatitudeOfTp1_deg\n\
                   0,1,2,3,4,5,6,0.7,0.0,0.0,0.7,1,2,3,99,99,99\n";
        let out = transform_to_canonical(raw);
        assert!(out.contains("time,pos_x"));
        assert!(out.lines().count() >= 5); // 3 banner + header + 1 data
    }

    #[test]
    fn _path_helper_resolves_under_workspace_root() {
        // Smoke test that workspace_root() finds Cargo.lock in the
        // running test environment — surfaces a regression early if a
        // future move breaks the relative depth assumption.
        let wr = workspace_root();
        assert!(
            wr.join("Cargo.lock").exists(),
            "Cargo.lock should exist at {wr:?}"
        );
    }
}
