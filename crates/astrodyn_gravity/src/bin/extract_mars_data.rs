//! Extract Mars, Sun, and Moon gravity fixtures from a JEOD source
//! checkout into `test_data/gravity/`.
//!
//! This is a **regen-only** path: it reads `$JEOD_HOME` or an explicit `--jeod-home <PATH>` argument, parses
//! `models/environment/gravity/data/src/mars_MRO110B2.cc`,
//! `models/environment/gravity/data/src/sun_spherical.cc`, and
//! `models/environment/gravity/data/src/moon_LP150Q.cc`, and writes the
//! committed binary blobs and JSON metadata sidecars consumed by
//! `astrodyn_gravity::fixtures`.
//!
//! Run after a JEOD upgrade or whenever the source data changes:
//!
//! ```bash
//! cargo run -p astrodyn_gravity --bin extract_mars_data
//! # or with an explicit JEOD path:
//! cargo run -p astrodyn_gravity --bin extract_mars_data -- --jeod-home /path/to/jeod
//! ```
//!
//! Outputs:
//! - `test_data/gravity/mars_mro110b2.bin` — Mars MRO110B2 SH coefficients
//!   (110×110, ~108 KB) in the production [`astrodyn_gravity::coefficients`]
//!   binary format.
//! - `test_data/gravity/mars_mro110b2.json` — metadata sidecar
//!   (source path, JEOD git rev, mu, radius, degree, order, tide_free).
//! - `test_data/gravity/sun_spherical.bin` — Sun point-mass encoded as a
//!   degree=1 SH (all coefficients zero) so callers may use the same
//!   binary loader uniformly. Only `mu` and `radius` are physically
//!   meaningful.
//! - `test_data/gravity/sun_spherical.json` — metadata sidecar.
//! - `test_data/gravity/moon_lp150q.bin` — Moon LP150Q SH coefficients
//!   (150×150, ~180 KB) in the production binary format. Used by Tier 3
//!   tests covering Earth–Moon dynamics.
//! - `test_data/gravity/moon_lp150q.json` — metadata sidecar.
//! - `test_data/gravity/moon_grail150.bin` — Moon GRAIL150 SH coefficients
//!   (150×150) — newer GRAIL-derived field used by SIM_dyncomp's
//!   third-body Moon source and by the gravity-gradient torque rigs.
//! - `test_data/gravity/moon_grail150.json` — metadata sidecar.
//!
//! Every sidecar carries `jeod_version`, `jeod_commit` (read from
//! `git rev-parse HEAD` of `$JEOD_HOME` at regen time; falls back to
//! `"unknown"` if the checkout is not a git tree), `generated_utc`,
//! `binary_file_bytes`, and `binary_file_sha256`. The hash is asserted by
//! `tests/fixture_metadata.rs` to catch drift between the committed
//! `.bin` and its recorded provenance.
//!
//! The binary prints each destination path on success.

use std::io::Write;
use std::path::{Path, PathBuf};

use astrodyn_gravity::coefficients::save_binary;
use astrodyn_gravity::jeod_cc::{load_from_jeod_cc, load_mu_from_jeod_cc};
use astrodyn_gravity::SphericalHarmonicsData;
use sha2::{Digest, Sha256};

/// Pinned JEOD version captured in every fixture sidecar.
const JEOD_VERSION: &str = "5.4";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jeod_root = resolve_jeod_root(&args).unwrap_or_else(|| {
        eprintln!(
            "extract_mars_data: JEOD source not found.\n\
             Pass `--jeod-home <PATH>` or set JEOD_HOME \
             (see CLAUDE.md \"Environment Setup\")."
        );
        std::process::exit(2);
    });
    assert!(
        jeod_root.exists(),
        "JEOD source root {} does not exist. Set JEOD_HOME \
         to a valid JEOD checkout.",
        jeod_root.display(),
    );

    let jeod_commit = read_git_rev(&jeod_root).unwrap_or_else(|| "unknown".to_string());
    let generated_utc = utc_now_iso8601();

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/gravity");
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
        panic!("Cannot create {}: {e}", out_dir.display());
    });

    extract_mars(&jeod_root, &jeod_commit, &generated_utc, &out_dir);
    extract_sun(&jeod_root, &jeod_commit, &generated_utc, &out_dir);
    extract_moon_lp150q(&jeod_root, &jeod_commit, &generated_utc, &out_dir);
    extract_moon_grail150(&jeod_root, &jeod_commit, &generated_utc, &out_dir);
}

fn extract_moon_grail150(jeod_root: &Path, jeod_commit: &str, generated_utc: &str, out_dir: &Path) {
    let rel = "models/environment/gravity/data/src/moon_GRAIL150.cc";
    let src_path = jeod_root.join(rel);
    let data = load_from_jeod_cc(&src_path).unwrap_or_else(|e| {
        panic!(
            "Failed to parse Moon GRAIL150 SH data from {}: {e:?}. \
             Ensure $JEOD_HOME points at a valid JEOD checkout containing this file.",
            src_path.display()
        );
    });

    let bin_path = out_dir.join("moon_grail150.bin");
    save_binary(&data, &bin_path)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", bin_path.display()));
    let (bin_size, bin_sha256) = read_size_and_sha256(&bin_path);

    let meta_path = out_dir.join("moon_grail150.json");
    write_metadata(
        &meta_path,
        rel,
        jeod_commit,
        generated_utc,
        "moon_grail150.bin",
        bin_size,
        &bin_sha256,
        Some(data.degree),
        Some(data.order),
        data.mu,
        data.radius,
        Some(data.tide_free),
        Some(data.tide_free_delta),
        "Moon GRAIL150 spherical harmonics gravity coefficients (degree=order=150). \
         Truncated from the GRAIL gggrx_0660pm_sha 660x660 field.",
    );

    println!(
        "wrote {} ({} bytes; sha256 {}) and {}",
        bin_path.display(),
        bin_size,
        bin_sha256,
        meta_path.display(),
    );
}

fn extract_moon_lp150q(jeod_root: &Path, jeod_commit: &str, generated_utc: &str, out_dir: &Path) {
    let rel = "models/environment/gravity/data/src/moon_LP150Q.cc";
    let src_path = jeod_root.join(rel);
    let data = load_from_jeod_cc(&src_path).unwrap_or_else(|e| {
        panic!(
            "Failed to parse Moon LP150Q SH data from {}: {e:?}. \
             Ensure $JEOD_HOME points at a valid JEOD checkout containing this file.",
            src_path.display()
        );
    });

    let bin_path = out_dir.join("moon_lp150q.bin");
    save_binary(&data, &bin_path)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", bin_path.display()));
    let (bin_size, bin_sha256) = read_size_and_sha256(&bin_path);

    let meta_path = out_dir.join("moon_lp150q.json");
    write_metadata(
        &meta_path,
        rel,
        jeod_commit,
        generated_utc,
        "moon_lp150q.bin",
        bin_size,
        &bin_sha256,
        Some(data.degree),
        Some(data.order),
        data.mu,
        data.radius,
        Some(data.tide_free),
        Some(data.tide_free_delta),
        "Moon LP150Q (Lunar Prospector) spherical harmonics gravity coefficients (degree=order=150).",
    );

    println!(
        "wrote {} ({} bytes; sha256 {}) and {}",
        bin_path.display(),
        bin_size,
        bin_sha256,
        meta_path.display(),
    );
}

fn extract_mars(jeod_root: &Path, jeod_commit: &str, generated_utc: &str, out_dir: &Path) {
    let rel = "models/environment/gravity/data/src/mars_MRO110B2.cc";
    let src_path = jeod_root.join(rel);
    let data = load_from_jeod_cc(&src_path).unwrap_or_else(|e| {
        panic!(
            "Failed to parse Mars SH data from {}: {e:?}. \
             Ensure $JEOD_HOME points at a valid JEOD checkout containing this file.",
            src_path.display()
        );
    });

    let bin_path = out_dir.join("mars_mro110b2.bin");
    save_binary(&data, &bin_path)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", bin_path.display()));
    let (bin_size, bin_sha256) = read_size_and_sha256(&bin_path);

    let meta_path = out_dir.join("mars_mro110b2.json");
    write_metadata(
        &meta_path,
        rel,
        jeod_commit,
        generated_utc,
        "mars_mro110b2.bin",
        bin_size,
        &bin_sha256,
        Some(data.degree),
        Some(data.order),
        data.mu,
        data.radius,
        Some(data.tide_free),
        Some(data.tide_free_delta),
        "Mars MRO110B2 spherical harmonics gravity coefficients (degree=order=110).",
    );

    println!(
        "wrote {} ({} bytes; sha256 {}) and {}",
        bin_path.display(),
        bin_size,
        bin_sha256,
        meta_path.display(),
    );
}

fn extract_sun(jeod_root: &Path, jeod_commit: &str, generated_utc: &str, out_dir: &Path) {
    let rel = "models/environment/gravity/data/src/sun_spherical.cc";
    let src_path = jeod_root.join(rel);
    let mu = load_mu_from_jeod_cc(&src_path).unwrap_or_else(|e| {
        panic!(
            "Failed to parse Sun mu from {}: {e:?}. \
             Ensure $JEOD_HOME points at a valid JEOD checkout containing this file.",
            src_path.display()
        );
    });
    let radius = load_radius_from_jeod_cc(&src_path).unwrap_or_else(|e| {
        panic!(
            "Failed to parse Sun radius from {}: {e:?}. \
             Ensure $JEOD_HOME points at a valid JEOD checkout containing this file.",
            src_path.display()
        );
    });

    // sun_spherical.cc is a point-mass entry (no degree / order / Cnm / Snm).
    // Encode as a degree=1 SH with all coefficients zero so the production
    // binary loader works uniformly. Only `mu` and `radius` are physically
    // meaningful — callers should use `load_sun_spherical_mu()`.
    let cnm: Vec<Vec<f64>> = vec![vec![0.0], vec![0.0, 0.0]];
    let snm: Vec<Vec<f64>> = vec![vec![0.0], vec![0.0, 0.0]];
    let data = SphericalHarmonicsData::new(
        1,      // degree
        1,      // order
        radius, // m
        mu,     // m^3/s^2
        cnm, snm, true, // tide_free (irrelevant for point-mass)
        0.0,  // tide_free_delta
    );

    let bin_path = out_dir.join("sun_spherical.bin");
    save_binary(&data, &bin_path)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", bin_path.display()));
    let (bin_size, bin_sha256) = read_size_and_sha256(&bin_path);

    let meta_path = out_dir.join("sun_spherical.json");
    write_metadata(
        &meta_path,
        rel,
        jeod_commit,
        generated_utc,
        "sun_spherical.bin",
        bin_size,
        &bin_sha256,
        None, // degree (semantically point-mass)
        None, // order
        mu,
        radius,
        None,
        None,
        "Sun point-mass gravity (mu and radius only). \
         Encoded as a degree=1 SH with all-zero coefficients so the production \
         binary loader works uniformly; only mu/radius are physically meaningful.",
    );

    println!(
        "wrote {} ({} bytes; sha256 {}) and {}",
        bin_path.display(),
        bin_size,
        bin_sha256,
        meta_path.display(),
    );
}

/// Parse `radius` from a JEOD `.cc` gravity file. Mirrors `load_mu_from_jeod_cc`
/// in `astrodyn_gravity::jeod_cc` but for the `radius` field.
fn load_radius_from_jeod_cc(path: &Path) -> Result<f64, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("I/O: {e}"))?;
    for line in content.lines() {
        let line = line.trim();
        if let Some(idx) = line.find("->radius = ") {
            let rest = &line[idx + "->radius = ".len()..];
            let expr: String = rest.chars().take_while(|c| *c != ';').collect();
            if let Some(val) = eval_simple_expr(&expr) {
                return Ok(val);
            }
        }
    }
    Err(format!("missing 'radius' in {}", path.display()))
}

fn eval_simple_expr(expr: &str) -> Option<f64> {
    let expr = expr.trim();
    if let Ok(val) = expr.parse::<f64>() {
        return Some(val);
    }
    if let Some(star_idx) = expr.find('*') {
        let lhs = expr[..star_idx].trim();
        let rhs = expr[star_idx + 1..]
            .trim()
            .trim_matches(|c| c == '(' || c == ')')
            .trim();
        if let (Ok(a), Ok(b)) = (lhs.parse::<f64>(), rhs.parse::<f64>()) {
            return Some(a * b);
        }
    }
    None
}

fn resolve_jeod_root(args: &[String]) -> Option<PathBuf> {
    if let Some(idx) = args.iter().position(|a| a == "--jeod-home") {
        if let Some(p) = args.get(idx + 1) {
            return Some(PathBuf::from(p));
        }
    }
    if let Ok(p) = std::env::var("JEOD_HOME") {
        return Some(PathBuf::from(p));
    }
    None
}

fn read_git_rev(jeod_root: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(jeod_root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Read a file's size and SHA-256 hex digest in one pass.
fn read_size_and_sha256(path: &Path) -> (u64, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    (bytes.len() as u64, format!("{:x}", hasher.finalize()))
}

/// Current UTC time formatted as `YYYY-MM-DDThh:mm:ssZ`.
fn utc_now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_unix_utc(secs)
}

// Howard Hinnant's civil-from-days algorithm: every intermediate (`doe`,
// `yoe`, `doy`, `mp`) is bounded by an explicit divisor (146 097, 400, …)
// well below `u32::MAX`. The truncations are exact by construction.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Hinnant civil-from-days intermediates bounded by divisor (<< u32::MAX)"
)]
fn format_unix_utc(secs: i64) -> String {
    let z = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let hour = sod / 3_600;
    let minute = (sod / 60) % 60;
    let second = sod % 60;

    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp.wrapping_sub(9) };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[allow(clippy::too_many_arguments)]
fn write_metadata(
    path: &Path,
    source_rel: &str,
    jeod_commit: &str,
    generated_utc: &str,
    binary_file: &str,
    binary_file_bytes: u64,
    binary_file_sha256: &str,
    degree: Option<usize>,
    order: Option<usize>,
    mu: f64,
    radius: f64,
    tide_free: Option<bool>,
    tide_free_delta: Option<f64>,
    note: &str,
) {
    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Cannot create {}: {e}", path.display()));
    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"schema_version\": 2,").unwrap();
    writeln!(f, "  \"source\": {},", json_str(source_rel)).unwrap();
    writeln!(f, "  \"jeod_version\": \"{JEOD_VERSION}\",").unwrap();
    writeln!(f, "  \"jeod_commit\": {},", json_str(jeod_commit)).unwrap();
    writeln!(f, "  \"generated_utc\": \"{generated_utc}\",").unwrap();
    writeln!(f, "  \"binary_file\": {},", json_str(binary_file)).unwrap();
    writeln!(f, "  \"binary_file_bytes\": {binary_file_bytes},").unwrap();
    writeln!(f, "  \"binary_file_sha256\": \"{binary_file_sha256}\",").unwrap();
    if let Some(d) = degree {
        writeln!(f, "  \"degree\": {d},").unwrap();
    }
    if let Some(o) = order {
        writeln!(f, "  \"order\": {o},").unwrap();
    }
    writeln!(f, "  \"mu_m3_per_s2\": {},", fmt_f64(mu)).unwrap();
    writeln!(f, "  \"radius_m\": {},", fmt_f64(radius)).unwrap();
    if let Some(t) = tide_free {
        writeln!(f, "  \"tide_free\": {t},").unwrap();
    }
    if let Some(d) = tide_free_delta {
        writeln!(f, "  \"tide_free_delta\": {},", fmt_f64(d)).unwrap();
    }
    writeln!(
        f,
        "  \"regen_command\": \"cargo run -p astrodyn_gravity --bin extract_mars_data\","
    )
    .unwrap();
    writeln!(f, "  \"note\": {}", json_str(note)).unwrap();
    writeln!(f, "}}").unwrap();
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn fmt_f64(x: f64) -> String {
    // Round-trippable f64 representation.
    format!("{x:?}")
}
