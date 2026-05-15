//! Extract spherical-harmonics gravity coefficients from a JEOD source
//! checkout into committed binary fixtures.
//!
//! This is a **regen-only** path: it reads `$JEOD_HOME` or an explicit `--jeod-home <PATH>` argument, parses each
//! `models/environment/gravity/data/src/earth_GGM*.cc` file via
//! [`astrodyn_gravity::jeod_cc::load_from_jeod_cc`], and writes
//! `test_data/gravity/{ggm02c,ggm05c}.bin` using the production
//! [`astrodyn_gravity::coefficients::save_binary`] format. A sidecar
//! `{label}.json` records source provenance (path, file size, mu, degree,
//! order) plus an audit trail (JEOD commit SHA, generation timestamp,
//! SHA-256 of the produced binary) so reviewers can verify that the
//! committed `.bin` matches a specific upstream revision without
//! re-running the regen.
//!
//! Run after a JEOD upgrade or whenever the coefficient files change:
//!
//! ```bash
//! cargo run -p astrodyn_gravity --bin extract_grav_coeffs
//! cargo run -p astrodyn_gravity --bin extract_grav_coeffs -- \
//!     --jeod-home /path/to/jeod --out-dir test_data/gravity
//! ```
//!
//! The binary prints a summary of each generated file on success.

#![forbid(unsafe_code)]

use std::io::Write;
use std::path::{Path, PathBuf};

use astrodyn_gravity::coefficients::save_binary;
use astrodyn_gravity::jeod_cc;
use sha2::{Digest, Sha256};

/// One coefficient source to extract.
struct Source {
    /// File name under `models/environment/gravity/data/src/`.
    cc_filename: &'static str,
    /// Output label (no extension).
    label: &'static str,
}

const SOURCES: &[Source] = &[
    Source {
        cc_filename: "earth_GGM02C.cc",
        label: "ggm02c",
    },
    Source {
        cc_filename: "earth_GGM05C.cc",
        label: "ggm05c",
    },
    Source {
        cc_filename: "earth_GEMT1.cc",
        label: "gemt1",
    },
];

/// Pinned JEOD version captured in every fixture sidecar. Update when the
/// project bumps to a new upstream JEOD release; the per-fixture
/// `jeod_commit` field (read from `git rev-parse HEAD` at regen time)
/// provides the exact tree identity.
const JEOD_VERSION: &str = "5.4";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jeod_root = resolve_jeod_root(&args).unwrap_or_else(|| {
        eprintln!(
            "extract_grav_coeffs: JEOD source not found.\n\
             Pass `--jeod-home <PATH>` or set JEOD_HOME \
             (see CLAUDE.md \"Environment Setup\")."
        );
        std::process::exit(2);
    });
    let out_dir = resolve_out_dir(&args);

    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));

    let jeod_commit = read_git_rev(&jeod_root).unwrap_or_else(|| "unknown".to_string());
    let generated_utc = utc_now_iso8601();

    println!("JEOD root: {}", jeod_root.display());
    println!("JEOD commit: {jeod_commit}");
    println!("Output dir: {}", out_dir.display());
    println!();

    for source in SOURCES {
        process_source(&jeod_root, &out_dir, source, &jeod_commit, &generated_utc);
    }
}

fn process_source(
    jeod_root: &Path,
    out_dir: &Path,
    source: &Source,
    jeod_commit: &str,
    generated_utc: &str,
) {
    let cc_path = jeod_root
        .join("models/environment/gravity/data/src")
        .join(source.cc_filename);
    assert!(
        cc_path.exists(),
        "JEOD coefficient source not found at {}.\n\
         Verify your JEOD checkout includes {}.",
        cc_path.display(),
        source.cc_filename,
    );

    let cc_size = std::fs::metadata(&cc_path)
        .unwrap_or_else(|e| panic!("stat {}: {e}", cc_path.display()))
        .len();

    let data = jeod_cc::load_from_jeod_cc(&cc_path)
        .unwrap_or_else(|e| panic!("parse {}: {e}", cc_path.display()));

    let bin_path = out_dir.join(format!("{}.bin", source.label));
    save_binary(&data, &bin_path).unwrap_or_else(|e| panic!("write {}: {e}", bin_path.display()));
    let bin_bytes =
        std::fs::read(&bin_path).unwrap_or_else(|e| panic!("read {}: {e}", bin_path.display()));
    let bin_size = bin_bytes.len() as u64;
    let bin_sha256 = sha256_hex(&bin_bytes);

    let meta_path = out_dir.join(format!("{}.json", source.label));
    write_metadata(
        &meta_path,
        source,
        cc_size,
        bin_size,
        &bin_sha256,
        data.degree,
        data.order,
        data.radius,
        data.mu,
        data.tide_free,
        data.tide_free_delta,
        jeod_commit,
        generated_utc,
    );

    println!(
        "  {} -> {} ({} bytes; degree={}, order={}, mu={:.6e}, radius={:.3})",
        cc_path
            .strip_prefix(jeod_root)
            .unwrap_or(&cc_path)
            .display(),
        bin_path.display(),
        bin_size,
        data.degree,
        data.order,
        data.mu,
        data.radius,
    );
    println!("    sha256   {bin_sha256}");
    println!("    metadata -> {}", meta_path.display());
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

fn resolve_out_dir(args: &[String]) -> PathBuf {
    if let Some(idx) = args.iter().position(|a| a == "--out-dir") {
        if let Some(p) = args.get(idx + 1) {
            return PathBuf::from(p);
        }
    }
    // Default: <astrodyn_gravity-manifest>/test_data/gravity
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/gravity")
}

/// Read `git rev-parse HEAD` from the JEOD checkout, returning `None`
/// when the directory is not a git checkout (e.g., tarball mirror) or
/// `git` is unavailable. Callers fall back to `"unknown"`.
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

/// Compute SHA-256 of a byte slice as a lowercase hex string. Matches
/// `sha256sum` output for the same bytes.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Current UTC time formatted as `YYYY-MM-DDThh:mm:ssZ` (ISO-8601, no
/// sub-second precision). Implemented locally so the regen binaries
/// stay free of a `chrono`/`time` dep.
fn utc_now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_unix_utc(secs)
}

/// Convert a Unix timestamp (seconds since 1970-01-01 UTC) to an
/// ISO-8601 string. Uses the canonical civil-from-days algorithm
/// (Howard Hinnant, 2013) so it stays correct across the proleptic
/// Gregorian calendar without leap-second adjustment.
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

    // Days from epoch -> civil date (Hinnant 2013).
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
    out_path: &Path,
    source: &Source,
    cc_size: u64,
    bin_size: u64,
    bin_sha256: &str,
    degree: usize,
    order: usize,
    radius: f64,
    mu: f64,
    tide_free: bool,
    tide_free_delta: f64,
    jeod_commit: &str,
    generated_utc: &str,
) {
    let source_rel = format!("models/environment/gravity/data/src/{}", source.cc_filename);
    let mut f = std::fs::File::create(out_path)
        .unwrap_or_else(|e| panic!("create {}: {e}", out_path.display()));
    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"schema_version\": 2,").unwrap();
    writeln!(f, "  \"label\": \"{}\",", source.label).unwrap();
    writeln!(f, "  \"source_file\": \"{}\",", source_rel).unwrap();
    writeln!(f, "  \"source_file_bytes\": {},", cc_size).unwrap();
    writeln!(f, "  \"jeod_version\": \"{JEOD_VERSION}\",").unwrap();
    writeln!(f, "  \"jeod_commit\": \"{jeod_commit}\",").unwrap();
    writeln!(f, "  \"generated_utc\": \"{generated_utc}\",").unwrap();
    writeln!(f, "  \"binary_file\": \"{}.bin\",", source.label).unwrap();
    writeln!(f, "  \"binary_file_bytes\": {},", bin_size).unwrap();
    writeln!(f, "  \"binary_file_sha256\": \"{bin_sha256}\",").unwrap();
    writeln!(f, "  \"degree\": {},", degree).unwrap();
    writeln!(f, "  \"order\": {},", order).unwrap();
    writeln!(f, "  \"radius_m\": {:?},", radius).unwrap();
    writeln!(f, "  \"mu_m3_per_s2\": {:?},", mu).unwrap();
    writeln!(f, "  \"tide_free\": {},", tide_free).unwrap();
    writeln!(f, "  \"tide_free_delta\": {:?},", tide_free_delta).unwrap();
    writeln!(
        f,
        "  \"generated_by\": \"cargo run -p astrodyn_gravity --bin extract_grav_coeffs\","
    )
    .unwrap();
    writeln!(
        f,
        "  \"note\": \"Regenerate after a JEOD upgrade or coefficient-file change. The .bin file uses the astrodyn_gravity::coefficients::save_binary format (magic JEOD, version 1). The binary_file_sha256 field is asserted by tests/fixture_metadata.rs to detect drift between the committed binary and its recorded provenance.\""
    )
    .unwrap();
    writeln!(f, "}}").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_unix_utc_known_values() {
        // 1970-01-01T00:00:00Z
        assert_eq!(format_unix_utc(0), "1970-01-01T00:00:00Z");
        // 2024-01-01T00:00:00Z = 1_704_067_200
        assert_eq!(format_unix_utc(1_704_067_200), "2024-01-01T00:00:00Z");
        // 2026-05-14T12:34:56Z = 1_778_762_096
        assert_eq!(format_unix_utc(1_778_762_096), "2026-05-14T12:34:56Z");
    }

    #[test]
    fn sha256_matches_known_vector() {
        // Empty input SHA-256.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
