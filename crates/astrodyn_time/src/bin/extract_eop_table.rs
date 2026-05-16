//! Extract the IERS EOP TAI→UT1 lookup table from a JEOD source
//! checkout into the committed binary fixture.
//!
//! This is a **regen-only** path: it reads `$JEOD_HOME` or an explicit
//! `--jeod-home <PATH>` argument, parses
//! `models/environment/time/data/src/tai_to_ut1.cc` via
//! [`astrodyn_time::EopTable::parse_jeod_cc`], and writes
//! `test_data/eop/iers_eop_c04.bin` using the production
//! [`astrodyn_time::EopTable::save_binary`] format. A sidecar
//! `iers_eop_c04.json` records source provenance (path, file size,
//! entry count, covered TAI TJT range) plus an audit trail (JEOD
//! commit SHA, generation timestamp, SHA-256 of the produced binary)
//! so reviewers can verify that the committed `.bin` matches a
//! specific upstream revision without re-running the regen.
//!
//! Run after a JEOD upgrade or whenever the IERS EOP series is
//! refreshed:
//!
//! ```bash
//! cargo run -p astrodyn_time --bin extract_eop_table
//! cargo run -p astrodyn_time --bin extract_eop_table -- \
//!     --jeod-home /path/to/jeod --out-dir test_data/eop
//! ```
//!
//! The binary prints a summary of the generated fixture on success.

#![forbid(unsafe_code)]

use std::io::Write;
use std::path::{Path, PathBuf};

use astrodyn_time::EopTable;
use sha2::{Digest, Sha256};

/// Pinned JEOD version captured in every fixture sidecar. Update when
/// the project bumps to a new upstream JEOD release; the
/// `jeod_commit` field (read from `git rev-parse HEAD` at regen time)
/// provides the exact tree identity.
const JEOD_VERSION: &str = "5.4";

/// Relative path from `$JEOD_HOME` to the generated TAI↔UT1 source.
const JEOD_TAI_TO_UT1_REL: &str = "models/environment/time/data/src/tai_to_ut1.cc";

/// Output label (basename, no extension) under the destination
/// directory.
const OUTPUT_LABEL: &str = "iers_eop_c04";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jeod_root = resolve_jeod_root(&args).unwrap_or_else(|| {
        eprintln!(
            "extract_eop_table: JEOD source not found.\n\
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

    let cc_path = jeod_root.join(JEOD_TAI_TO_UT1_REL);
    assert!(
        cc_path.exists(),
        "JEOD TAI↔UT1 source not found at {}.\n\
         Verify your JEOD checkout includes {JEOD_TAI_TO_UT1_REL}.",
        cc_path.display()
    );
    let cc_size = std::fs::metadata(&cc_path)
        .unwrap_or_else(|e| panic!("stat {}: {e}", cc_path.display()))
        .len();
    let src = std::fs::read_to_string(&cc_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", cc_path.display()));
    let table = EopTable::parse_jeod_cc(&src)
        .unwrap_or_else(|e| panic!("parse {}: {e}", cc_path.display()));

    let bin_path = out_dir.join(format!("{OUTPUT_LABEL}.bin"));
    table
        .save_binary(&bin_path)
        .unwrap_or_else(|e| panic!("write {}: {e}", bin_path.display()));
    let bin_bytes =
        std::fs::read(&bin_path).unwrap_or_else(|e| panic!("read {}: {e}", bin_path.display()));
    let bin_size = bin_bytes.len() as u64;
    let bin_sha256 = sha256_hex(&bin_bytes);
    let entry_count = table.len();
    let first_tjt = table.first_tai_tjt();
    let last_tjt = table.last_tai_tjt();

    let meta_path = out_dir.join(format!("{OUTPUT_LABEL}.json"));
    write_metadata(
        &meta_path,
        cc_size,
        bin_size,
        &bin_sha256,
        entry_count,
        first_tjt,
        last_tjt,
        &jeod_commit,
        &generated_utc,
    );

    println!(
        "  {} -> {} ({} bytes; {entry_count} daily entries; TAI TJT {first_tjt}..{last_tjt})",
        cc_path
            .strip_prefix(&jeod_root)
            .unwrap_or(&cc_path)
            .display(),
        bin_path.display(),
        bin_size,
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
    // Default: <astrodyn_time-manifest>/test_data/eop
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/eop")
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

// Howard Hinnant's civil-from-days algorithm: every intermediate
// (`doe`, `yoe`, `doy`, `mp`) is bounded by an explicit divisor
// (146 097, 400, …) well below `u32::MAX`. The truncations are exact
// by construction.
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
    cc_size: u64,
    bin_size: u64,
    bin_sha256: &str,
    entry_count: usize,
    first_tjt: f64,
    last_tjt: f64,
    jeod_commit: &str,
    generated_utc: &str,
) {
    let mut f = std::fs::File::create(out_path)
        .unwrap_or_else(|e| panic!("create {}: {e}", out_path.display()));
    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"schema_version\": 1,").unwrap();
    writeln!(f, "  \"label\": \"{OUTPUT_LABEL}\",").unwrap();
    writeln!(f, "  \"source_file\": \"{JEOD_TAI_TO_UT1_REL}\",").unwrap();
    writeln!(f, "  \"source_file_bytes\": {cc_size},").unwrap();
    writeln!(f, "  \"jeod_version\": \"{JEOD_VERSION}\",").unwrap();
    writeln!(f, "  \"jeod_commit\": \"{jeod_commit}\",").unwrap();
    writeln!(f, "  \"generated_utc\": \"{generated_utc}\",").unwrap();
    writeln!(f, "  \"binary_file\": \"{OUTPUT_LABEL}.bin\",").unwrap();
    writeln!(f, "  \"binary_file_bytes\": {bin_size},").unwrap();
    writeln!(f, "  \"binary_file_sha256\": \"{bin_sha256}\",").unwrap();
    writeln!(f, "  \"entry_count\": {entry_count},").unwrap();
    writeln!(f, "  \"tai_tjt_first\": {first_tjt:?},").unwrap();
    writeln!(f, "  \"tai_tjt_last\": {last_tjt:?},").unwrap();
    writeln!(
        f,
        "  \"generated_by\": \"cargo run -p astrodyn_time --bin extract_eop_table\","
    )
    .unwrap();
    writeln!(
        f,
        "  \"note\": \"Regenerate after a JEOD upgrade or IERS EOP refresh. The .bin file uses the EopTable::save_binary format (magic EOPT, version 1). The IERS source series is EOP 14 C04 (jeod/models/environment/time/data/eopc04_14_IAU2000.62-now); JEOD's parser.py converts UT1-UTC to UT1-TAI by subtracting the day's leap-second value.\""
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
