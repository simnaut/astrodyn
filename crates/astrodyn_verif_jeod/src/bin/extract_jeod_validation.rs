//! Extract / verify the `astrodyn_math` validation fixtures.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "regen-tool sizes and counts fit exactly in f64 mantissa and target int width"
)]
//!
//! This is a **regen-only** path: it reads `$JEOD_HOME` (or an explicit
//! `--jeod-home <PATH>` argument), parses
//!
//! - `models/utils/orbital_elements/verif/SIM_orb_elem/Modified_data/orb_ell_in.py`
//!   → `test_data/jeod_validation/orbital_vectors.bin` (binary, ~5000 records)
//! - `models/dynamics/derived_state/verif/unit_tests/euler_derived_state_ut.cc`
//!   → `test_data/jeod_validation/euler_cases.json` (deduplicated 1-case
//!   dataset).
//! - `models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt`
//!   → verbatim copy at `test_data/gravity/grav_geospherical_verif_out.txt`
//!   (40 lines, plain text; consumed by
//!   `crate::gravity_verif::load_gravity_test_cases`).
//! - `models/environment/time/data/Leap_Second.dat`
//!   → verbatim copy at `test_data/time/Leap_Second.dat` (consumed by
//!   `crate::leap_second::load_leap_second_table`).
//! - `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/mass.py`
//!   → verbatim copy at `test_data/body_init/iss_mass.py` (consumed by
//!   `astrodyn_verif_jeod::mass_data::load_mass_data`).
//!
//! Run after a JEOD upgrade or whenever the source data changes:
//!
//! ```bash
//! cargo run -p astrodyn_verif_jeod --bin extract_jeod_validation
//! # or with an explicit JEOD path:
//! cargo run -p astrodyn_verif_jeod --bin extract_jeod_validation -- --jeod-home /path/to/jeod
//! ```
//!
//! Every output file is paired with a JSON sidecar (either inline for
//! files whose contents and provenance share a JSON document, or a
//! `<file>.meta.json` for verbatim text mirrors) carrying
//! `jeod_version`, `jeod_commit` (`git rev-parse HEAD` of `$JEOD_HOME`
//! at regen time; falls back to `"unknown"` if the checkout is not a
//! git tree), `generated_utc`, and (for binary / canonical files) the
//! file's byte count + SHA-256. The hash is asserted by the workspace-
//! level `tests/fixture_metadata.rs` so a regen that drops or
//! desynchronises a sidecar fails CI.

use std::io::Write;
use std::path::{Path, PathBuf};

use astrodyn_verif_jeod::euler_test::{encode_euler_cases_json, parse_euler_test_cases_cc};
use astrodyn_verif_jeod::orbital_data::{
    encode_orbital_vectors_bin, parse_orbital_test_vectors_py,
};
use sha2::{Digest, Sha256};

/// Pinned JEOD version captured in every fixture sidecar.
const JEOD_VERSION: &str = "5.4";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jeod_root = resolve_jeod_root(&args).unwrap_or_else(|| {
        eprintln!(
            "extract_jeod_validation: JEOD source not found.\n\
             Pass `--jeod-home <PATH>` or set JEOD_HOME (see CLAUDE.md \"Environment Setup\")."
        );
        std::process::exit(2);
    });
    assert!(
        jeod_root.exists(),
        "JEOD source root {} does not exist. Set JEOD_HOME to a valid JEOD checkout.",
        jeod_root.display(),
    );

    let workspace = workspace_root();
    let verif_data = workspace.join("crates/astrodyn_verif_jeod/test_data");
    let out_dir = verif_data.join("jeod_validation");
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
        panic!("Cannot create {}: {e}", out_dir.display());
    });

    let jeod_commit = read_git_rev(&jeod_root).unwrap_or_else(|| "unknown".to_string());
    let generated_utc = utc_now_iso8601();

    extract_orbital_vectors(&jeod_root, &out_dir, &jeod_commit, &generated_utc);
    extract_euler_cases(&jeod_root, &out_dir, &jeod_commit, &generated_utc);
    extract_grav_geospherical_verif_out(
        &jeod_root,
        &workspace.join("crates/astrodyn_gravity/test_data/gravity"),
        &jeod_commit,
        &generated_utc,
    );
    copy_verbatim(
        &jeod_root,
        "models/environment/time/data/Leap_Second.dat",
        &workspace.join("crates/astrodyn_time/test_data/Leap_Second.dat"),
        &jeod_commit,
        &generated_utc,
    );
    copy_verbatim(
        &jeod_root,
        "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/mass.py",
        &verif_data.join("body_init").join("iss_mass.py"),
        &jeod_commit,
        &generated_utc,
    );
}

/// Read `git rev-parse HEAD` from the JEOD checkout. Returns `None` when
/// the directory is not a git checkout (tarball mirror) or `git` is
/// unavailable; callers fall back to `"unknown"`.
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn utc_now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_unix_utc(secs)
}

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

fn copy_verbatim(jeod_root: &Path, rel: &str, dst: &Path, jeod_commit: &str, generated_utc: &str) {
    let src = jeod_root.join(rel);
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Cannot create {}: {e}", parent.display()));
    }
    std::fs::copy(&src, dst)
        .unwrap_or_else(|e| panic!("Cannot copy {} -> {}: {e}", src.display(), dst.display()));
    let bytes_buf = std::fs::read(dst).unwrap_or_else(|e| panic!("read {}: {e}", dst.display()));
    let sha = sha256_hex(&bytes_buf);
    let bytes = bytes_buf.len() as u64;
    println!(
        "  {} -> {} ({bytes} bytes; sha256 {sha}; verbatim copy)",
        rel,
        dst.display(),
    );
    write_verbatim_sidecar(dst, rel, jeod_commit, generated_utc, bytes, &sha);
}

fn extract_grav_geospherical_verif_out(
    jeod_root: &Path,
    out_dir: &Path,
    jeod_commit: &str,
    generated_utc: &str,
) {
    let rel = "models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt";
    let src = jeod_root.join(rel);
    std::fs::create_dir_all(out_dir)
        .unwrap_or_else(|e| panic!("Cannot create {}: {e}", out_dir.display()));
    let dst = out_dir.join("grav_geospherical_verif_out.txt");
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("Cannot copy {} -> {}: {e}", src.display(), dst.display()));
    let bytes_buf = std::fs::read(&dst).unwrap_or_else(|e| panic!("read {}: {e}", dst.display()));
    let sha = sha256_hex(&bytes_buf);
    let bytes = bytes_buf.len() as u64;
    println!(
        "  {} -> {} ({bytes} bytes; sha256 {sha}; verbatim copy)",
        rel,
        dst.display(),
    );
    write_verbatim_sidecar(&dst, rel, jeod_commit, generated_utc, bytes, &sha);
}

/// Write a `<file>.meta.json` next to a verbatim-mirrored fixture so the
/// audit trail (JEOD source path + commit + size + SHA-256) lives next
/// to the file it describes.
fn write_verbatim_sidecar(
    dst: &Path,
    source_rel: &str,
    jeod_commit: &str,
    generated_utc: &str,
    bytes: u64,
    sha256: &str,
) {
    let file_name = dst
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("(unknown)");
    let meta_path = dst.with_file_name(format!("{file_name}.meta.json"));
    let meta = format!(
        "{{\n  \"schema_version\": 2,\n  \
         \"source\": \"{source_rel}\",\n  \
         \"jeod_version\": \"{JEOD_VERSION}\",\n  \
         \"jeod_commit\": \"{jeod_commit}\",\n  \
         \"generated_utc\": \"{generated_utc}\",\n  \
         \"verbatim_copy\": true,\n  \
         \"reference_file\": \"{file_name}\",\n  \
         \"reference_file_bytes\": {bytes},\n  \
         \"reference_file_sha256\": \"{sha256}\",\n  \
         \"note\": \"Verbatim mirror of a JEOD source file. Regenerate with: cargo run -p astrodyn_verif_jeod --bin extract_jeod_validation\"\n}}\n",
    );
    std::fs::write(&meta_path, meta)
        .unwrap_or_else(|e| panic!("write {}: {e}", meta_path.display()));
}

fn extract_orbital_vectors(
    jeod_root: &Path,
    out_dir: &Path,
    jeod_commit: &str,
    generated_utc: &str,
) {
    let rel = "models/utils/orbital_elements/verif/SIM_orb_elem/Modified_data/orb_ell_in.py";
    let src = jeod_root.join(rel);
    let content = std::fs::read_to_string(&src)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", src.display()));
    let vectors = parse_orbital_test_vectors_py(&content);
    assert!(
        !vectors.is_empty(),
        "{}: parsed 0 orbital vectors — JEOD source may have changed format",
        src.display(),
    );

    let bin_path = out_dir.join("orbital_vectors.bin");
    let blob = encode_orbital_vectors_bin(&vectors);
    let mut f = std::fs::File::create(&bin_path)
        .unwrap_or_else(|e| panic!("Cannot create {}: {e}", bin_path.display()));
    f.write_all(&blob)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", bin_path.display()));
    let bin_sha256 = sha256_hex(&blob);
    println!(
        "  {} -> {} ({} bytes; sha256 {}; {} vectors)",
        rel,
        bin_path.display(),
        blob.len(),
        bin_sha256,
        vectors.len(),
    );

    let meta_path = out_dir.join("orbital_vectors.json");
    let meta = format!(
        "{{\n  \"schema_version\": 2,\n  \"source\": \"{rel}\",\n  \
         \"jeod_version\": \"{JEOD_VERSION}\",\n  \
         \"jeod_commit\": \"{jeod_commit}\",\n  \
         \"generated_utc\": \"{generated_utc}\",\n  \
         \"binary_file\": \"orbital_vectors.bin\",\n  \
         \"binary_file_bytes\": {},\n  \
         \"binary_file_sha256\": \"{bin_sha256}\",\n  \
         \"vector_count\": {},\n  \
         \"format\": \"u32 count (LE) followed by count * 6 * f64 (LE) = pos[3] vel[3] in SI\",\n  \
         \"note\": \"Regenerate with: cargo run -p astrodyn_verif_jeod --bin extract_jeod_validation\"\n}}\n",
        blob.len(),
        vectors.len(),
    );
    std::fs::write(&meta_path, meta)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", meta_path.display()));
    println!("  metadata -> {}", meta_path.display());
}

fn extract_euler_cases(jeod_root: &Path, out_dir: &Path, jeod_commit: &str, generated_utc: &str) {
    let rel = "models/dynamics/derived_state/verif/unit_tests/euler_derived_state_ut.cc";
    let src = jeod_root.join(rel);
    let content = std::fs::read_to_string(&src)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", src.display()));
    let mut cases = parse_euler_test_cases_cc(&content);
    cases.dedup();
    assert!(
        !cases.is_empty(),
        "{}: parsed 0 Euler test cases — JEOD source may have changed format",
        src.display(),
    );

    // `euler_cases.json` is both the data file and its own metadata —
    // there is no `.bin` sidecar to hash separately. Audit-trail fields
    // (schema_version, jeod_commit, generated_utc) are embedded inline
    // by `encode_euler_cases_json`.
    let json_path = out_dir.join("euler_cases.json");
    let json = encode_euler_cases_json(&cases, jeod_commit, generated_utc);
    std::fs::write(&json_path, &json)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", json_path.display()));
    println!(
        "  {} -> {} ({} bytes; {} case(s) after dedup)",
        rel,
        json_path.display(),
        json.len(),
        cases.len(),
    );
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

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            return dir;
        }
        if !dir.pop() {
            panic!("workspace_root: Cargo.lock not found in any ancestor of CARGO_MANIFEST_DIR");
        }
    }
}
