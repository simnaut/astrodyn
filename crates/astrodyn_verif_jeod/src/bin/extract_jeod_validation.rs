//! Extract / verify the `astrodyn_math` validation fixtures.
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
//!   `astrodyn_verif_jeod::gravity_verif::load_gravity_test_cases`).
//! - `models/environment/time/data/Leap_Second.dat`
//!   → verbatim copy at `test_data/time/Leap_Second.dat` (consumed by
//!   `astrodyn_verif_jeod::leap_second::load_leap_second_table`).
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
//! Outputs: `test_data/jeod_validation/orbital_vectors.bin` plus an
//! `orbital_vectors.json` metadata sidecar (source path, vector count).

use std::io::Write;
use std::path::{Path, PathBuf};

use astrodyn_verif_jeod::euler_test::{encode_euler_cases_json, parse_euler_test_cases_cc};
use astrodyn_verif_jeod::orbital_data::{
    encode_orbital_vectors_bin, parse_orbital_test_vectors_py,
};

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

    extract_orbital_vectors(&jeod_root, &out_dir);
    extract_euler_cases(&jeod_root, &out_dir);
    extract_grav_geospherical_verif_out(
        &jeod_root,
        &workspace.join("crates/astrodyn_gravity/test_data/gravity"),
    );
    copy_verbatim(
        &jeod_root,
        "models/environment/time/data/Leap_Second.dat",
        &workspace.join("crates/astrodyn_time/test_data/Leap_Second.dat"),
    );
    copy_verbatim(
        &jeod_root,
        "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/mass.py",
        &verif_data.join("body_init").join("iss_mass.py"),
    );
}

fn copy_verbatim(jeod_root: &Path, rel: &str, dst: &Path) {
    let src = jeod_root.join(rel);
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Cannot create {}: {e}", parent.display()));
    }
    std::fs::copy(&src, dst)
        .unwrap_or_else(|e| panic!("Cannot copy {} -> {}: {e}", src.display(), dst.display()));
    let bytes = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {} -> {} ({} bytes; verbatim copy)",
        rel,
        dst.display(),
        bytes
    );
}

fn extract_grav_geospherical_verif_out(jeod_root: &Path, out_dir: &Path) {
    let rel = "models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt";
    let src = jeod_root.join(rel);
    std::fs::create_dir_all(out_dir)
        .unwrap_or_else(|e| panic!("Cannot create {}: {e}", out_dir.display()));
    let dst = out_dir.join("grav_geospherical_verif_out.txt");
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("Cannot copy {} -> {}: {e}", src.display(), dst.display()));
    let bytes = std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {} -> {} ({} bytes; verbatim copy)",
        rel,
        dst.display(),
        bytes
    );
}

fn extract_orbital_vectors(jeod_root: &Path, out_dir: &Path) {
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
    println!(
        "  {} -> {} ({} bytes; {} vectors)",
        rel,
        bin_path.display(),
        blob.len(),
        vectors.len(),
    );

    let meta_path = out_dir.join("orbital_vectors.json");
    let meta = format!(
        "{{\n  \"source\": \"{rel}\",\n  \"vector_count\": {},\n  \"bytes\": {},\n  \
         \"format\": \"u32 count (LE) followed by count * 6 * f64 (LE) = pos[3] vel[3] in SI\",\n  \
         \"note\": \"Regenerate with: cargo run -p astrodyn_verif_jeod --bin extract_jeod_validation\"\n}}\n",
        vectors.len(),
        blob.len(),
    );
    std::fs::write(&meta_path, meta)
        .unwrap_or_else(|e| panic!("Cannot write {}: {e}", meta_path.display()));
    println!("  metadata -> {}", meta_path.display());
}

fn extract_euler_cases(jeod_root: &Path, out_dir: &Path) {
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

    let json_path = out_dir.join("euler_cases.json");
    let json = encode_euler_cases_json(&cases);
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
