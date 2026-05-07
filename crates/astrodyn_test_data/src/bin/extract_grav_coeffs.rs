//! Extract spherical-harmonics gravity coefficients from a JEOD source
//! checkout into committed binary fixtures.
//!
//! This is a **regen-only** path: it reads `$JEOD_HOME` or an explicit `--jeod-home <PATH>` argument, parses each
//! `models/environment/gravity/data/src/earth_GGM*.cc` file via
//! [`astrodyn_test_data::jeod_cc::load_from_jeod_cc`], and writes
//! `test_data/gravity/{ggm02c,ggm05c}.bin` using the production
//! [`astrodyn_gravity::coefficients::save_binary`] format. A sidecar
//! `{label}.json` records source provenance (path, file size, mu, degree,
//! order) so reviewers can audit drift.
//!
//! Run after a JEOD upgrade or whenever the coefficient files change:
//!
//! ```bash
//! cargo run -p astrodyn_test_data --bin extract_grav_coeffs
//! cargo run -p astrodyn_test_data --bin extract_grav_coeffs -- \
//!     --jeod-home /path/to/jeod --out-dir test_data/gravity
//! ```
//!
//! The binary prints a summary of each generated file on success.

use std::io::Write;
use std::path::{Path, PathBuf};

use astrodyn_gravity::coefficients::save_binary;
use astrodyn_test_data::jeod_cc;

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

    println!("JEOD root: {}", jeod_root.display());
    println!("Output dir: {}", out_dir.display());
    println!();

    for source in SOURCES {
        process_source(&jeod_root, &out_dir, source);
    }
}

fn process_source(jeod_root: &Path, out_dir: &Path, source: &Source) {
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
    let bin_size = std::fs::metadata(&bin_path)
        .unwrap_or_else(|e| panic!("stat {}: {e}", bin_path.display()))
        .len();

    let meta_path = out_dir.join(format!("{}.json", source.label));
    write_metadata(
        &meta_path,
        source,
        cc_size,
        bin_size,
        data.degree,
        data.order,
        data.radius,
        data.mu,
        data.tide_free,
        data.tide_free_delta,
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
    // Default: <workspace>/test_data/gravity
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            break;
        }
        if !dir.pop() {
            // Legacy fallback (matches `tier3_csv::test_data_path` behavior).
            return PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/gravity");
        }
    }
    dir.join("test_data").join("gravity")
}

#[allow(clippy::too_many_arguments)]
fn write_metadata(
    out_path: &Path,
    source: &Source,
    cc_size: u64,
    bin_size: u64,
    degree: usize,
    order: usize,
    radius: f64,
    mu: f64,
    tide_free: bool,
    tide_free_delta: f64,
) {
    let source_rel = format!("models/environment/gravity/data/src/{}", source.cc_filename);
    let mut f = std::fs::File::create(out_path)
        .unwrap_or_else(|e| panic!("create {}: {e}", out_path.display()));
    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"schema_version\": 1,").unwrap();
    writeln!(f, "  \"label\": \"{}\",", source.label).unwrap();
    writeln!(f, "  \"source_file\": \"{}\",", source_rel).unwrap();
    writeln!(f, "  \"source_file_bytes\": {},", cc_size).unwrap();
    writeln!(f, "  \"binary_file\": \"{}.bin\",", source.label).unwrap();
    writeln!(f, "  \"binary_file_bytes\": {},", bin_size).unwrap();
    writeln!(f, "  \"degree\": {},", degree).unwrap();
    writeln!(f, "  \"order\": {},", order).unwrap();
    writeln!(f, "  \"radius_m\": {:?},", radius).unwrap();
    writeln!(f, "  \"mu_m3_per_s2\": {:?},", mu).unwrap();
    writeln!(f, "  \"tide_free\": {},", tide_free).unwrap();
    writeln!(f, "  \"tide_free_delta\": {:?},", tide_free_delta).unwrap();
    writeln!(
        f,
        "  \"generated_by\": \"cargo run -p astrodyn_test_data --bin extract_grav_coeffs\","
    )
    .unwrap();
    writeln!(
        f,
        "  \"note\": \"Regenerate after a JEOD upgrade or coefficient-file change. The .bin file uses the astrodyn_gravity::coefficients::save_binary format (magic JEOD, version 1).\""
    )
    .unwrap();
    writeln!(f, "}}").unwrap();
}
