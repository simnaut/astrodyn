//! Extract `SIM_PFIXPOSN_VERIF` seeds from a JEOD source checkout into
//! `test_data/planet_pfixposn_seeds.json`.
//!
//! This is a **regen-only** path: it reads `$JEOD_HOME` or an explicit `--jeod-home <PATH>` argument, parses the
//! three `add_read` blocks from
//! `models/utils/planet_fixed/planet_fixed_posn/verif/SIM_PFIXPOSN_VERIF/SET_test/RUN_pfixposn_test/input.py`,
//! and writes the committed JSON consumed by
//! `astrodyn_planet::geodetic_verif::load_planet_fixed_verif_cases`.
//!
//! Run after a JEOD upgrade or whenever the SIM's input is amended:
//!
//! ```bash
//! cargo run -p astrodyn_planet --bin extract_planet_pfixposn -- \
//!     --jeod-home /path/to/jeod
//! ```
//!
//! The binary prints the destination path on success.

use std::io::Write;

use regex::Regex;
use sha2::{Digest, Sha256};

/// Pinned JEOD version captured in the fixture sidecar.
const JEOD_VERSION: &str = "5.4";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jeod_root = resolve_jeod_root(&args).unwrap_or_else(|| {
        eprintln!(
            "extract_planet_pfixposn: JEOD source not found.\n\
             Pass `--jeod-home <PATH>` or set JEOD_HOME \
             (see CLAUDE.md \"Environment Setup\")."
        );
        std::process::exit(2);
    });

    let input_py = jeod_root.join(
        "models/utils/planet_fixed/planet_fixed_posn/\
         verif/SIM_PFIXPOSN_VERIF/SET_test/RUN_pfixposn_test/input.py",
    );
    let content = std::fs::read_to_string(&input_py).unwrap_or_else(|e| {
        panic!(
            "Cannot read {} (under JEOD_HOME = {}): {e}",
            input_py.display(),
            jeod_root.display(),
        )
    });

    let cases = parse_input_py(&content);
    assert!(
        cases.len() >= 3,
        "expected three add_read blocks in {}, found {}",
        input_py.display(),
        cases.len(),
    );

    let jeod_commit = read_git_rev(&jeod_root).unwrap_or_else(|| "unknown".to_string());
    let generated_utc = utc_now_iso8601();

    let out_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data/planet_pfixposn_seeds.json");
    let mut f = std::fs::File::create(&out_path)
        .unwrap_or_else(|e| panic!("Cannot create {}: {e}", out_path.display()));
    write_json(&mut f, &cases, &jeod_commit, &generated_utc);
    drop(f);
    let sha = sha256_hex_of_file(&out_path);
    println!(
        "wrote {} ({} cases; sha256 {})",
        out_path.display(),
        cases.len(),
        sha,
    );
}

/// Read `git rev-parse HEAD` from the JEOD checkout. Returns `None` when
/// the directory is not a git checkout (tarball mirror) or `git` is
/// unavailable; callers fall back to `"unknown"`.
fn read_git_rev(jeod_root: &std::path::Path) -> Option<String> {
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

fn sha256_hex_of_file(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
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

fn resolve_jeod_root(args: &[String]) -> Option<std::path::PathBuf> {
    if let Some(idx) = args.iter().position(|a| a == "--jeod-home") {
        if let Some(p) = args.get(idx + 1) {
            return Some(std::path::PathBuf::from(p));
        }
    }
    if let Ok(p) = std::env::var("JEOD_HOME") {
        return Some(std::path::PathBuf::from(p));
    }
    None
}

#[derive(Debug)]
enum Case {
    Cart {
        read_time: f64,
        cart_m: [f64; 3],
    },
    Spher {
        read_time: f64,
        altitude_m: f64,
        latitude_rad: f64,
        longitude_rad: f64,
    },
    Ellip {
        read_time: f64,
        altitude_m: f64,
        latitude_rad: f64,
        longitude_rad: f64,
    },
}

fn parse_input_py(content: &str) -> Vec<Case> {
    let block_re = Regex::new(
        r#"(?ms)read\s*=\s*([0-9eE+.\-]+)\s*\n\s*trick\.add_read\s*\(\s*read\s*,\s*"""(?P<body>.*?)"""\s*\)"#,
    )
    .expect("regex for add_read block");
    let mut cases = Vec::new();
    for caps in block_re.captures_iter(content) {
        let read_time: f64 = caps[1]
            .parse()
            .unwrap_or_else(|e| panic!("malformed read time {:?}: {e}", &caps[1]));
        let body = &caps["body"];
        if body.contains("update_from_cart") {
            let xyz = parse_array3(body, r"earth\.cartesian_pos\s*=\s*\[([^\]]+)\]")
                .unwrap_or_else(|| panic!("cartesian_pos not parseable in:\n{body}"));
            cases.push(Case::Cart {
                read_time,
                cart_m: xyz,
            });
        } else if body.contains("update_from_spher") {
            cases.push(Case::Spher {
                read_time,
                altitude_m: parse_assign(
                    body,
                    r"earth\.spherical_pos\.altitude\s*=\s*([\-0-9eE+.]+)",
                ),
                latitude_rad: parse_assign(
                    body,
                    r"earth\.spherical_pos\.latitude\s*=\s*([\-0-9eE+.]+)",
                ),
                longitude_rad: parse_assign(
                    body,
                    r"earth\.spherical_pos\.longitude\s*=\s*([\-0-9eE+.]+)",
                ),
            });
        } else if body.contains("update_from_ellip") {
            cases.push(Case::Ellip {
                read_time,
                altitude_m: parse_assign(
                    body,
                    r"earth\.elliptical_pos\.altitude\s*=\s*([\-0-9eE+.]+)",
                ),
                latitude_rad: parse_assign(
                    body,
                    r"earth\.elliptical_pos\.latitude\s*=\s*([\-0-9eE+.]+)",
                ),
                longitude_rad: parse_assign(
                    body,
                    r"earth\.elliptical_pos\.longitude\s*=\s*([\-0-9eE+.]+)",
                ),
            });
        }
    }
    cases
}

fn parse_array3(text: &str, pattern: &str) -> Option<[f64; 3]> {
    let re = Regex::new(pattern).ok()?;
    let caps = re.captures(text)?;
    let parts: Vec<f64> = caps[1]
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    if parts.len() != 3 {
        return None;
    }
    Some([parts[0], parts[1], parts[2]])
}

fn parse_assign(text: &str, pattern: &str) -> f64 {
    let re = Regex::new(pattern).expect("valid regex");
    let caps = re
        .captures(text)
        .unwrap_or_else(|| panic!("pattern {pattern:?} did not match in:\n{text}"));
    caps[1]
        .parse()
        .unwrap_or_else(|e| panic!("failed to parse {:?}: {e}", &caps[1]))
}

fn write_json(out: &mut std::fs::File, cases: &[Case], jeod_commit: &str, generated_utc: &str) {
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema_version\": 2,").unwrap();
    writeln!(
        out,
        "  \"source\": \"models/utils/planet_fixed/planet_fixed_posn/verif/SIM_PFIXPOSN_VERIF/SET_test/RUN_pfixposn_test/input.py\","
    )
    .unwrap();
    writeln!(out, "  \"jeod_version\": \"{JEOD_VERSION}\",").unwrap();
    writeln!(out, "  \"jeod_commit\": \"{jeod_commit}\",").unwrap();
    writeln!(out, "  \"generated_utc\": \"{generated_utc}\",").unwrap();
    writeln!(
        out,
        "  \"note\": \"PlanetFixedPosition verification seeds. Regenerate with: cargo run -p astrodyn_planet --bin extract_planet_pfixposn -- --jeod-home $JEOD_HOME\","
    )
    .unwrap();
    writeln!(out, "  \"cases\": [").unwrap();
    for (i, c) in cases.iter().enumerate() {
        let comma = if i + 1 < cases.len() { "," } else { "" };
        match c {
            Case::Cart { read_time, cart_m } => writeln!(
                out,
                "    {{\"kind\": \"cartesian\",  \"read_time\": {}, \"cart_m\":   [{}, {}, {}]}}{}",
                fmt(*read_time),
                fmt(cart_m[0]),
                fmt(cart_m[1]),
                fmt(cart_m[2]),
                comma,
            ),
            Case::Spher {
                read_time,
                altitude_m,
                latitude_rad,
                longitude_rad,
            } => writeln!(
                out,
                "    {{\"kind\": \"spherical\",  \"read_time\": {}, \"altitude_m\": {}, \"latitude_rad\": {}, \"longitude_rad\": {}}}{}",
                fmt(*read_time),
                fmt(*altitude_m),
                fmt(*latitude_rad),
                fmt(*longitude_rad),
                comma,
            ),
            Case::Ellip {
                read_time,
                altitude_m,
                latitude_rad,
                longitude_rad,
            } => writeln!(
                out,
                "    {{\"kind\": \"elliptical\", \"read_time\": {}, \"altitude_m\": {},  \"latitude_rad\": {},    \"longitude_rad\": {}}}{}",
                fmt(*read_time),
                fmt(*altitude_m),
                fmt(*latitude_rad),
                fmt(*longitude_rad),
                comma,
            ),
        }
        .unwrap();
    }
    writeln!(out, "  ]").unwrap();
    writeln!(out, "}}").unwrap();
}

fn fmt(x: f64) -> String {
    // Round-trippable f64 representation; trims unnecessary fractional zeros.
    let s = format!("{x:?}");
    s
}
