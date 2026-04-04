//! Generates a Markdown cross-validation error report from Tier 3 test results.
//!
//! Usage:
//!   cargo run -p jeod_test_data --bin tier3_report
//!
//! Reads JSON files from `target/tier3_crossval/` (written by `CrossvalReport`)
//! during `cargo test`) and writes `target/tier3_report.md`.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            return dir;
        }
        if !dir.pop() {
            return PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        }
    }
}

struct TestResult {
    test: String,
    position: Option<[f64; 3]>,
    velocity: Option<[f64; 3]>,
    acceleration: Option<[f64; 3]>,
    quaternion: Option<[f64; 4]>,
    quat_angle: Option<f64>,
    ang_vel: Option<[f64; 3]>,
    ang_accel: Option<[f64; 3]>,
    position_tol: Option<[f64; 3]>,
    velocity_tol: Option<[f64; 3]>,
    acceleration_tol: Option<[f64; 3]>,
    quaternion_tol: Option<[f64; 4]>,
    quat_angle_tol: Option<f64>,
    ang_vel_tol: Option<[f64; 3]>,
    ang_accel_tol: Option<[f64; 3]>,
    extras: Vec<(String, f64, Option<f64>, String)>,
}

fn parse_json(s: &str) -> Option<TestResult> {
    let test = extract_string(s, "test")?;
    Some(TestResult {
        test,
        position: parse_vec3(s, "position"),
        velocity: parse_vec3(s, "velocity"),
        acceleration: parse_vec3(s, "acceleration"),
        quaternion: parse_vec4(s, "quaternion"),
        quat_angle: parse_f64(s, "quat_angle"),
        ang_vel: parse_vec3(s, "ang_vel"),
        ang_accel: parse_vec3(s, "ang_accel"),
        position_tol: parse_vec3(s, "position_tol"),
        velocity_tol: parse_vec3(s, "velocity_tol"),
        acceleration_tol: parse_vec3(s, "acceleration_tol"),
        quaternion_tol: parse_vec4(s, "quaternion_tol"),
        quat_angle_tol: parse_f64(s, "quat_angle_tol"),
        ang_vel_tol: parse_vec3(s, "ang_vel_tol"),
        ang_accel_tol: parse_vec3(s, "ang_accel_tol"),
        extras: parse_extras(s),
    })
}

fn extract_string(s: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":\"", key);
    let start = s.find(&pattern)? + pattern.len();
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_string())
}

fn parse_number_at(s: &str) -> Option<(f64, usize)> {
    let s = s.trim_start();
    if s.starts_with("null") {
        return None;
    }
    let end = s
        .find(|c: char| {
            c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E' && !c.is_ascii_digit()
        })
        .unwrap_or(s.len());
    s[..end].parse().ok().map(|v| (v, end))
}

fn parse_f64(s: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let start = s.find(&pattern)? + pattern.len();
    parse_number_at(&s[start..]).map(|(v, _)| v)
}

fn parse_vec3(s: &str, key: &str) -> Option<[f64; 3]> {
    let pattern = format!("\"{}\":", key);
    let start = s.find(&pattern)? + pattern.len();
    let rest = s[start..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    let arr_start = rest.find('[')?;
    let arr_end = rest.find(']')?;
    let inner = &rest[arr_start + 1..arr_end];
    let nums: Vec<f64> = inner
        .split(',')
        .filter_map(|n| n.trim().parse().ok())
        .collect();
    if nums.len() == 3 {
        Some([nums[0], nums[1], nums[2]])
    } else {
        None
    }
}

fn parse_vec4(s: &str, key: &str) -> Option<[f64; 4]> {
    let pattern = format!("\"{}\":", key);
    let start = s.find(&pattern)? + pattern.len();
    let rest = s[start..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    let arr_start = rest.find('[')?;
    let arr_end = rest.find(']')?;
    let inner = &rest[arr_start + 1..arr_end];
    let nums: Vec<f64> = inner
        .split(',')
        .filter_map(|n| n.trim().parse().ok())
        .collect();
    if nums.len() == 4 {
        Some([nums[0], nums[1], nums[2], nums[3]])
    } else {
        None
    }
}

fn parse_extras(s: &str) -> Vec<(String, f64, Option<f64>, String)> {
    let mut result = Vec::new();
    let Some(start) = s.find("\"extras\":") else {
        return result;
    };
    let rest = &s[start..];
    let Some(arr_start) = rest.find('[') else {
        return result;
    };
    let Some(arr_end) = rest.rfind(']') else {
        return result;
    };
    let arr = &rest[arr_start + 1..arr_end];
    let mut pos = 0;
    while let Some(obj_start) = arr[pos..].find('{') {
        let Some(obj_end) = arr[pos + obj_start..].find('}') else {
            break;
        };
        let obj = &arr[pos + obj_start..pos + obj_start + obj_end + 1];
        if let (Some(var), Some(val)) = (extract_string(obj, "var"), parse_f64(obj, "val")) {
            let tol = parse_f64(obj, "tol");
            let unit = extract_string(obj, "unit").unwrap_or_default();
            result.push((var, val, tol, unit));
        }
        pos = pos + obj_start + obj_end + 1;
    }
    result
}

fn f3(v: f64) -> String {
    format!("{v:.3e}")
}

fn f3_opt(v: Option<f64>) -> String {
    match v {
        Some(v) => f3(v),
        None => "—".to_string(),
    }
}

fn main() {
    let root = workspace_root();
    let data_dir = root.join("target").join("tier3_crossval");
    let output_path = root.join("target").join("tier3_report.md");

    if !data_dir.exists() {
        eprintln!("No cross-validation data found in {}", data_dir.display());
        eprintln!("Run: cargo test --workspace -- tier3_");
        std::process::exit(1);
    }

    let mut entries: Vec<TestResult> = Vec::new();
    let mut files: Vec<_> = fs::read_dir(&data_dir)
        .expect("failed to read tier3_crossval directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort_by_key(|e| e.file_name());

    for entry in &files {
        let content = fs::read_to_string(entry.path()).expect("failed to read JSON file");
        if let Some(result) = parse_json(&content) {
            entries.push(result);
        } else {
            eprintln!("Warning: failed to parse {}", entry.path().display());
        }
    }

    if entries.is_empty() {
        eprintln!("No valid JSON files found in {}", data_dir.display());
        std::process::exit(1);
    }

    entries.sort_by(|a, b| a.test.cmp(&b.test));

    let mut out = fs::File::create(&output_path).expect("failed to create tier3_report.md");

    writeln!(out, "# Tier 3 Cross-Validation Error Report").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "{} tests reported.", entries.len()).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "All values are max absolute per-component errors across the trajectory."
    )
    .unwrap();

    // ── Translational state ──
    writeln!(out).unwrap();
    writeln!(out, "## Translational State").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Test | pos_x (m) | pos_y (m) | pos_z (m) | vel_x (m/s) | vel_y (m/s) | vel_z (m/s) | acc_x (m/s²) | acc_y (m/s²) | acc_z (m/s²) |"
    ).unwrap();
    writeln!(
        out,
        "|------|-----------|-----------|-----------|-------------|-------------|-------------|--------------|--------------|--------------|"
    ).unwrap();

    for e in &entries {
        let short = e.test.replace("tier3_", "");
        let p = e.position.unwrap_or([f64::NAN; 3]);
        let v = e.velocity.unwrap_or([f64::NAN; 3]);
        let a = e.acceleration.unwrap_or([f64::NAN; 3]);
        let has_trans = e.position.is_some() || e.velocity.is_some() || e.acceleration.is_some();
        if !has_trans {
            continue;
        }
        let fc = |val: f64| -> String {
            if val.is_nan() {
                "—".to_string()
            } else {
                f3(val)
            }
        };
        writeln!(
            out,
            "| {short} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            fc(p[0]),
            fc(p[1]),
            fc(p[2]),
            fc(v[0]),
            fc(v[1]),
            fc(v[2]),
            fc(a[0]),
            fc(a[1]),
            fc(a[2]),
        )
        .unwrap();
    }

    // ── Translational tolerances ──
    writeln!(out).unwrap();
    writeln!(out, "### Translational Tolerances").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Test | pos_x (m) | pos_y (m) | pos_z (m) | vel_x (m/s) | vel_y (m/s) | vel_z (m/s) | acc_x (m/s²) | acc_y (m/s²) | acc_z (m/s²) |"
    ).unwrap();
    writeln!(
        out,
        "|------|-----------|-----------|-----------|-------------|-------------|-------------|--------------|--------------|--------------|"
    ).unwrap();

    for e in &entries {
        let has_tol =
            e.position_tol.is_some() || e.velocity_tol.is_some() || e.acceleration_tol.is_some();
        if !has_tol {
            continue;
        }
        let short = e.test.replace("tier3_", "");
        let p = e.position_tol.unwrap_or([f64::NAN; 3]);
        let v = e.velocity_tol.unwrap_or([f64::NAN; 3]);
        let a = e.acceleration_tol.unwrap_or([f64::NAN; 3]);
        let fc = |val: f64| -> String {
            if val.is_nan() {
                "—".to_string()
            } else {
                f3(val)
            }
        };
        writeln!(
            out,
            "| {short} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            fc(p[0]),
            fc(p[1]),
            fc(p[2]),
            fc(v[0]),
            fc(v[1]),
            fc(v[2]),
            fc(a[0]),
            fc(a[1]),
            fc(a[2]),
        )
        .unwrap();
    }

    // ── Rotational state ──
    writeln!(out).unwrap();
    writeln!(out, "## Rotational State").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Test | q_w | q_x | q_y | q_z | q_angle (rad) | ω_x (rad/s) | ω_y (rad/s) | ω_z (rad/s) | α_x (rad/s²) | α_y (rad/s²) | α_z (rad/s²) |"
    ).unwrap();
    writeln!(
        out,
        "|------|-----|-----|-----|-----|---------------|-------------|-------------|-------------|--------------|--------------|--------------|"
    ).unwrap();

    for e in &entries {
        let has_rot = e.quaternion.is_some() || e.ang_vel.is_some() || e.ang_accel.is_some();
        if !has_rot {
            continue;
        }
        let short = e.test.replace("tier3_", "");
        let q = e.quaternion.unwrap_or([f64::NAN; 4]);
        let w = e.ang_vel.unwrap_or([f64::NAN; 3]);
        let a = e.ang_accel.unwrap_or([f64::NAN; 3]);
        let fc = |val: f64| -> String {
            if val.is_nan() {
                "—".to_string()
            } else {
                f3(val)
            }
        };
        writeln!(
            out,
            "| {short} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            fc(q[0]),
            fc(q[1]),
            fc(q[2]),
            fc(q[3]),
            f3_opt(e.quat_angle),
            fc(w[0]),
            fc(w[1]),
            fc(w[2]),
            fc(a[0]),
            fc(a[1]),
            fc(a[2]),
        )
        .unwrap();
    }

    // ── Rotational tolerances ──
    writeln!(out).unwrap();
    writeln!(out, "### Rotational Tolerances").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Test | q_w | q_x | q_y | q_z | q_angle (rad) | ω_x (rad/s) | ω_y (rad/s) | ω_z (rad/s) | α_x (rad/s²) | α_y (rad/s²) | α_z (rad/s²) |"
    ).unwrap();
    writeln!(
        out,
        "|------|-----|-----|-----|-----|---------------|-------------|-------------|-------------|--------------|--------------|--------------|"
    ).unwrap();

    for e in &entries {
        let has_tol =
            e.quaternion_tol.is_some() || e.ang_vel_tol.is_some() || e.ang_accel_tol.is_some();
        if !has_tol {
            continue;
        }
        let short = e.test.replace("tier3_", "");
        let q = e.quaternion_tol.unwrap_or([f64::NAN; 4]);
        let w = e.ang_vel_tol.unwrap_or([f64::NAN; 3]);
        let a = e.ang_accel_tol.unwrap_or([f64::NAN; 3]);
        let fc = |val: f64| -> String {
            if val.is_nan() {
                "—".to_string()
            } else {
                f3(val)
            }
        };
        writeln!(
            out,
            "| {short} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            fc(q[0]),
            fc(q[1]),
            fc(q[2]),
            fc(q[3]),
            f3_opt(e.quat_angle_tol),
            fc(w[0]),
            fc(w[1]),
            fc(w[2]),
            fc(a[0]),
            fc(a[1]),
            fc(a[2]),
        )
        .unwrap();
    }

    // ── Extras ──
    let has_extras = entries.iter().any(|e| !e.extras.is_empty());
    if has_extras {
        writeln!(out).unwrap();
        writeln!(out, "## Test-Specific Metrics").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "| Test | Variable | Max Error | Tolerance | Unit |").unwrap();
        writeln!(out, "|------|----------|-----------|-----------|------|").unwrap();

        for e in &entries {
            if e.extras.is_empty() {
                continue;
            }
            let short = e.test.replace("tier3_", "");
            for (i, (var, val, tol, unit)) in e.extras.iter().enumerate() {
                let test_col = if i == 0 { short.as_str() } else { "" };
                writeln!(
                    out,
                    "| {test_col} | {var} | {} | {} | {unit} |",
                    f3(*val),
                    f3_opt(*tol),
                )
                .unwrap();
            }
        }
    }

    let total_metrics: usize = entries
        .iter()
        .map(|e| {
            let mut n = e.extras.len();
            if e.position.is_some() {
                n += 3;
            }
            if e.velocity.is_some() {
                n += 3;
            }
            if e.acceleration.is_some() {
                n += 3;
            }
            if e.quaternion.is_some() {
                n += 4;
            }
            if e.quat_angle.is_some() {
                n += 1;
            }
            if e.ang_vel.is_some() {
                n += 3;
            }
            if e.ang_accel.is_some() {
                n += 3;
            }
            n
        })
        .sum();

    writeln!(out).unwrap();
    writeln!(
        out,
        "*{total_metrics} total metrics across {} tests.*",
        entries.len()
    )
    .unwrap();

    eprintln!("Wrote {}", output_path.display());
}
