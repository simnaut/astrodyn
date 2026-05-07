//! Generates a Markdown cross-validation error report from Tier 3 test results.
//!
//! Usage:
//!   cargo run -p astrodyn_test_data --bin tier3_report -- [--freeze-baselines]
//!
//! Reads JSON files from `target/tier3_crossval/` (written by `CrossvalReport`
//! during `cargo test`) and extracts tolerances from test source files.
//!
//! Always writes `target/tier3_report.md` (the rolling report).
//!
//! When `--freeze-baselines` is passed, also writes the frozen snapshots:
//!   - `test_data/baselines.json` — per-test, per-component max-errors
//!   - `test_data/baselines.md`   — human-readable mirror of the above
//!
//! These snapshots anchor the **physics-invariance policy** of GitHub issue
//! #101's type-system refactor: every refactor-only phase must satisfy
//! `max_error_new ≤ max(baseline · 1.0 + 1e-12 · magnitude, 1e-12)`.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use astrodyn_test_data::crossval::json_escape;

/// Slice `content` by byte range, snapping endpoints to valid UTF-8 char
/// boundaries. Needed because test source files can contain multi-byte chars
/// (e.g. box-drawing `═`) and naive `&content[a..b]` with computed offsets
/// (e.g. `name_pos ± N`) can panic if an endpoint lands inside a codepoint.
fn slice_safe(content: &str, start: usize, end: usize) -> &str {
    let mut s = start.min(content.len());
    let mut e = end.min(content.len());
    while s > 0 && !content.is_char_boundary(s) {
        s -= 1;
    }
    while e < content.len() && !content.is_char_boundary(e) {
        e += 1;
    }
    &content[s..e]
}

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
    // Tolerances extracted from test source files
    position_tol: Option<[f64; 3]>,
    velocity_tol: Option<[f64; 3]>,
    quat_angle_tol: Option<f64>,
    ang_vel_tol: Option<[f64; 3]>,
    // Extras: (var, val, tol_from_source, unit)
    extras: Vec<(String, f64, Option<f64>, String)>,
}

// ── JSON parsing ──

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
        position_tol: None,
        velocity_tol: None,
        quat_angle_tol: None,
        ang_vel_tol: None,
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
            let unit = extract_string(obj, "unit").unwrap_or_default();
            // tol=None here; filled from source later
            result.push((var, val, None, unit));
        }
        pos = pos + obj_start + obj_end + 1;
    }
    result
}

// ── Source tolerance extraction ──

/// Parse a float from a Rust source fragment (handles scientific notation).
fn parse_rust_float(s: &str) -> Option<f64> {
    let s = s.trim().trim_end_matches([',', ')', ']']);
    s.parse().ok()
}

/// Extract an [f64; 3] array literal from a string like `[1.37e-6, 2.154e-6, 1.826e-6]`
/// or `[1e-15; 3]`.
fn extract_array3(s: &str) -> Option<[f64; 3]> {
    let start = s.find('[')?;
    let end = s[start..].find(']')? + start;
    let inner = s[start + 1..end].trim();
    // Check for [val; 3] syntax
    if let Some(semi) = inner.find(';') {
        let val = parse_rust_float(inner[..semi].trim())?;
        return Some([val; 3]);
    }
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() == 3 {
        let a = parse_rust_float(parts[0])?;
        let b = parse_rust_float(parts[1])?;
        let c = parse_rust_float(parts[2])?;
        Some([a, b, c])
    } else {
        None
    }
}

/// Extract a scalar float from the first argument after the opening paren,
/// e.g. from `assert_quat_angle(4.426e-8)`.
fn extract_scalar_arg(s: &str) -> Option<f64> {
    let start = s.find('(')?;
    let end = s[start..].find(')')? + start;
    parse_rust_float(s[start + 1..end].trim())
}

/// For a given test name, search source files for assert_position/velocity/etc.
/// calls and extract their tolerance values.
fn extract_source_tolerances(
    test_name: &str,
    source_contents: &[(String, String)],
) -> SourceTolerances {
    let mut tols = SourceTolerances::default();

    // Find lines near the test name string literal
    for (path, content) in source_contents {
        // Find the test name in the source
        let name_pattern = format!("\"{}\"", test_name);
        let Some(name_pos) = content.find(&name_pattern) else {
            continue;
        };

        // Look in a window around the test name (the function body)
        // Go back to find function start and forward to find function end
        let search_start = name_pos.saturating_sub(3000);
        let search_end = name_pos + 5000;
        let window = slice_safe(content, search_start, search_end);

        // Extract assert_position([...])
        if let Some(pos) = window.find("assert_position(") {
            let rest = &window[pos..];
            tols.position = extract_array3(rest);
        }

        // Extract assert_velocity([...])
        if let Some(pos) = window.find("assert_velocity(") {
            let rest = &window[pos..];
            tols.velocity = extract_array3(rest);
        }

        // Extract assert_quat_angle(val)
        if let Some(pos) = window.find("assert_quat_angle(") {
            let rest = &window[pos..];
            tols.quat_angle = extract_scalar_arg(rest);
        }

        // Extract assert_ang_vel([...])
        if let Some(pos) = window.find("assert_ang_vel(") {
            let rest = &window[pos..];
            tols.ang_vel = extract_array3(rest);
        }

        if tols.position.is_some() || tols.velocity.is_some() {
            tols.source_file = Some(path.clone());
            break;
        }
    }

    // If assert_* calls use variable names (shared helper pattern),
    // try to find the call site that passes the test name and extract
    // array literals from nearby arguments.
    if tols.position.is_none() {
        for (_path, content) in source_contents {
            let name_pattern = format!("\"{}\"", test_name);
            let Some(name_pos) = content.find(&name_pattern) else {
                continue;
            };
            // Search backwards from the test name for array literals in the same call
            let call_start = name_pos.saturating_sub(500);
            let call_end = name_pos + 200;
            let call_window = slice_safe(content, call_start, call_end);

            // Look for array literals that might be tolerance arguments
            let mut arrays: Vec<[f64; 3]> = Vec::new();
            let mut search_pos = 0;
            while let Some(bracket) = call_window[search_pos..].find('[') {
                let abs_pos = search_pos + bracket;
                if let Some(arr) = extract_array3(&call_window[abs_pos..]) {
                    arrays.push(arr);
                }
                search_pos = abs_pos + 1;
            }

            // Heuristic: the last two arrays before the test name are likely pos_tol, vel_tol
            if arrays.len() >= 2 {
                tols.position = Some(arrays[arrays.len() - 2]);
                tols.velocity = Some(arrays[arrays.len() - 1]);
            } else if arrays.len() == 1 {
                tols.position = Some(arrays[0]);
            }

            if tols.position.is_some() {
                break;
            }
        }
    }

    tols
}

#[derive(Default)]
struct SourceTolerances {
    position: Option<[f64; 3]>,
    velocity: Option<[f64; 3]>,
    quat_angle: Option<f64>,
    ang_vel: Option<[f64; 3]>,
    source_file: Option<String>,
}

/// Extract tolerances for extras from source.
/// Looks for `assert!(EXPR < LITERAL, "metric_name")` near the test name.
fn extract_extras_tolerances(
    test_name: &str,
    extras: &mut [(String, f64, Option<f64>, String)],
    source_contents: &[(String, String)],
) {
    // Find which file contains the test name, then search near the test name
    // for add_extra("NAME",...) followed by assert!(... < LITERAL, "NAME").
    for (_path, content) in source_contents {
        let name_pattern = format!("\"{}\"", test_name);
        let Some(name_pos) = content.find(&name_pattern) else {
            continue;
        };

        // Search within ±3000 chars of the test name (covers the function body
        // for both inline tests and shared helpers)
        let window_start = name_pos.saturating_sub(3000);
        let window_end = name_pos + 5000;
        let window = slice_safe(content, window_start, window_end);

        for extra in extras.iter_mut() {
            if extra.2.is_some() {
                continue;
            }
            let add_extra_pattern = format!("add_extra(\"{}\",", extra.0);
            let assert_pattern = format!("\"{}\")", extra.0);

            // Find the add_extra call within the window
            if let Some(ae_pos) = window.find(&add_extra_pattern) {
                // Look for assert within ~500 chars after the add_extra
                let nearby = slice_safe(window, ae_pos, ae_pos + 500);

                for line in nearby.lines() {
                    if line.contains(&assert_pattern) && line.contains("assert!") {
                        if let Some(lt_pos) = line.rfind('<') {
                            let after_lt = &line[lt_pos + 1..];
                            if let Some(comma_pos) = after_lt.find(',') {
                                let literal_str = after_lt[..comma_pos].trim();
                                if let Some(val) = parse_rust_float(literal_str) {
                                    extra.2 = Some(val);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Second pass: for any extras still missing, search the entire file.
        // This handles shared helpers where the add_extra+assert are far from
        // the test name call site. Only take the FIRST match in the file to
        // avoid cross-contamination from other tests using the same metric name.
        for extra in extras.iter_mut() {
            if extra.2.is_some() {
                continue;
            }
            let add_extra_pattern = format!("add_extra(\"{}\",", extra.0);
            let assert_pattern = format!("\"{}\")", extra.0);

            if let Some(ae_pos) = content.find(&add_extra_pattern) {
                let nearby = slice_safe(content, ae_pos, ae_pos + 500);
                for line in nearby.lines() {
                    if line.contains(&assert_pattern) && line.contains("assert!") {
                        if let Some(lt_pos) = line.rfind('<') {
                            let after_lt = &line[lt_pos + 1..];
                            if let Some(comma_pos) = after_lt.find(',') {
                                let literal_str = after_lt[..comma_pos].trim();
                                if let Some(val) = parse_rust_float(literal_str) {
                                    extra.2 = Some(val);
                                }
                            }
                        }
                    }
                }
            }
        }

        break; // found the file containing this test
    }
}

// ── Formatting ──

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let freeze_baselines = args.iter().any(|a| a == "--freeze-baselines");
    for arg in &args {
        if arg != "--freeze-baselines" {
            eprintln!("unknown argument: {arg}");
            eprintln!("usage: tier3_report [--freeze-baselines]");
            std::process::exit(2);
        }
    }

    let root = workspace_root();
    let data_dir = root.join("target").join("tier3_crossval");
    let output_path = root.join("target").join("tier3_report.md");

    if !data_dir.exists() {
        eprintln!("No cross-validation data found in {}", data_dir.display());
        eprintln!("Run: cargo test --workspace -- tier3_");
        std::process::exit(1);
    }

    // Load JSON reports
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

    // Load test source files for tolerance extraction
    let source_contents = load_test_sources(&root);
    eprintln!(
        "Loaded {} test source files for tolerance extraction",
        source_contents.len()
    );

    // Extract tolerances from source for each test
    let mut missing_tols = 0;
    for entry in &mut entries {
        let tols = extract_source_tolerances(&entry.test, &source_contents);
        entry.position_tol = tols.position;
        entry.velocity_tol = tols.velocity;
        entry.quat_angle_tol = tols.quat_angle;
        entry.ang_vel_tol = tols.ang_vel;

        // Extract extras tolerances from assert!(var < LITERAL, "metric_name")
        extract_extras_tolerances(&entry.test, &mut entry.extras, &source_contents);

        if entry.position.is_some() && entry.position_tol.is_none() {
            missing_tols += 1;
            eprintln!(
                "  Warning: no position tolerance found in source for {}",
                entry.test
            );
        }
    }
    if missing_tols > 0 {
        eprintln!("{missing_tols} tests missing source-extracted tolerances");
    }

    // ── Generate report ──

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
        "| Test | pos_x (m) | pos_y (m) | pos_z (m) | vel_x (m/s) | vel_y (m/s) | vel_z (m/s) |"
    )
    .unwrap();
    writeln!(
        out,
        "|------|-----------|-----------|-----------|-------------|-------------|-------------|"
    )
    .unwrap();

    for e in &entries {
        let has_tol = e.position_tol.is_some() || e.velocity_tol.is_some();
        if !has_tol {
            continue;
        }
        let short = e.test.replace("tier3_", "");
        let p = e.position_tol.unwrap_or([f64::NAN; 3]);
        let v = e.velocity_tol.unwrap_or([f64::NAN; 3]);
        let fc = |val: f64| -> String {
            if val.is_nan() {
                "—".to_string()
            } else {
                f3(val)
            }
        };
        writeln!(
            out,
            "| {short} | {} | {} | {} | {} | {} | {} |",
            fc(p[0]),
            fc(p[1]),
            fc(p[2]),
            fc(v[0]),
            fc(v[1]),
            fc(v[2]),
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
        "| Test | q_angle (rad) | ω_x (rad/s) | ω_y (rad/s) | ω_z (rad/s) |"
    )
    .unwrap();
    writeln!(
        out,
        "|------|---------------|-------------|-------------|-------------|"
    )
    .unwrap();

    for e in &entries {
        let has_tol = e.quat_angle_tol.is_some() || e.ang_vel_tol.is_some();
        if !has_tol {
            continue;
        }
        let short = e.test.replace("tier3_", "");
        let w = e.ang_vel_tol.unwrap_or([f64::NAN; 3]);
        let fc = |val: f64| -> String {
            if val.is_nan() {
                "—".to_string()
            } else {
                f3(val)
            }
        };
        writeln!(
            out,
            "| {short} | {} | {} | {} | {} |",
            f3_opt(e.quat_angle_tol),
            fc(w[0]),
            fc(w[1]),
            fc(w[2]),
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

    if freeze_baselines {
        let baselines_dir = root.join("crates/astrodyn_verif_jeod/test_data");
        fs::create_dir_all(&baselines_dir)
            .expect("failed to create crates/astrodyn_verif_jeod/test_data/");
        let json_path = baselines_dir.join("baselines.json");
        let md_path = baselines_dir.join("baselines.md");
        write_baselines_json(&json_path, &entries);
        write_baselines_md(&md_path, &entries);
        eprintln!("Wrote {}", json_path.display());
        eprintln!("Wrote {}", md_path.display());
    }
}

// ── Baseline serialization ──

/// Write the frozen baseline snapshot in JSON. Hand-rolled serializer (no
/// `serde` dep) because the schema is simple and the binary should stay
/// lightweight.
fn write_baselines_json(path: &std::path::Path, entries: &[TestResult]) {
    let mut out = fs::File::create(path).expect("failed to create baselines.json");
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema_version\": 1,").unwrap();
    writeln!(
        out,
        "  \"note\": \"Tier 3 per-test, per-component max absolute errors. Frozen snapshot; see CLAUDE.md \\\"Baseline freeze\\\" policy.\","
    )
    .unwrap();
    writeln!(out, "  \"tests\": {{").unwrap();
    for (i, e) in entries.iter().enumerate() {
        let comma = if i + 1 < entries.len() { "," } else { "" };
        writeln!(out, "    \"{}\": {{", json_escape(&e.test)).unwrap();
        write_opt_vec3_json(&mut out, "position_m", e.position);
        write_opt_vec3_json(&mut out, "velocity_m_per_s", e.velocity);
        write_opt_vec3_json(&mut out, "acceleration_m_per_s2", e.acceleration);
        write_opt_f64_json(&mut out, "quat_angle_rad", e.quat_angle);
        write_opt_vec3_json(&mut out, "ang_vel_rad_per_s", e.ang_vel);
        write_opt_vec3_json(&mut out, "ang_accel_rad_per_s2", e.ang_accel);
        // extras
        write!(out, "      \"extras\": [").unwrap();
        for (j, (name, value, _tol, unit)) in e.extras.iter().enumerate() {
            let csep = if j + 1 < e.extras.len() { "," } else { "" };
            write!(
                out,
                "{{\"name\":\"{}\",\"value\":{},\"unit\":\"{}\"}}{}",
                json_escape(name),
                fmt_f64(*value),
                json_escape(unit),
                csep
            )
            .unwrap();
        }
        writeln!(out, "]").unwrap();
        writeln!(out, "    }}{comma}").unwrap();
    }
    writeln!(out, "  }}").unwrap();
    writeln!(out, "}}").unwrap();
}

fn write_opt_vec3_json(out: &mut fs::File, key: &str, v: Option<[f64; 3]>) {
    if let Some([a, b, c]) = v {
        writeln!(
            out,
            "      \"{}\": [{}, {}, {}],",
            key,
            fmt_f64(a),
            fmt_f64(b),
            fmt_f64(c)
        )
        .unwrap();
    }
}

fn write_opt_f64_json(out: &mut fs::File, key: &str, v: Option<f64>) {
    if let Some(x) = v {
        writeln!(out, "      \"{}\": {},", key, fmt_f64(x)).unwrap();
    }
}

/// Format an f64 for baseline serialization. Finite values use scientific
/// notation with 17-digit precision — the minimum needed to round-trip an
/// `f64` exactly — so baselines file contents survive re-serialization
/// bit-identically. Non-finite values map to JSON `null`.
fn fmt_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.17e}")
    } else {
        "null".to_string()
    }
}

/// Write the human-readable mirror of the JSON baselines.
fn write_baselines_md(path: &std::path::Path, entries: &[TestResult]) {
    let mut out = fs::File::create(path).expect("failed to create baselines.md");
    writeln!(out, "# Tier 3 Baselines (frozen)").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Per-test, per-component max absolute errors captured at the Phase 0 freeze point of"
    )
    .unwrap();
    writeln!(
        out,
        "GitHub issue #101. See `CLAUDE.md` §\"Baseline freeze\" for the invariance policy."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "{} tests recorded.", entries.len()).unwrap();
    writeln!(out).unwrap();

    for e in entries {
        writeln!(out, "## `{}`", e.test).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "| Metric | X / value | Y | Z | Unit |").unwrap();
        writeln!(out, "|--------|-----------|---|---|------|").unwrap();
        if let Some([x, y, z]) = e.position {
            writeln!(
                out,
                "| position | {} | {} | {} | m |",
                fmt_f64(x),
                fmt_f64(y),
                fmt_f64(z)
            )
            .unwrap();
        }
        if let Some([x, y, z]) = e.velocity {
            writeln!(
                out,
                "| velocity | {} | {} | {} | m/s |",
                fmt_f64(x),
                fmt_f64(y),
                fmt_f64(z)
            )
            .unwrap();
        }
        if let Some([x, y, z]) = e.acceleration {
            writeln!(
                out,
                "| acceleration | {} | {} | {} | m/s² |",
                fmt_f64(x),
                fmt_f64(y),
                fmt_f64(z)
            )
            .unwrap();
        }
        if let Some(a) = e.quat_angle {
            writeln!(out, "| quat_angle | {} |  |  | rad |", fmt_f64(a)).unwrap();
        }
        if let Some([x, y, z]) = e.ang_vel {
            writeln!(
                out,
                "| ang_vel | {} | {} | {} | rad/s |",
                fmt_f64(x),
                fmt_f64(y),
                fmt_f64(z)
            )
            .unwrap();
        }
        if let Some([x, y, z]) = e.ang_accel {
            writeln!(
                out,
                "| ang_accel | {} | {} | {} | rad/s² |",
                fmt_f64(x),
                fmt_f64(y),
                fmt_f64(z)
            )
            .unwrap();
        }
        for (name, value, _tol, unit) in &e.extras {
            writeln!(out, "| {} | {} |  |  | {} |", name, fmt_f64(*value), unit).unwrap();
        }
        writeln!(out).unwrap();
    }
}

/// Load all tier3 test source files.
fn load_test_sources(root: &std::path::Path) -> Vec<(String, String)> {
    let mut sources = Vec::new();
    let crates_dir = root.join("crates");

    if let Ok(crate_entries) = fs::read_dir(&crates_dir) {
        for crate_entry in crate_entries.flatten() {
            let tests_dir = crate_entry.path().join("tests");
            if !tests_dir.is_dir() {
                continue;
            }
            if let Ok(test_files) = fs::read_dir(&tests_dir) {
                for test_file in test_files.flatten() {
                    let path = test_file.path();
                    if path
                        .file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("tier3_"))
                        && path.extension().is_some_and(|e| e == "rs")
                    {
                        if let Ok(content) = fs::read_to_string(&path) {
                            sources.push((path.display().to_string(), content));
                        }
                    }
                }
            }
        }
    }

    // Also check root tests/ directory
    let root_tests = root.join("tests");
    if root_tests.is_dir() {
        if let Ok(test_files) = fs::read_dir(&root_tests) {
            for test_file in test_files.flatten() {
                let path = test_file.path();
                if path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("tier3_"))
                    && path.extension().is_some_and(|e| e == "rs")
                {
                    if let Ok(content) = fs::read_to_string(&path) {
                        sources.push((path.display().to_string(), content));
                    }
                }
            }
        }
    }

    sources
}
