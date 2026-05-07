//! Enforce the Tier 3 baseline-invariance policy from issue #101.
//!
//! Compares per-test, per-component max absolute errors in
//! `target/tier3_crossval/*.json` (written by the Tier 3 test run) against
//! the frozen snapshot in `test_data/baselines.json`. Exits non-zero with a
//! diff report when any component exceeds its tolerance.
//!
//! Policy (from `CLAUDE.md` §"Baseline freeze"):
//!
//! ```text
//! max_error_new <= max(baseline * 1.0 + 1e-12 * |baseline|, 1e-12)
//! ```
//!
//! "magnitude" in the original spec is the nominal state scale; we use
//! `|baseline|` as a conservative proxy (stricter than spec when baseline is
//! much smaller than nominal state — which is exactly the refactor-phase
//! regime where we want the tightest guard).
//!
//! Usage:
//!   cargo run -p astrodyn_test_data --bin tier3_baseline_diff \
//!       [--allow-missing NAME]... [--allow-missing-from FILE]
//!
//! `--allow-missing` declares a baseline test that is *intentionally* absent
//! from the current run (e.g. CI's fast Tier 3 lane excludes the 17-minute
//! `tier3_earth_moon_clem` test). Repeatable. Names must match exactly.
//! Baseline tests missing from the run that are NOT on the allow-list are
//! hard failures — silently dropping a Tier 3 test is a regression.
//!
//! `--allow-missing-from FILE` reads test names from a config file (one name
//! per line, `#` and blank lines ignored). Combines additively with
//! `--allow-missing` flags. Both PR and main CI read from the same config
//! file so the slow-test list has a single source of truth.
//!
//! Prerequisite: `target/tier3_crossval/*.json` must exist (run
//! `cargo nextest run --workspace -E 'test(tier3_)'` first).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const RELATIVE_SLACK: f64 = 1e-12;
const ABSOLUTE_FLOOR: f64 = 1e-12;

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

/// A metric value: either a per-component vector or a scalar.
#[derive(Clone, Debug)]
enum Metric {
    Vec3([f64; 3]),
    Scalar(f64),
}

/// Normalized per-test record — same layout for baseline and current-run.
#[derive(Default, Debug)]
struct TestRecord {
    metrics: BTreeMap<String, Metric>,
}

/// Parse `target/tier3_crossval/<test>.json` (current-run format).
///
/// Current-run JSON keys: `position`, `velocity`, `acceleration`,
/// `quat_angle`, `ang_vel`, `ang_accel`; extras as
/// `{"var":..., "val":..., "unit":...}`.
fn parse_current_run(s: &str) -> Option<(String, TestRecord)> {
    let test = extract_string(s, "test")?;
    let mut rec = TestRecord::default();

    // Key rename: current-run → canonical (matches baseline keys without units).
    for (src, dst) in [
        ("position", "position_m"),
        ("velocity", "velocity_m_per_s"),
        ("acceleration", "acceleration_m_per_s2"),
        ("ang_vel", "ang_vel_rad_per_s"),
        ("ang_accel", "ang_accel_rad_per_s2"),
    ] {
        if let Some(v) = parse_vec3(s, src) {
            rec.metrics.insert(dst.to_string(), Metric::Vec3(v));
        }
    }
    if let Some(v) = parse_f64(s, "quat_angle") {
        rec.metrics
            .insert("quat_angle_rad".to_string(), Metric::Scalar(v));
    }

    for (name, val) in parse_extras(s, "var", "val") {
        rec.metrics.insert(name, Metric::Scalar(val));
    }

    Some((test, rec))
}

/// Parse `test_data/baselines.json` into `{test_name: TestRecord}`.
///
/// Baseline JSON keys already include units (`position_m`, `velocity_m_per_s`,
/// etc.); extras use `name`/`value` (not `var`/`val`).
fn parse_baselines(s: &str) -> BTreeMap<String, TestRecord> {
    let mut out = BTreeMap::new();
    let Some(tests_start) = s.find("\"tests\":") else {
        return out;
    };
    let rest = &s[tests_start..];
    let Some(obj_start) = rest.find('{') else {
        return out;
    };
    // Walk top-level test entries by tracking brace depth.
    let bytes = rest.as_bytes();
    let mut i = obj_start + 1;
    let mut depth = 1;
    while i < bytes.len() && depth > 0 {
        let c = bytes[i];
        if c == b'}' {
            depth -= 1;
            if depth == 0 {
                break;
            }
        }
        if c != b'"' {
            i += 1;
            continue;
        }
        // Found a key: scan to closing quote (skip escapes).
        let key_start = i + 1;
        let mut j = key_start;
        while j < bytes.len() && bytes[j] != b'"' {
            if bytes[j] == b'\\' {
                j += 2;
            } else {
                j += 1;
            }
        }
        if j >= bytes.len() {
            break;
        }
        let key = &rest[key_start..j];
        // Skip past closing quote + colon.
        let mut k = j + 1;
        while k < bytes.len() && bytes[k] != b':' {
            k += 1;
        }
        k += 1;
        // Find the matching object for this key's value (if it's an object).
        while k < bytes.len() && bytes[k].is_ascii_whitespace() {
            k += 1;
        }
        if k >= bytes.len() || bytes[k] != b'{' {
            i = k;
            continue;
        }
        if depth != 1 {
            i = k;
            continue;
        }
        // depth==1 + value is object → this is a test entry.
        let entry_start = k;
        let mut d2 = 1;
        let mut m = k + 1;
        while m < bytes.len() && d2 > 0 {
            match bytes[m] {
                b'{' => d2 += 1,
                b'}' => d2 -= 1,
                b'"' => {
                    let mut n = m + 1;
                    while n < bytes.len() && bytes[n] != b'"' {
                        if bytes[n] == b'\\' {
                            n += 2;
                        } else {
                            n += 1;
                        }
                    }
                    m = n;
                }
                _ => {}
            }
            m += 1;
        }
        let entry = &rest[entry_start..m];
        let rec = parse_baseline_entry(entry);
        out.insert(key.to_string(), rec);
        i = m;
    }
    out
}

fn parse_baseline_entry(entry: &str) -> TestRecord {
    let mut rec = TestRecord::default();
    for key in [
        "position_m",
        "velocity_m_per_s",
        "acceleration_m_per_s2",
        "ang_vel_rad_per_s",
        "ang_accel_rad_per_s2",
    ] {
        if let Some(v) = parse_vec3(entry, key) {
            rec.metrics.insert(key.to_string(), Metric::Vec3(v));
        }
    }
    if let Some(v) = parse_f64(entry, "quat_angle_rad") {
        rec.metrics
            .insert("quat_angle_rad".to_string(), Metric::Scalar(v));
    }
    for (name, val) in parse_extras(entry, "name", "value") {
        rec.metrics.insert(name, Metric::Scalar(val));
    }
    rec
}

// ── Shared JSON parsing helpers ──

fn extract_string(s: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = s.find(&pat)? + pat.len();
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_string())
}

fn parse_number_at(s: &str) -> Option<f64> {
    let s = s.trim_start();
    if s.starts_with("null") {
        return None;
    }
    let end = s
        .find(|c: char| {
            c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E' && !c.is_ascii_digit()
        })
        .unwrap_or(s.len());
    s[..end].parse().ok()
}

fn parse_f64(s: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let start = s.find(&pat)? + pat.len();
    parse_number_at(&s[start..])
}

fn parse_vec3(s: &str, key: &str) -> Option<[f64; 3]> {
    let pat = format!("\"{key}\":");
    let start = s.find(&pat)? + pat.len();
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

/// Extract `{name_key: string, value_key: number, ...}` pairs from an
/// `"extras":[...]` array.
fn parse_extras(s: &str, name_key: &str, value_key: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    let Some(start) = s.find("\"extras\":") else {
        return out;
    };
    let rest = &s[start..];
    let Some(arr_start) = rest.find('[') else {
        return out;
    };
    // Find matching close bracket (rfind is fine here; extras arrays don't
    // nest arrays inside objects).
    let Some(arr_end) = rest.rfind(']') else {
        return out;
    };
    let arr = &rest[arr_start + 1..arr_end];
    let mut pos = 0;
    while let Some(obj_start) = arr[pos..].find('{') {
        let obj_abs = pos + obj_start;
        let Some(obj_end_rel) = arr[obj_abs..].find('}') else {
            break;
        };
        let obj = &arr[obj_abs..obj_abs + obj_end_rel + 1];
        if let (Some(name), Some(value)) =
            (extract_string(obj, name_key), parse_f64(obj, value_key))
        {
            out.push((name, value));
        }
        pos = obj_abs + obj_end_rel + 1;
    }
    out
}

// ── Comparison logic ──

/// Tolerance per the #101 invariance policy: values may only grow by 1 ulp
/// relative to the frozen baseline, with a `1e-12` absolute floor.
fn tolerance(baseline: f64) -> f64 {
    (baseline * 1.0 + RELATIVE_SLACK * baseline.abs()).max(ABSOLUTE_FLOOR)
}

#[derive(Debug)]
struct Violation {
    test: String,
    metric: String,
    component: Option<usize>, // 0/1/2 for Vec3, None for Scalar
    baseline: f64,
    current: f64,
    tolerance: f64,
}

fn compare(
    baseline: &BTreeMap<String, TestRecord>,
    current: &BTreeMap<String, TestRecord>,
    allow_missing: &BTreeSet<String>,
) -> (Vec<Violation>, Vec<String>, Vec<String>, Vec<String>) {
    let mut violations = Vec::new();
    let mut missing_from_run = Vec::new();
    let mut allowed_missing = Vec::new();
    let mut new_in_run = Vec::new();

    for (test, base_rec) in baseline {
        let Some(cur_rec) = current.get(test) else {
            if allow_missing.contains(test) {
                allowed_missing.push(test.clone());
            } else {
                missing_from_run.push(test.clone());
            }
            continue;
        };
        for (metric, base_val) in &base_rec.metrics {
            let Some(cur_val) = cur_rec.metrics.get(metric) else {
                // Baseline had this metric, current run doesn't — treat as a
                // missing-measurement regression (hard fail).
                violations.push(Violation {
                    test: test.clone(),
                    metric: format!("{metric} (metric absent from current run)"),
                    component: None,
                    baseline: match base_val {
                        Metric::Scalar(v) => *v,
                        Metric::Vec3(v) => v.iter().copied().fold(0.0_f64, f64::max),
                    },
                    current: f64::NAN,
                    tolerance: f64::NAN,
                });
                continue;
            };
            match (base_val, cur_val) {
                (Metric::Vec3(b), Metric::Vec3(c)) => {
                    for i in 0..3 {
                        let tol = tolerance(b[i]);
                        if c[i] > tol {
                            violations.push(Violation {
                                test: test.clone(),
                                metric: metric.clone(),
                                component: Some(i),
                                baseline: b[i],
                                current: c[i],
                                tolerance: tol,
                            });
                        }
                    }
                }
                (Metric::Scalar(b), Metric::Scalar(c)) => {
                    let tol = tolerance(*b);
                    if *c > tol {
                        violations.push(Violation {
                            test: test.clone(),
                            metric: metric.clone(),
                            component: None,
                            baseline: *b,
                            current: *c,
                            tolerance: tol,
                        });
                    }
                }
                _ => {
                    violations.push(Violation {
                        test: test.clone(),
                        metric: format!("{metric} (shape mismatch: baseline vs current)"),
                        component: None,
                        baseline: f64::NAN,
                        current: f64::NAN,
                        tolerance: f64::NAN,
                    });
                }
            }
        }
    }

    for test in current.keys() {
        if !baseline.contains_key(test) {
            new_in_run.push(test.clone());
        }
    }

    (violations, missing_from_run, allowed_missing, new_in_run)
}

fn comp_label(c: Option<usize>) -> &'static str {
    match c {
        Some(0) => "[0]",
        Some(1) => "[1]",
        Some(2) => "[2]",
        _ => "",
    }
}

fn parse_args() -> Result<BTreeSet<String>, String> {
    let mut allow_missing = BTreeSet::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--allow-missing" => match args.next() {
                Some(name) => {
                    allow_missing.insert(name);
                }
                None => return Err("--allow-missing requires a test name".to_string()),
            },
            "--allow-missing-from" => match args.next() {
                Some(path) => {
                    load_allow_missing_file(&path, &mut allow_missing)?;
                }
                None => {
                    return Err("--allow-missing-from requires a file path".to_string());
                }
            },
            other => {
                return Err(format!(
                    "unknown argument: {other}\n\
                     usage: tier3_baseline_diff \\\n\
                            [--allow-missing NAME]... [--allow-missing-from FILE]"
                ));
            }
        }
    }
    Ok(allow_missing)
}

fn load_allow_missing_file(path: &str, allow_missing: &mut BTreeSet<String>) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("--allow-missing-from: cannot read {path}: {e}"))?;
    for line in raw.lines() {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if !trimmed.is_empty() {
            allow_missing.insert(trimmed.to_string());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let allow_missing = match parse_args() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    let root = workspace_root();
    let baselines_path = root.join("test_data").join("baselines.json");
    let current_dir = root.join("target").join("tier3_crossval");

    let baselines_raw = match fs::read_to_string(&baselines_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read {}: {e}", baselines_path.display());
            return ExitCode::from(2);
        }
    };
    let baselines = parse_baselines(&baselines_raw);
    if baselines.is_empty() {
        eprintln!(
            "{} parsed empty — aborting (fix parser or regenerate baselines)",
            baselines_path.display()
        );
        return ExitCode::from(2);
    }

    if !current_dir.exists() {
        eprintln!("no tier3 crossval output at {}", current_dir.display());
        eprintln!("run: cargo nextest run --workspace -E 'test(tier3_)' first");
        return ExitCode::from(2);
    }

    let mut current: BTreeMap<String, TestRecord> = BTreeMap::new();
    let mut files: Vec<_> = fs::read_dir(&current_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", current_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort_by_key(|e| e.file_name());
    for entry in &files {
        let content = fs::read_to_string(entry.path())
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", entry.path().display()));
        match parse_current_run(&content) {
            Some((test, rec)) => {
                current.insert(test, rec);
            }
            None => {
                eprintln!("warning: failed to parse {}", entry.path().display());
            }
        }
    }

    let (violations, missing, allowed, new) = compare(&baselines, &current, &allow_missing);

    // Warn about unused allow-missing entries (typos, stale CI config).
    let stale_allow: Vec<&String> = allow_missing
        .iter()
        .filter(|n| !baselines.contains_key(n.as_str()))
        .collect();
    if !stale_allow.is_empty() {
        eprintln!("warning: --allow-missing names not present in baselines (typo?):");
        for n in &stale_allow {
            eprintln!("  ? {n}");
        }
        eprintln!();
    }

    if !allowed.is_empty() {
        eprintln!(
            "note: {} baseline tests allowed-missing from current run:",
            allowed.len()
        );
        for t in &allowed {
            eprintln!("  ~ {t}");
        }
        eprintln!();
    }

    if !new.is_empty() {
        eprintln!(
            "note: {} tests in current run are not in baselines:",
            new.len()
        );
        for t in &new {
            eprintln!("  + {t}");
        }
        eprintln!(
            "  (re-freeze with `cargo run -p astrodyn_test_data --bin tier3_report -- --freeze-baselines`)"
        );
        eprintln!();
    }

    if missing.is_empty() && violations.is_empty() {
        println!(
            "baseline-diff: OK ({} matched; {} allowed-missing; {} new)",
            baselines.len() - allowed.len(),
            allowed.len(),
            new.len()
        );
        return ExitCode::SUCCESS;
    }

    if !missing.is_empty() {
        eprintln!(
            "FAIL: {} baseline tests missing from current run:",
            missing.len()
        );
        for t in &missing {
            eprintln!("  - {t}");
        }
        eprintln!();
    }

    if !violations.is_empty() {
        eprintln!(
            "FAIL: {} component(s) exceed baseline tolerance:",
            violations.len()
        );
        eprintln!("  formula: tolerance = max(baseline + 1e-12*|baseline|, 1e-12)");
        eprintln!();
        eprintln!(
            "  {:<50}  {:<24}  {:>14}  {:>14}  {:>14}  {:>14}",
            "test", "metric", "baseline", "current", "tolerance", "excess"
        );
        for v in &violations {
            let metric = format!("{}{}", v.metric, comp_label(v.component));
            let excess = if v.tolerance.is_finite() {
                v.current - v.tolerance
            } else {
                f64::NAN
            };
            eprintln!(
                "  {:<50}  {:<24}  {:>14.3e}  {:>14.3e}  {:>14.3e}  {:>14.3e}",
                v.test, metric, v.baseline, v.current, v.tolerance, excess
            );
        }
        eprintln!();
        eprintln!("policy (CLAUDE.md §\"Baseline freeze\"): refactor-only phases must not widen");
        eprintln!("baselines. If the delta is justified by a physics change, document the reason");
        eprintln!("in the PR body and refreeze with `--freeze-baselines`.");
    }

    ExitCode::from(1)
}
