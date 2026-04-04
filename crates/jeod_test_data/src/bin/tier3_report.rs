//! Generates a Markdown cross-validation error report from Tier 3 test results.
//!
//! Usage:
//!   cargo run -p jeod_test_data --bin tier3_report
//!
//! Reads JSON files from `target/tier3_crossval/` (written by `crossval_report()`
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

struct Metric {
    var: String,
    val: f64,
    unit: String,
}

struct TestResult {
    test: String,
    metrics: Vec<Metric>,
}

/// Minimal JSON parser — avoids adding serde as a dependency.
/// Parses: {"test":"name","metrics":[{"var":"x","val":1.23e-4,"unit":"m"}, ...]}
fn parse_json(s: &str) -> Option<TestResult> {
    let test = extract_string(s, "test")?;
    let metrics_start = s.find("\"metrics\":")?;
    let arr_start = s[metrics_start..].find('[')?;
    let arr_end = s[metrics_start..].rfind(']')?;
    let arr = &s[metrics_start + arr_start + 1..metrics_start + arr_end];

    let mut metrics = Vec::new();
    let mut pos = 0;
    while let Some(obj_start) = arr[pos..].find('{') {
        let obj_end = arr[pos + obj_start..].find('}')?;
        let obj = &arr[pos + obj_start..pos + obj_start + obj_end + 1];
        let var = extract_string(obj, "var")?;
        let val = extract_number(obj, "val")?;
        let unit = extract_string(obj, "unit").unwrap_or_default();
        metrics.push(Metric { var, val, unit });
        pos = pos + obj_start + obj_end + 1;
    }

    Some(TestResult { test, metrics })
}

fn extract_string(s: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":\"", key);
    let start = s.find(&pattern)? + pattern.len();
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_string())
}

fn extract_number(s: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let start = s.find(&pattern)? + pattern.len();
    let rest = s[start..].trim_start();
    let end = rest
        .find(|c: char| {
            c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E' && !c.is_ascii_digit()
        })
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
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
    writeln!(out, "{} tests reported metrics.", entries.len()).unwrap();
    writeln!(out).unwrap();

    // Per-test table
    writeln!(out, "## Per-Test Results").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Test | Variable | Max Error | Unit |").unwrap();
    writeln!(out, "|------|----------|-----------|------|").unwrap();

    for entry in &entries {
        let short = entry.test.replace("tier3_", "");
        for (i, m) in entry.metrics.iter().enumerate() {
            let test_col = if i == 0 { &short } else { "" };
            writeln!(
                out,
                "| {test_col} | {} | {:.6e} | {} |",
                m.var, m.val, m.unit
            )
            .unwrap();
        }
    }

    // Worst by variable
    writeln!(out).unwrap();
    writeln!(out, "## Worst Errors by Variable").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Variable | Worst Error | Unit | Test |").unwrap();
    writeln!(out, "|----------|-------------|------|------|").unwrap();

    let mut worst: Vec<(String, f64, String, String)> = Vec::new();
    for entry in &entries {
        for m in &entry.metrics {
            let key = m.var.clone();
            if let Some(existing) = worst.iter_mut().find(|w| w.0 == key) {
                if m.val > existing.1 {
                    existing.1 = m.val;
                    existing.2 = m.unit.clone();
                    existing.3 = entry.test.replace("tier3_", "");
                }
            } else {
                worst.push((key, m.val, m.unit.clone(), entry.test.replace("tier3_", "")));
            }
        }
    }
    worst.sort_by(|a, b| a.0.cmp(&b.0));

    for (var, val, unit, test) in &worst {
        writeln!(out, "| {var} | {val:.6e} | {unit} | {test} |").unwrap();
    }

    // Top 50 largest errors
    writeln!(out).unwrap();
    writeln!(out, "## Largest Errors").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Error | Unit | Variable | Test |").unwrap();
    writeln!(out, "|-------|------|----------|------|").unwrap();

    let mut all: Vec<(f64, String, String, String)> = Vec::new();
    for entry in &entries {
        for m in &entry.metrics {
            all.push((
                m.val,
                m.unit.clone(),
                m.var.clone(),
                entry.test.replace("tier3_", ""),
            ));
        }
    }
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (val, unit, var, test) in all.iter().take(50) {
        writeln!(out, "| {val:.6e} | {unit} | {var} | {test} |").unwrap();
    }

    writeln!(out).unwrap();
    writeln!(
        out,
        "*{} total metrics across {} tests.*",
        all.len(),
        entries.len()
    )
    .unwrap();

    eprintln!("Wrote {}", output_path.display());
}
