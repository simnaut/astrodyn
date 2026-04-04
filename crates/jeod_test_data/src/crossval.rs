//! Structured cross-validation error reporting for Tier 3 tests.
//!
//! Each test calls [`crossval_report`] with its error metrics. The function
//! writes a JSON file per test to `target/tier3_crossval/<test_name>.json`,
//! overwriting on each run. The report script (`scripts/tier3_report.sh`)
//! reads all files in that directory to produce a summary.

use std::io::Write;
use std::path::PathBuf;

fn output_dir() -> PathBuf {
    // Walk up from the crate directory to find the workspace root (where Cargo.lock lives),
    // then use target/tier3_crossval/ under it.
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            break;
        }
        if !dir.pop() {
            dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            break;
        }
    }
    dir.join("target").join("tier3_crossval")
}

/// Report cross-validation error metrics for a single test.
///
/// `test_name` should match the `#[test]` function name (e.g., `"tier3_simulation_run2_3dof"`).
///
/// Each metric is a `(&str, f64, &str)` tuple: `(variable_name, value, unit)`.
///
/// Example:
/// ```ignore
/// crossval_report("tier3_simulation_run2_3dof", &[
///     ("position", 2.214764e-6, "m"),
///     ("velocity", 2.324993e-9, "m/s"),
/// ]);
/// ```
pub fn crossval_report(test_name: &str, metrics: &[(&str, f64, &str)]) {
    let dir = output_dir();
    let _ = std::fs::create_dir_all(&dir);

    let path = dir.join(format!("{test_name}.json"));

    // Build JSON by hand to avoid serde dependency.
    let mut json = format!(r#"{{"test":"{test_name}","metrics":["#);
    for (i, (var, val, unit)) in metrics.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            r#"{{"var":"{var}","val":{val:.6e},"unit":"{unit}"}}"#
        ));
    }
    json.push_str("]}");

    let mut file = std::fs::File::create(&path).expect("failed to create tier3_crossval JSON file");
    file.write_all(json.as_bytes())
        .expect("failed to write tier3_crossval JSON file");
}
