//! Structured cross-validation error reporting for Tier 3 tests.
//!
//! Each test calls [`crossval_report`] with its error metrics. The function
//! writes a JSON file per test to `target/tier3_crossval/<test_name>.json`,
//! overwriting on each run. The report binary (`tier3_report`) reads all
//! files in that directory to produce a summary.

use std::io::Write;
use std::path::PathBuf;

fn output_dir() -> PathBuf {
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
/// `test_name` should match the `#[test]` function name.
///
/// Each metric is a `(&str, f64, f64, &str)` tuple:
/// `(variable_name, value, tolerance, unit)`.
///
/// Use `f64::INFINITY` for tolerance when there is no explicit threshold
/// (e.g., informational metrics, exact-match checks).
///
/// Example:
/// ```ignore
/// crossval_report("tier3_simulation_run2_3dof", &[
///     ("position", max_pos_error, 0.5, "m"),
///     ("velocity", max_vel_error, 0.001, "m/s"),
/// ]);
/// ```
pub fn crossval_report(test_name: &str, metrics: &[(&str, f64, f64, &str)]) {
    let dir = output_dir();
    let _ = std::fs::create_dir_all(&dir);

    let path = dir.join(format!("{test_name}.json"));

    let mut json = format!(r#"{{"test":"{test_name}","metrics":["#);
    for (i, (var, val, tol, unit)) in metrics.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let tol_str = if tol.is_finite() {
            format!("{tol:.6e}")
        } else {
            "null".to_string()
        };
        json.push_str(&format!(
            r#"{{"var":"{var}","val":{val:.6e},"tol":{tol_str},"unit":"{unit}"}}"#
        ));
    }
    json.push_str("]}");

    let mut file = std::fs::File::create(&path).expect("failed to create tier3_crossval JSON file");
    file.write_all(json.as_bytes())
        .expect("failed to write tier3_crossval JSON file");
}
