//! Schema-flexible CSV reader for ad-hoc test formats.
//!
//! The typed CSV catalogue lives in `astrodyn_verif_jeod::tier3_csv` (and
//! its predecessor `astrodyn_verif_jeod::dyncomp_csv`); those loaders own
//! the standard JEOD reference logs (`SIM_dyncomp`, `SIM_LVLH`, …).
//! This module provides the one-off / line-by-line escape hatch for
//! archetype-B tests whose format is unique enough that defining a
//! typed loader for them would have a single user.
//!
//! The function is intentionally minimal: read every line, skip
//! blanks and the header, return parsed `Vec<Vec<f64>>`. Callers
//! convert to their own record struct with field indices documented
//! at the call site.
//!
//! Per the project's "no graceful skip" rule, malformed numeric
//! fields panic with the exact `path:line:col` and offending token
//! rather than silently turning into `f64::NAN`.

use std::path::Path;

/// Read a CSV file and return one `Vec<f64>` per non-empty data row,
/// skipping the first line (header).
///
/// On read failure, panics with a helpful message that includes the
/// command to regenerate test data via Docker. On parse failure of
/// any field, panics with the offending `path:line:col` and the raw
/// token so the malformed input is never silently coerced to `NaN`.
pub fn read_lines(path: &Path, sim_label: &str) -> Vec<Vec<f64>> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {sim_label} CSV from {}: {e}\n\
             Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            path.display()
        )
    });
    let path_display = path.display();
    let mut rows = Vec::new();
    // 1-based line numbers in the panic message match what an editor
    // jumps to. `i` here is 0-based over `lines()`, so add 1.
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let line_no = i + 1;
        let row: Vec<f64> = line
            .split(',')
            .enumerate()
            .map(|(col, s)| {
                let token = s.trim();
                // 1-based column number to match editor / grep output
                // conventions (`enumerate()` is 0-based).
                let col_no = col + 1;
                token.parse::<f64>().unwrap_or_else(|e| {
                    panic!(
                        "CSV parse error at {path_display}:{line_no}:{col_no} \
                         (sim {sim_label}): {token:?} — {e}"
                    )
                })
            })
            .collect();
        rows.push(row);
    }
    rows
}
