//! Schema-flexible CSV reader for ad-hoc test formats.
//!
//! Phase 7's [`recipes::verification::csv_loader`](super::super::verification)
//! owns the typed CSV loaders for the standard JEOD reference logs
//! (`SIM_dyncomp`, `SIM_LVLH`, etc.). This module provides the
//! one-off / line-by-line escape hatch for archetype-B tests whose
//! format is unique enough that defining a typed loader for them
//! would have a single user.
//!
//! The function is intentionally minimal: read every line, skip
//! blanks and the header, return parsed `Vec<Vec<f64>>`. Callers
//! convert to their own record struct with field indices documented
//! at the call site.

use std::path::Path;

/// Read a CSV file and return one `Vec<f64>` per non-empty data row,
/// skipping the first line (header).
///
/// On read failure, panics with a helpful message that includes the
/// command to regenerate test data via Docker.
pub fn read_lines(path: &Path, sim_label: &str) -> Vec<Vec<f64>> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {sim_label} CSV from {}: {e}\n\
             Generate with: docker run --rm -v $(pwd)/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            path.display()
        )
    });
    let mut rows = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let row: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        rows.push(row);
    }
    rows
}
