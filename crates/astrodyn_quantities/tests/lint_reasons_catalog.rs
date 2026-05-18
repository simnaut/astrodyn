//! Coverage test for the `astrodyn_quantities::lint_reasons` catalog.
//!
//! Every canonical string in [`astrodyn_quantities::lint_reasons`] must
//! still appear verbatim in at least one `reason = "..."` literal in the
//! workspace. If a future refactor renames a sub-theme at the call sites
//! but forgets to update the catalog, this test fails — preventing a
//! stale catalog entry from silently drifting away from the actual
//! audit-log strings.
//!
//! Conversely, this is also a *minimum-cluster* check: catalog entries
//! only exist for sub-themes that actually recur verbatim across the
//! workspace. The minimum threshold (2 occurrences) protects against
//! single-site rationales being centralized for no benefit.

use astrodyn_quantities::lint_reasons::clippy_float_cmp;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Each catalog constant must appear in at least this many `reason = "..."`
/// literals in the workspace. A cluster smaller than this doesn't justify
/// centralization — the issue's deduplication argument only holds for
/// genuinely recurring phrasings.
const MIN_OCCURRENCES: usize = 2;

#[test]
fn every_catalog_entry_has_at_least_min_occurrences() {
    let workspace_root = find_workspace_root();
    let rust_files = collect_rust_files(&workspace_root);

    let catalog: BTreeMap<&str, &str> = BTreeMap::from([
        ("TYPED_RAW_PARITY", clippy_float_cmp::TYPED_RAW_PARITY),
        (
            "TIER3_LITERAL_ANALYTIC",
            clippy_float_cmp::TIER3_LITERAL_ANALYTIC,
        ),
        (
            "BEVY_PARITY_STATE_FIELDS",
            clippy_float_cmp::BEVY_PARITY_STATE_FIELDS,
        ),
        (
            "BEVY_PARITY_TIME_FIELDS",
            clippy_float_cmp::BEVY_PARITY_TIME_FIELDS,
        ),
    ]);

    let mut failures = Vec::new();
    for (name, value) in &catalog {
        // We look for the catalog text wrapped in `reason = "..."` so we
        // only count attribute literals, not stray prose in doc comments.
        let needle = format!("reason = \"{}\"", value);
        let count = rust_files
            .iter()
            .filter(|path| {
                // Don't count the catalog's own source file or this test —
                // the catalog defines the string, it doesn't *use* it.
                let p = path.to_string_lossy();
                !p.ends_with("lint_reasons.rs") && !p.ends_with("lint_reasons_catalog.rs")
            })
            .map(|path| {
                fs::read_to_string(path)
                    .map(|src| src.matches(&needle).count())
                    .unwrap_or(0)
            })
            .sum::<usize>();

        if count < MIN_OCCURRENCES {
            failures.push(format!(
                "  {name}: found {count} occurrence(s) of `{needle}`, expected at least \
                 {MIN_OCCURRENCES}.\n    Either:\n      (a) the catalog entry is stale — \
                 the canonical phrasing was renamed at call sites; update the const value, \
                 or\n      (b) the cluster shrank below the centralization threshold; \
                 remove the const from the catalog."
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "lint_reasons catalog drift detected:\n{}",
        failures.join("\n"),
    );
}

/// Walk up from `CARGO_MANIFEST_DIR` until we find the workspace root
/// (the directory containing the top-level `Cargo.toml` with a `[workspace]`
/// table).
fn find_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dir = manifest_dir.as_path();
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(contents) = fs::read_to_string(&candidate) {
                if contents.contains("[workspace]") {
                    return dir.to_path_buf();
                }
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => panic!(
                "could not find workspace root above CARGO_MANIFEST_DIR ({})",
                manifest_dir.display()
            ),
        }
    }
}

/// Collect every `.rs` file under `crates/`, `src/`, and `tests/` in the
/// workspace root. Skips `target/` and any hidden directory.
fn collect_rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for sub in ["crates", "src", "tests"] {
        let dir = root.join(sub);
        if dir.exists() {
            walk(&dir, &mut out);
        }
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(it) => it,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name.starts_with('.') || name == "target" {
            continue;
        }
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
