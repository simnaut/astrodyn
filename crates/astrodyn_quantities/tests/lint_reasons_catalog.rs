//! Coverage test for the `astrodyn_quantities::lint_reasons` catalog.
//!
//! Every canonical string in [`astrodyn_quantities::lint_reasons`] must
//! still appear verbatim in at least [`MIN_OCCURRENCES`] (= 2) `reason
//! = "..."` literals inside real `#[allow(...)]` / `#![allow(...)]`
//! attributes in the workspace. If a future refactor renames a
//! sub-theme at the call sites but forgets to update the catalog, this
//! test fails — preventing a stale catalog entry from silently drifting
//! away from the actual audit-log strings.
//!
//! Conversely, this is also a *minimum-cluster* check: catalog entries
//! only exist for sub-themes that actually recur verbatim across the
//! workspace. The [`MIN_OCCURRENCES`] threshold protects against
//! single-site rationales being centralized for no benefit.
//!
//! Matches are counted only inside parsed attribute spans, not by raw
//! substring search, so a `reason = "..."` literal that appears in a
//! `//` comment, a doc comment, or a non-`allow` attribute does not
//! count toward the cluster size. Each `reason = "..."` literal inside
//! an attribute span counts as one occurrence; two sibling
//! `#[allow(...)]` attributes in the same file therefore count as two,
//! matching the audit-log semantics ("two real bypass sites").

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
        // We only count `reason = "<value>"` literals that live inside a
        // real `#[allow(...)]` or `#![allow(...)]` attribute span (see
        // `count_reason_in_allow_attrs`). A raw substring search would
        // also match `reason = "..."` text in `//` line comments, doc
        // comments, or unrelated string literals — that's exactly the
        // false-positive surface the audit-log invariant cares about
        // distinguishing.
        let count: usize = rust_files
            .iter()
            .filter(|path| {
                // Don't count the catalog's own source file or this test —
                // the catalog defines the string, it doesn't *use* it.
                let p = path.to_string_lossy();
                !p.ends_with("lint_reasons.rs") && !p.ends_with("lint_reasons_catalog.rs")
            })
            .map(|path| {
                fs::read_to_string(path)
                    .map(|src| count_reason_in_allow_attrs(&src, value))
                    .unwrap_or(0)
            })
            .sum();

        if count < MIN_OCCURRENCES {
            failures.push(format!(
                "  {name}: found {count} `#[allow(... reason = \"{value}\")]` occurrence(s), \
                 expected at least {MIN_OCCURRENCES}.\n    Either:\n      (a) the catalog \
                 entry is stale — the canonical phrasing was renamed at call sites; update \
                 the const value, or\n      (b) the cluster shrank below the centralization \
                 threshold; remove the const from the catalog."
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "lint_reasons catalog drift detected:\n{}",
        failures.join("\n"),
    );
}

/// Count occurrences of `reason = "<needle_value>"` that sit inside a
/// real `#[allow(...)]` or `#![allow(...)]` attribute span.
///
/// Walks `src` in a single linear pass, tracking the parenthesis depth
/// of an open `#[allow(` / `#![allow(` token across multi-line
/// attribute blocks (the canonical layout in this workspace, where
/// `clippy::float_cmp` and `reason = "..."` sit on separate lines
/// inside the opener). A `//` line comment short-circuits to
/// end-of-line, so `reason = "..."` text inside a `//` or `///`
/// comment is never counted.
///
/// We do not parse block comments (`/* ... */`); the catalog strings
/// are long, specific audit-log phrasings that don't show up inside
/// block-commented code in practice, and proper handling (nested
/// block comments, string literals containing `*/`, etc.) buys
/// nothing on the real codebase. A `reason = "<catalog value>"`
/// inside `/* ... */` would inflate the cluster size, which is the
/// harmless direction for a minimum-cluster check.
///
/// We also do not parse arbitrary string literals: a `(` appearing
/// inside a non-catalog `reason = "..."` value would erroneously
/// inflate `depth`. The audit-log invariant is structural — every
/// `reason = "..."` in this workspace lives inside an `#[allow]`
/// attribute by policy — so this risk is limited to a future
/// contributor adding `reason = "...with ( in it..."` to a *non*-allow
/// attribute, which would only affect this counter if the
/// non-balanced paren preceded a `#[allow(` site. None of that
/// happens today; if it ever does, the symptom is a single
/// false-positive count, not a CI break.
fn count_reason_in_allow_attrs(src: &str, needle_value: &str) -> usize {
    let needle = format!("reason = \"{needle_value}\"");
    let needle_bytes = needle.as_bytes();
    let bytes = src.as_bytes();

    // The scanner walks `src` as bytes (not as `char`s) so a UTF-8
    // multi-byte sequence (e.g. an em-dash inside a comment or
    // string literal) doesn't trip char-boundary slicing in `src[i..]`.
    // Each ASCII delimiter we care about — `#`, `/`, `(`, `)`, `[`,
    // `]`, `!`, the bytes of `allow(` and of the needle — is a
    // single byte under UTF-8, so byte-level comparisons are
    // exactly as precise as char-level ones for our matching needs.

    let mut count = 0usize;
    let mut depth: u32 = 0;
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip a `//` line comment to end-of-line. Doc comments
        // (`///`, `//!`) share this prefix and are handled the same
        // way — neither can carry executable attribute syntax.
        if bytes_starts_with(bytes, i, b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Detect an attribute opener at this position. We require the
        // exact `#[allow(` or `#![allow(` prefix so unrelated
        // attributes (`#[cfg(...)]`, `#[derive(...)]`, …) don't bump
        // `depth` — the audit-log invariant is specifically about
        // `allow` bypasses.
        if bytes[i] == b'#' {
            if bytes_starts_with(bytes, i, b"#[allow(") {
                depth += 1;
                i += b"#[allow(".len();
                continue;
            }
            if bytes_starts_with(bytes, i, b"#![allow(") {
                depth += 1;
                i += b"#![allow(".len();
                continue;
            }
        }

        if depth > 0 {
            // Inside an `#[allow(...)]` span: track balanced parens so
            // a nested `(` (e.g. a sub-attribute argument) doesn't
            // close the outer span prematurely.
            if bytes[i] == b'(' {
                depth += 1;
                i += 1;
                continue;
            }
            if bytes[i] == b')' {
                depth -= 1;
                i += 1;
                continue;
            }
            // Match the needle only inside the span; jump past it on
            // success so two adjacent occurrences (unlikely in
            // practice but easy to handle) each count once.
            if bytes_starts_with(bytes, i, needle_bytes) {
                count += 1;
                i += needle_bytes.len();
                continue;
            }
        }

        i += 1;
    }

    count
}

/// Byte-level `starts_with` at offset `at`. Avoids the char-boundary
/// requirement of `str` slicing, which matters when `src` contains
/// multi-byte UTF-8 chars (e.g. em-dashes in comments) that happen to
/// land on an iteration boundary in `count_reason_in_allow_attrs`.
fn bytes_starts_with(haystack: &[u8], at: usize, prefix: &[u8]) -> bool {
    at + prefix.len() <= haystack.len() && &haystack[at..at + prefix.len()] == prefix
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

#[cfg(test)]
mod scanner_tests {
    //! Self-checks for `count_reason_in_allow_attrs`. These exercise the
    //! invariants the file-level doc-comment makes load-bearing: comment
    //! lines don't count, non-`allow` attributes don't count, and
    //! sibling `#[allow]` attributes in the same file each count once.

    use super::count_reason_in_allow_attrs;

    const VALUE: &str = "typed-vs-raw parity tests assert bit-exact identity at the type boundary";

    #[test]
    fn counts_multiline_allow_attribute() {
        let src = r#"
#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary"
)]
mod tests {}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 1);
    }

    #[test]
    fn counts_single_line_allow_attribute() {
        let src = r#"
#[allow(clippy::float_cmp, reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary")]
fn foo() {}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 1);
    }

    #[test]
    fn counts_inner_allow_attribute() {
        let src = r#"
#![allow(
    clippy::float_cmp,
    reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary"
)]
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 1);
    }

    #[test]
    fn skips_line_comment_with_attribute_text() {
        let src = r#"
// #[allow(clippy::float_cmp, reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary")]
fn foo() {}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 0);
    }

    #[test]
    fn skips_doc_comment_with_attribute_text() {
        let src = r#"
/// Example: `#[allow(clippy::float_cmp, reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary")]`
fn foo() {}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 0);
    }

    #[test]
    fn skips_non_allow_attribute() {
        // A hypothetical (non-existent in stable Rust) attribute that
        // happens to use the same `reason = "..."` shape — our scanner
        // must not count it, because the audit-log invariant is
        // specifically about `allow` bypasses.
        let src = r#"
#[some_other_attr(
    reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary"
)]
fn foo() {}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 0);
    }

    #[test]
    fn counts_two_sibling_allow_attributes_in_same_file() {
        // Two distinct `#[allow]` sites in the same file count as two
        // occurrences, matching the audit-log semantics: each site is
        // its own bypass with its own justification.
        let src = r#"
#[allow(clippy::float_cmp, reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary")]
fn foo() {}

#[allow(
    clippy::float_cmp,
    reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary"
)]
fn bar() {}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 2);
    }

    #[test]
    fn em_dash_in_comment_does_not_panic() {
        // Real workspace files use em-dashes (U+2014, 3 UTF-8 bytes)
        // freely in `// JEOD_INV: …` source tags. A byte-level scanner
        // that occasionally slices `str` mid-codepoint would panic
        // here; the byte-only `bytes_starts_with` helper exists to
        // prevent exactly that.
        let src = "// JEOD_INV: TS.01 — see invariant catalog\nfn foo() {}\n";
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 0);
    }
}
