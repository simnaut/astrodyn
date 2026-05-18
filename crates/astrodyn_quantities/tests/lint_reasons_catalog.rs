//! Coverage test for the `astrodyn_quantities::lint_reasons` catalog.
//!
//! Every canonical string registered in
//! [`astrodyn_quantities::lint_reasons::clippy_float_cmp::ALL`] must
//! still appear verbatim in at least [`MIN_OCCURRENCES`] (= 2) `reason
//! = "..."` literals inside real `#[allow(...)]` / `#![allow(...)]`
//! attributes in the workspace.
//!
//! What this test catches:
//!
//! - A catalog string dropping below `MIN_OCCURRENCES` — e.g. a rename
//!   or typo applied at *most* sites that shrinks the verbatim cluster
//!   to one or zero matches.
//! - A catalog entry with zero matching attribute literals — a stale
//!   constant that no longer corresponds to any real bypass site.
//!
//! What this test does **not** catch:
//!
//! - Partial drift in a large cluster: a sub-theme used at six sites
//!   stays green even if three are reworded, because the remaining
//!   three still satisfy `≥ MIN_OCCURRENCES`. The catalog string is
//!   then technically still accurate for those three, but the others
//!   have silently diverged.
//! - A paraphrased call site: the scan is a verbatim substring match,
//!   so a near-miss at a fresh site is invisible to it.
//!
//! Treat this test as a stale-catalog detector and a low-bar typo
//! trip, not a uniform-wording enforcer. Uniform wording is enforced by
//! review (copy the catalog string verbatim; when you edit a catalog
//! string, grep the workspace for the old wording).
//!
//! Matches are counted only inside parsed attribute spans, not by raw
//! substring search, so a `reason = "..."` literal that appears in a
//! `//` comment, a doc comment, or a non-`allow` attribute does not
//! count toward the cluster size. Each `reason = "..."` literal inside
//! an attribute span counts as one occurrence; two sibling
//! `#[allow(...)]` attributes in the same file therefore count as two,
//! matching the audit-log semantics ("two real bypass sites").

use astrodyn_quantities::lint_reasons::clippy_float_cmp;
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

    // Iterate over the catalog's own `ALL` slice rather than
    // hand-duplicating it here. A new `pub const` added to
    // `clippy_float_cmp` only enters the coverage check once it's also
    // registered in `ALL` — see that slice's doc comment for the
    // contract.
    let mut failures = Vec::new();
    for (name, value) in clippy_float_cmp::ALL {
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
/// inside the opener).
///
/// The scanner explicitly skips four lexical contexts so a `reason =
/// "<catalog value>"` literal embedded in *any* of them does **not**
/// count toward the cluster size:
///
/// - `//` line comments (including `///` and `//!` doc comments) —
///   short-circuit to end-of-line.
/// - `/* ... */` block comments — Rust permits nesting, so the scanner
///   tracks an integer block-comment depth and only resumes scanning
///   when the depth returns to zero.
/// - Cooked string literals `"..."` — the scanner consumes characters
///   until the matching unescaped closing quote, honouring `\"` so an
///   escaped quote inside the literal doesn't close it.
/// - Raw string literals `r"..."` / `r#"..."#` / `r##"..."##` … — the
///   scanner remembers how many `#` hashes opened the literal and
///   closes only on a matching `"<same-count-of-#>` suffix. Raw
///   strings don't honour `\"`, mirroring Rust's lexer.
///
/// Skipping these contexts matters because example snippets in doc
/// comments (`///`), assertion-failure messages (`assert!(..., "...
/// #[allow(... reason = \"...\")] ...")`), or block-commented stubs
/// can legitimately quote the canonical catalog phrasings without
/// representing a real bypass site. The earlier scanner counted
/// those occurrences and could keep a stale catalog entry alive even
/// after every genuine `#[allow]` site was removed; the four extra
/// skip states close that hole.
///
/// The four skip states are mutually exclusive and dominate the
/// `#[allow(` opener test: a scanner that is currently inside a
/// string literal does not start a new attribute span even if the
/// literal's bytes spell `#[allow(`. That ordering is the load-bearing
/// invariant for the false-positive cases (`scanner_tests::*` covers
/// the cooked-string, block-comment, and raw-string variants
/// explicitly).
fn count_reason_in_allow_attrs(src: &str, needle_value: &str) -> usize {
    let needle = format!("reason = \"{needle_value}\"");
    let needle_bytes = needle.as_bytes();
    let bytes = src.as_bytes();

    // The scanner walks `src` as bytes (not as `char`s) so a UTF-8
    // multi-byte sequence (e.g. an em-dash inside a comment or
    // string literal) doesn't trip char-boundary slicing in `src[i..]`.
    // Each ASCII delimiter we care about — `#`, `/`, `(`, `)`, `[`,
    // `]`, `!`, `"`, `*`, the bytes of `allow(` and of the needle —
    // is a single byte under UTF-8, so byte-level comparisons are
    // exactly as precise as char-level ones for our matching needs.

    let mut count = 0usize;
    let mut depth: u32 = 0;
    // Rust block comments nest, so `/* /* */ */` is a single comment.
    // Track the open-block depth and only resume scanning when it
    // returns to zero.
    let mut block_comment_depth: u32 = 0;
    // Cooked-string state: when `in_cooked_string` is true, consume
    // bytes until an unescaped `"`. Tracks `cooked_escape_next` so a
    // backslash escapes the next byte (including a `"`).
    let mut in_cooked_string = false;
    let mut cooked_escape_next = false;
    // Raw-string state: when `Some(n)`, we are inside `r##…"…"##` with
    // exactly `n` opening hashes; close only on a `"` followed by
    // exactly `n` `#` bytes. Raw strings do not honour `\"`, mirroring
    // Rust's lexer (so `r"…\""` doesn't exist — the first unescaped
    // quote ends the literal).
    let mut raw_string_hashes: Option<usize> = None;
    let mut i = 0usize;

    while i < bytes.len() {
        // Highest-priority state: inside a block comment. Nothing else
        // is scanned until the matching `*/` closes the outermost
        // open block. A nested `/*` bumps the depth.
        if block_comment_depth > 0 {
            if bytes_starts_with(bytes, i, b"/*") {
                block_comment_depth += 1;
                i += 2;
                continue;
            }
            if bytes_starts_with(bytes, i, b"*/") {
                block_comment_depth -= 1;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        // Second-priority state: inside a raw string literal. Only a
        // closing `"<n hashes>` exits; the hash-count must match the
        // opener exactly. Raw strings ignore backslash escapes.
        if let Some(n_hashes) = raw_string_hashes {
            if bytes[i] == b'"' {
                // Check for `n_hashes` trailing `#` after the quote.
                let end = i + 1;
                let has_enough = end + n_hashes <= bytes.len()
                    && bytes[end..end + n_hashes].iter().all(|b| *b == b'#');
                if has_enough {
                    raw_string_hashes = None;
                    i = end + n_hashes;
                    continue;
                }
            }
            i += 1;
            continue;
        }

        // Third-priority state: inside a cooked string literal.
        // Honour `\<anything>` as an escape so `\"` doesn't close.
        if in_cooked_string {
            if cooked_escape_next {
                cooked_escape_next = false;
                i += 1;
                continue;
            }
            match bytes[i] {
                b'\\' => {
                    cooked_escape_next = true;
                    i += 1;
                    continue;
                }
                b'"' => {
                    in_cooked_string = false;
                    i += 1;
                    continue;
                }
                _ => {
                    i += 1;
                    continue;
                }
            }
        }

        // Skip a `//` line comment to end-of-line. Doc comments
        // (`///`, `//!`) share this prefix and are handled the same
        // way — neither can carry executable attribute syntax.
        if bytes_starts_with(bytes, i, b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Enter a block comment.
        if bytes_starts_with(bytes, i, b"/*") {
            block_comment_depth += 1;
            i += 2;
            continue;
        }

        // Enter a raw string literal `r"…"` or `r#"…"#` etc. Count
        // the `#` hashes between the `r` and the opening quote so the
        // closer can match exactly. A bare `r` inside an identifier
        // (e.g. `for`, `var_r`, `r_value`) is *not* a raw string —
        // those distinguish themselves by the next byte being an
        // identifier character rather than `#` or `"`, which makes the
        // `bytes[j] == b'"'` check below fail. Raw identifier syntax
        // (`r#type`) similarly fails the check because the byte after
        // the trailing hashes is an identifier byte, not `"`.
        if bytes[i] == b'r' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b'#' {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'"' {
                raw_string_hashes = Some(j - (i + 1));
                i = j + 1;
                continue;
            }
        }

        // Enter a cooked string literal. We don't distinguish
        // byte-strings (`b"…"`) here because their internal escape
        // grammar matches cooked strings closely enough that we
        // never miscount the catalog needle (which is a plain UTF-8
        // string anyway).
        if bytes[i] == b'"' {
            in_cooked_string = true;
            cooked_escape_next = false;
            i += 1;
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

    #[test]
    fn skips_cooked_string_literal_with_attribute_text() {
        // A cooked string literal that happens to spell out a fake
        // `#[allow(... reason = "...")]` snippet (e.g. as part of an
        // assertion-failure message or an error-formatting helper)
        // must not count as a live bypass — the audit-log invariant
        // is structural, not textual, and the bytes inside a string
        // literal are not an attribute.
        let src = r#"
fn foo() {
    let s = "example: #[allow(clippy::float_cmp, reason = \"typed-vs-raw parity tests assert bit-exact identity at the type boundary\")]";
    let _ = s;
}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 0);
    }

    #[test]
    fn skips_block_comment_with_attribute_text() {
        // A `/* ... */` block comment may carry example code that
        // quotes the canonical catalog phrasing verbatim. Those bytes
        // are not a real bypass site and must not count.
        let src = r#"
/* example: #[allow(clippy::float_cmp, reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary")] */
fn foo() {}
"#;
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 0);
    }

    #[test]
    fn skips_raw_string_literal_with_attribute_text() {
        // A raw string literal `r##"..."##` can contain an attribute
        // snippet without escaping its quotes — these are common in
        // test fixtures and macro-input docs. The scanner must close
        // raw strings on the matching hash count so the embedded
        // `#[allow(... reason = "...")]` doesn't count as a live
        // bypass. The literal itself uses three opening hashes so
        // that the embedded `"##` inside the catalog example doesn't
        // prematurely terminate the test fixture.
        let src = "let s = r###\"#[allow(clippy::float_cmp, reason = \"typed-vs-raw parity tests assert bit-exact identity at the type boundary\")]\"###;";
        assert_eq!(count_reason_in_allow_attrs(src, VALUE), 0);
    }
}
