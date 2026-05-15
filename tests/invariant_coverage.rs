//! Invariant coverage verification.
//!
//! Ensures bidirectional consistency between `docs/JEOD_invariants.md` and
//! `// JEOD_INV: XX.YY` tags in source code:
//!
//! 1. Every invariant marked `enforced`, `partial`, or `structural` (with a
//!    file reference) in the catalog MUST have at least one corresponding
//!    `// JEOD_INV: XX.YY` tag in source.
//!
//! 2. Every `// JEOD_INV: XX.YY` tag in source MUST reference an invariant
//!    that exists in the catalog.
//!
//! 3. (Informational, seed gate from #531) Every `enforced` invariant row
//!    SHOULD have at least one `#[should_panic]` negative test that drives
//!    the misconfiguration and verifies the panic message — this proves
//!    the `assert!`/`panic!` site actually fires rather than being a
//!    silently-dead branch. The negative test is recognised by a
//!    `// JEOD_INV: XX.YY` tag inside or immediately preceding a
//!    `#[should_panic]` test function. This third leg is informational
//!    this round (prints a `::warning::`-style coverage report but does
//!    not fail the build); subsequent PRs add the body of negative tests
//!    per catalog section, and the assertion is promoted to a hard gate
//!    once the catalog is fully covered.
//!
//! Fails CI if someone:
//! - Marks an invariant as `enforced` without adding a source tag
//! - Removes a source tag without updating the catalog status
//! - Adds a source tag referencing a nonexistent invariant ID

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Parse JEOD_invariants.md and return a map of tag → status for all invariants.
fn parse_catalog() -> BTreeMap<String, String> {
    let md_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/JEOD_invariants.md");
    assert!(
        md_path.exists(),
        "docs/JEOD_invariants.md not found at {}. \
         This file is the authoritative invariant catalog.",
        md_path.display()
    );

    let content = fs::read_to_string(&md_path).expect("Failed to read JEOD_invariants.md");
    let mut invariants = BTreeMap::new();

    for line in content.lines() {
        // Table rows look like: | DB.05 | description | ... | status |
        if !line.starts_with("| ") {
            continue;
        }
        let cols: Vec<&str> = line.split('|').map(str::trim).collect();
        // cols[0] = "" (before first |), cols[1] = tag, ... cols[N-1] = "" (after last |)
        // "Our Status" is always the last non-empty data column (cols[N-2]).
        if cols.len() < 4 {
            continue;
        }
        let tag = cols[1];
        let status = cols[cols.len() - 2]; // last data column before trailing ""

        // Skip header rows
        if tag == "Tag" || tag.starts_with("---") {
            continue;
        }

        // Validate tag format: SECTION.NUMBER (e.g., DB.05, GV.12)
        if !tag.contains('.') {
            continue;
        }

        invariants.insert(tag.to_string(), status.to_string());
    }

    invariants
}

/// Recursively find all `// JEOD_INV: XX.YY` tags in Rust source files under crates/ and src/.
fn find_source_tags() -> BTreeMap<String, Vec<String>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut tags: BTreeMap<String, Vec<String>> = BTreeMap::new();

    collect_tags_recursive(&manifest_dir.join("crates"), &mut tags);
    collect_tags_recursive(&manifest_dir.join("src"), &mut tags);
    tags
}

fn collect_tags_recursive(dir: &Path, tags: &mut BTreeMap<String, Vec<String>>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_tags_recursive(&path, tags);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let rel_path = path.strip_prefix(manifest_dir).unwrap_or(&path);

            for (line_num, line) in content.lines().enumerate() {
                // Find all JEOD_INV: XX.YY patterns on this line
                for tag in extract_inv_tags(line) {
                    let location = format!("{}:{}", rel_path.display(), line_num + 1);
                    tags.entry(tag).or_default().push(location);
                }
            }
        }
    }
}

fn extract_inv_tags(line: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut search = line;
    while let Some(idx) = search.find("JEOD_INV: ") {
        let after = &search[idx + 10..]; // skip "JEOD_INV: "
                                         // Extract tag: letters, then dot, then digits
        let tag_end = after
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '.')
            .unwrap_or(after.len());
        let tag = &after[..tag_end];
        if tag.contains('.') && tag.len() >= 4 {
            tags.push(tag.to_string());
        }
        search = &after[tag_end..];
    }
    tags
}

/// Direction 1: Every invariant with status `enforced`, `partial`, or
/// `structural` (containing a file reference) must have at least one
/// source tag.
#[test]
fn catalog_to_source_coverage() {
    let catalog = parse_catalog();
    let source_tags = find_source_tags();
    let mut missing = Vec::new();

    for (tag, status) in &catalog {
        let needs_tag = status.starts_with("enforced")
            || status.starts_with("partial")
            || (status.starts_with("structural") && status.contains(".rs"));

        if needs_tag && !source_tags.contains_key(tag) {
            missing.push(format!(
                "  {tag}: marked as `{status}` but no // JEOD_INV: {tag} found in source"
            ));
        }
    }

    assert!(
        missing.is_empty(),
        "Invariants marked enforced/partial/structural in JEOD_invariants.md \
         but missing source tags:\n{}\n\n\
         Fix: add `// JEOD_INV: XX.YY` comment at the enforcement site, \
         or update the catalog status to `deferred` or `n/a`.",
        missing.join("\n")
    );
}

/// Direction 2: Every `// JEOD_INV: XX.YY` tag in source must reference
/// an invariant that exists in the catalog.
#[test]
fn source_to_catalog_coverage() {
    let catalog = parse_catalog();
    let source_tags = find_source_tags();
    let mut orphans = Vec::new();

    for (tag, locations) in &source_tags {
        if !catalog.contains_key(tag) {
            orphans.push(format!(
                "  {tag}: tagged at {} but not in JEOD_invariants.md",
                locations.join(", ")
            ));
        }
    }

    assert!(
        orphans.is_empty(),
        "Source tags reference invariants not in the catalog:\n{}\n\n\
         Fix: add the invariant to docs/JEOD_invariants.md, \
         or remove the orphaned source tag.",
        orphans.join("\n")
    );
}

/// Direction 3: every "Our Status" column must begin with one of the five
/// recognized status words. Protects against typos drifting into the catalog.
#[test]
fn catalog_status_values_are_valid() {
    let catalog = parse_catalog();
    let valid = ["enforced", "partial", "structural", "deferred", "n/a"];
    let mut invalid = Vec::new();
    for (tag, status) in &catalog {
        if !valid.iter().any(|v| status.starts_with(v)) {
            invalid.push(format!(
                "  {tag}: status `{status}` starts with none of {valid:?}"
            ));
        }
    }
    assert!(
        invalid.is_empty(),
        "Invariants with unrecognized status prefix:\n{}\n\n\
         Fix: the `Our Status` column must start with one of \
         `enforced`, `partial`, `structural`, `deferred`, or `n/a`.",
        invalid.join("\n")
    );
}

/// Direction 4: if a status row cites a Rust file (`something.rs`), the file
/// must actually exist in the repo. Catches renames and deletions that left
/// the catalog stale.
#[test]
fn catalog_file_references_exist() {
    let catalog = parse_catalog();
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut missing = Vec::new();

    // Resolve bare filenames relative to the top-level crate `src/` and every
    // workspace member under `crates/*/src`, discovered at runtime so adding a
    // crate doesn't require touching this list.
    let mut roots = vec![manifest_dir.join("src")];
    let crates_dir = manifest_dir.join("crates");
    if let Ok(entries) = fs::read_dir(&crates_dir) {
        roots.extend(entries.filter_map(|entry| {
            let entry = entry.ok()?;
            let src_dir = entry.path().join("src");
            src_dir.is_dir().then_some(src_dir)
        }));
    }

    for (tag, status) in &catalog {
        for token in extract_rs_paths(status) {
            // Drop any `:...` suffix — line numbers (`:141`), ranges (`:141-142`),
            // and function-name hints (`:add_mass_point`) all just annotate the
            // file reference; none of them affect whether the file itself exists.
            let path_part = token.split(':').next().unwrap();
            // Try the path as-written, then under each discovered source root.
            let candidates: Vec<_> = std::iter::once(manifest_dir.join(path_part))
                .chain(roots.iter().map(|root| root.join(path_part)))
                .collect();
            let exists = candidates.iter().any(|p| p.exists());
            if !exists {
                missing.push(format!(
                    "  {tag}: cites `{token}` but the file does not exist under any known crate root"
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "Catalog rows cite .rs files that are missing (renamed or deleted?):\n{}\n\n\
         Fix: update the `Our Status` column to point at the current location, \
         or move the invariant to `deferred`/`n/a` if the code no longer exists.",
        missing.join("\n")
    );
}

/// Extract `*.rs` filenames/paths from a status-column string, ignoring backticks
/// and surrounding punctuation.
fn extract_rs_paths(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in s.split(|c: char| {
        c.is_whitespace() || c == '`' || c == '(' || c == ')' || c == ',' || c == ';'
    }) {
        let trimmed = raw.trim_end_matches('.');
        // Accept `foo.rs`, plus any `:`-separated annotation: line numbers
        // (`foo.rs:141`), line ranges (`foo.rs:141-142`), and function-name
        // hints (`foo.rs:add_mass_point`) are all treated as valid file refs.
        let stripped = trimmed.trim_start_matches('(');
        let candidate = stripped.trim_end_matches(['.', ')', ',']);
        let parts: Vec<&str> = candidate.split(':').collect();
        if !parts.is_empty() && parts[0].ends_with(".rs") && !parts[0].is_empty() {
            out.push(candidate.to_string());
        }
    }
    out
}

/// Direction 5: each source tag site must have descriptive text on the same
/// line or the immediately preceding comment line — enforces the CLAUDE.md
/// rule that tags must describe what the code does, not just appear.
#[test]
fn source_tag_comments_are_nontrivial() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bare = Vec::new();
    check_tag_comments_recursive(&manifest_dir.join("crates"), manifest_dir, &mut bare);
    check_tag_comments_recursive(&manifest_dir.join("src"), manifest_dir, &mut bare);

    assert!(
        bare.is_empty(),
        "JEOD_INV tags without descriptive comment text (rule from CLAUDE.md):\n{}\n\n\
         Fix: extend the `// JEOD_INV: XX.YY` comment with a short phrase describing \
         what the code actually enforces or how it diverges from JEOD.",
        bare.join("\n")
    );
}

fn check_tag_comments_recursive(dir: &Path, manifest_root: &Path, bare: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            check_tag_comments_recursive(&path, manifest_root, bare);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            let rel_path = path.strip_prefix(manifest_root).unwrap_or(&path);
            let lines: Vec<&str> = content.lines().collect();

            for (line_num, line) in lines.iter().enumerate() {
                let Some(idx) = line.find("JEOD_INV:") else {
                    continue;
                };
                // Accept description anywhere on the line (before or after the tag),
                // or on the immediately preceding comment line. This matches the
                // common doc-comment patterns `/// ... (JEOD_INV: XX.YY)` and
                // `// <description>\n// JEOD_INV: XX.YY`.
                let mut context = String::new();
                // Preceding comment line.
                if line_num > 0 {
                    let prev = lines[line_num - 1].trim_start();
                    if prev.starts_with("//") || prev.starts_with("///") {
                        context.push_str(prev);
                    }
                }
                // Text on the same line, excluding the "JEOD_INV: XX.YY" substring.
                let before = &line[..idx];
                let after = &line[idx..];
                // Skip "JEOD_INV:", whitespace, then the tag token.
                let after_tag = after["JEOD_INV:".len()..].trim_start();
                let tag_end = after_tag
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '.')
                    .unwrap_or(after_tag.len());
                let rest = &after_tag[tag_end..];
                context.push(' ');
                context.push_str(before);
                context.push(' ');
                context.push_str(rest);

                if context.chars().filter(|c| c.is_alphanumeric()).count() < 15 {
                    bare.push(format!(
                        "  {}:{}: bare tag `{}`",
                        rel_path.display(),
                        line_num + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
}

/// Verify no duplicate invariant IDs in the catalog.
#[test]
fn no_duplicate_catalog_ids() {
    let md_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/JEOD_invariants.md");
    let content = fs::read_to_string(&md_path).expect("Failed to read JEOD_invariants.md");
    let mut seen = BTreeSet::new();
    let mut duplicates = Vec::new();

    for line in content.lines() {
        if !line.starts_with("| ") {
            continue;
        }
        let cols: Vec<&str> = line.split('|').map(str::trim).collect();
        if cols.len() < 3 {
            continue;
        }
        let tag = cols[1];
        if tag == "Tag" || tag.starts_with("---") || !tag.contains('.') {
            continue;
        }
        if !seen.insert(tag.to_string()) {
            duplicates.push(tag.to_string());
        }
    }

    assert!(
        duplicates.is_empty(),
        "Duplicate invariant IDs in JEOD_invariants.md: {:?}",
        duplicates
    );
}

/// Scan source files for `#[should_panic]` test functions whose
/// surrounding context (immediately preceding lines or function body)
/// carries a `// JEOD_INV: XX.YY` tag, and return the set of catalog
/// IDs that have at least one such negative test.
///
/// Matching window: any `JEOD_INV:` tag within the test function's
/// body, OR on any of the 6 lines immediately preceding the
/// `#[should_panic]` attribute (covers a leading doc comment, the
/// `#[test]` attribute, and a tag-bearing comment above it).
///
/// Tags appearing only in unrelated source comments (i.e. outside a
/// `#[should_panic]`-attributed test) are NOT counted as negative
/// tests — the existing `// JEOD_INV: GV.04` tag at the
/// panic-emitting `assert!` call site is documentation of the
/// enforcement site, not proof that the panic actually fires.
fn find_negative_tests() -> BTreeMap<String, Vec<String>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut tagged: BTreeMap<String, Vec<String>> = BTreeMap::new();

    let mut roots = vec![
        manifest_dir.join("crates"),
        manifest_dir.join("src"),
        manifest_dir.join("tests"),
    ];
    // Also scan per-crate `tests/` directories explicitly — the recursive
    // walk below picks them up via `crates/<crate>/tests/`, but listing
    // the top-level `tests/` separately ensures the workspace-root
    // integration tests are covered too.
    let crates_dir = manifest_dir.join("crates");
    if let Ok(entries) = fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            let tests_dir = entry.path().join("tests");
            if tests_dir.is_dir() {
                roots.push(tests_dir);
            }
        }
    }

    for root in roots {
        scan_negative_tests_recursive(&root, manifest_dir, &mut tagged);
    }
    tagged
}

fn scan_negative_tests_recursive(
    dir: &Path,
    manifest_root: &Path,
    tagged: &mut BTreeMap<String, Vec<String>>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_negative_tests_recursive(&path, manifest_root, tagged);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            let rel_path = path.strip_prefix(manifest_root).unwrap_or(&path);
            let lines: Vec<&str> = content.lines().collect();

            for (idx, line) in lines.iter().enumerate() {
                let trimmed = line.trim_start();
                if !trimmed.starts_with("#[should_panic") {
                    continue;
                }
                // Collect tags from a window: 6 lines above through the
                // closing brace of the function body. The brace search
                // counts depth from the first `{` on or after the
                // `#[should_panic]` line.
                let window_start = idx.saturating_sub(6);
                let body_end = find_test_fn_body_end(&lines, idx);
                let window_end = body_end.unwrap_or(lines.len().min(idx + 80));

                let mut window_tags = BTreeSet::new();
                for window_line in &lines[window_start..window_end] {
                    for tag in extract_inv_tags(window_line) {
                        window_tags.insert(tag);
                    }
                }
                let location = format!("{}:{}", rel_path.display(), idx + 1);
                for tag in window_tags {
                    tagged.entry(tag).or_default().push(location.clone());
                }
            }
        }
    }
}

/// Given a `#[should_panic]` attribute line at `attr_idx`, find the
/// line index just past the test function's closing brace by simple
/// brace counting. Returns `None` if no balanced body is found within
/// 200 lines (defensive — keeps the scanner fast on malformed input).
fn find_test_fn_body_end(lines: &[&str], attr_idx: usize) -> Option<usize> {
    let mut depth: i32 = 0;
    let mut started = false;
    let limit = lines.len().min(attr_idx + 200);
    for (i, line) in lines.iter().enumerate().take(limit).skip(attr_idx) {
        for c in line.chars() {
            match c {
                '{' => {
                    depth += 1;
                    started = true;
                }
                '}' => {
                    depth -= 1;
                }
                _ => {}
            }
        }
        if started && depth == 0 {
            return Some(i + 1);
        }
    }
    None
}

/// Direction 6 (informational seed from #531): every `enforced`
/// catalog row should have at least one negative test driving the
/// misconfiguration. Emits a `::warning::`-style coverage report and
/// PASSES regardless — the bidirectional gate is staged in chunks
/// (this PR seeds the matcher + a handful of tests; subsequent PRs
/// fill in the per-section bodies; the final PR in the chain promotes
/// this to a hard `assert!`).
///
/// Rationale: today the bidirectional `tag ↔ catalog` checks above
/// only prove the tag *exists* at the enforcement site. They do not
/// prove the `assert!` actually fires under misconfiguration — an
/// AI-authored refactor could neuter a panic site to a silent
/// `if … return` and CI would stay green. A `#[should_panic(expected
/// = "…")]` negative test that drives the failure and pins the panic
/// message text closes that gap.
#[test]
fn enforced_rows_have_negative_tests_or_warn() {
    let catalog = parse_catalog();
    let neg_tests = find_negative_tests();

    let mut covered_enforced: Vec<&String> = Vec::new();
    let mut missing_enforced: Vec<&String> = Vec::new();
    let mut covered_partial: Vec<&String> = Vec::new();
    let mut missing_partial: Vec<&String> = Vec::new();

    for (tag, status) in &catalog {
        // Skip rows that are not in scope for negative tests:
        // - `structural` is enforced at compile time (deleted ctor,
        //   value member, typestate witness) — there is no runtime
        //   panic to drive.
        // - `deferred` and `n/a` are out of scope by definition.
        let scope_enforced = status.starts_with("enforced");
        let scope_partial = status.starts_with("partial");
        if !scope_enforced && !scope_partial {
            continue;
        }
        let has_test = neg_tests.contains_key(tag);
        match (scope_enforced, has_test) {
            (true, true) => covered_enforced.push(tag),
            (true, false) => missing_enforced.push(tag),
            (false, true) => covered_partial.push(tag),
            (false, false) => missing_partial.push(tag),
        }
    }

    let enforced_total = covered_enforced.len() + missing_enforced.len();
    let partial_total = covered_partial.len() + missing_partial.len();

    eprintln!();
    eprintln!("=== Negative-test coverage (#531 seed gate, informational) ===");
    eprintln!(
        "enforced: {}/{} rows have a #[should_panic] negative test",
        covered_enforced.len(),
        enforced_total
    );
    eprintln!(
        "partial:  {}/{} rows have a #[should_panic] negative test (best-effort)",
        covered_partial.len(),
        partial_total
    );
    if !missing_enforced.is_empty() {
        eprintln!();
        eprintln!(
            "::warning:: {} `enforced` row(s) lack a #[should_panic] negative test.",
            missing_enforced.len()
        );
        eprintln!(
            "::warning:: Add `#[should_panic(expected = \"<substring>\")]` test(s) near \
             the enforcement site, tagged with `// JEOD_INV: XX.YY`."
        );
        eprintln!("Missing (enforced):");
        for tag in &missing_enforced {
            eprintln!("  - {tag}");
        }
    }
    if !missing_partial.is_empty() {
        eprintln!();
        eprintln!(
            "(`partial` rows without negative tests — informational, lower priority): {}",
            missing_partial
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    eprintln!("================================================================");
    eprintln!();

    // Pass unconditionally this round. The gate is promoted to an
    // assertion once the catalog body lands (tracked as follow-up
    // sub-issues of #517 / #531).
    //
    // Sanity check: the matcher must locate at least the seed tests
    // landed alongside this gate; if zero rows are covered, the
    // scanner is broken and the informational-mode escape hatch is
    // hiding the breakage.
    assert!(
        !covered_enforced.is_empty(),
        "Negative-test scanner found zero covered `enforced` rows. \
         The seed tests in `gravity_controls.rs`, `mass.rs`, and \
         `crates/astrodyn_quantities/tests/quat_normalize_panic.rs` \
         should each contribute coverage; if all of them are missing \
         the scanner has regressed (window, brace-counting, or root \
         set), not the test suite."
    );
}

/// Print a coverage summary (informational, not an assertion).
/// Run with: `cargo test --test invariant_coverage coverage_summary -- --ignored --nocapture`
#[test]
#[ignore]
fn coverage_summary() {
    let catalog = parse_catalog();
    let source_tags = find_source_tags();

    let total = catalog.len();
    let enforced = catalog
        .values()
        .filter(|s| s.starts_with("enforced"))
        .count();
    let partial = catalog
        .values()
        .filter(|s| s.starts_with("partial"))
        .count();
    let structural = catalog
        .values()
        .filter(|s| s.starts_with("structural"))
        .count();
    let deferred = catalog
        .values()
        .filter(|s| s.starts_with("deferred"))
        .count();
    let na = catalog.values().filter(|s| s.starts_with("n/a")).count();
    let not_enforced = catalog
        .values()
        .filter(|s| s.starts_with("not enforced"))
        .count();
    let tagged_count = source_tags.len();
    let tag_sites: usize = source_tags.values().map(|v| v.len()).sum();

    eprintln!();
    eprintln!("=== JEOD Invariant Coverage ===");
    eprintln!("Catalog:     {total} invariants");
    eprintln!("  enforced:  {enforced}");
    eprintln!("  partial:   {partial}");
    eprintln!("  structural:{structural}");
    eprintln!("  deferred:  {deferred}");
    eprintln!("  n/a:       {na}");
    eprintln!("  not enforced: {not_enforced}");
    eprintln!("Source tags: {tagged_count} unique IDs, {tag_sites} total sites");

    let neg_tests = find_negative_tests();
    let neg_covered: usize = catalog
        .iter()
        .filter(|(tag, status)| status.starts_with("enforced") && neg_tests.contains_key(*tag))
        .count();
    eprintln!("Negative tests (`#[should_panic]` + JEOD_INV tag): {neg_covered}/{enforced} enforced rows covered");
    eprintln!();

    // Per-section breakdown.
    let mut per_section: BTreeMap<String, [usize; 5]> = BTreeMap::new();
    for (tag, status) in &catalog {
        let Some(dot) = tag.find('.') else { continue };
        let section = tag[..dot].to_string();
        let counts = per_section.entry(section).or_insert([0; 5]);
        let idx = if status.starts_with("enforced") {
            0
        } else if status.starts_with("partial") {
            1
        } else if status.starts_with("structural") {
            2
        } else if status.starts_with("deferred") {
            3
        } else {
            4
        };
        counts[idx] += 1;
    }
    eprintln!("--- Per section (enforced / partial / structural / deferred / n/a) ---");
    for (section, counts) in &per_section {
        eprintln!(
            "  {:>3}: {:>3} / {:>3} / {:>3} / {:>3} / {:>3}  (total {})",
            section,
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            counts.iter().sum::<usize>()
        );
    }
    eprintln!("===============================");
    eprintln!();
}
