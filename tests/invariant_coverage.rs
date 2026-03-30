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
    eprintln!("===============================");
    eprintln!();
}
