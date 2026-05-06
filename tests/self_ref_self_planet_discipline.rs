//! Lint: enforce TS.01 — `<SelfRef>` and `<SelfPlanet>` wildcards may only
//! appear at per-entity storage boundaries.
//!
//! Catalogued in `docs/JEOD_invariants.md` row TS.01. The wildcards
//! `jeod_quantities::frame::SelfRef` and `jeod_quantities::frame::SelfPlanet`
//! exist because per-entity storage decides the vehicle/planet identity at
//! runtime — Bevy `Component`s, `Message`s, runner `SimBody`/`VehicleConfig`/
//! `VehicleOutput` fields, and the dynamic-registry-erased return types in
//! `jeod_sim::derived` / `jeod_sim::atmosphere` / `jeod_sim::planet_config`.
//! All other code paths (system functions, public APIs, `jeod_*` algorithm
//! kernels) carry `<V: Vehicle>` / `<P: Planet>` parameters that flow from
//! the call site, never `<SelfRef>` / `<SelfPlanet>` minted afresh in a
//! system body.
//!
//! The auto-memory rule "No default type parameters on Planet-aware types"
//! (`<P: Planet = Earth>` and friends) is the inverse-direction sister
//! rule: defaults silently relax to `<SelfPlanet>` whenever inference has
//! no constraint, hiding a missing pinning decision; explicit wildcards at
//! runtime-resolved boundaries are not. Together the two rules close the
//! loop on the wildcard discipline.
//!
//! # Allow-list mechanism
//!
//! Each `SelfRef` / `SelfPlanet` token appearing in a workspace `.rs`
//! file must be covered by one of:
//!
//! 1. **Inline annotation** on the same line: `// JEOD_INV: TS.01` or
//!    `// allowed: <reason>` (the latter shared with the
//!    `scripts/check_no_escape_hatches.sh` policing of typed-quantity
//!    bypass constructors).
//! 2. **Preceding pure-comment annotation**: any of the immediately
//!    preceding consecutive comment lines (`//`, `///`, or `//!`) that
//!    sit between the token and the previous non-comment code line
//!    contains `JEOD_INV: TS.01` or `allowed:`.
//! 3. **File-level annotation**: the file contains a `JEOD_INV: TS.01`
//!    or `allowed:` mention anywhere in its first 80 lines (typically a
//!    module-doc `//!` opener that documents the file as a storage
//!    boundary, e.g. `src/components.rs`, `crates/jeod_runner/src/
//!    simulation/types.rs`).
//!
//! Documentation comments (`///`, `//!`) and string literals never trip
//! the lint — only "real" code uses do. The point is that mentioning
//! `SelfRef` in a doc comment that explains the boundary is fine; minting
//! a `SelfRef`-tagged value in a system body is not.
//!
//! # When the lint fires
//!
//! The lint catches the bug class the issue described: "a drive-by
//! `let foo: RelativeState<SelfRef, SelfRef> = …` inside a non-storage
//! code path silently degrades type safety with no compiler signal."
//! Adding such a line to a non-storage file (one without a TS.01
//! file-level marker) without an inline `// JEOD_INV: TS.01` tag will
//! fail this test with the offending file path, line number, and source
//! snippet.
//!
//! # Adding a new storage-boundary site
//!
//! When a new genuinely runtime-resolved boundary appears (e.g. another
//! Bevy `Component` newtype that must wrap a `<V>`-parametric typed
//! sibling), tag the type definition with `// JEOD_INV: TS.01 — <short
//! rationale>` and either (a) place the tag on a line directly above the
//! token, or (b) add a file-level `JEOD_INV: TS.01` mention to the
//! module-doc opener if many uses cluster in the same file. The lint's
//! preceding-comment scan covers (a); the first-80-lines scan covers (b).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const TARGET_TOKENS: &[&str] = &["SelfRef", "SelfPlanet"];
const TS01_TAG: &str = "JEOD_INV: TS.01";
const ALLOWED_PREFIX: &str = "allowed:";
const FILE_LEVEL_PROBE_LINES: usize = 80;

/// Recursively collect every `.rs` file under the given root that the
/// workspace builds. Skips `target/` (cargo build artifacts) and any
/// hidden directories (`.git`, `.claude`, etc).
fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') || name == "target" {
                continue;
            }
        }
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Returns `true` if the file's first [`FILE_LEVEL_PROBE_LINES`] lines
/// mention `JEOD_INV: TS.01` or `allowed:` inside a comment (`//`,
/// `///`, or `//!`). This is the per-file opt-in for storage-boundary
/// modules.
fn file_level_marker_present(content: &str) -> bool {
    for line in content.lines().take(FILE_LEVEL_PROBE_LINES) {
        let trimmed = line.trim_start();
        if !(trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("//!"))
        {
            continue;
        }
        if trimmed.contains(TS01_TAG) || trimmed.contains(ALLOWED_PREFIX) {
            return true;
        }
    }
    false
}

/// Strip line/block comments and string literals from a single line so
/// the token search only sees real code. Multi-line block comments and
/// multi-line strings are not handled here — uses of `SelfRef` /
/// `SelfPlanet` inside them are vanishingly rare and would be flagged
/// for manual review if encountered. The simple per-line filter is
/// sufficient for the doc-comment + string-literal cases that dominate
/// the false-positive surface.
fn strip_comments_and_strings(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut in_str = false;
    let mut in_char = false;
    while i < bytes.len() {
        let b = bytes[i];
        if !in_str && !in_char {
            // Line-comment start: drop the rest of the line.
            if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                break;
            }
            if b == b'"' {
                in_str = true;
                i += 1;
                continue;
            }
            if b == b'\'' {
                // Naive — a `'static` lifetime is benign here because the
                // only thing we care about is whether the matching token
                // appears as actual code. Lifetime tokens cannot contain
                // `SelfRef` / `SelfPlanet`, so leaving char-mode "open"
                // until the next `'` is safe even when the next `'` is
                // on a later line; we just don't enter string-suppression
                // unless we see a real char delimiter pair.
                in_char = true;
                i += 1;
                continue;
            }
            out.push(b as char);
        } else if in_str {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
        } else if in_char {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == b'\'' {
                in_char = false;
            }
        }
        i += 1;
    }
    out
}

/// Returns `true` if `line` is a pure-comment line (whitespace + `//`,
/// `///`, or `//!` followed by content), not a code line with a trailing
/// comment. Used to walk back through preceding annotations.
fn is_pure_comment_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//")
}

/// Returns `true` if `line` (after trimming) is empty.
fn is_blank_line(line: &str) -> bool {
    line.trim().is_empty()
}

/// Walk preceding lines looking for a `JEOD_INV: TS.01` or `allowed:`
/// annotation in any pure-comment line in the immediate comment block.
/// Stops at the first non-comment, non-blank line.
fn preceding_annotation_present(lines: &[&str], use_idx: usize) -> bool {
    let mut i = use_idx;
    while i > 0 {
        i -= 1;
        let line = lines[i];
        if is_blank_line(line) {
            // A blank line breaks the comment block above the use.
            return false;
        }
        if is_pure_comment_line(line) {
            if line.contains(TS01_TAG) || line.contains(ALLOWED_PREFIX) {
                return true;
            }
            continue;
        }
        // Any non-comment, non-blank code line: the use is not part of
        // a comment-annotated block.
        return false;
    }
    false
}

/// Returns `true` if the inline annotation on `line` (whether on the
/// same code or as a trailing `// allowed: …`) covers this use.
fn inline_annotation_present(line: &str) -> bool {
    line.contains(TS01_TAG) || line.contains(ALLOWED_PREFIX)
}

/// Find the byte offsets of every occurrence of any token in
/// `TARGET_TOKENS` inside `code` that is bordered by non-identifier
/// characters (so `SelfRef` matches but `MySelfRef` does not).
fn find_token_uses(code: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    for &token in TARGET_TOKENS {
        let bytes = code.as_bytes();
        let tb = token.as_bytes();
        let mut start = 0;
        while let Some(pos) = code[start..].find(token) {
            let abs = start + pos;
            let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
            let after_idx = abs + tb.len();
            let after_ok = after_idx == bytes.len() || !is_ident_byte(bytes[after_idx]);
            if before_ok && after_ok {
                hits.push(abs);
            }
            start = abs + tb.len();
        }
    }
    hits
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[derive(Debug)]
struct Violation {
    path: PathBuf,
    line: usize,
    snippet: String,
}

/// Inspect every `.rs` file under the workspace root and return a list
/// of `SelfRef` / `SelfPlanet` uses that lack the required allow-list
/// annotation.
fn scan_workspace() -> Vec<Violation> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    // The crate-local manifest dir is the workspace root for this
    // top-level test, but be explicit about the directories we scan so
    // the lint stays scoped to source code (no `target/`, `vendor/`,
    // generated artifacts, etc).
    for sub in ["src", "crates", "tests"] {
        collect_rs_files(&manifest_dir.join(sub), &mut files);
    }
    files.sort();

    let mut violations = Vec::new();
    let self_path = manifest_dir
        .join("tests")
        .join("self_ref_self_planet_discipline.rs");

    for path in files {
        // Self-skip: this lint file mentions the tokens in its own
        // documentation. We use the file-level marker mechanism to
        // exempt it (the module doc above already mentions `SelfRef`
        // and `SelfPlanet`), but be defensive — if someone strips the
        // marker, the lint should still pass on its own source.
        if path == self_path {
            continue;
        }

        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let file_marker = file_level_marker_present(&content);

        let lines: Vec<&str> = content.lines().collect();
        for (line_idx, raw_line) in lines.iter().enumerate() {
            // Skip pure-comment lines outright — mentions of the
            // tokens in `///` / `//!` doc comments are by design.
            if is_pure_comment_line(raw_line) {
                continue;
            }
            let stripped = strip_comments_and_strings(raw_line);
            let hits = find_token_uses(&stripped);
            if hits.is_empty() {
                continue;
            }
            // We have at least one real code use. Check coverage.
            if file_marker {
                continue;
            }
            if inline_annotation_present(raw_line) {
                continue;
            }
            if preceding_annotation_present(&lines, line_idx) {
                continue;
            }
            violations.push(Violation {
                path: path.clone(),
                line: line_idx + 1,
                snippet: raw_line.trim().to_string(),
            });
        }
    }
    violations
}

/// Direction 1 (the lint itself): every `SelfRef` / `SelfPlanet` code
/// use in the workspace must be covered by an inline tag, a preceding
/// pure-comment tag, or a file-level marker.
#[test]
fn self_ref_self_planet_uses_are_at_storage_boundaries() {
    let violations = scan_workspace();

    if !violations.is_empty() {
        let mut msg = String::from(
            "Found `SelfRef` / `SelfPlanet` uses outside the documented \
             per-entity storage boundary (TS.01).\n\n\
             These wildcards exist because per-entity storage decides the \
             vehicle/planet identity at runtime; system code paths must \
             carry `<V: Vehicle>` / `<P: Planet>` parameters instead.\n\n\
             Each violation below either needs:\n\
             - an inline `// JEOD_INV: TS.01 — <reason>` annotation on \
             the same line; or\n\
             - a `// JEOD_INV: TS.01 — <reason>` annotation on the \
             immediately preceding pure-comment line; or\n\
             - if the entire file is a storage-boundary module (Bevy \
             component file, runner state types, etc.), a `JEOD_INV: \
             TS.01` mention in the module-doc opener (`//!` block) in \
             the first 80 lines.\n\n\
             Violations:\n",
        );
        for v in &violations {
            msg.push_str(&format!(
                "  {}:{}: {}\n",
                v.path.display(),
                v.line,
                v.snippet
            ));
        }
        panic!("{}", msg);
    }
}

/// Direction 2: the documented allow-list files (Bevy component newtype
/// module, runner state-types module, the typed-frame definitions
/// module, the dynamic-registry-erased producers in `jeod_sim`) must
/// retain their file-level TS.01 markers — so a future cleanup can't
/// silently strip the marker without the lint catching the regression
/// that immediately follows.
///
/// Each entry below is a (file, why-it-must-have-the-marker) pair.
/// Adding an entry here is a stronger guarantee than the bidirectional
/// `invariant_coverage::catalog_to_source_coverage` test: that test
/// just requires *some* `JEOD_INV: TS.01` tag exists somewhere in the
/// workspace, while this list pins the marker at the canonical
/// boundaries.
#[test]
fn canonical_storage_boundary_files_carry_ts01_marker() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let pinned: &[(&str, &str)] = &[
        (
            "src/components.rs",
            "Bevy `Component` newtype boundary — `<SelfRef>` tags on \
             RotationalStateC / MassPropertiesC / TotalForceC / \
             FrameDerivativesC / GravityTorqueC / StructuralTransformC / \
             ExternalTorqueC / FlatPlateConfigC.",
        ),
        (
            "crates/jeod_runner/src/simulation/types.rs",
            "Runner storage boundary — `SimBody.flat_plate_state: \
             FlatPlateState<SelfRef>`, `SimBody.atmospheric_state: \
             AtmosphereState<SelfPlanet>`, `VehicleOutput.orbital_elements: \
             OrbitalElements<SelfPlanet>`.",
        ),
        (
            "crates/jeod_quantities/src/frame.rs",
            "Definition site for `SelfRef` and `SelfPlanet` — the \
             phantom markers themselves and their docstrings.",
        ),
        (
            "crates/jeod_sim/src/derived.rs",
            "Dynamic-registry-erased return types: \
             `compute_orbital_elements -> OrbitalElements<SelfPlanet>`, \
             paired with the planet-pinned `_typed` siblings.",
        ),
        (
            "crates/jeod_sim/src/atmosphere.rs",
            "Dynamic-registry-erased return types: \
             `evaluate_atmosphere -> AtmosphereState<SelfPlanet>`, \
             paired with the planet-pinned `evaluate_atmosphere_typed`.",
        ),
        (
            "crates/jeod_sim/src/planet_config.rs",
            "`PlanetConfig::mu_typed -> GravParam<SelfPlanet>` — \
             entity-resolved planet identity.",
        ),
    ];

    let mut missing: BTreeSet<String> = BTreeSet::new();
    for (rel, _why) in pinned {
        let path = manifest_dir.join(rel);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("required TS.01 boundary file `{rel}` is unreadable: {e}"));
        if !file_level_marker_present(&content) {
            missing.insert((*rel).to_string());
        }
    }

    assert!(
        missing.is_empty(),
        "Canonical TS.01 storage-boundary files lost their module-doc \
         marker — TS.01 file-level allow-list is not enforced for these \
         files until the marker returns:\n\n  {}\n\n\
         Restore a `JEOD_INV: TS.01 — <reason>` line in the `//!` opener \
         of each file (within the first 80 lines).",
        missing.iter().cloned().collect::<Vec<_>>().join("\n  ")
    );
}
