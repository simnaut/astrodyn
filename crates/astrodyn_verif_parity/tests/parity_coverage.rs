//! Coverage CI guard for issue #389 — keeps the bevy parity test set a
//! superset of every Tier 3 topic.
//!
//! Walks `crates/astrodyn_verif_jeod/tests/tier3_*.rs` and
//! `crates/astrodyn_verif_parity/tests/bevy_parity_*.rs`, strips the
//! family prefixes (`tier3_`, optional `sim_`, `bevy_parity_`), and
//! asserts:
//!
//! ```text
//! tier3_topics ⊂ bevy_parity_topics ∪ KNOWN_PARITY_GAPS
//! ```
//!
//! A new `tier3_*` test that lands without a matching `bevy_parity_*`
//! sibling and isn't documented in [`KNOWN_PARITY_GAPS`] fails CI here,
//! preventing the silent-regression mode the issue's matrix table
//! describes.
//!
//! There are two granularities of "deliberate gap":
//!
//! 1. **Topic-level** — the whole `bevy_parity_<topic>.rs` file is
//!    missing. Document by adding `<topic>` to [`KNOWN_PARITY_GAPS`].
//! 2. **Per-test** — the wrapper file exists and covers most
//!    scenarios, but one individual `#[test]` is `#[ignore]`d with a
//!    `parity-gap:` reason. A topic-level entry would *hide* this
//!    sub-gap (the coverage test is filename-based and treats the
//!    topic as covered the moment the file exists), so per-test
//!    ignores are tracked separately in
//!    [`KNOWN_PER_TEST_PARITY_GAPS`]. Each `#[ignore = "parity-gap:
//!    …"]` annotation under `bevy_parity_*.rs` must be allow-listed
//!    by its full test-function name, and each allow-list entry must
//!    correspond to a real ignored test — the test enforces both
//!    directions to prevent silent regression in either.

use std::collections::BTreeSet;
use std::path::Path;

/// Topics whose Tier 3 sibling exists but whose parity counterpart is
/// deliberately absent (or `#[ignore]`d) for a documented structural
/// reason. The entries here are the "intentional gap" set the
/// `parity_coverage` test exempts.
///
/// Two flavors of gap live in this list:
///
/// 1. **Deferred** — the bridge or recipe layer doesn't cover the
///    topic *yet*, but a parity wrapper is expected once the blocker
///    lifts. These entries MUST link to the tracking issue (typically
///    `#389` or a follow-up) so the gap can be closed and the entry
///    dropped when the issue lands.
/// 2. **Permanent** — the topic is structurally out of scope for the
///    `VerificationCaseParityExt` trait (no trajectory CSV, pure
///    analytical/solver test, structural mass-tree composition). The
///    reason field states *why* the topic doesn't fit; no issue link
///    is required because there is no follow-up planned.
///
/// The set is intentionally small. Issue #389 closes the bulk of the
/// deferred cluster; entries that remain are either narrowly-scoped
/// follow-ups or permanent out-of-scope cases.
const KNOWN_PARITY_GAPS: &[(&str, &str)] = &[
    // ── Multi-planet scenarios: the bridge spawns all bodies under a
    //    single `<P>` today, so cases that integrate in two
    //    planet-inertial frames need a non-generic dispatch (#389
    //    risk note).
    (
        "apollo8_frame_switch",
        "multi-planet scenario (Earth ⇄ Moon frame switch) — bridge needs \
         non-generic Planet dispatch (#389 risk)",
    ),
    (
        "apollo_mass_tree",
        "Apollo lunar transfer (Earth ⇄ Moon) — same multi-planet gap as \
         apollo8_frame_switch",
    ),
    (
        "apollo_trajectory",
        "Apollo lunar transfer (Earth ⇄ Moon) — same multi-planet gap as \
         apollo8_frame_switch",
    ),
    (
        "earth_moon",
        "Earth ⇄ Moon dual-body sim — multi-planet gap (#389 risk)",
    ),
    (
        "mars_orbit",
        "Mars-centered scenario — bridge today fixes <P=Earth> across the \
         scenario; multi-planet generic dispatch tracked as #389 follow-up",
    ),
    (
        "mercury",
        "Heliocentric Mercury orbit — same single-planet bridge gap as \
         mars_orbit (#389 risk)",
    ),
    (
        "planetary",
        "Multi-planet planetary integration sim — bridge gap (#389 risk)",
    ),
    // ── Mass-tree-only structural tests with no JEOD CSV.
    (
        "attach_mass",
        "structural mass-tree composition test — no trajectory CSV, \
         doesn't fit VerificationCase shape",
    ),
    (
        "complex_attach_detach",
        "structural mass-tree composition test — no trajectory CSV",
    ),
    (
        "contact",
        "structural contact-pair test — no trajectory CSV",
    ),
    // ── Pure analytical / math-comparison tests with no propagation.
    (
        "battin",
        "Battin/Lambert solver test — analytical, not a propagation scenario",
    ),
    (
        "integ_analytical",
        "analytical integrator-comparison test — no Bevy parity counterpart",
    ),
    (
        "integ_comparison",
        "analytical integrator-comparison test — no Bevy parity counterpart",
    ),
    (
        "integ_gj_orders",
        "GJ-order sweep — analytical, depends on pre-recipe `gj` factory",
    ),
    // ── Pre-recipe tier3 siblings: the `VerificationCase` factory
    //    doesn't exist yet, so the parity trait has nothing to drive.
    //    Recipe migration is tracked as a follow-up to #389. The
    //    long tail below covers every pre-recipe topic in the
    //    workspace today; each entry collapses to "wrap once the
    //    recipe lands", and the matching follow-up can drop the entry
    //    when the wrapper file is created.
    (
        "dyncomp_combinations",
        "pre-recipe family aggregator — multiple sub-cases need recipe \
         factories before parity can drive them (#389 follow-up)",
    ),
    (
        "dyncomp_run9",
        "pre-recipe sibling — recipe factory for run9 not yet defined \
         (#389 follow-up)",
    ),
    (
        "dyncomp_run_attach_to_ref_frame",
        "pre-recipe sibling exercising attach_to_frame — recipe factory \
         not yet defined; needs `pre_step` Bevy support too (#389 follow-up)",
    ),
    (
        "drag_6dof",
        "pre-recipe sibling — drag-family recipe factories not yet defined \
         (#389 follow-up)",
    ),
    (
        "drag_analytical",
        "analytical drag verification — out of trait scope (no propagation)",
    ),
    (
        "drag_rot_verif",
        "pre-recipe sibling — drag-rotation recipe factory not yet defined \
         (#389 follow-up)",
    ),
    (
        "drag_ver",
        "pre-recipe sibling — drag-family recipe factory not yet defined \
         (#389 follow-up)",
    ),
    (
        "drag_verif",
        "pre-recipe sibling — drag-family recipe factory not yet defined \
         (#389 follow-up)",
    ),
    (
        "force_torque_response",
        "pre-recipe sibling exercising external forces/torques — \
         recipe factory not yet defined (#389 follow-up)",
    ),
    (
        "lsode",
        "pre-recipe sibling for LSODE integrator — recipe factory not yet \
         defined (#389 follow-up); LSODE integrator may need its own \
         per-step state component on the Bevy side",
    ),
    (
        "orbelem_comprehensive",
        "pre-recipe sibling — comprehensive sweep recipe not yet defined \
         (#389 follow-up)",
    ),
    (
        "orbinit_docker",
        "pre-recipe sibling exercising orbital-element initialization — \
         recipe factory not yet defined (#389 follow-up)",
    ),
    (
        "orbinit_edge",
        "pre-recipe edge-case sibling — recipe factory not yet defined \
         (#389 follow-up)",
    ),
    (
        "orbinit_families",
        "pre-recipe sibling — sweep across orbit families, recipe factory \
         not yet defined (#389 follow-up)",
    ),
    (
        "orbinit_roundtrip",
        "pre-recipe sibling — Cartesian↔Keplerian round-trip, recipe not yet \
         defined (#389 follow-up)",
    ),
    (
        "ref_attach",
        "pre-recipe sibling exercising attach_to_frame — recipe factory \
         not yet defined (#389 follow-up)",
    ),
    (
        "relative_extended",
        "pre-recipe sibling (same family as `relative`) — needs new \
         CsvReference variant; follow-up to #389",
    ),
    (
        "solar_beta_extended",
        "pre-recipe sibling — extended cases recipe factory not yet defined \
         (#389 follow-up)",
    ),
    (
        "time_docker",
        "pre-recipe sibling exercising time-scale conversions — recipe \
         factory not yet defined (#389 follow-up)",
    ),
    (
        "time_reversal",
        "pre-recipe sibling exercising time reversal — recipe factory \
         not yet defined (#389 follow-up)",
    ),
    (
        "timescale",
        "pre-recipe sibling — recipe factory not yet defined (#389 follow-up)",
    ),
    // ── dyncomp run3-run10: most have recipe factories
    //    (sim_dyncomp::run3a_sh4x4, run4_3rd_body, run7a_*, run10a_*, …)
    //    but several rely on `pre_step` for ephemeris updates and the
    //    wrapper hasn't been added yet. Tracked individually so each
    //    can be dropped from the gap list as its wrapper lands.
    //
    // dyncomp_run2 covered by `bevy_parity_dyncomp_run2_3dof.rs` (the
    // pilot wrapper); the prefix-match in `is_covered_by_parity` lets
    // it satisfy this entry implicitly, so it is not listed here.
    //
    // dyncomp_run6 covered by `bevy_parity_dyncomp_run6.rs` — drag
    // family (run6a_const_density_drag, run6b_drag,
    // run6b_drag_rotated_struct, run6b_drag_aero_traj).
];

/// Per-test parity gaps: individual `#[test]` functions inside a
/// `bevy_parity_*.rs` file that are `#[ignore]`d with a `parity-gap:`
/// reason. Entries are keyed by the full Rust function name (matching
/// `fn <name>()` in the source). Each entry must point at a real
/// ignored test, and every `#[ignore = "parity-gap: …"]` annotation in
/// the parity test set must appear here — the meta-test below enforces
/// both directions so a sub-scenario gap can't silently regress into
/// "fully covered" just because the topic-level wrapper file exists.
const KNOWN_PER_TEST_PARITY_GAPS: &[(&str, &str)] = &[
    // `lvlh_extended::periodicity` uses `dt = period / 560`, which is
    // irrational in seconds (≈9.917 s). The runner integrates with the
    // raw f64; the Bevy adapter routes time through
    // `Time<Fixed>::advance_by(Duration::from_secs_f64(dt))`, which
    // rounds to integer nanoseconds. The two paths diverge in the LSBs
    // of position after the first few ticks even though the
    // `astrodyn_*` math is identical. Re-enabling the wrapper needs a
    // Bevy-side time-advance path that preserves full f64 dt precision.
    (
        "bevy_parity_lvlh_periodicity",
        "irrational dt loses precision through Time<Fixed>'s Duration \
         round-trip; needs Bevy-side f64 time advance",
    ),
];

#[test]
fn parity_topics_are_a_superset_of_tier3_topics() {
    let workspace_root = workspace_root();
    let tier3_topics = collect_topics(
        &workspace_root.join("crates/astrodyn_verif_jeod/tests"),
        "tier3_",
    );
    assert!(
        !tier3_topics.is_empty(),
        "no tier3 tests discovered in crates/astrodyn_verif_jeod/tests/ — coverage \
         test cannot run"
    );

    let parity_topics = collect_topics(
        &workspace_root.join("crates/astrodyn_verif_parity/tests"),
        "bevy_parity_",
    );

    let allowed: BTreeSet<&'static str> = KNOWN_PARITY_GAPS.iter().map(|(t, _)| *t).collect();

    let mut uncovered: Vec<String> = Vec::new();
    for topic in &tier3_topics {
        if is_covered_by_parity(topic, &parity_topics) {
            continue;
        }
        if allowed.contains(topic.as_str()) {
            continue;
        }
        uncovered.push(topic.clone());
    }

    if !uncovered.is_empty() {
        let mut msg = String::new();
        msg.push_str(
            "tier3 topics without a matching bevy_parity_* sibling (and not in \
             KNOWN_PARITY_GAPS):\n",
        );
        for t in &uncovered {
            msg.push_str(&format!("  - {t}\n"));
        }
        msg.push_str(
            "\nFix by either:\n  \
             1. Adding `crates/astrodyn_verif_parity/tests/bevy_parity_<topic>.rs`, or\n  \
             2. Documenting the gap in `KNOWN_PARITY_GAPS` (parity_coverage.rs) with a reason.\n",
        );
        panic!("{msg}");
    }

    // Surface stale `KNOWN_PARITY_GAPS` entries so a topic that has
    // since been covered (or removed from tier3) doesn't sit in the
    // exemption list forever.
    let mut stale: Vec<&str> = Vec::new();
    for (topic, _reason) in KNOWN_PARITY_GAPS {
        if !tier3_topics.contains(*topic) {
            stale.push(topic);
        }
    }
    assert!(
        stale.is_empty(),
        "KNOWN_PARITY_GAPS contains topics that no longer exist in tier3_*.rs: {stale:?}\n  \
         Either restore the missing tier3 test or drop the exemption.",
    );

    // Cross-list redundancy: a topic with an existing parity wrapper
    // shouldn't *also* be in `KNOWN_PARITY_GAPS` (the gap entry would
    // be a lie). Catches the "added the wrapper but forgot to drop the
    // gap entry" failure mode.
    let mut redundant: Vec<&str> = Vec::new();
    for (topic, _reason) in KNOWN_PARITY_GAPS {
        if is_covered_by_parity(topic, &parity_topics) {
            redundant.push(topic);
        }
    }
    assert!(
        redundant.is_empty(),
        "KNOWN_PARITY_GAPS lists topics that already have a parity wrapper: {redundant:?}\n  \
         Drop the redundant gap entry — the wrapper supersedes it.",
    );
}

#[test]
fn per_test_parity_gaps_match_ignored_wrappers() {
    let workspace_root = workspace_root();
    let parity_dir = workspace_root.join("crates/astrodyn_verif_parity/tests");
    let discovered = collect_per_test_parity_gaps(&parity_dir);

    let allow: BTreeSet<&'static str> =
        KNOWN_PER_TEST_PARITY_GAPS.iter().map(|(t, _)| *t).collect();
    let discovered_names: BTreeSet<String> = discovered.iter().map(|(t, _)| t.clone()).collect();

    // Every `#[ignore = "parity-gap: …"]` in the parity test set must
    // be allow-listed by full test name. A new ignored wrapper that
    // lands without an entry here fails CI — preventing the silent
    // regression mode the topic-level coverage check can't catch.
    let mut unlisted: Vec<&str> = Vec::new();
    for (name, _) in &discovered {
        if !allow.contains(name.as_str()) {
            unlisted.push(name.as_str());
        }
    }
    assert!(
        unlisted.is_empty(),
        "bevy_parity_*.rs tests carry `#[ignore = \"parity-gap: …\"]` but are not \
         listed in KNOWN_PER_TEST_PARITY_GAPS: {unlisted:?}\n  \
         Add each test by its full Rust function name to KNOWN_PER_TEST_PARITY_GAPS \
         with a reason mirroring the ignore annotation.",
    );

    // And conversely: stale allow-list entries must drop. A test that
    // was previously ignored and is now either active or removed
    // shouldn't keep a parity-gap exemption silently.
    let mut stale: Vec<&str> = Vec::new();
    for (name, _) in KNOWN_PER_TEST_PARITY_GAPS {
        if !discovered_names.contains(*name) {
            stale.push(name);
        }
    }
    assert!(
        stale.is_empty(),
        "KNOWN_PER_TEST_PARITY_GAPS references tests that no longer carry an \
         `#[ignore = \"parity-gap: …\"]` annotation (or no longer exist): {stale:?}\n  \
         Drop the stale entries — either the wrapper is now active or the test was \
         renamed/removed.",
    );
}

/// Decide whether a tier3 topic is covered by some `bevy_parity_*.rs`
/// file. The matching rule is exact-or-prefix: tier3 `dyncomp_run2`
/// counts as covered when a parity wrapper exists named
/// `bevy_parity_dyncomp_run2.rs` (exact) or
/// `bevy_parity_dyncomp_run2_3dof.rs` (prefix). The tier3 file groups
/// related test functions while parity wrappers tend to split per
/// scenario flavor; the prefix rule handles that asymmetry without
/// forcing either side to rename.
fn is_covered_by_parity(tier3_topic: &str, parity_topics: &BTreeSet<String>) -> bool {
    let prefix_with_sep = format!("{tier3_topic}_");
    parity_topics
        .iter()
        .any(|p| p == tier3_topic || p.starts_with(&prefix_with_sep))
}

/// Walk `dir` for files matching `<prefix>*.rs`, strip the prefix, and
/// return the resulting topic set. Also strips a leading `sim_` from
/// `tier3_` topics so e.g. `tier3_sim_dyncomp_run2.rs` → `dyncomp_run2`,
/// matching the issue's matrix-table convention.
fn collect_topics(dir: &Path, prefix: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("utf-8 path");
        if let Some(topic) = stem.strip_prefix(prefix) {
            // For tier3 tests, strip the inner `sim_` infix used by
            // the SIM_*-style scenario names so the topic strings line
            // up between `tier3_sim_dyncomp_run2` and
            // `bevy_parity_dyncomp_run2`.
            let topic = topic.strip_prefix("sim_").unwrap_or(topic);
            out.insert(topic.to_string());
        }
    }
    out
}

/// Resolve the workspace root from `CARGO_MANIFEST_DIR` (set by Cargo
/// for tests). The verif_parity crate sits at
/// `<root>/crates/astrodyn_verif_parity`, so the workspace root is two
/// directories up.
fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by Cargo when running integration tests");
    Path::new(&manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is at <root>/crates/<name>")
        .to_path_buf()
}

/// Scan every `bevy_parity_*.rs` file in `dir` for `#[test]` functions
/// annotated with `#[ignore = "parity-gap: …"]` and return the
/// `(function_name, reason)` pairs. The parser is intentionally
/// narrow — it matches the exact annotation shape the codebase uses
/// (the doc comment in this file documents the contract) rather than
/// trying to handle arbitrary attribute syntax. Anything that doesn't
/// fit the shape is silently skipped, which is the conservative choice:
/// an unparseable annotation simply won't be allow-listed, so CI fails
/// loudly with a clear hint about the expected form.
fn collect_per_test_parity_gaps(dir: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("utf-8 path");
        if !stem.starts_with("bevy_parity_") {
            continue;
        }
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        out.extend(parse_parity_gap_ignores(&src));
    }
    out.sort();
    out
}

/// Extract `(function_name, reason)` pairs from `src` for every
/// `#[ignore = "parity-gap: <reason>"]` annotation attached to a
/// `#[test]` function. The two attributes may appear in either order
/// and may be separated by arbitrary whitespace, comments, or
/// continuation lines (`"…\ …"`). A function name is anchored on the
/// first `fn <ident>(` after the attribute pair.
fn parse_parity_gap_ignores(src: &str) -> Vec<(String, String)> {
    const MARKER: &str = "#[ignore";
    const PARITY_TAG: &str = "parity-gap:";
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut search_from = 0;
    while let Some(rel) = src[search_from..].find(MARKER) {
        let attr_start = search_from + rel;
        // Locate the closing `]` of this `#[ignore(…)]`/`#[ignore = "…"]`
        // attribute. The reason string can span multiple physical lines
        // via Rust's `"…\ …"` continuation, so we scan for the `]`
        // delimiter rather than relying on a newline.
        let Some(attr_end_rel) = src[attr_start..].find(']') else {
            break;
        };
        let attr_end = attr_start + attr_end_rel + 1;
        let attr = &src[attr_start..attr_end];
        search_from = attr_end;

        let Some(tag_pos) = attr.find(PARITY_TAG) else {
            continue;
        };
        // Reason is the substring between `parity-gap:` and the closing
        // quote of the ignore string, with whitespace and Rust string
        // continuations (`\<newline><spaces>`) collapsed to single
        // spaces. The collapsed form is purely informational — the
        // test asserts on function-name presence, not reason text —
        // but a clean reason makes the panic message readable.
        let after_tag = &attr[tag_pos + PARITY_TAG.len()..];
        let Some(end_quote_rel) = after_tag.rfind('"') else {
            continue;
        };
        let reason_raw = &after_tag[..end_quote_rel];
        let reason: String = reason_raw.split_whitespace().collect::<Vec<_>>().join(" ");

        // Find the next `fn <ident>(` after the attribute. Skip over
        // any further attributes (`#[test]`, doc comments) that sit
        // between the `#[ignore]` and the function.
        let tail = &bytes[attr_end..];
        let Some(fn_name) = find_following_fn_name(tail) else {
            continue;
        };
        out.push((fn_name, reason));
    }
    out
}

/// Find the next `fn <ident>(` after the start of `tail`, returning
/// `<ident>`. Skips whitespace, line/block comments, and intervening
/// attributes — `#[test]`/`#[should_panic]` may legally sit between
/// `#[ignore]` and the function header.
fn find_following_fn_name(tail: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(tail).ok()?;
    // The pattern is permissive: any chunk of the form `fn <ident>(`
    // anywhere in the next ~512 chars is the target. Constraining the
    // search window protects against accidentally pairing an
    // `#[ignore]` with a `fn` from the *next* function block if the
    // intermediate `fn` somehow disappears.
    let window = &s[..s.len().min(512)];
    let mut idx = 0;
    while idx < window.len() {
        let rest = &window[idx..];
        if let Some(stripped) = rest.strip_prefix("fn ") {
            // Read the identifier up to the next `(` or whitespace.
            let end = stripped
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(stripped.len());
            if end == 0 {
                return None;
            }
            return Some(stripped[..end].to_string());
        }
        // Advance by one character (UTF-8 safe via `char_indices`).
        idx += rest.chars().next()?.len_utf8();
    }
    None
}
