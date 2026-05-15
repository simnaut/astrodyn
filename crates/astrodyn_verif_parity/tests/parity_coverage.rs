//! Coverage CI guard for issue #389 — keeps the bevy parity test set a
//! superset of every Tier 3 topic.
//!
//! Walks `crates/astrodyn_verif_jeod/tests/tier3_*.rs`,
//! `crates/astrodyn_verif_nesc/tests/tier3_*.rs`, and
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
//! To document a deliberate gap: add the topic to `KNOWN_PARITY_GAPS`
//! with a `#[ignore = "parity-gap: <reason>"]` on the wrapper test (or
//! omit the wrapper entirely). The intent is that *every* gap is
//! either solved or named explicitly.

use std::collections::BTreeSet;
use std::path::Path;

/// Tier 3 topics whose parity wrapper is *expected to land* once a
/// concrete blocker lifts (recipe factory, multi-planet dispatch, etc.).
/// Each entry should link to the tracking issue (#389 + follow-ups) so
/// the entry can be dropped when the wrapper file is created. Split out
/// from the umbrella `KNOWN_PARITY_GAPS` (#485 L2) so the "this should
/// eventually go away" set is auditable distinct from the structurally-
/// permanent set below.
///
/// **Note** (#485 M4): the gap-count is one entry larger than the count
/// of orphaned parity wrappers because some entries here have *no*
/// corresponding parity wrapper at all (they are pure-deferred); others
/// document a recipe-factory follow-up rather than a missing wrapper.
/// The `is_covered_by_parity` prefix rule lets a single wrapper file
/// satisfy multiple closely-related tier3 topics, which is why the
/// raw file counts do not need to match 1:1.
const DEFERRED_GAPS: &[(&str, &str)] = &[
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
    // ── Pre-recipe tier3 siblings: the `VerificationCase` factory
    //    doesn't exist yet, so the parity trait has nothing to drive.
    //    Recipe migration is tracked as a follow-up to #389. Each
    //    entry can be dropped when the wrapper file is created.
    (
        "dyncomp_run_attach_to_ref_frame",
        "pre-recipe sibling exercising attach_to_frame — recipe factory \
         not yet defined; needs `pre_step` Bevy support too (#389 follow-up)",
    ),
    (
        "drag_ver",
        "pre-recipe sibling — drag-family recipe factory not yet defined \
         (#389 follow-up)",
    ),
    (
        "lsode",
        "pre-recipe sibling for LSODE integrator — recipe factory not yet \
         defined (#389 follow-up); LSODE integrator may need its own \
         per-step state component on the Bevy side",
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

/// Tier 3 topics that are structurally out of scope for the
/// `VerificationCaseParityExt` trait — no JEOD trajectory CSV to compare
/// against, pure analytical/solver test, structural mass-tree composition.
/// No follow-up is planned: the topic exists in tier3 because it
/// exercises owner-crate logic the parity trait was never meant to cover.
const PERMANENT_GAPS: &[(&str, &str)] = &[
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
    (
        "drag_analytical",
        "analytical drag verification — out of trait scope (no propagation)",
    ),
];

/// Union of the two gap arrays. The coverage check unions both into the
/// `allowed` set, then sweeps each separately for stale / redundant
/// entries; deferred entries that have been wrapped (or whose tier3
/// topic was deleted) must be dropped from `DEFERRED_GAPS`, and similarly
/// for `PERMANENT_GAPS`. Splitting the arrays preserves the audit story
/// while keeping the lint behavior identical.
fn known_parity_gaps() -> impl Iterator<Item = &'static (&'static str, &'static str)> {
    DEFERRED_GAPS.iter().chain(PERMANENT_GAPS.iter())
}

#[test]
fn parity_topics_are_a_superset_of_tier3_topics() {
    let workspace_root = workspace_root();
    // Tier 3 tests live in two crates today: astrodyn_verif_jeod (JEOD/Trick
    // cross-validation) and astrodyn_verif_nesc (NESC GN&C check cases).
    // Both directories follow the same `tier3_*.rs` naming and feed into
    // the same parity-coverage assertion.
    let tier3_topics_jeod = collect_topics(
        &workspace_root.join("crates/astrodyn_verif_jeod/tests"),
        "tier3_",
    );
    let tier3_topics_nesc = collect_topics(
        &workspace_root.join("crates/astrodyn_verif_nesc/tests"),
        "tier3_",
    );
    let tier3_topics: BTreeSet<String> = tier3_topics_jeod
        .union(&tier3_topics_nesc)
        .cloned()
        .collect();
    assert!(
        !tier3_topics.is_empty(),
        "no tier3 tests discovered in either crates/astrodyn_verif_jeod/tests/ \
         or crates/astrodyn_verif_nesc/tests/ — coverage test cannot run"
    );

    let parity_topics = collect_topics(
        &workspace_root.join("crates/astrodyn_verif_parity/tests"),
        "bevy_parity_",
    );

    let allowed: BTreeSet<&'static str> = known_parity_gaps().map(|(t, _)| *t).collect();

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
             DEFERRED_GAPS or PERMANENT_GAPS):\n",
        );
        for t in &uncovered {
            msg.push_str(&format!("  - {t}\n"));
        }
        msg.push_str(
            "\nFix by either:\n  \
             1. Adding `crates/astrodyn_verif_parity/tests/bevy_parity_<topic>.rs`, or\n  \
             2. Documenting the gap in `DEFERRED_GAPS` (wrapper expected to land) \
             or `PERMANENT_GAPS` (structurally out of scope) in parity_coverage.rs \
             with a reason.\n",
        );
        panic!("{msg}");
    }

    // Surface stale gap entries so a topic that has since been covered
    // (or removed from tier3) doesn't sit in the exemption list
    // forever. Both arrays are swept independently so the diagnostic
    // names which array carried the stale entry.
    let mut stale_deferred: Vec<&str> = Vec::new();
    for (topic, _reason) in DEFERRED_GAPS {
        if !tier3_topics.contains(*topic) {
            stale_deferred.push(topic);
        }
    }
    assert!(
        stale_deferred.is_empty(),
        "DEFERRED_GAPS contains topics that no longer exist in tier3_*.rs: \
         {stale_deferred:?}\n  \
         Either restore the missing tier3 test or drop the exemption.",
    );
    let mut stale_permanent: Vec<&str> = Vec::new();
    for (topic, _reason) in PERMANENT_GAPS {
        if !tier3_topics.contains(*topic) {
            stale_permanent.push(topic);
        }
    }
    assert!(
        stale_permanent.is_empty(),
        "PERMANENT_GAPS contains topics that no longer exist in tier3_*.rs: \
         {stale_permanent:?}\n  \
         Either restore the missing tier3 test or drop the exemption.",
    );

    // Surface redundant gap entries: a topic listed here that already
    // has a `bevy_parity_*.rs` wrapper should drop the exemption — the
    // wrapper satisfies the superset invariant on its own. Deferred
    // entries that get covered are the natural close-out path; a
    // permanent entry that gets covered indicates the entry was
    // misclassified and the wrapper is real coverage.
    let mut redundant_deferred: Vec<&str> = Vec::new();
    for (topic, _reason) in DEFERRED_GAPS {
        if is_covered_by_parity(topic, &parity_topics) {
            redundant_deferred.push(topic);
        }
    }
    assert!(
        redundant_deferred.is_empty(),
        "DEFERRED_GAPS contains topics that already have a bevy_parity_*.rs wrapper: \
         {redundant_deferred:?}\n  \
         Drop the exemption — the wrapper file covers the topic.",
    );
    let mut redundant_permanent: Vec<&str> = Vec::new();
    for (topic, _reason) in PERMANENT_GAPS {
        if is_covered_by_parity(topic, &parity_topics) {
            redundant_permanent.push(topic);
        }
    }
    assert!(
        redundant_permanent.is_empty(),
        "PERMANENT_GAPS contains topics that already have a bevy_parity_*.rs wrapper: \
         {redundant_permanent:?}\n  \
         Drop the exemption — the topic was misclassified; the wrapper file is real coverage.",
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
