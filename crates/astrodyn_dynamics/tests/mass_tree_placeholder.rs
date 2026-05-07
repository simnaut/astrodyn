//! Mass-tree composite-mass cross-check against JEOD's
//! `models/dynamics/dyn_body/verif/SIM_verif_attach_detach/` reference CSVs.
//!
//! This is a unit-level test of [`MassTree::attach`] / [`MassTree::detach`] /
//! [`MassTree::attach_with_reroot`] against three placeholder vehicles
//! (1 kg / 2 kg / 3 kg, named `veh{1,2,3}`) — the exact mass tree JEOD's
//! verification SIM exercises. The signal validated is the single quantity
//! that is fully determined by the mass-tree topology:
//! `dyn_body.mass.composite_properties.mass`. JEOD logs it directly, so
//! the JEOD CSV is convenient ground truth for the kernel — no
//! `Simulation::step()` is involved.
//!
//! The full-pipeline cross-validation of the same SIM (translational and
//! rotational propagation through `Simulation::step()`, attach/detach
//! routed through the production [`Simulation::attach`] /
//! [`Simulation::detach`] API, momentum conservation via
//! `combine_states_at_attach`) lives in
//! `crates/astrodyn_runner/tests/tier3_sim_attach_detach_trajectory.rs`. That
//! file is the Tier 3 contract; this one is its mass-tree algebra
//! companion.
//!
//! ## Runs validated
//!
//! - **RUN_simple_attach_detach**: veh1→veh2 at t=10s, detached at t=20s.
//!   After t=20 the run does frame-only operations (no mass changes), so
//!   composite masses stay at their base values.
//!
//! - **RUN_complex_attach_detach**: veh1→veh2 at t=10, veh1→veh3 at
//!   t=32.777 (chained: re-roots veh2 under veh3 because veh1 is
//!   already attached to veh2 — JEOD's
//!   `dyn_body_attach.cc::attach_child` 521→567 path; ported as
//!   [`MassTree::attach_with_reroot`]), veh1↔veh2 detach at t=50,
//!   veh1→veh2 re-attach at t=55. Composite masses across the
//!   topology timeline are validated at every CSV row through the
//!   end of the run.
//!
//! - **RUN_compute_child_derivative**: the chained attaches at t=1
//!   (veh1→veh2) and t=2 (veh1→veh3) fire the same re-rooting code
//!   path. After the second attach all three vehicles share v3's
//!   composite. veh1 ↔ veh3 detach at t=15 then re-detach at t=45
//!   (the input.py issues two `veh1.detach_from_3` events; the
//!   second is a no-op in JEOD because veh1 is no longer rooted
//!   under veh3 — JEOD's `BodyDetachSpecific::apply` warns and
//!   returns false). The composite-mass timeline is validated end to
//!   end here too.
//!
//! [`MassTree::attach_with_reroot`]: astrodyn_dynamics::MassTree::attach_with_reroot
//! [`Simulation::attach`]: https://docs.rs/astrodyn_runner

use astrodyn_dynamics::{MassProperties, MassTree};

fn test_data_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../astrodyn_verif_jeod/test_data")
}

/// One CSV row: time plus the three composite masses.
#[derive(Debug, Clone, Copy)]
struct MassRow {
    time: f64,
    veh1: f64,
    veh2: f64,
    veh3: f64,
}

/// Load the `attach_detach_ASCII` CSV. The Trick DRAscii logger writes:
///
/// ```text
/// time,veh1.dyn_body.mass.composite_properties.mass,veh2...,veh3...
/// ```
fn load_csv(filename: &str) -> Vec<MassRow> {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "JEOD reference data not found at {}.\n\
         Generate with:\n\
         docker run --rm \\\n\
           -v $(pwd)/test_data:/output \\\n\
           -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \\\n\
           jeod-trick",
        path.display()
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let mut rows = Vec::new();
    // Parse each data row strictly: exactly 4 columns, each a valid f64.
    // `filter_map(... .ok())` would silently drop malformed columns or rows
    // and hide a corrupted reference CSV, so we assert the shape instead.
    for (idx, line) in content.lines().skip(1).enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').map(str::trim).collect();
        assert_eq!(
            fields.len(),
            4,
            "CSV {} line {}: expected 4 columns, found {}: {:?}",
            path.display(),
            idx + 2,
            fields.len(),
            trimmed
        );
        let parse = |col: usize, name: &str| -> f64 {
            fields[col].parse().unwrap_or_else(|e| {
                panic!(
                    "CSV {} line {}: invalid {name} value {:?}: {e}",
                    path.display(),
                    idx + 2,
                    fields[col]
                )
            })
        };
        rows.push(MassRow {
            time: parse(0, "time"),
            veh1: parse(1, "veh1"),
            veh2: parse(2, "veh2"),
            veh3: parse(3, "veh3"),
        });
    }
    assert!(
        !rows.is_empty(),
        "CSV {} contained no data rows",
        path.display()
    );
    rows
}

/// Build a fresh mass tree with three disconnected vehicles matching the
/// Modified_data/veh{1,2,3}.py initial masses from JEOD.
fn build_three_vehicles() -> (MassTree, usize, usize, usize) {
    let mut tree = MassTree::new();
    let v1 = tree.add_root("veh1".into(), MassProperties::new(1.0));
    let v2 = tree.add_root("veh2".into(), MassProperties::new(2.0));
    let v3 = tree.add_root("veh3".into(), MassProperties::new(3.0));
    (tree, v1, v2, v3)
}

/// Tolerance on composite mass (kg). JEOD's DRAscii logger writes f64 with
/// default formatting (~17 sig figures), so the floor is machine epsilon.
/// Our composite_mass is computed by summation of exact f64 inputs, so the
/// error should be < 1e-12 kg.
const MASS_TOL: f64 = 1e-12;

/// Compare our computed composite masses to the JEOD row, assert, and return
/// the per-row max absolute delta across the three vehicles.
fn assert_masses(row: &MassRow, v1: f64, v2: f64, v3: f64) -> f64 {
    let d1 = (row.veh1 - v1).abs();
    let d2 = (row.veh2 - v2).abs();
    let d3 = (row.veh3 - v3).abs();
    assert!(
        d1 < MASS_TOL,
        "t={:.3}s veh1: ours={v1}, JEOD={}, diff={d1:.2e}",
        row.time,
        row.veh1
    );
    assert!(
        d2 < MASS_TOL,
        "t={:.3}s veh2: ours={v2}, JEOD={}, diff={d2:.2e}",
        row.time,
        row.veh2
    );
    assert!(
        d3 < MASS_TOL,
        "t={:.3}s veh3: ours={v3}, JEOD={}, diff={d3:.2e}",
        row.time,
        row.veh3
    );
    d1.max(d2).max(d3)
}

// ════════════════════════════════════════════════════════════════════
// RUN_simple_attach_detach
// ════════════════════════════════════════════════════════════════════

/// JEOD event times from `SET_test/RUN_simple_attach_detach/input.py`:
/// - t=10.0: `veh1.attach_to_2.active = True`   → veh1 (root) attached to veh2.
/// - t=20.0: `veh1.detach_from_2.active = True` → veh1 detached from veh2.
/// - t=30, 35, 40, 50: frame-only operations (no mass change).
const SIMPLE_ATTACH_TIME: f64 = 10.0;
const SIMPLE_DETACH_TIME: f64 = 20.0;

// Mass-tree algebra check: the kernel's composite-mass output must match
// JEOD's logged composite at every row of the verification SIM. The full
// trajectory (with `Simulation::step()`-driven translation, rotation, and
// momentum-conserving attach/detach) is exercised by the Tier 3 sibling at
// `crates/astrodyn_runner/tests/tier3_sim_attach_detach_trajectory.rs`.
#[test]
fn mass_tree_simple_attach_detach() {
    let rows = load_csv("attach_detach_simple_attach_detach.csv");

    // Sanity: initial state at t=0 must match baseline masses.
    let t0 = &rows[0];
    assert_masses(t0, 1.0, 2.0, 3.0);

    // Build tree and step through the recorded timeline, applying
    // attach/detach at their scheduled times. JEOD fires each action exactly
    // once (trick.add_read scheduled events), so we use one-shot flags — a
    // simple `!attached` gate would re-fire attach after detach and mask any
    // asymmetry if `attach`/`detach` ever stopped being perfect inverses.
    let (mut tree, v1, v2, v3) = build_three_vehicles();
    let mut attach_fired = false;
    let mut detach_fired = false;

    // JEOD's DRAscii samples at 0.5s starting from t=0. Events fire at
    // the beginning of the second (t=10.000 or t=20.000 exactly).
    // In the CSV the *same-second* row should reflect the event's effect.
    for row in &rows {
        // Apply events that are due by this row's timestamp. Tight
        // inequality keeps the event firing on the row at which JEOD's
        // trick.add_read(t, ...) executes.
        if !attach_fired && row.time >= SIMPLE_ATTACH_TIME {
            // BodyAttachAligned: veh1.node12 ↔ veh2.node21. node12 lives at
            // (10, 0, 0) in veh1 struct; node21 at (0, 0, 0) in veh2 struct.
            // With node21's orientation being YPR(180°,0,0) (180° yaw), the
            // attach_aligned transform is well-defined and yields a specific
            // offset/rotation, but the *composite mass* signal we validate
            // is independent of these geometric details — it's the sum of
            // core masses in the subtree. So a direct `attach` is fine here.
            tree.attach(
                v1,
                v2,
                glam::DVec3::ZERO, // mass is pose-invariant
                glam::DMat3::IDENTITY,
            );
            attach_fired = true;
        }
        if !detach_fired && row.time >= SIMPLE_DETACH_TIME {
            tree.detach(v1);
            detach_fired = true;
        }

        let m1 = tree.get(v1).composite_properties.mass;
        let m2 = tree.get(v2).composite_properties.mass;
        // v3 is untouched by this run, but still queried each row so the
        // assertion catches any cross-root contamination from attach/detach.
        let m3 = tree.get(v3).composite_properties.mass;

        // JEOD's `update_mass_properties` runs at the dynamics rate
        // (DYNAMICS=0.01s), so by the log-cycle boundary the event's
        // effect is always visible. We compare directly.
        assert_masses(row, m1, m2, m3);
    }
}

// ════════════════════════════════════════════════════════════════════
// RUN_complex_attach_detach
// ════════════════════════════════════════════════════════════════════

/// JEOD event times from `SET_test/RUN_complex_attach_detach/input.py`:
///   t=10.0:    `veh1.attach_to_2.active = True`
///                — root subject; tree shape `v2{v1}`.
///   t=32.777:  `veh1.attach_to_3.active = True`
///                — chained: re-roots veh2 (veh1's existing root) under
///                  veh3 (`MassTree::attach_with_reroot`). New tree
///                  shape `v3{v2{v1}}`.
///   t=50.0:    `veh1.detach_from_2.active = True`
///                — direct `MassTree::detach(v1)`.
///   t=55.0:    `veh1.attach_to_2b.active = True`
///                — chained: subject (v1) is a fresh tree root, but
///                  parent (v2) is interior to v3's tree, so the
///                  topology adds v1 under v2 inside v3's tree.
///                  No re-rooting (subject is a root); the destination
///                  is the chained one.
///   t=60.0:    `veh1.detach_from_3.active = True`
///                — JEOD's `BodyDetachSpecific::apply` routes through
///                  `dyn_detach_from->remove_mass_body(*mass_subject)`
///                  in `dyn_body_detach.cc:165-234`, which walks up
///                  from `v1.mass.links` to find the immediate child
///                  of `v3.mass` (i.e. v2), notices v2 has a DynBody
///                  owner, and re-routes to `v3.detach(v2_dynbody)`.
///                  Net effect on the mass tree: the v3 ↔ v2 edge is
///                  cut — `MassTree::detach(v2)`.
const COMPLEX_ATTACH_V1_V2_TIME: f64 = 10.0;
const COMPLEX_RECHAIN_V1_V3_TIME: f64 = 32.777;
const COMPLEX_DETACH_V1_FROM_V2_TIME: f64 = 50.0;
const COMPLEX_REATTACH_V1_V2_TIME: f64 = 55.0;
const COMPLEX_DETACH_FROM_V3_TIME: f64 = 60.0;

/// Cross-validate the composite-mass timeline of
/// `RUN_complex_attach_detach` end to end. The signal is:
///
/// | window           | expected (m1, m2, m3)             |
/// |------------------|-----------------------------------|
/// | `[0, 10)`        | (1, 2, 3) — three free roots      |
/// | `[10, 32.777)`   | (1, 3, 3) — v1 attached to v2     |
/// | `[32.777, 50)`   | (1, 3, 6) — v2 re-rooted under v3 |
/// | `[50, 55)`       | (1, 2, 5) — v1 detached from v2   |
/// | `[55, 60)`       | (1, 3, 6) — v1 re-attached to v2  |
/// | `[60, 65]`       | (1, 3, 3) — v3 ↔ v2 edge cut      |
//
// Mass-tree algebra check across the chained-reroot timeline. Re-rooting,
// detach, and re-attach are the kernel paths under test; full-pipeline
// trajectory coverage of this run is deferred to its Tier 3 sibling at
// `crates/astrodyn_runner/tests/tier3_sim_attach_detach_trajectory.rs` (and
// `tier3_sim_complex_attach_detach.rs` for the complex schedule).
#[test]
fn mass_tree_complex_attach_detach() {
    let rows = load_csv("attach_detach_complex_attach_detach.csv");
    let t0 = &rows[0];
    assert_masses(t0, 1.0, 2.0, 3.0);

    let (mut tree, v1, v2, v3) = build_three_vehicles();
    let mut attach_v1_v2_fired = false;
    let mut rechain_v1_v3_fired = false;
    let mut detach_v1_v2_fired = false;
    let mut reattach_v1_v2_fired = false;
    let mut detach_from_v3_fired = false;

    for row in &rows {
        // Apply each scheduled event at the row whose timestamp first
        // satisfies `row.time >= event_time` — JEOD's
        // `trick.add_read(t, ...)` semantics fire each action exactly
        // once at the event time, before the next CSV log cycle.
        if !attach_v1_v2_fired && row.time >= COMPLEX_ATTACH_V1_V2_TIME {
            tree.attach_with_reroot(v1, v2, glam::DVec3::ZERO, glam::DMat3::IDENTITY);
            attach_v1_v2_fired = true;
        }
        if !rechain_v1_v3_fired && row.time >= COMPLEX_RECHAIN_V1_V3_TIME {
            // Subject (v1) is already a child of v2; the kernel re-
            // roots v2 under v3. Geometric offset is irrelevant for
            // the composite-mass signal.
            tree.attach_with_reroot(v1, v3, glam::DVec3::ZERO, glam::DMat3::IDENTITY);
            rechain_v1_v3_fired = true;
        }
        if !detach_v1_v2_fired && row.time >= COMPLEX_DETACH_V1_FROM_V2_TIME {
            tree.detach(v1);
            detach_v1_v2_fired = true;
        }
        if !reattach_v1_v2_fired && row.time >= COMPLEX_REATTACH_V1_V2_TIME {
            tree.attach_with_reroot(v1, v2, glam::DVec3::ZERO, glam::DMat3::IDENTITY);
            reattach_v1_v2_fired = true;
        }
        if !detach_from_v3_fired && row.time >= COMPLEX_DETACH_FROM_V3_TIME {
            // JEOD `remove_mass_body(v1.mass)` walks up from v1 to
            // find the immediate child of v3 (v2 — v1 is two levels
            // under v3 here), notices v2 has a DynBody owner, and
            // routes through `v3.detach(v2_dynbody)`. The mass-tree
            // outcome is `MassTree::detach(v2)`.
            tree.detach(v2);
            detach_from_v3_fired = true;
        }

        let m1 = tree.get(v1).composite_properties.mass;
        let m2 = tree.get(v2).composite_properties.mass;
        let m3 = tree.get(v3).composite_properties.mass;
        assert_masses(row, m1, m2, m3);
    }
}

// ════════════════════════════════════════════════════════════════════
// RUN_compute_child_derivative
// ════════════════════════════════════════════════════════════════════

/// JEOD event times from
/// `SET_test/RUN_compute_child_derivative/input.py`:
///   t=1:  `veh1.attach_to_2.active = True`
///   t=2:  `veh1.attach_to_3.active = True` (chained reroot of v2 under v3)
///   t=15: `veh1.detach_from_3.active = True`
///           — same `remove_mass_body` re-route as the complex run
///             at t=60. Walks v1 → ... → immediate child of v3 (v2),
///             v2 has a DynBody owner, so it routes through
///             `v3.detach(v2_dynbody)`. Mass-tree effect:
///             `MassTree::detach(v2)`.
///   t=45: `veh1.detach_from_3.active = True` (a *second* time)
///           — by t=45 v2 is no longer in v3's tree (the t=15 event
///             cut that edge) and v1's mass-body tree path no longer
///             contains v3 at all. JEOD's `remove_mass_body` walks up
///             from v1 looking for the immediate child of v3, fails,
///             and the action `MessageHandler::inform`s + returns
///             without mutating the tree. Mass-tree effect: no-op.
const CHILD_DERIV_ATTACH_V1_V2_TIME: f64 = 1.0;
const CHILD_DERIV_RECHAIN_V1_V3_TIME: f64 = 2.0;
const CHILD_DERIV_DETACH_FROM_V3_TIME_FIRST: f64 = 15.0;
const CHILD_DERIV_DETACH_FROM_V3_TIME_SECOND: f64 = 45.0;

/// Cross-validate the composite-mass timeline of
/// `RUN_compute_child_derivative` end to end. Same signal shape as
/// the complex run, with a different schedule:
///
/// | window           | expected (m1, m2, m3)             |
/// |------------------|-----------------------------------|
/// | `[0, 1)`         | (1, 2, 3) — three free roots      |
/// | `[1, 2)`         | (1, 3, 3) — v1 attached to v2     |
/// | `[2, 15)`        | (1, 3, 6) — v2 re-rooted under v3 |
/// | `[15, 65]`       | (1, 3, 3) — v3 ↔ v2 edge cut      |
//
// Mass-tree algebra check for the second chained-reroot schedule. Asserts
// the duplicate-detach no-op and the re-route via `remove_mass_body`. The
// pipeline-level trajectory cross-check for SIM_verif_attach_detach lives
// in `crates/astrodyn_runner/tests/tier3_sim_attach_detach_trajectory.rs`.
#[test]
fn mass_tree_child_derivative_attach_detach() {
    let rows = load_csv("attach_detach_child_deriv_attach_detach.csv");
    assert_masses(&rows[0], 1.0, 2.0, 3.0);

    let (mut tree, v1, v2, v3) = build_three_vehicles();
    let mut attach_v1_v2_fired = false;
    let mut rechain_v1_v3_fired = false;
    let mut detach_first_fired = false;
    let mut detach_second_fired = false;

    for row in &rows {
        if !attach_v1_v2_fired && row.time >= CHILD_DERIV_ATTACH_V1_V2_TIME {
            tree.attach_with_reroot(v1, v2, glam::DVec3::ZERO, glam::DMat3::IDENTITY);
            attach_v1_v2_fired = true;
        }
        if !rechain_v1_v3_fired && row.time >= CHILD_DERIV_RECHAIN_V1_V3_TIME {
            tree.attach_with_reroot(v1, v3, glam::DVec3::ZERO, glam::DMat3::IDENTITY);
            rechain_v1_v3_fired = true;
        }
        if !detach_first_fired && row.time >= CHILD_DERIV_DETACH_FROM_V3_TIME_FIRST {
            tree.detach(v2);
            detach_first_fired = true;
        }
        if !detach_second_fired && row.time >= CHILD_DERIV_DETACH_FROM_V3_TIME_SECOND {
            // JEOD remove_mass_body no-op when the requested
            // attachment doesn't exist any more — see comment on
            // `CHILD_DERIV_DETACH_FROM_V3_TIME_SECOND`.
            //
            // We deliberately *don't* call `tree.detach(v2)` here
            // (panics when v2 has no parent). The flag flip is
            // sufficient to assert the absence of a mass-tree
            // mutation: composite masses must remain (1, 3, 3) post-
            // event, which the row-by-row assertion below verifies.
            detach_second_fired = true;
        }

        let m1 = tree.get(v1).composite_properties.mass;
        let m2 = tree.get(v2).composite_properties.mass;
        let m3 = tree.get(v3).composite_properties.mass;
        assert_masses(row, m1, m2, m3);
    }
}
