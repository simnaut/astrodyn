//! Tier 3: SIM_verif_attach_detach — dyn-body mass-tree composite mass.
//!
//! Cross-validates the composite mass of each of three vehicles (`veh1`,
//! `veh2`, `veh3`) over time against JEOD's
//! `models/dynamics/dyn_body/verif/SIM_verif_attach_detach/` simulation.
//!
//! This is the **mass-tree slice** of a larger dynamics test. JEOD's run
//! exercises:
//! - `BodyAttachAligned` / `BodyDetach` (mass-tree attach/detach)
//! - `DynBody::attach_to_frame` (reference-frame attach — not mass-tree)
//! - Translational + rotational propagation
//!
//! Our port currently has only the mass-tree portion (`MassTree::attach` /
//! `detach`). We therefore validate the single signal that is 100%
//! determined by the mass tree: `dyn_body.mass.composite_properties.mass`.
//!
//! ## Runs validated
//!
//! - **RUN_simple_attach_detach**: veh1→veh2 at t=10s, detached at t=20s.
//!   After t=20 the run does frame-only operations (no mass changes), so
//!   composite masses stay at their base values.
//!
//! - **RUN_complex_attach_detach** and **RUN_compute_child_derivative** are
//!   deliberately **not** validated end-to-end here — they exercise chained
//!   attachments (`veh1 → veh2 → veh3`) whose root-propagation semantics
//!   (`MassBody::attach_to` automatically re-roots the attaching body's
//!   tree) are not yet implemented in our port. The CSVs are still generated
//!   for future use and sanity-checked at t=0.

use jeod_dynamics::{MassProperties, MassTree};

fn test_data_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
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

#[test]
fn tier3_sim_attach_detach_simple() {
    let rows = load_csv("attach_detach_simple_attach_detach.csv");

    // Sanity: initial state at t=0 must match baseline masses.
    let t0 = &rows[0];
    let mut max_err = assert_masses(t0, 1.0, 2.0, 3.0);

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
        max_err = max_err.max(assert_masses(row, m1, m2, m3));
    }

    let mut report = jeod_test_data::crossval::CrossvalReport::compute(
        "tier3_sim_attach_detach_simple",
        &[],
        &[],
    );
    report.add_extra("composite_mass_max_err", max_err, "kg");
    report.write();
}

// ════════════════════════════════════════════════════════════════════
// Sanity-only checks for complex / child_derivative runs
// ════════════════════════════════════════════════════════════════════

/// Verify the CSV for the complex run is present and has correct t=0 state.
/// Full mass-tree validation is deferred — `attach_to` auto-reroots the
/// attaching body's root, which our `MassTree::attach` does not yet model.
#[test]
fn tier3_sim_attach_detach_complex_t0() {
    let rows = load_csv("attach_detach_complex_attach_detach.csv");
    assert_masses(&rows[0], 1.0, 2.0, 3.0);
}

/// Same sanity check for the compute_child_derivative run.
#[test]
fn tier3_sim_attach_detach_child_derivative_t0() {
    let rows = load_csv("attach_detach_child_deriv_attach_detach.csv");
    assert_masses(&rows[0], 1.0, 2.0, 3.0);
}
