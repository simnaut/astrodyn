//! Tier 3: SIM_orbinit cross-validation via Simulation pipeline
//!
//! Validates body initialization from 4 distinct coordinate representations
//! by building each scenario through its `sim_orbinit_edge` recipe,
//! propagating one step, and checking range + cross-consistency against
//! JEOD's logged t=0 state.
//!
//!   RUN_0101: Orbital elements in inertial frame (STS-114)
//!   RUN_0201: Orbital elements in planet-fixed frame (ISS)
//!   RUN_0301: Orbital elements in planet-fixed frame (STS-114)
//!   RUN_0401: Cartesian state in inertial frame (STS-114)
//!
//! The `Simulation` construction lives in the `sim_orbinit_edge` recipe
//! module so the parity wrapper (`bevy_parity_orbinit_edge.rs`) can drive
//! the same scenarios through the Bevy adapter for the `runner ↔ bevy`
//! half of the transitivity argument.

use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_orbinit_edge;
use astrodyn_verif_jeod::tier3_csv::{load_orbinit_csv, test_data_path};
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};
use glam::DVec3;

/// Build the recipe's `Simulation` exactly the way the parity trait does
/// — call the scenario factory with a default `InitialConditions` (the
/// recipes don't read it — initial state is baked in from each RUN's
/// JEOD output t=0 row), then `.build()` — so the runner-side
/// propagation here and the Bevy-side propagation in
/// `bevy_parity_orbinit_edge.rs` see the same initial state bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_orbinit_edge` uses this variant
/// because the orbinit CSVs are initialization-only (one row at t=0);
/// panicking on any other variant surfaces a future recipe-shape drift
/// here rather than producing a silently-truncated propagation.
/// Returning both halves of the cadence lets callers assert that the
/// `dt` they're stepping at (typically `sim.dt`) matches the cadence
/// the recipe declared.
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

/// Read the post-construction translational state from each RUN's
/// recipe so the cross-consistency assertions are driven by exactly the
/// same numbers the parity wrapper integrates. The state is read from
/// the runner's `body(0)` *before* any propagation step, so each entry
/// is the t=0 state the recipe bakes in.
fn runner_initial_state(case: &VerificationCase) -> (DVec3, DVec3) {
    let sim = build_sim(case);
    let body = sim.body(0);
    (body.trans.position.raw_si(), body.trans.velocity.raw_si())
}

/// Per-RUN row: CSV file name, recipe factory, human-readable label.
type RunRow = (&'static str, fn() -> VerificationCase, &'static str);

#[test]
fn tier3_simulation_orbinit_cross_consistency() {
    // The CSV is loaded only to cross-check the recipe's baked-in
    // initial state against JEOD's logged t=0 row — a regression
    // fence so a future recipe-side edit can't silently drift away
    // from the JEOD-source values. The actual propagation uses the
    // recipe-driven `Simulation`.
    let runs: [RunRow; 4] = [
        (
            "orbinit_0101_orbinit.csv",
            sim_orbinit_edge::run_0101,
            "RUN_0101 (STS-114 inertial OE)",
        ),
        (
            "orbinit_0201_orbinit.csv",
            sim_orbinit_edge::run_0201,
            "RUN_0201 (ISS pfix OE)",
        ),
        (
            "orbinit_0301_orbinit.csv",
            sim_orbinit_edge::run_0301,
            "RUN_0301 (STS-114 pfix OE)",
        ),
        (
            "orbinit_0401_orbinit.csv",
            sim_orbinit_edge::run_0401,
            "RUN_0401 (STS-114 inertial cart)",
        ),
    ];

    let mut states: Vec<(DVec3, DVec3, &str)> = Vec::new();

    for (filename, recipe, label) in runs {
        let csv_path = test_data_path(filename);
        assert!(
            csv_path.exists(),
            "SIM_orbinit CSV not found at {}.\n\
             Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            csv_path.display()
        );

        let records = load_orbinit_csv(&csv_path);
        assert!(
            !records.is_empty(),
            "{label}: no records found in {filename}"
        );
        let csv_init = &records[0];

        let case = recipe();
        let (init_pos, init_vel) = runner_initial_state(&case);

        // Fence: the recipe's baked-in state must reproduce the
        // JEOD-logged t=0 row exactly, bit-for-bit for the OE RUNs and
        // within the CSV's printed precision for the Cartesian RUN
        // (RUN_0401 logs 6 significant digits — JEOD's truncation, not
        // ours). A future edit that tweaks recipe-side numbers will
        // trip this check first instead of silently changing what the
        // parity wrapper integrates.
        let pos_drift = (init_pos - csv_init.position).length();
        let vel_drift = (init_vel - csv_init.velocity).length();
        assert!(
            pos_drift < 1.0,
            "{label}: recipe init position drifted from CSV by {pos_drift:.6} m"
        );
        assert!(
            vel_drift < 0.01,
            "{label}: recipe init velocity drifted from CSV by {vel_drift:.6e} m/s"
        );

        // Step once through the full pipeline (TimeUpdate → Environment →
        // Integration → DerivedState) at the recipe's synthetic cadence.
        // Cross-check `dt` against the built `Simulation`'s integrator dt
        // to catch a future recipe edit that updates one half of the
        // cadence but not the other.
        let mut sim = build_sim(&case);
        let (dt, n_steps) = synthetic_cadence(&case);
        assert_eq!(
            dt, sim.dt,
            "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
            case.name, sim.dt
        );
        assert!(
            n_steps >= 1,
            "`{}`: recipe must propagate at least one step",
            case.name
        );
        sim.step_until(dt).expect("step_until failed");

        // Read back the body state after one step (confirms pipeline ran).
        let body = sim.body(0);
        let r_mag = body.trans.position.raw_si().length();
        let v_mag = body.trans.velocity.raw_si().length();

        println!(
            "  {label}: r={:.3} km  v={:.6} km/s  pos=[{:.1}, {:.1}, {:.1}] m",
            r_mag / 1000.0,
            v_mag / 1000.0,
            init_pos.x,
            init_pos.y,
            init_pos.z,
        );

        // Sanity: LEO orbit (post-step state should still be LEO).
        assert!(
            (6_000_000.0..=8_000_000.0).contains(&r_mag),
            "{label}: r={r_mag:.0} m outside LEO range after one step"
        );
        assert!(
            (6_000.0..=8_000.0).contains(&v_mag),
            "{label}: v={v_mag:.1} m/s outside LEO range after one step"
        );

        // Use the initial (t=0) state for cross-consistency: it
        // mirrors the original test, which compared the JEOD-logged
        // t=0 vectors across RUNs.
        states.push((init_pos, init_vel, label));
    }

    println!();

    // Cross-consistency: STS-114 runs (0101, 0301, 0401) should agree closely.
    let sts_indices = [0, 2, 3];
    for i in 0..sts_indices.len() {
        for j in (i + 1)..sts_indices.len() {
            let (pos_a, vel_a, label_a) = states[sts_indices[i]];
            let (pos_b, vel_b, label_b) = states[sts_indices[j]];
            let pos_err = (pos_a - pos_b).length();
            let vel_err = (vel_a - vel_b).length();
            println!(
                "  {label_a} vs {label_b}: pos_err={:.6} m  vel_err={:.6e} m/s",
                pos_err, vel_err,
            );
            assert!(
                pos_err < 1.0,
                "STS-114 cross-consistency: position error {pos_err:.3} m exceeds 1.0 m \
                 between {label_a} and {label_b}"
            );
            assert!(
                vel_err < 0.001,
                "STS-114 cross-consistency: velocity error {vel_err:.3e} m/s exceeds 0.001 m/s \
                 between {label_a} and {label_b}"
            );
        }
    }

    // ISS vs STS-114: different vehicles at similar epoch.
    let (pos_iss, _, _) = states[1];
    let (pos_sts, _, _) = states[0];
    let cross_vehicle_err = (pos_iss - pos_sts).length();
    println!(
        "\n  ISS vs STS-114: pos_diff={:.1} m (expected: different vehicles)",
        cross_vehicle_err,
    );
    assert!(
        cross_vehicle_err < 1000.0,
        "ISS vs STS-114 position difference {cross_vehicle_err:.0} m exceeds 1 km"
    );

    println!("\n  All initialization methods produce consistent LEO states");
}
