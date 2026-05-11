//! Tier 3: NESC GN&C Lunar Check Case 8 (NRHO) — 7-day NRHO propagation
//! against the published NESC reference trajectory (translation + attitude).
//!
//! Source: <https://nescacademy.nasa.gov/flightsim/2023/cc08>.
//!
//! Force model (per case spec, see `crate::cc8` for the full IC table):
//!
//! - Moon: GRAIL 8×8 spherical harmonics + DE440 libration.
//! - Earth: point-mass third body (DE-driven position each step).
//! - Sun: point-mass third body (DE-driven position each step).
//! - No SRP, no drag, no gravity-gradient torque.
//!
//! Attitude propagates open-loop (no torques) — NESC publishes the IC for
//! cross-validation of the rotational integrator path.
//!
//! ## Reference
//!
//! Today's reference is **sim_01** of the eight NESC participating
//! propagators (six of eight agree on the IC at t=0 to ≥ 9 decimal
//! places; sim_03 and sim_06 are anomalies). See
//! `crates/astrodyn_verif_nesc/README.md` for the methodology proposal
//! that replaces sim_01 with a consensus-of-six median + spread
//! reference once landed.
//!
//! ## Tolerances
//!
//! Set per the CLAUDE.md "observed-max × 1.05" rule against sim_01,
//! captured from `target/tier3_crossval/tier3_nesc_cc8_nrho.json`.
//! These are sim_01-specific baselines, not physics-truth tolerances —
//! the consensus methodology will widen them to the inter-sim spread
//! envelope.
//!
//! ## Attitude convention note (issue #454, resolved)
//!
//! Initial propagation runs showed a ~π rad quaternion drift vs sim_01
//! over the 7-day, 605° body-frame spin. Root cause was a convention
//! mismatch: NESC-RP-23-01853 §7.4.1 publishes the IC and reference-CSV
//! quaternions as **right-transformative**, but our `JeodQuat` is
//! left-transformative. `cc8.rs` now conjugates the vector part at both
//! the IC boundary and the CSV parser. Post-fix attitude drift over the
//! full 7-day run is ~0.15 rad (~8.4°), consistent with the
//! inter-propagator numerical noise the consensus methodology will
//! calibrate against in a follow-up.

use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_nesc::{cc8::cc8_builder, cc8::cc8_reference, CrossvalReport, StateLog};

#[test]
fn tier3_nesc_cc8_nrho() {
    // Build → step → cross-check at every CC8 checkpoint (60 s cadence).
    let mut sim = cc8_builder().build().expect("CC8 builder");
    let cc8_ref = cc8_reference();
    assert!(
        !cc8_ref.is_empty(),
        "CC8 reference trajectory is empty; regenerate via \
         `cargo run -p astrodyn_verif_nesc --bin extract_nesc`."
    );

    let mut ours: Vec<StateLog> = Vec::with_capacity(cc8_ref.len());
    for chk in &cc8_ref {
        sim.step_until(chk.time).expect("step_until");
        let body = sim.body(0);
        let rot_raw = body
            .rot
            .as_ref()
            .map(astrodyn::typed_bridge::rot_typed_to_raw);
        ours.push(StateLog {
            time: chk.time,
            position: Some(body.trans.position.raw_si()),
            velocity: Some(body.trans.velocity.raw_si()),
            quaternion: rot_raw.as_ref().map(|r| r.quaternion.to_glam()),
            ang_vel: rot_raw.as_ref().map(|r| r.ang_vel_body),
            ..Default::default()
        });
    }

    let report = CrossvalReport::compute("tier3_nesc_cc8_nrho", &ours, &cc8_ref);
    report.write();

    // Tolerances are observed-max × 1.05 against sim_01 (CLAUDE.md rule).
    // Captured 2026-05-10 from target/tier3_crossval/tier3_nesc_cc8_nrho.json.
    // The consensus methodology in the README will widen these to the
    // inter-sim spread envelope once landed.
    report.assert_position([6.1e1, 1.24e2, 9.5e1]); // m, ~150 m magnitude over 7 days
    report.assert_velocity([1.17e-2, 2.78e-2, 3.31e-2]); // m/s
    report.assert_ang_vel([2.6e-8, 2.4e-6, 3.5e-7]); // rad/s
                                                     // Attitude max-error angle over the full 7-day run vs sim_01,
                                                     // observed-max × 1.05 (0.1465 × 1.05 ≈ 0.154). See the module-level
                                                     // "Attitude convention note" for the underlying fix.
    report.assert_quat_angle(0.154); // rad
}
