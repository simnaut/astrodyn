// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: 6-DOF drag verification tests.
//!
//! These tests exercise aerodynamic drag with rotational dynamics through
//! `Simulation::step()`, verifying that drag interacts correctly with the
//! attitude state. All tests use analytical verification.
//!
//! No Docker reference data required. The `Simulation` construction lives
//! in the `sim_drag_6dof` recipe module so the parity wrapper
//! (`bevy_parity_drag_6dof.rs`) can drive the same scenarios through the
//! Bevy adapter for the `runner ↔ bevy` half of the transitivity argument.

use astrodyn::recipes::helpers::energy_conservation::specific_orbital_energy;
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_drag_6dof;
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};

/// Earth gravitational parameter (m³/s²) — JEOD `earth_GGM05C.cc`.
/// Matches the value the `sim_drag_6dof` recipe uses so the analytical
/// assertions reconstruct the recipe-encoded initial state exactly.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Earth mean equatorial radius (m) — JEOD `earth.cc`.
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq;

/// Build the recipe's `Simulation` exactly the way the parity trait does
/// — call the scenario factory with a default `InitialConditions`, then
/// `.build()` — so the runner-side propagation here and the Bevy-side
/// propagation in `bevy_parity_drag_6dof.rs` see the same initial state
/// bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_drag_6dof` uses this variant because
/// the family is analytical-only; panicking on any other variant
/// surfaces a future recipe-shape drift here rather than producing a
/// silently-truncated propagation.
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

/// 6-DOF with rotation: drag should still remove orbital energy.
///
/// For the ballistic drag model (constant Cd*A), attitude does not affect
/// the drag magnitude. The force always opposes the relative velocity
/// regardless of body orientation. This test verifies that a spinning body
/// still experiences proper orbital energy dissipation.
#[test]
fn tier3_drag_with_rotation_energy_loss() {
    let case = sim_drag_6dof::drag_with_rotation_energy_loss();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    // Reconstruct the recipe-encoded initial position/velocity locally
    // so the closed-form initial energy matches the recipe's t=0 state
    // exactly (the recipe places the body at (R_ORBIT, 0, 0) with the
    // local circular speed along +Y).
    let r = R_EARTH + 400_000.0;
    let v = (MU_EARTH / r).sqrt();
    let pos = glam::DVec3::new(r, 0.0, 0.0);
    let vel = glam::DVec3::new(0.0, v, 0.0);
    let e_initial = specific_orbital_energy(pos, vel, MU_EARTH);
    let initial_ang_vel_mag = glam::DVec3::new(0.01, -0.005, 0.003).length();

    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let e_final = specific_orbital_energy(
        body.trans.position.raw_si(),
        body.trans.velocity.raw_si(),
        MU_EARTH,
    );

    println!("  Initial energy: {e_initial:.6e} J/kg");
    println!("  Final energy:   {e_final:.6e} J/kg");
    println!("  Energy change:  {:.6e} J/kg", e_final - e_initial);

    // Energy must decrease due to drag
    assert!(
        e_final < e_initial,
        "Drag must remove orbital energy even with rotation: \
         E_final={e_final:.6e} >= E_initial={e_initial:.6e}"
    );

    // Also verify the body is still spinning (angular velocity not damped to zero
    // by ballistic drag, since ballistic drag produces no torque)
    let rot = body
        .rot
        .as_ref()
        .expect("6-DOF body should have rotational state");
    let final_ang_vel_mag = rot.ang_vel_body.raw_si().length();

    println!("  Initial |omega|: {initial_ang_vel_mag:.6e} rad/s");
    println!("  Final   |omega|: {final_ang_vel_mag:.6e} rad/s");

    // Ballistic drag should not produce torque, so angular velocity magnitude
    // should be approximately conserved (point-mass gravity produces no torque
    // either, since the gravity gradient flag is off).
    let ang_vel_change = (final_ang_vel_mag - initial_ang_vel_mag).abs() / initial_ang_vel_mag;
    assert!(
        ang_vel_change < 0.01,
        "Angular velocity magnitude should be conserved (no torque): \
         relative change = {ang_vel_change:.6e}"
    );
}

/// Ballistic drag should produce the same trajectory regardless of attitude.
///
/// Since ballistic drag (constant Cd*A) applies force proportional to
/// relative velocity in the inertial frame, two bodies with different
/// attitudes but same translational state should experience the same
/// translational trajectory. The force direction changes in the body frame,
/// but in the inertial frame it is always anti-velocity.
#[test]
fn tier3_drag_attitude_invariance_ballistic() {
    let case_identity = sim_drag_6dof::drag_attitude_invariance_identity();
    let case_rotated = sim_drag_6dof::drag_attitude_invariance_rotated();

    let mut sim1 = build_sim(&case_identity);
    let mut sim2 = build_sim(&case_rotated);
    let (dt1, n_steps1) = synthetic_cadence(&case_identity);
    let (dt2, n_steps2) = synthetic_cadence(&case_rotated);
    assert_eq!(
        dt1, sim1.dt,
        "`{}`: recipe SyntheticTimes dt ({dt1}) and Simulation dt ({}) drifted apart",
        case_identity.name, sim1.dt
    );
    assert_eq!(
        dt2, sim2.dt,
        "`{}`: recipe SyntheticTimes dt ({dt2}) and Simulation dt ({}) drifted apart",
        case_rotated.name, sim2.dt
    );
    // Both legs must propagate for the same number of integration ticks
    // — the attitude-invariance assertion compares their final
    // translational states, which is only meaningful when the two
    // trajectories were stepped through the same cadence.
    assert_eq!(
        (dt1, n_steps1),
        (dt2, n_steps2),
        "attitude-invariance legs disagree on cadence: \
         identity=({dt1}, {n_steps1}), rotated=({dt2}, {n_steps2})"
    );

    sim1.step_n(n_steps1).expect("step_n failed");
    sim2.step_n(n_steps2).expect("step_n failed");

    let body1 = sim1.body(0);
    let body2 = sim2.body(0);

    let pos_diff = (body1.trans.position.raw_si() - body2.trans.position.raw_si()).length();
    let vel_diff = (body1.trans.velocity.raw_si() - body2.trans.velocity.raw_si()).length();

    println!("  Position difference: {pos_diff:.6e} m");
    println!("  Velocity difference: {vel_diff:.6e} m/s");

    // For ballistic drag, the translational trajectories should be identical
    // (within numerical precision). The body frame force direction differs, but
    // when transformed back to inertial, it is the same.
    assert!(
        pos_diff < 1e-3,
        "Ballistic drag trajectory should be attitude-invariant: pos_diff={pos_diff:.6e} m"
    );
    assert!(
        vel_diff < 1e-6,
        "Ballistic drag trajectory should be attitude-invariant: vel_diff={vel_diff:.6e} m/s"
    );
}
