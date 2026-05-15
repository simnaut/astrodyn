//! Tier 3: Analytical verification of SIM_force_torque physics.

#![allow(
    clippy::float_cmp,
    reason = "Tier 3 tests assert bit-exact recovery of literal-built / analytic state values"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Tier 3 step counts and indices fit exactly in f64 mantissa and usize"
)]
//!
//! JEOD's `models/dynamics/dyn_body/verif/SIM_force_torque/` is an
//! empty-space test rig that verifies force and torque accumulation,
//! non-transmitted forces, and the equations of motion by scheduling
//! time-stamped force/torque changes and checking that the body
//! traces out a predictable trajectory.
//!
//! The scenario factories live in
//! [`astrodyn_verif_jeod::run_verification::sim_force_torque_response`];
//! each tier3 function below materializes one of those recipes
//! through [`VerificationCaseExt::run_and_assert`] semantics by
//! re-using the runner-side builder + propagation path, then reads
//! body 0's final state and asserts the closed-form analytical
//! identity that recipe targets. The matching `bevy_parity_*`
//! wrapper drives the same recipes through the Bevy adapter and
//! asserts `runner ↔ bevy` bit-identity at every synthetic record.
//!
//! JEOD scenario mapping (`SIM_force_torque/SET_test/RUN_test/input.py`):
//! - The scheduled `trick.add_read` blocks successively apply a z-axis
//!   force that yields 1 m/s², 2 m/s², 3 m/s² on the composite body,
//!   then play with non-transmitted forces, yaw/roll maneuvers, etc.
//! - The tests below pick out the four primary physics verifications
//!   in that sim: uniform F/m translation, τ/I rotation, independence
//!   of the two when the force is applied at the center of mass, and
//!   the symmetric ±F impulse that returns the body to rest.

use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_force_torque_response;
use astrodyn_verif_jeod::verification::{InitialConditions, VerificationCase};

use astrodyn::SimulationBuilder;
use astrodyn_runner::builder::SimulationBuilderExt;
use glam::DVec3;

/// Materialize one of the recipes into a runtime `Simulation` and
/// propagate it through every synthetic record (driving any `pre_step`
/// closure in the process). Returns the post-propagation simulation so
/// the test can read body 0's final state for its closed-form
/// assertion. Mirrors the runner-side shape of
/// [`astrodyn_verif_jeod::run_verification::VerificationCaseExt::run_and_assert`],
/// minus the JEOD CSV comparison (none of these recipes pair with a
/// real reference) and minus the tolerance asserts (each test asserts
/// directly on the closed-form identity below).
fn run_recipe(case: &VerificationCase) -> Simulation {
    // Empty-space scenarios derive their initial conditions from the
    // recipe constants, not from a JEOD CSV. Pass a default
    // `InitialConditions` so the factory signature is satisfied
    // without re-parsing anything; the recipe ignores the parameter.
    let init = InitialConditions::default();
    let sb: SimulationBuilder = (case.scenario)(&init);
    let dt = sb.dt;
    let mut sim = sb
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` failed validation: {e:?}", case.name));
    let mut pre_step = case.pre_step.map(|(builder, _cadence)| builder(&init));

    let (sync_dt, num_steps) = match case.reference {
        astrodyn_verif_jeod::verification::CsvReference::SyntheticTimes { dt, num_steps } => {
            (dt, num_steps)
        }
        ref other => panic!(
            "tier3_sim_force_torque_response: recipe `{}` reference must be \
             SyntheticTimes; got {other:?}",
            case.name
        ),
    };
    assert_eq!(
        sync_dt, dt,
        "recipe `{}`: SyntheticTimes.dt ({sync_dt}) must match \
         SimulationBuilder.dt ({dt}) so the closed-form analytical \
         constants line up with the integration cadence",
        case.name,
    );

    for step in 1..=num_steps {
        let t = (step as f64) * dt;
        if let Some(hook) = pre_step.as_mut() {
            hook(&mut sim, t);
        }
        sim.step()
            .unwrap_or_else(|e| panic!("`{}`: step at t={t} failed: {e}", case.name));
    }
    sim
}

// ── Test 1: F = m·a on a free body ───────────────────────────────

/// Constant force on a free body produces uniform acceleration
/// `a = F/m`. After applying F for T seconds starting from rest:
///   v(T) = F·T/m
///   x(T) = 0.5 · F·T² / m
#[test]
fn tier3_force_constant_acceleration() {
    let mass = 100.0;
    let t_total = 10.0;
    let force = DVec3::new(3.0, -2.0, 1.0);

    let sim = run_recipe(&sim_force_torque_response::force_constant_acceleration());
    let body = sim.body(0);

    let expected_v = force * t_total / mass;
    let expected_x = 0.5 * force * t_total * t_total / mass;

    let err_v = (body.trans.velocity.raw_si() - expected_v).length();
    let err_x = (body.trans.position.raw_si() - expected_x).length();

    let rel_v = err_v / expected_v.length();
    let rel_x = err_x / expected_x.length();

    println!(
        "  F=m*a: v={:?}, expected={:?}, rel_err={rel_v:.3e}",
        body.trans.velocity.raw_si(),
        expected_v
    );
    println!(
        "         x={:?}, expected={:?}, rel_err={rel_x:.3e}",
        body.trans.position.raw_si(),
        expected_x
    );

    // RK4 is exact for linear ODEs up to roundoff; tolerances are tight.
    assert!(rel_v < 1.0e-12, "velocity rel err {rel_v:.3e}");
    assert!(rel_x < 1.0e-12, "position rel err {rel_x:.3e}");
}

// ── Test 2: τ = I·α on a symmetric body (constant torque) ─────────

/// Constant torque about a principal axis of a body with diagonal
/// inertia:
///   omega(T) = τ·T / I (on that axis; zero on others)
///   theta(T) = 0.5 · τ·T² / I
/// For an initially aligned body, after 10 s with `τ_x = 1 N·m` and
/// `I_x = 10`:
///   omega_x = 1 rad/s, theta_x = 5 rad.
#[test]
fn tier3_torque_constant_angular_acceleration() {
    let i_x = 10.0;
    let t_total = 10.0;
    let tau = DVec3::new(1.0, 0.0, 0.0);

    let sim = run_recipe(&sim_force_torque_response::torque_constant_angular_acceleration());
    let body = sim.body(0);
    let omega = body.rot.as_ref().unwrap().ang_vel_body.raw_si();

    let expected_omega_x = tau.x * t_total / i_x;

    let rel_err = (omega.x - expected_omega_x).abs() / expected_omega_x.abs();
    println!(
        "  tau=I*alpha: omega={omega:?}, expected_x={expected_omega_x}, rel_err={rel_err:.3e}"
    );

    assert!(rel_err < 1.0e-10, "omega_x rel err {rel_err:.3e}");

    // Perpendicular components stay zero (diagonal inertia, torque on
    // principal axis, no initial rate).
    assert!(
        omega.y.abs() < 1.0e-12,
        "omega_y should be zero, got {}",
        omega.y
    );
    assert!(
        omega.z.abs() < 1.0e-12,
        "omega_z should be zero, got {}",
        omega.z
    );
}

// ── Test 3: force at CoM decouples translation and rotation ────────

/// When a force is applied through the center of mass and an
/// independent torque is applied on the body, translation and
/// rotation are decoupled:
///   linear motion follows v = F·t/m regardless of torque
///   angular motion follows omega = τ·t/I regardless of force
#[test]
fn tier3_force_and_torque_decoupled() {
    let mass = 100.0;
    let i_z = 10.0;
    let t_total = 5.0;
    let force = DVec3::new(2.0, 0.0, 0.0);
    let tau = DVec3::new(0.0, 0.0, 1.0);

    // Case A: only force.
    let sim_a = run_recipe(&sim_force_torque_response::force_and_torque_decoupled_force());
    let v_a = sim_a.body(0).trans.velocity.raw_si();
    let omega_a = sim_a.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();

    // Case B: only torque.
    let sim_b = run_recipe(&sim_force_torque_response::force_and_torque_decoupled_torque());
    let v_b = sim_b.body(0).trans.velocity.raw_si();
    let omega_b = sim_b.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();

    // Case C: both.
    let sim_c = run_recipe(&sim_force_torque_response::force_and_torque_decoupled_both());
    let v_c = sim_c.body(0).trans.velocity.raw_si();
    let omega_c = sim_c.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();

    println!("  A (force only): v={:?}, omega={:?}", v_a, omega_a);
    println!("  B (torque only): v={:?}, omega={:?}", v_b, omega_b);
    println!("  C (both): v={:?}, omega={:?}", v_c, omega_c);

    // Decoupling: case C's translation matches A, rotation matches B.
    let dv = (v_c - v_a).length();
    let domega = (omega_c - omega_b).length();

    assert!(
        dv < 1.0e-12,
        "Case C translation should match Case A: |dv|={dv:.3e}"
    );
    assert!(
        domega < 1.0e-12,
        "Case C rotation should match Case B: |domega|={domega:.3e}"
    );

    // Case B should have zero velocity (torque alone doesn't translate).
    assert!(
        v_b.length() < 1.0e-12,
        "Torque-only should produce no translation: |v|={}",
        v_b.length()
    );
    // Case A should have zero angular velocity (force at CoM doesn't rotate).
    assert!(
        omega_a.length() < 1.0e-12,
        "Force at CoM should produce no rotation: |omega|={}",
        omega_a.length()
    );

    // Absolute magnitudes match F=ma and τ=I·α analytical predictions.
    let expected_v_x = force.x * t_total / mass;
    let expected_omega_z = tau.z * t_total / i_z;
    assert!((v_a.x - expected_v_x).abs() < 1.0e-12);
    assert!((omega_b.z - expected_omega_z).abs() < 1.0e-12);
}

// ── Test 4: symmetric +F/-F impulse pair returns body to rest ──────

/// Mirrors JEOD's SIM_force_torque scheduling: from `t=7` to `t=9`
/// the input.py applies a force that reverses from +F to -F so the
/// net velocity change is zero after the symmetric impulse pair.
///
/// Here: 5 seconds at +F, then 5 seconds at -F. Velocity returns to
/// zero, but position remains positive because the body continues to
/// drift while decelerating; analytically the final displacement is
/// twice the position reached at `t = HALF_DURATION_S`. A symmetric
/// triangle of acceleration is enough to assert the body behaves
/// linearly.
#[test]
fn tier3_force_symmetric_impulse_returns_to_rest() {
    let mass = 100.0;
    let force = DVec3::new(5.0, 0.0, 0.0);
    let half_duration = 5.0;

    let sim = run_recipe(&sim_force_torque_response::force_symmetric_impulse());
    let body = sim.body(0);
    let v_end = body.trans.velocity.raw_si();
    let x_end = body.trans.position.raw_si();

    // Velocity should be back to zero within roundoff.
    println!("  v_end = {v_end:?}");
    println!("  x_end = {x_end:?}");
    assert!(
        v_end.length() < 1.0e-10,
        "Velocity should return to rest: |v|={}",
        v_end.length()
    );

    // Position must be nonzero (body has drifted).  Analytical:
    //   v during [0,5]:  (F/m)·t peaks at v_mid = 0.25 m/s
    //   x at t=5:        0.5 · a · 5² = 0.625 m
    //   v during [5,10]: v_mid - (F/m)·(t-5), zero at t=10
    //   x at t=10:       x(5) + v_mid·5 - 0.5·a·5² = 1.25 m
    let expected_x = 2.0 * 0.5 * force.x / mass * half_duration * half_duration;
    let rel = (x_end.x - expected_x).abs() / expected_x;
    println!(
        "  x_end.x = {}, expected = {}, rel_err = {rel:.3e}",
        x_end.x, expected_x
    );
    assert!(
        rel < 1.0e-12,
        "symmetric impulse position rel err {rel:.3e}"
    );
}
