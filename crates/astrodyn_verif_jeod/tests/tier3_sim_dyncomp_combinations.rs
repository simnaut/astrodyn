// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: Analytical physics-combination tests inspired by SIM_dyncomp RUNs.
//!
//! The numbered SIM_dyncomp RUN_* scenarios already have Docker-backed
//! cross-validation tests (tier3_sim_dyncomp_run2..run10). This file adds
//! analytical verification tests for physics combinations and laws that are
//! exercised by those RUNs but which admit closed-form / conservation-law
//! verification without a JEOD reference CSV.
//!
//! The `Simulation` construction lives in the `sim_dyncomp_combinations`
//! recipe module so the parity wrapper
//! (`bevy_parity_dyncomp_combinations.rs`) can drive the same scenarios
//! through the Bevy adapter for the `runner ↔ bevy` half of the
//! transitivity argument. Each test reads recipe-encoded values back
//! off the built `Simulation` (rather than duplicating literals) so
//! recipe edits stay locally consistent with their analytical
//! assertions.
//!
//! JEOD scenario mapping:
//! - tier3_dyncomp_point_mass_3dof_conservation:
//!   RUN_2 family (point-mass gravity): energy + angular momentum
//!   conservation for Keplerian orbits.
//! - tier3_dyncomp_point_mass_plus_thirdbody_conservation:
//!   RUN_7A/7B analog (third-body torque effect from Sun/Moon): third bodies
//!   do not conserve orbital angular momentum, but total energy stays bounded
//!   over short spans. Uses point-mass Earth; full SH adds secular drift but
//!   is not required to exhibit third-body torques.
//! - tier3_dyncomp_drag_point_mass_monotonic_decay:
//!   RUN_6/RUN_7C/7D analog (drag with gravity): semi-major axis must trend
//!   monotonically downward under drag. Uses point-mass Earth; SH adds a
//!   secular J2 SMA correction but is not required for monotonic decay.
//! - tier3_dyncomp_6dof_rigid_body_invariance:
//!   RUN_8A (torque-free rotation in orbit): inertial angular momentum is
//!   conserved, body-frame omega varies (Euler's equations).
//! - tier3_dyncomp_external_force_impulse_response:
//!   RUN_9C/9D (external force): delta-v = F * dt / m during force window.
//! - tier3_dyncomp_external_torque_impulse_response:
//!   RUN_9A/9B/9C (external torque): delta-omega = tau * dt / I on axis.
//! - tier3_dyncomp_attitude_stability_major_axis:
//!   Related to RUN_8B (LVLH rate) attitude propagation: spin about the
//!   major principal axis is stable (intermediate-axis theorem).

use astrodyn::recipes::helpers::energy_conservation::specific_orbital_energy;
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_dyncomp_combinations;
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};

/// Earth gravitational parameter (m³/s²) — JEOD `earth_GGM05C.cc`.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Build the recipe's `Simulation` by calling the scenario factory with
/// a default `InitialConditions`, then `.build()`.
///
/// `VerificationCaseParityExt::run_and_assert_parity` derives its init
/// via `initial_conditions_from(&ref_states[0])` rather than
/// `Default::default()`. Every recipe in this file pairs with
/// `CsvReference::SyntheticTimes`, for which the loader fills each
/// generated `StateLog` with `time: t, ..Default::default()`; at
/// `i = 0` that gives `time = 0.0` and `None`/`DVec3::ZERO` for every
/// other field, so `initial_conditions_from(ref_states[0])` collapses
/// to `InitialConditions::default()` bit-for-bit. Passing
/// `Default::default()` here is therefore equivalent to what the
/// parity wrapper does *for these cases*, which is why the runner-side
/// propagation here and the Bevy-side propagation in
/// `bevy_parity_dyncomp_combinations.rs` see the same initial state.
/// If a future recipe in this file switches off `SyntheticTimes` or
/// starts honoring `InitialConditions`, switch this call site to
/// derive the init the same way `run_and_assert_parity` does.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_dyncomp_combinations` uses this
/// variant because the family is analytical-only; panicking on any
/// other variant surfaces a future recipe-shape drift here rather than
/// producing a silently-truncated propagation.
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

// ─── Test 1: Point-mass Kepler conservation ───

/// 3-DOF point-mass orbit (RUN_2 configuration without reference data):
/// specific orbital energy and angular momentum are conserved by Kepler
/// dynamics. Any drift is numerical integrator error.
#[test]
fn tier3_dyncomp_point_mass_3dof_conservation() {
    let case = sim_dyncomp_combinations::point_mass_3dof_conservation();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    let body0 = sim.body(0);
    let pos0 = body0.trans.position.raw_si();
    let vel0 = body0.trans.velocity.raw_si();
    let e0 = specific_orbital_energy(pos0, vel0, MU_EARTH);
    let h0 = pos0.cross(vel0);

    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let e1 = specific_orbital_energy(
        body.trans.position.raw_si(),
        body.trans.velocity.raw_si(),
        MU_EARTH,
    );
    let h1 = body
        .trans
        .position
        .raw_si()
        .cross(body.trans.velocity.raw_si());

    let de = (e1 - e0).abs() / e0.abs();
    let dh = (h1 - h0).length() / h0.length();

    println!("  3 orbits Kepler: relative dE={de:.3e}, relative dH={dh:.3e}");

    // RK4 at dt=10s: observed well below these bounds.
    assert!(de < 1.0e-7, "Kepler energy drift {de:.3e} too large");
    assert!(
        dh < 1.0e-8,
        "Kepler angular momentum drift {dh:.3e} too large"
    );
}

// ─── Test 2: Point-mass Earth + third-body produces non-conservation of h ───

/// Analytical analog of RUN_7A/7B: Sun + Moon third-body gravity exerts
/// torques about Earth, so orbital angular momentum about Earth is NOT
/// strictly conserved. Its direction drifts (nodal regression, inclination
/// wobble), yet total orbital energy stays bounded over short spans.
///
/// This test intentionally uses point-mass Earth gravity only. Full
/// spherical-harmonic Earth gravity would add secular drift, but it is not
/// required to demonstrate the third-body torque effect checked here.
#[test]
fn tier3_dyncomp_point_mass_plus_thirdbody_conservation() {
    let case = sim_dyncomp_combinations::point_mass_plus_thirdbody_conservation();
    let mut sim = build_sim(&case);
    let (_dt, n_steps) = synthetic_cadence(&case);

    let body0 = sim.body(0);
    let pos0 = body0.trans.position.raw_si();
    let vel0 = body0.trans.velocity.raw_si();
    let e0 = specific_orbital_energy(pos0, vel0, MU_EARTH);
    let h0 = pos0.cross(vel0);

    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let e1 = specific_orbital_energy(
        body.trans.position.raw_si(),
        body.trans.velocity.raw_si(),
        MU_EARTH,
    );
    let h1 = body
        .trans
        .position
        .raw_si()
        .cross(body.trans.velocity.raw_si());

    // Orbital energy about Earth should remain bounded (~third-body magnitude
    // times one orbit).  Angular momentum *direction* should shift measurably.
    let relative_de = (e1 - e0).abs() / e0.abs();
    let dh_angle = {
        let cos_th = h0.dot(h1) / (h0.length() * h1.length());
        cos_th.clamp(-1.0, 1.0).acos()
    };

    println!("  1 orbit SH+3body: relative dE={relative_de:.3e}, dH_angle={dh_angle:.3e} rad");

    // Energy change from third bodies over one LEO orbit is tiny but not zero.
    assert!(
        relative_de < 1.0e-5,
        "Third-body relative energy drift {relative_de:.3e} too large"
    );

    // Verify the third-body torques had a measurable effect: the angular
    // momentum vector must have moved more than pure-Kepler numerical noise.
    assert!(
        dh_angle > 1.0e-10,
        "Third-body torque should tilt orbital plane; dH_angle={dh_angle:.3e} is pure noise"
    );
}

// ─── Test 3: drag leads to monotonic decay of SMA ───

/// Analytical analog of RUN_6/RUN_7C/7D (gravity + drag): in a point-mass
/// Earth + drag LEO orbit, the semi-major axis must trend monotonically
/// downward. We sample at orbital-period intervals (filtering out the
/// in-orbit oscillation of instantaneous position) and verify strict
/// monotonic decrease.
///
/// This test intentionally uses point-mass Earth gravity only. Full
/// spherical-harmonic Earth gravity would add secular J2 effects but is
/// not required to demonstrate monotonic SMA decay under drag.
#[test]
fn tier3_dyncomp_drag_point_mass_monotonic_decay() {
    let case = sim_dyncomp_combinations::drag_point_mass_monotonic_decay();
    let mut sim = build_sim(&case);
    let (_dt, n_total_steps) = synthetic_cadence(&case);

    // The recipe's `num_steps` covers five orbital periods. Recover
    // the per-orbit tick count by dividing — both sides are integer-
    // truncated identically, so this matches the recipe's
    // `steps_per_orbit(dt)` helper without re-importing it.
    let steps_per_orbit = n_total_steps / 5;
    assert!(
        steps_per_orbit > 0 && steps_per_orbit * 5 == n_total_steps,
        "`{}`: SyntheticTimes count {n_total_steps} not a clean 5×steps_per_orbit",
        case.name,
    );

    let mut sma_samples = Vec::new();
    for _ in 0..5 {
        sim.step_n(steps_per_orbit).expect("step_n failed");
        let body = sim.body(0);
        let e = specific_orbital_energy(
            body.trans.position.raw_si(),
            body.trans.velocity.raw_si(),
            MU_EARTH,
        );
        let a = -MU_EARTH / (2.0 * e);
        sma_samples.push(a);
    }

    println!("  SMA per orbit: {:?}", sma_samples);

    // Strictly monotonic decrease.
    for window in sma_samples.windows(2) {
        assert!(
            window[1] < window[0],
            "SMA did not decrease monotonically: {} -> {}",
            window[0],
            window[1]
        );
    }

    // Total decay should be non-trivial (> 10 m over 5 orbits at this density).
    let total_decay = sma_samples[0] - sma_samples[sma_samples.len() - 1];
    assert!(
        total_decay > 10.0,
        "Total SMA decay {total_decay:.2} m is implausibly small"
    );
}

// ─── Test 4: torque-free rigid-body rotation conserves inertial H ───

/// RUN_8A physics (spherical Earth gravity + no torque + asymmetric inertia):
/// For a rigid body with no applied torque, Euler's equations give the
/// body-frame omega a time-varying path, but the *inertial-frame* angular
/// momentum vector is rigorously conserved.
#[test]
fn tier3_dyncomp_6dof_rigid_body_invariance() {
    let case = sim_dyncomp_combinations::rigid_body_invariance_6dof();
    let mut sim = build_sim(&case);
    let (_dt, n_steps) = synthetic_cadence(&case);

    // Recover the recipe's inertia + initial omega from the built body
    // so the closed-form initial angular momentum tracks whatever the
    // recipe sets at t=0. Duplicating the recipe literals here would
    // let the assertion silently drift if the recipe edits the inertia
    // without the test noticing. `Simulation::body_mass` returns the
    // typed `MassPropertiesTyped<SelfRef>`; demote through the kernel-
    // boundary helper because `VehicleOutput` doesn't carry mass.
    let mass0_typed = sim
        .body_mass(0)
        .expect("6-DOF body must have mass properties");
    let inertia = astrodyn::typed_bridge::mass_typed_to_raw(mass0_typed).inertia;
    let body0 = sim.body(0);
    let rot0 = body0
        .rot
        .as_ref()
        .expect("6-DOF body must have rotational state");
    let omega0_body = rot0.ang_vel_body.raw_si();
    let q0 = rot0.q_inertial_body.to_jeod_quat();
    let t0 = q0.left_quat_to_transformation(); // inertial→body
    let h_body0 = inertia * omega0_body;
    let h_inertial_0 = t0.transpose() * h_body0;

    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let rot = body
        .rot
        .as_ref()
        .expect("6-DOF body must have rotational state");
    let q1 = rot.q_inertial_body.to_jeod_quat();
    let omega1_body = rot.ang_vel_body.raw_si();
    let t1 = q1.left_quat_to_transformation();
    let h_body1 = inertia * omega1_body;
    let h_inertial_1 = t1.transpose() * h_body1;

    let dh_rel = (h_inertial_1 - h_inertial_0).length() / h_inertial_0.length();
    let mag_rel = (h_body1.length() - h_body0.length()).abs() / h_body0.length();

    println!(
        "  60s torque-free: |H_inertial| conservation {dh_rel:.3e}, |H_body| mag error {mag_rel:.3e}"
    );

    assert!(
        dh_rel < 1.0e-6,
        "RootInertial H not conserved: relative error {dh_rel:.3e}"
    );
    assert!(
        mag_rel < 1.0e-6,
        "|H| magnitude not conserved: relative error {mag_rel:.3e}"
    );

    // Sanity: with asymmetric inertia and tipped omega, body-frame omega DOES
    // change (otherwise the test would be trivial).
    let domega = (omega1_body - omega0_body).length();
    assert!(
        domega > 1.0e-4,
        "body omega did not evolve; test trivially passes: |domega|={domega:.3e}"
    );
}

// ─── Test 5: external force delta-v ───

/// RUN_9C physics (external force application): during a constant force window,
/// the body's velocity increment equals F*dt/m along the force direction.
#[test]
fn tier3_dyncomp_external_force_impulse_response() {
    use sim_dyncomp_combinations::{
        EXTERNAL_FORCE_IMPULSE_DURATION_S, EXTERNAL_FORCE_IMPULSE_INERTIAL_N,
        EXTERNAL_FORCE_IMPULSE_MASS_KG,
    };
    let case = sim_dyncomp_combinations::external_force_impulse_response();
    let case_ref = sim_dyncomp_combinations::external_force_impulse_kepler_reference();
    let mut sim = build_sim(&case);
    let mut ref_sim = build_sim(&case_ref);
    let (dt, n_steps) = synthetic_cadence(&case);
    let (dt_ref, n_steps_ref) = synthetic_cadence(&case_ref);
    assert_eq!(
        (dt, n_steps),
        (dt_ref, n_steps_ref),
        "force-impulse and reference cadences must agree: \
         forced=({dt}, {n_steps}), reference=({dt_ref}, {n_steps_ref})"
    );
    // Cross-check that the SyntheticTimes horizon exactly covers the
    // exposed force-window duration. Catches silent drift if a future
    // recipe edit changes one but not the other.
    let computed_duration = (n_steps as f64) * dt;
    assert!(
        (computed_duration - EXTERNAL_FORCE_IMPULSE_DURATION_S).abs() < 1e-12,
        "force-impulse SyntheticTimes horizon ({computed_duration} s) drifted from \
         exposed EXTERNAL_FORCE_IMPULSE_DURATION_S ({EXTERNAL_FORCE_IMPULSE_DURATION_S} s)"
    );

    let v_before = sim.body(0).trans.velocity.raw_si();

    // Propagate both sims through the force window.
    sim.step_n(n_steps).expect("step_n failed");
    ref_sim.step_n(n_steps_ref).expect("step_n failed");

    let v_after = sim.body(0).trans.velocity.raw_si();
    let v_reference = ref_sim.body(0).trans.velocity.raw_si();

    let force_delta_v = v_after - v_reference;
    let expected_dv = EXTERNAL_FORCE_IMPULSE_INERTIAL_N * EXTERNAL_FORCE_IMPULSE_DURATION_S
        / EXTERNAL_FORCE_IMPULSE_MASS_KG;

    let err = (force_delta_v - expected_dv).length();
    let rel_err = err / expected_dv.length();

    println!(
        "  Impulse: measured dv={:?}, expected={:?}, rel_err={rel_err:.3e}",
        force_delta_v, expected_dv
    );

    // Generous tolerance: RK4 integration of combined force+gravity introduces
    // second-order cross-terms, but the first-order F*t/m should dominate.
    assert!(
        rel_err < 1.0e-4,
        "External-force delta-v error {rel_err:.3e} too large"
    );

    // Delta-v direction should match force direction.
    let cos_align = force_delta_v
        .normalize()
        .dot(EXTERNAL_FORCE_IMPULSE_INERTIAL_N.normalize());
    assert!(
        cos_align > 0.9999,
        "delta-v direction {force_delta_v:?} not aligned with force {EXTERNAL_FORCE_IMPULSE_INERTIAL_N:?}: cos={cos_align}"
    );

    // Sanity that the forced sim did pick up the force in the first place.
    let total_delta_v = v_after - v_before;
    assert!(
        (total_delta_v - force_delta_v).length() > 0.0,
        "forced sim must include gravity contribution; total dv = {total_delta_v:?}"
    );
}

// ─── Test 6: external torque delta-omega ───

/// RUN_9A physics (external torque application): during a constant torque
/// window, the body-frame angular velocity increment about a principal axis
/// equals tau*dt/I.
#[test]
fn tier3_dyncomp_external_torque_impulse_response() {
    use sim_dyncomp_combinations::{
        EXTERNAL_TORQUE_IMPULSE_BODY_NM, EXTERNAL_TORQUE_IMPULSE_DURATION_S,
        EXTERNAL_TORQUE_IMPULSE_INERTIA_X_KGM2,
    };
    let case = sim_dyncomp_combinations::external_torque_impulse_response();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);

    // Cross-check the SyntheticTimes horizon against the exposed
    // torque-window duration. Catches silent drift if a future recipe
    // edit changes one but not the other.
    let computed_duration = (n_steps as f64) * dt;
    assert!(
        (computed_duration - EXTERNAL_TORQUE_IMPULSE_DURATION_S).abs() < 1e-12,
        "torque-impulse SyntheticTimes horizon ({computed_duration} s) drifted from \
         exposed EXTERNAL_TORQUE_IMPULSE_DURATION_S ({EXTERNAL_TORQUE_IMPULSE_DURATION_S} s)"
    );

    sim.step_n(n_steps).expect("step_n failed");

    let omega_after = sim.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();
    // Expected: omega_x = tau_x * dt / I_xx (y, z remain ~zero).
    let expected_omega_x = EXTERNAL_TORQUE_IMPULSE_BODY_NM.x * EXTERNAL_TORQUE_IMPULSE_DURATION_S
        / EXTERNAL_TORQUE_IMPULSE_INERTIA_X_KGM2;

    let err_x = (omega_after.x - expected_omega_x).abs();
    let rel_err = err_x / expected_omega_x.abs();
    println!(
        "  Torque impulse: omega={:?}, expected omega_x={expected_omega_x:.6}, rel_err={rel_err:.3e}",
        omega_after
    );

    assert!(
        rel_err < 1.0e-6,
        "Torque delta-omega error {rel_err:.3e} too large"
    );
    assert!(
        omega_after.y.abs() < 1.0e-8,
        "omega_y should remain zero, got {}",
        omega_after.y
    );
    assert!(
        omega_after.z.abs() < 1.0e-8,
        "omega_z should remain zero, got {}",
        omega_after.z
    );
}

// ─── Test 7: major-axis spin stability (intermediate-axis theorem) ───

/// Related to RUN_8B (rotational propagation): the intermediate-axis theorem
/// (Dzhanibekov effect) predicts stable rotation about the axis of maximum
/// inertia and unstable rotation about the intermediate axis.
///
/// Here we verify the *stable* case: a body spinning about its major
/// (largest-inertia) axis with small perpendicular perturbations must keep
/// nearly all its angular momentum along that axis.
#[test]
fn tier3_dyncomp_attitude_stability_major_axis() {
    let case = sim_dyncomp_combinations::attitude_stability_major_axis();
    let mut sim = build_sim(&case);
    let (_dt, n_steps) = synthetic_cadence(&case);

    // Propagate one step at a time and track the maximum |omega_perp|.
    let mut max_perp = 0.0_f64;
    for _ in 0..n_steps {
        sim.step_n(1).expect("step_n failed");
        let omega = sim.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();
        let perp = (omega.x.powi(2) + omega.y.powi(2)).sqrt();
        if perp > max_perp {
            max_perp = perp;
        }
    }

    let omega_final = sim.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();
    println!(
        "  Major-axis spin: omega_final={:?}, max |omega_perp|={max_perp:.6} rad/s",
        omega_final
    );

    // The perpendicular components of omega oscillate but stay bounded near
    // the initial 0.01 rad/s perturbation.  For stable major-axis rotation the
    // bound should stay well below the spin rate (1 rad/s).
    assert!(
        max_perp < 0.05,
        "Major-axis spin unstable: |omega_perp| grew to {max_perp:.3}"
    );
    // Spin about z must remain dominant.
    assert!(
        omega_final.z.abs() > 0.99,
        "Z-axis spin should be preserved, got {:.4}",
        omega_final.z
    );
}
