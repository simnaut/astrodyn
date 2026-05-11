//! Tier 3: Extended solar-beta tests (analytical).
//!
//! These tests build controlled orbit/Sun geometries and propagate through
//! `Simulation::step()` with a fake Sun source at a chosen position. The
//! reported solar-beta angle is checked against the closed-form value
//! β = asin(ĥ · ŝ).
//!
//! * `tier3_solar_beta_equatorial_at_equinox` — equatorial orbit with Sun in
//!   the orbital (equatorial) plane → β ≈ 0.
//! * `tier3_solar_beta_polar_orbit` — polar orbit. Over one year, β can swing
//!   across the full [-90°, +90°] range. We verify three snapshot geometries.
//! * `tier3_solar_beta_iss_orbit` — ISS inclination (51.6°); for Sun on the
//!   +Z axis β = π/2 − inclination, for Sun in the orbital plane β = 0.
//! * `tier3_solar_beta_sun_in_orbital_plane` — Sun direction in the orbital
//!   plane regardless of orbit inclination → β = 0.
//! * `tier3_solar_beta_sun_perpendicular_to_plane` — Sun along the orbit
//!   normal → |β| = 90°.
//! * `tier3_solar_beta_bounded` — for every propagated checkpoint of a mid-
//!   inclination orbit with a fixed Sun position, |β| ≤ 90°.
//!
//! No Docker reference data required. The `Simulation` construction lives
//! in the `sim_solar_beta_extended` recipe module so the parity wrapper
//! (`bevy_parity_solar_beta_extended.rs`) can drive the same scenarios
//! through the Bevy adapter for the `runner ↔ bevy` half of the
//! transitivity argument.

use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_solar_beta_extended;
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};

/// Build the recipe's `Simulation` exactly the way the parity trait does
/// — call the scenario factory with a default `InitialConditions`, then
/// `.build()` — so the runner-side propagation here and the Bevy-side
/// propagation in `bevy_parity_solar_beta_extended.rs` see the same
/// initial state bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Read the body's `solar_beta` after the most recent propagation step,
/// panicking with the case name if it's not populated. Shared by every
/// test in this file so the per-case bodies stay focused on the
/// closed-form assertion.
fn read_beta(sim: &Simulation, case_name: &str) -> f64 {
    sim.body(0)
        .solar_beta
        .unwrap_or_else(|| panic!("`{case_name}`: solar_beta not computed"))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_solar_beta_extended` uses this
/// variant because the family is analytical-only; panicking on any
/// other variant surfaces a future recipe-shape drift here rather than
/// producing a silently-truncated propagation. Returning both halves
/// of the cadence lets callers assert that the `dt` they're stepping
/// at (typically `sim.dt`) matches the cadence the recipe declared —
/// catches a future edit that updates the builder dt but forgets the
/// `SyntheticTimes` dt (or vice versa).
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

#[test]
fn tier3_solar_beta_equatorial_at_equinox() {
    // Equatorial circular orbit → orbit normal = +Z.
    // Sun in the equatorial (x–y) plane → ŝ ⊥ ĥ → β = 0.
    let case = sim_solar_beta_extended::equatorial_at_equinox();
    let mut sim = build_sim(&case);

    // Drive the scan from the recipe's `SyntheticTimes` cadence — the
    // same `(dt, num_steps)` the parity wrapper uses on the Bevy side,
    // so this loop and the bit-identity assertion step in lockstep.
    // Cross-check `dt` against the built `Simulation`'s integrator dt
    // to catch a future recipe edit that updates one half of the
    // cadence but not the other.
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    let mut max_beta = 0.0_f64;
    for step in 1..=n_steps {
        sim.step_until(step as f64 * dt).expect("step_until failed");
        max_beta = max_beta.max(read_beta(&sim, case.name).abs());
    }

    // The Sun-direction deviates slightly from +X as the body orbits (since
    // sun_direction = sun_position - body_position). For r/SUN_DISTANCE ~ 4.5e-5
    // the worst-case projection onto ĥ = +Z is zero (coplanar), so β is
    // limited by floating-point noise and the small offset: well under 1e-4 rad.
    assert!(
        max_beta < 1e-4,
        "equatorial+in-plane sun: max |beta| = {max_beta:.3e} rad exceeds 1e-4"
    );
}

#[test]
fn tier3_solar_beta_polar_orbit() {
    // Polar orbit about Earth — orbit plane contains +Z, so orbit normal is
    // perpendicular to +Z. Over a full year the Sun sweeps the ecliptic; we
    // verify three snapshot geometries by instantiating distinct simulations.
    //
    //   (1) Sun along +X (equinox-like):   ĥ·ŝ = 0 → β = 0
    //   (2) Sun along +Y:                  β = ±90° if orbit normal = ±Y
    //   (3) Sun along +Z:                  Sun is in the orbital plane → β = 0
    let cases = [
        (sim_solar_beta_extended::polar_sun_x(), 0.0_f64),
        (sim_solar_beta_extended::polar_sun_y(), 90.0_f64),
        (sim_solar_beta_extended::polar_sun_z(), 0.0_f64),
    ];

    for (case, expected_deg) in cases {
        let mut sim = build_sim(&case);
        sim.step_until(sim.dt).expect("step_until failed"); // derived state is populated after first step
        let beta = read_beta(&sim, case.name);
        let expected = expected_deg.to_radians();
        assert!(
            (beta.abs() - expected).abs() < 1e-4,
            "`{}`: |beta| = {} rad, expected {}",
            case.name,
            beta.abs(),
            expected
        );
    }
}

#[test]
fn tier3_solar_beta_iss_orbit() {
    // ISS-like circular orbit at 51.6° inclination. With
    //   r = (r, 0, 0), v = (0, v cos i, v sin i)
    //   h = r × v = (0, -r v sin i, r v cos i)
    //   ĥ = (0, -sin i, cos i)
    //
    // For Sun along +Z:    ĥ·ẑ = cos i  →  β = asin(cos i) = π/2 − i
    // For Sun along +X:    ĥ·x̂ = 0      →  β = 0
    // For Sun along −Y:    ĥ·(−ŷ) = sin i →  β = asin(sin i) = i
    //
    // We verify all three cases.
    let inc = 51.6_f64.to_radians();

    // Case 1: Sun in the equatorial plane (+X). β ≈ 0.
    {
        let case = sim_solar_beta_extended::iss_sun_x();
        let mut sim = build_sim(&case);
        sim.step_until(sim.dt).expect("step_until failed");
        let beta = read_beta(&sim, case.name);
        assert!(
            beta.abs() < 1e-4,
            "ISS orbit + Sun in equatorial plane: |beta| = {beta} rad (expected ≈0)"
        );
    }

    // Case 2: Sun along +Z → β = π/2 − i.
    {
        let case = sim_solar_beta_extended::iss_sun_z();
        let mut sim = build_sim(&case);
        sim.step_until(sim.dt).expect("step_until failed");
        let beta = read_beta(&sim, case.name);
        let expected = std::f64::consts::FRAC_PI_2 - inc;
        assert!(
            (beta - expected).abs() < 1e-4,
            "ISS orbit + Sun along +Z: beta = {beta} rad (expected {expected} = π/2 − i)"
        );
    }

    // Case 3: Sun along −Y → β = i. This exercises the bound that β can
    // reach the full inclination angle at a favorable Sun direction.
    {
        let case = sim_solar_beta_extended::iss_sun_neg_y();
        let mut sim = build_sim(&case);
        sim.step_until(sim.dt).expect("step_until failed");
        let beta = read_beta(&sim, case.name);
        assert!(
            (beta - inc).abs() < 1e-4,
            "ISS orbit + Sun along −Y: beta = {beta} rad (expected {inc} = inclination)"
        );
    }
}

#[test]
fn tier3_solar_beta_sun_in_orbital_plane() {
    // Sun direction exactly in the orbital plane → ĥ · ŝ = 0 → β = 0.
    // Check this for a 30° inclination orbit where the Sun sits exactly at
    // the ascending node direction (in the equatorial plane the node is +X).
    let case = sim_solar_beta_extended::sun_in_orbital_plane();
    let mut sim = build_sim(&case);
    sim.step_until(sim.dt).expect("step_until failed");
    let beta = read_beta(&sim, case.name);
    assert!(
        beta.abs() < 1e-4,
        "Sun in orbital plane: |beta| = {beta} rad exceeds 1e-4"
    );
}

#[test]
fn tier3_solar_beta_sun_perpendicular_to_plane() {
    // Sun along the orbit normal → ĥ · ŝ = ±1 → |β| = π/2.
    // Construct an equatorial orbit (normal = +Z) and place Sun along +Z.
    let case = sim_solar_beta_extended::sun_perpendicular_to_plane();
    let mut sim = build_sim(&case);
    sim.step_until(sim.dt).expect("step_until failed");
    let beta = read_beta(&sim, case.name);

    let pi_2 = std::f64::consts::FRAC_PI_2;
    // Tolerance accounts for the small LEO offset from the origin vs the Sun
    // at 1 AU: the body-Sun direction is slightly off +Z by r/AU ≈ 4.7e-5 rad.
    assert!(
        (beta.abs() - pi_2).abs() < 1e-4,
        "Sun perpendicular to orbit plane: |beta| = {} rad (expected π/2 = {})",
        beta.abs(),
        pi_2
    );
}

#[test]
fn tier3_solar_beta_bounded() {
    // For any orbit and any Sun position, |β| ≤ π/2. We propagate an
    // inclined orbit for several periods and verify this bound at every
    // integration step. (The tighter bound mentioned in the spec —
    // π/2 + inclination — does not apply to β itself, which is always a
    // signed angle in [-π/2, +π/2] by construction; our asserted bound is
    // the mathematical limit |β| ≤ π/2.)
    let case = sim_solar_beta_extended::bounded();
    let mut sim = build_sim(&case);

    // Same recipe-cadence + drift-check pattern as the
    // equatorial-at-equinox scan above.
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    let pi_2 = std::f64::consts::FRAC_PI_2;
    let mut max_abs_beta = 0.0_f64;
    for step in 1..=n_steps {
        sim.step_until(step as f64 * dt).expect("step_until failed");
        let beta = read_beta(&sim, case.name);
        // Absolute bound from the asin definition.
        assert!(
            beta.abs() <= pi_2 + 1e-12,
            "step {step}: |beta| = {} exceeds π/2",
            beta.abs()
        );
        max_abs_beta = max_abs_beta.max(beta.abs());
    }

    // Confirm the β swing is nontrivial for this geometry so the test
    // actually exercises the bound rather than asserting near zero.
    assert!(
        max_abs_beta > 0.05,
        "max |beta| = {max_abs_beta} rad — geometry too degenerate to test bound"
    );
}
