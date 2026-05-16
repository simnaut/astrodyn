//! Tier 3: Orbit initialization families -- conservation verification

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
//! Exercises `Simulation::step()` end-to-end for diverse orbit families:
//! circular, eccentric, retrograde, equatorial, polar, hyperbolic, and
//! near-parabolic. Since no Docker reference data exists for these cases,
//! verification uses analytical invariants:
//!
//! - Specific orbital energy conservation: E = v²/2 − μ/r
//! - Specific angular momentum conservation: |h| = |r × v|
//! - Radius constancy for circular orbits
//! - Periapsis/apoapsis radius bounds for elliptic orbits
//!
//! The `Simulation` construction lives in the
//! [`sim_orbinit_families`] recipe module so the parity wrapper
//! (`bevy_parity_orbinit_families.rs`) can drive the same scenarios
//! through the Bevy adapter for the `runner ↔ bevy` half of the
//! transitivity argument.

use astrodyn::recipes::helpers::energy_conservation::specific_orbital_energy;
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_orbinit_families;
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};
use glam::DVec3;

/// Earth gravitational parameter (m³/s²) — same const-folded literal
/// the recipe uses, so the expected-vs-recovered conservation bounds
/// resolve against the same bit-pattern that drove the initial state.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Earth equatorial radius (m) — same const-folded literal the recipe
/// uses. Tier3 geometric bounds (radius checks, altitude expectations)
/// resolve against the same bit-pattern that seeded the initial state.
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq();

/// Build the recipe's `Simulation` exactly the way the parity trait
/// does — call the scenario factory with a default `InitialConditions`,
/// then `.build()` — so the runner-side propagation here and the
/// Bevy-side propagation in `bevy_parity_orbinit_families.rs` see the
/// same initial state bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_orbinit_families` uses this variant
/// because the family is analytical-only; panicking on any other
/// variant surfaces a future recipe-shape drift here rather than
/// producing a silently-truncated propagation. Returning both halves
/// of the cadence lets callers assert that the `dt` they're stepping
/// at (`sim.dt`) matches the cadence the recipe declared — catches a
/// future edit that updates the builder dt but forgets the
/// `SyntheticTimes` dt (or vice versa).
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

/// Specific orbital energy delegate that resolves to
/// [`recipes::helpers::energy_conservation::specific_orbital_energy`].
fn specific_energy(pos: DVec3, vel: DVec3, mu: f64) -> f64 {
    specific_orbital_energy(pos, vel, mu)
}

/// Compute specific angular momentum magnitude: |h| = |r × v|.
fn specific_ang_momentum(pos: DVec3, vel: DVec3) -> f64 {
    pos.cross(vel).length()
}

/// Conservation verification results — radius extremes tracked during
/// propagation. Returned by [`verify_conservation`] so callers can
/// assert family-specific radius bounds (circular constancy,
/// periapsis/apoapsis windows) without a second propagation pass.
struct ConservationResult {
    min_r: f64,
    max_r: f64,
}

/// Propagate and verify energy and angular momentum conservation.
///
/// Also tracks min/max radius across all steps so callers can assert
/// radius bounds without a separate propagation pass.
///
/// Energy error is relative when |E₀| is large, but switches to
/// absolute error normalized by μ/r₀ when |E₀| is small (near-parabolic
/// orbits where E₀ ≈ 0 makes relative error ill-conditioned).
fn verify_conservation(
    sim: &mut Simulation,
    n_steps: usize,
    label: &str,
    energy_tol: f64,
    h_tol: f64,
) -> ConservationResult {
    let body0 = sim.body(0);
    let energy_0 = specific_energy(
        body0.trans.position.raw_si(),
        body0.trans.velocity.raw_si(),
        MU_EARTH,
    );
    let h0 = specific_ang_momentum(body0.trans.position.raw_si(), body0.trans.velocity.raw_si());

    // For near-parabolic orbits, |E₀| can be near zero, making relative
    // energy error ill-conditioned (inf/NaN). Use μ/r₀ as a stable scale.
    let r0 = body0.trans.position.raw_si().length();
    let energy_scale = if energy_0.abs() > MU_EARTH / r0 * 1e-6 {
        energy_0.abs() // standard relative error
    } else {
        MU_EARTH / r0 // stable scale for near-parabolic
    };

    let mut max_energy_err = 0.0_f64;
    let mut max_h_err = 0.0_f64;
    let mut min_r = r0;
    let mut max_r = r0;

    for step in 1..=n_steps {
        sim.step().expect("step failed");
        let body = sim.body(0);
        let energy_now = specific_energy(
            body.trans.position.raw_si(),
            body.trans.velocity.raw_si(),
            MU_EARTH,
        );
        let h_now =
            specific_ang_momentum(body.trans.position.raw_si(), body.trans.velocity.raw_si());
        let r_now = body.trans.position.raw_si().length();

        let energy_err = ((energy_now - energy_0) / energy_scale).abs();
        let h_rel = ((h_now - h0) / h0).abs();

        max_energy_err = max_energy_err.max(energy_err);
        max_h_err = max_h_err.max(h_rel);
        min_r = min_r.min(r_now);
        max_r = max_r.max(r_now);

        if step % 100 == 0 || step == n_steps {
            println!(
                "  {label} step {step}/{n_steps}: E_err={energy_err:.3e}, h_rel={h_rel:.3e}, \
                 r={:.1} km, v={:.3} km/s",
                body.trans.position.raw_si().length() / 1000.0,
                body.trans.velocity.raw_si().length() / 1000.0,
            );
        }
    }

    println!(
        "  {label}: max E_err={max_energy_err:.3e} (tol {energy_tol:.1e}), \
         max h_rel={max_h_err:.3e} (tol {h_tol:.1e})"
    );

    assert!(
        max_energy_err < energy_tol,
        "{label}: energy conservation failed: max error {max_energy_err:.6e} \
         exceeds tolerance {energy_tol:.1e}"
    );
    assert!(
        max_h_err < h_tol,
        "{label}: angular momentum conservation failed: max relative error {max_h_err:.6e} \
         exceeds tolerance {h_tol:.1e}"
    );

    ConservationResult { min_r, max_r }
}

/// Drive `sim` for the recipe's full SyntheticTimes cadence and assert
/// energy + angular momentum conservation. Shared by every family
/// test; case-specific geometric invariants (radius bounds, plane
/// confinement, polar passage, …) wrap this with extra checks. The
/// `dt`-vs-`sim.dt` assert catches a future edit that updates the
/// recipe builder dt without updating the matching SyntheticTimes dt.
fn run_and_verify_conservation(
    case: &VerificationCase,
    sim: &mut Simulation,
    label: &str,
    energy_tol: f64,
    h_tol: f64,
) -> ConservationResult {
    let (dt, n_steps) = synthetic_cadence(case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );
    verify_conservation(sim, n_steps, label, energy_tol, h_tol)
}

// ======================================================================
// Circular LEO
// ======================================================================

#[test]
fn tier3_orbinit_circular_leo() {
    let case = sim_orbinit_families::circular_leo();
    let (a, _e, i, _raan, _argp, _nu) = sim_orbinit_families::elements::circular_leo();
    let r = a; // e = 0 → r = a

    let mut sim = build_sim(&case);

    println!(
        "Tier 3: Circular LEO (a={a:.0} m, e=0, i={:.1} deg)",
        i.to_degrees()
    );

    let result = run_and_verify_conservation(&case, &mut sim, "circular_leo", 1e-10, 1e-10);

    // Additional check: radius should stay nearly constant for circular orbit.
    let max_rel_err = ((result.max_r - r).abs().max((r - result.min_r).abs())) / r;
    println!(
        "  Radius: initial={r:.1} m, min={:.1} m, max={:.1} m, max_rel_err={max_rel_err:.3e}",
        result.min_r, result.max_r
    );
    assert!(
        max_rel_err < 1e-8,
        "Circular orbit radius varied during propagation: min={:.6e}, max={:.6e}, max_rel_err={max_rel_err:.6e}",
        result.min_r,
        result.max_r
    );
}

// ======================================================================
// Eccentric orbit (e=0.3)
// ======================================================================

#[test]
fn tier3_orbinit_eccentric() {
    let case = sim_orbinit_families::eccentric();
    let (a, e, i, _raan, _argp, _nu) = sim_orbinit_families::elements::eccentric();

    let mut sim = build_sim(&case);

    println!(
        "Tier 3: Eccentric orbit (a={a:.0} m, e={e}, i={:.1} deg)",
        i.to_degrees()
    );

    let result = run_and_verify_conservation(&case, &mut sim, "eccentric_e03", 2.2e-10, 1e-10);

    // Verify periapsis/apoapsis bounds over the full propagation window.
    let r_peri = a * (1.0 - e);
    let r_apo = a * (1.0 + e);
    println!(
        "  Radius bounds: min={:.1} m (peri={r_peri:.1}), max={:.1} m (apo={r_apo:.1})",
        result.min_r, result.max_r
    );
    assert!(
        result.min_r >= r_peri * 0.999 && result.max_r <= r_apo * 1.001,
        "Radius outside [{r_peri:.0}, {r_apo:.0}] m bounds: min={:.0}, max={:.0}",
        result.min_r,
        result.max_r
    );
}

// ======================================================================
// Highly eccentric orbit (e=0.7)
// ======================================================================

#[test]
fn tier3_orbinit_highly_eccentric() {
    let case = sim_orbinit_families::highly_eccentric();
    let (a, e, i, _raan, _argp, _nu) = sim_orbinit_families::elements::highly_eccentric();

    let mut sim = build_sim(&case);

    println!(
        "Tier 3: Highly eccentric (a={a:.0} m, e={e}, i={:.1} deg)",
        i.to_degrees()
    );

    run_and_verify_conservation(&case, &mut sim, "eccentric_e07", 5.2e-9, 1e-10);
}

// ======================================================================
// Retrograde orbit (i > 90 deg)
// ======================================================================

#[test]
fn tier3_orbinit_retrograde() {
    let case = sim_orbinit_families::retrograde();
    let (a, e, i, _raan, _argp, _nu) = sim_orbinit_families::elements::retrograde();

    let mut sim = build_sim(&case);

    println!(
        "Tier 3: Retrograde orbit (a={a:.0} m, e={e}, i={:.1} deg)",
        i.to_degrees()
    );

    run_and_verify_conservation(&case, &mut sim, "retrograde", 1e-10, 1e-10);

    // Verify orbit is retrograde: angular momentum Z component should be negative
    let body = sim.body(0);
    let h = body
        .trans
        .position
        .raw_si()
        .cross(body.trans.velocity.raw_si());
    assert!(
        h.z < 0.0,
        "Retrograde orbit should have negative h_z, got {:.3e}",
        h.z
    );
}

// ======================================================================
// Equatorial orbit (i ~ 0, RAAN is singular)
// ======================================================================

#[test]
fn tier3_orbinit_equatorial() {
    let case = sim_orbinit_families::equatorial();
    let (a, e, _i, _raan, _argp, _nu) = sim_orbinit_families::elements::equatorial();

    let mut sim = build_sim(&case);

    println!("Tier 3: Equatorial orbit (a={a:.0} m, e={e}, i=0)");

    run_and_verify_conservation(&case, &mut sim, "equatorial", 1e-10, 1e-10);

    // Verify orbit stays in equatorial plane over the full propagation
    // window. A second sim build is required because the first call
    // has already consumed the cadence — `step_n`-style integer
    // counters don't rewind.
    let mut eq_sim = build_sim(&case);
    let (_, n_steps) = synthetic_cadence(&case);
    let mut max_z_frac = 0.0_f64;
    for _ in 0..n_steps {
        eq_sim.step().expect("step failed");
        let body = eq_sim.body(0);
        let pos = body.trans.position.raw_si();
        let z_frac = pos.z.abs() / pos.length();
        max_z_frac = max_z_frac.max(z_frac);
    }
    println!("  Equatorial: max |z|/r = {max_z_frac:.3e}");
    assert!(
        max_z_frac < 1e-12,
        "Equatorial orbit left the equatorial plane: max |z|/r={max_z_frac:.3e}"
    );
}

// ======================================================================
// Polar orbit (i = 90 deg)
// ======================================================================

#[test]
fn tier3_orbinit_polar() {
    let case = sim_orbinit_families::polar();
    let (a, e, _i, _raan, _argp, _nu) = sim_orbinit_families::elements::polar();

    let mut sim = build_sim(&case);

    println!("Tier 3: Polar orbit (a={a:.0} m, e={e}, i=90 deg)");

    run_and_verify_conservation(&case, &mut sim, "polar", 1e-10, 1e-10);

    // Propagate another full cadence to verify the orbit passes over
    // the poles: track the maximum |z|/r ratio across steps. The
    // conservation pass left `sim` at the end of its first cadence;
    // stepping the same number of integer steps again lets us scan a
    // second pass without rebuilding.
    let (_, n_steps) = synthetic_cadence(&case);
    let mut max_z_frac = 0.0_f64;
    for _ in 0..n_steps {
        sim.step().expect("step failed");
        let body = sim.body(0);
        let pos = body.trans.position.raw_si();
        let z_frac = pos.z.abs() / pos.length();
        max_z_frac = max_z_frac.max(z_frac);
    }
    println!("  Polar: max |z|/r = {max_z_frac:.4}");
    assert!(
        max_z_frac > 0.5,
        "Polar orbit should pass over the poles (|z| > r/2): max |z|/r = {max_z_frac:.4}"
    );

    // For polar orbit, angular momentum should be perpendicular to Z:
    // h_z = x*vy − y*vx should be ~0 for i=90.
    let body = sim.body(0);
    let pos = body.trans.position.raw_si();
    let vel = body.trans.velocity.raw_si();
    let r_mag = pos.length();
    let h_z = pos.x * vel.y - pos.y * vel.x;
    let h_mag = specific_ang_momentum(pos, vel);
    let h_z_frac = h_z.abs() / h_mag;
    println!(
        "  Polar: h_z/|h| = {h_z_frac:.3e}, r={:.1} km",
        r_mag / 1000.0
    );
    assert!(
        h_z_frac < 1e-10,
        "Polar orbit h_z should be ~0: h_z/|h|={h_z_frac:.3e}"
    );
}

// ======================================================================
// Hyperbolic orbit (e > 1)
// ======================================================================

#[test]
fn tier3_orbinit_hyperbolic() {
    let case = sim_orbinit_families::hyperbolic();
    let (a, e, _i, _raan, _argp, _nu) = sim_orbinit_families::elements::hyperbolic();
    let r_peri = R_EARTH + 300_000.0;

    let mut sim = build_sim(&case);

    println!(
        "Tier 3: Hyperbolic orbit (a={a:.0} m, e={e}, r_peri={:.0} km)",
        r_peri / 1000.0
    );

    // Energy should be positive for hyperbolic
    let body0 = sim.body(0);
    let e0 = specific_energy(
        body0.trans.position.raw_si(),
        body0.trans.velocity.raw_si(),
        MU_EARTH,
    );
    assert!(
        e0 > 0.0,
        "Hyperbolic orbit should have positive energy: E={e0:.6e}"
    );

    run_and_verify_conservation(&case, &mut sim, "hyperbolic", 1e-10, 1e-10);

    // Verify the body is moving away (radius increasing after periapsis)
    let body = sim.body(0);
    let r_final = body.trans.position.raw_si().length();
    assert!(
        r_final > r_peri * 1.1,
        "Hyperbolic orbit should be escaping: r_final={:.0} m, r_peri={:.0} m",
        r_final,
        r_peri
    );
}

// ======================================================================
// Near-parabolic orbit (e ~ 1)
// ======================================================================

#[test]
fn tier3_orbinit_near_parabolic() {
    let case = sim_orbinit_families::near_parabolic();
    let (a, e, _i, _raan, _argp, _nu) = sim_orbinit_families::elements::near_parabolic();
    let r_peri = R_EARTH + 500_000.0;

    let mut sim = build_sim(&case);

    println!(
        "Tier 3: Near-parabolic orbit (a={a:.0} m, e={e}, r_peri={:.0} km)",
        r_peri / 1000.0
    );

    // Near-parabolic orbits have near-zero energy
    let body0 = sim.body(0);
    let e0 = specific_energy(
        body0.trans.position.raw_si(),
        body0.trans.velocity.raw_si(),
        MU_EARTH,
    );
    println!("  Initial energy: {e0:.6e} J/kg (should be near zero)");

    // Relaxed tolerance for near-parabolic: numerical sensitivity is higher
    run_and_verify_conservation(&case, &mut sim, "near_parabolic", 1e-9, 1e-10);
}
