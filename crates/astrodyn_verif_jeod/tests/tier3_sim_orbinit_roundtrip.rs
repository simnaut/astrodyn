//! Tier 3: Orbit initialization round-trip tests
//!
//! For each orbit family, initializes from orbital elements, propagates through
//! `Simulation::step()`, then recovers orbital elements from the propagated
//! state and verifies they match the originals.
//!
//! This exercises the full pipeline end-to-end: element-to-Cartesian
//! initialization, RK4 integration with point-mass gravity, and
//! Cartesian-to-element recovery -- ensuring no information is lost
//! or corrupted through the pipeline.
//!
//! The `Simulation` construction lives in the
//! [`sim_orbinit_roundtrip`] recipe module so the parity wrapper
//! (`bevy_parity_orbinit_roundtrip.rs`) can drive the same scenarios
//! through the Bevy adapter for the `runner ↔ bevy` half of the
//! transitivity argument.

use astrodyn::recipes::helpers::state_helpers::angle_diff;
use astrodyn::OrbitalElements;
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_orbinit_roundtrip;
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};

/// Earth gravitational parameter (m³/s²) — same const-folded literal
/// the recipe uses, so the expected-vs-recovered comparison resolves
/// against the same bit-pattern that drove the initial state.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Build the recipe's `Simulation` exactly the way the parity trait
/// does — call the scenario factory with a default `InitialConditions`,
/// then `.build()` — so the runner-side propagation here and the
/// Bevy-side propagation in `bevy_parity_orbinit_roundtrip.rs` see the
/// same initial state bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_orbinit_roundtrip` uses this variant
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

/// Propagate `sim` for the recipe's full SyntheticTimes cadence, then
/// recover `OrbitalElements` from the final body state.
fn propagate_and_recover(
    case: &VerificationCase,
    sim: &mut Simulation,
) -> OrbitalElements<astrodyn::Earth> {
    let (dt, n_steps) = synthetic_cadence(case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    use astrodyn::{F64Ext, PlanetInertial, Vec3Ext};
    OrbitalElements::<astrodyn::Earth>::from_cartesian_typed(
        F64Ext::m3_per_s2(MU_EARTH),
        body.trans
            .position
            .raw_si()
            .m_at::<PlanetInertial<astrodyn::Earth>>(),
        body.trans
            .velocity
            .raw_si()
            .m_per_s_at::<PlanetInertial<astrodyn::Earth>>(),
    )
    .expect("from_cartesian_typed failed after propagation")
}

/// Shape/orientation closed-form check shared by every round-trip
/// test except the near-circular one (which compares specific energy
/// instead, see [`tier3_orbinit_roundtrip_circular`]).
///
/// Asserts: semi-major axis (a), eccentricity (e), inclination (i), and
/// conditionally RAAN (non-equatorial) and argument of periapsis
/// (non-circular). Anomalies are excluded because they evolve with time
/// and do not return to their initial values unless the propagation
/// time matches the period exactly.
#[allow(clippy::too_many_arguments)]
fn assert_roundtrip_elements(
    label: &str,
    recovered: &OrbitalElements<astrodyn::Earth>,
    a: f64,
    e: f64,
    i: f64,
    raan: f64,
    argp: f64,
    a_tol: f64,
    e_tol: f64,
    angle_tol: f64,
) {
    println!("  {label}: recovered elements");
    println!(
        "    a: {:.6e} (expected {:.6e}, err {:.3e})",
        recovered.semi_major_axis,
        a,
        (recovered.semi_major_axis - a).abs()
    );
    println!(
        "    e: {:.10} (expected {:.10}, err {:.3e})",
        recovered.e_mag,
        e,
        (recovered.e_mag - e).abs()
    );
    println!(
        "    i: {:.8} rad (expected {:.8}, err {:.3e})",
        recovered.inclination,
        i,
        (recovered.inclination - i).abs()
    );

    let a_err = (recovered.semi_major_axis - a).abs() / a.abs();
    assert!(
        a_err < a_tol,
        "{label}: semi_major_axis relative error {a_err:.6e} exceeds tolerance {a_tol:.1e}"
    );

    let e_err = (recovered.e_mag - e).abs();
    assert!(
        e_err < e_tol,
        "{label}: eccentricity error {e_err:.6e} exceeds tolerance {e_tol:.1e}"
    );

    let i_err = (recovered.inclination - i).abs();
    assert!(
        i_err < angle_tol,
        "{label}: inclination error {i_err:.6e} rad exceeds tolerance {angle_tol:.1e}"
    );

    // RAAN — undefined at the equatorial singularity (line of nodes
    // vanishes when `i ≈ 0` or `i ≈ π`).
    if i > 1e-6 && (std::f64::consts::PI - i) > 1e-6 {
        let raan_err = angle_diff(recovered.long_asc_node, raan);
        assert!(
            raan_err < angle_tol,
            "{label}: RAAN error {raan_err:.6e} rad exceeds tolerance {angle_tol:.1e}"
        );
    }

    // Argument of periapsis — undefined for circular orbits (periapsis
    // direction is unconstrained when `e ≈ 0`).
    if e > 1e-6 {
        let argp_err = angle_diff(recovered.arg_periapsis, argp);
        assert!(
            argp_err < angle_tol,
            "{label}: arg_periapsis error {argp_err:.6e} rad exceeds tolerance {angle_tol:.1e}"
        );
    }
}

// ======================================================================
// Circular LEO round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_circular() {
    let case = sim_orbinit_roundtrip::circular();
    let (_, _e_init, i, raan, _argp, _nu) = sim_orbinit_roundtrip::elements::circular();

    let mut sim = build_sim(&case);

    // For circular orbits, `from_cartesian` switches branches at
    // `e_mag < 1e-13` (setting `a = r_mag` in the circular branch).
    // Tiny numerical eccentricity after integration can cross this
    // threshold, changing how `semi_major_axis` is computed. Instead,
    // compare specific energy `E = v²/2 − μ/r`, which is branch-
    // independent and stable for near-circular orbits.
    let body0 = sim.body(0);
    let energy_0 = body0.trans.velocity.raw_si().length_squared() / 2.0
        - MU_EARTH / body0.trans.position.raw_si().length();

    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    use astrodyn::{F64Ext, PlanetInertial, Vec3Ext};
    let oe_recovered = OrbitalElements::<astrodyn::Earth>::from_cartesian_typed(
        F64Ext::m3_per_s2(MU_EARTH),
        body.trans
            .position
            .raw_si()
            .m_at::<PlanetInertial<astrodyn::Earth>>(),
        body.trans
            .velocity
            .raw_si()
            .m_per_s_at::<PlanetInertial<astrodyn::Earth>>(),
    )
    .expect("from_cartesian_typed failed after propagation");

    let energy_now = body.trans.velocity.raw_si().length_squared() / 2.0
        - MU_EARTH / body.trans.position.raw_si().length();
    let energy_err = (energy_now - energy_0).abs() / energy_0.abs();
    println!("  circular_leo: energy_rel_err={energy_err:.3e}");
    assert!(
        energy_err < 1e-10,
        "circular_leo: energy relative error {energy_err:.6e} exceeds tolerance 1e-10"
    );

    // Eccentricity should remain near zero
    let e_err = oe_recovered.e_mag;
    println!("  circular_leo: recovered e={e_err:.3e}");
    assert!(
        e_err < 1e-8,
        "circular_leo: eccentricity {e_err:.6e} exceeds tolerance 1e-8"
    );

    // Inclination
    let i_err = (oe_recovered.inclination - i).abs();
    assert!(
        i_err < 1e-8,
        "circular_leo: inclination error {i_err:.6e} rad exceeds tolerance 1e-8"
    );

    // RAAN
    let raan_err = angle_diff(oe_recovered.long_asc_node, raan);
    assert!(
        raan_err < 1e-8,
        "circular_leo: RAAN error {raan_err:.6e} rad exceeds tolerance 1e-8"
    );
}

// ======================================================================
// Eccentric orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_eccentric() {
    let case = sim_orbinit_roundtrip::eccentric();
    let (a, e, i, raan, argp, _nu) = sim_orbinit_roundtrip::elements::eccentric();
    let mut sim = build_sim(&case);
    let oe = propagate_and_recover(&case, &mut sim);
    assert_roundtrip_elements(
        "eccentric_e03",
        &oe,
        a,
        e,
        i,
        raan,
        argp,
        1e-10,
        1e-10,
        1e-8,
    );
}

// ======================================================================
// Retrograde orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_retrograde() {
    let case = sim_orbinit_roundtrip::retrograde();
    let (a, e, i, raan, argp, _nu) = sim_orbinit_roundtrip::elements::retrograde();
    let mut sim = build_sim(&case);
    let oe = propagate_and_recover(&case, &mut sim);
    assert_roundtrip_elements("retrograde", &oe, a, e, i, raan, argp, 1e-10, 1e-10, 1e-8);
}

// ======================================================================
// Equatorial orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_equatorial() {
    let case = sim_orbinit_roundtrip::equatorial();
    let (a, e, i, raan, argp, _nu) = sim_orbinit_roundtrip::elements::equatorial();
    let mut sim = build_sim(&case);
    let oe = propagate_and_recover(&case, &mut sim);
    assert_roundtrip_elements("equatorial", &oe, a, e, i, raan, argp, 1e-10, 1e-10, 1e-8);
}

// ======================================================================
// Polar orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_polar() {
    let case = sim_orbinit_roundtrip::polar();
    let (a, e, i, raan, argp, _nu) = sim_orbinit_roundtrip::elements::polar();
    let mut sim = build_sim(&case);
    let oe = propagate_and_recover(&case, &mut sim);
    assert_roundtrip_elements("polar", &oe, a, e, i, raan, argp, 1e-10, 1e-10, 1e-8);
}

// ======================================================================
// Highly eccentric (Molniya-like) round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_molniya() {
    let case = sim_orbinit_roundtrip::molniya();
    let (a, e, i, raan, argp, _nu) = sim_orbinit_roundtrip::elements::molniya();
    let mut sim = build_sim(&case);
    let oe = propagate_and_recover(&case, &mut sim);
    assert_roundtrip_elements(
        "molniya", &oe, a, e, i, raan, argp,
        // Per-element tolerances relaxed slightly for the long-period
        // high-eccentricity case (integrator budget over ~24-hour orbit).
        1e-9, 1e-9, 1e-7,
    );
}

// ======================================================================
// Hyperbolic orbit round-trip (short propagation, recover elements)
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_hyperbolic() {
    let case = sim_orbinit_roundtrip::hyperbolic();
    let (a, e, i, raan, argp, _nu) = sim_orbinit_roundtrip::elements::hyperbolic();
    let mut sim = build_sim(&case);
    let oe = propagate_and_recover(&case, &mut sim);
    assert_roundtrip_elements("hyperbolic", &oe, a, e, i, raan, argp, 1e-10, 1e-10, 1e-10);
}
