//! Tier 3: SIM_orb_elem comprehensive -- 7 orbit families via Simulation pipeline
//!
//! Builds each scenario through its `sim_orbelem_comprehensive` recipe,
//! propagates for the recipe's declared `SyntheticTimes` cadence (one
//! tiny-dt step), and compares the resulting orbital elements against
//! the JEOD-logged columns at t=0.
//!
//! The `Simulation` construction lives in the
//! `sim_orbelem_comprehensive` recipe module so the parity wrapper
//! (`bevy_parity_orbelem_comprehensive.rs`) can drive the same scenarios
//! through the Bevy adapter for the `runner ↔ bevy` half of the
//! transitivity argument.

use astrodyn::recipes::helpers::state_helpers::angle_diff;
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_orbelem_comprehensive;
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};
use glam::DVec3;

/// Full record parsed from the verification CSV (all 21 columns).
struct VerifRecord {
    semi_major_axis: f64,
    semiparam: f64,
    e_mag: f64,
    inclination: f64,
    arg_periapsis: f64,
    long_asc_node: f64,
    r_mag: f64,
    vel_mag: f64,
    true_anom: f64,
    mean_anom: f64,
    mean_motion: f64,
    orbital_anom: f64,
    orb_energy: f64,
    orb_ang_momentum: f64,
    position: DVec3,
    velocity: DVec3,
}

/// Parse the t=0 row from a verification CSV.
fn load_verif_record(csv_name: &str) -> VerifRecord {
    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "SIM_OrbElem verification CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let content = std::fs::read_to_string(&csv_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_OrbElem CSV from {}: {e}",
            csv_path.display()
        )
    });

    let line = content
        .lines()
        .nth(1)
        .expect("CSV must have at least one data row");
    let f: Vec<&str> = line.split(',').collect();
    assert!(f.len() >= 21, "expected >=21 columns, got {}", f.len());
    let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };

    VerifRecord {
        semi_major_axis: p(1),
        semiparam: p(2),
        e_mag: p(3),
        inclination: p(4),
        arg_periapsis: p(5),
        long_asc_node: p(6),
        r_mag: p(7),
        vel_mag: p(8),
        true_anom: p(9),
        mean_anom: p(10),
        mean_motion: p(11),
        orbital_anom: p(12),
        orb_energy: p(13),
        orb_ang_momentum: p(14),
        position: DVec3::new(p(15), p(16), p(17)),
        velocity: DVec3::new(p(18), p(19), p(20)),
    }
}

/// Build the recipe's `Simulation` exactly the way the parity trait does
/// — call the scenario factory with a default `InitialConditions` (the
/// recipes don't read it — initial state is baked in from each case's
/// JEOD-output t=0 row), then `.build()` — so the runner-side
/// propagation here and the Bevy-side propagation in
/// `bevy_parity_orbelem_comprehensive.rs` see the same initial state
/// bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_orbelem_comprehensive` uses this
/// variant because the orbelem verification CSVs are initialization-only
/// (one row at t=0); panicking on any other variant surfaces a future
/// recipe-shape drift here rather than producing a silently-truncated
/// propagation. Returning both halves of the cadence lets callers
/// assert that the `dt` they're stepping at (`sim.dt`) matches the
/// cadence the recipe declared.
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

fn assert_close(name: &str, computed: f64, reference: f64, rel_tol: f64, abs_tol: f64) {
    let diff = (computed - reference).abs();
    let tol = if reference.abs() > abs_tol {
        rel_tol * reference.abs()
    } else {
        abs_tol
    };
    assert!(
        diff <= tol,
        "{name}: computed={computed:.15e}, reference={reference:.15e}, \
         diff={diff:.6e}, tol={tol:.6e}"
    );
}

fn assert_angle_close(name: &str, computed: f64, reference: f64, abs_tol: f64) {
    let diff = angle_diff(computed, reference);
    assert!(
        diff <= abs_tol,
        "{name}: computed={computed:.15e}, reference={reference:.15e}, \
         angle_diff={diff:.6e}, tol={abs_tol:.6e}"
    );
}

/// Build a Simulation from the recipe, propagate for the declared
/// SyntheticTimes cadence, and compare orbital elements against the
/// JEOD-logged t=0 row from `csv_name`. The recipe's baked-in initial
/// state is cross-checked against the CSV's position/velocity so a
/// future recipe-side edit can't silently drift away from the
/// JEOD-source values.
fn verify_orbit_family(
    case: VerificationCase,
    csv_name: &str,
    label: &str,
    skip_degenerate_scalars: bool,
) {
    let rec = load_verif_record(csv_name);

    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    // Fence: the recipe's baked-in state must reproduce the JEOD-logged
    // t=0 row to f64 precision. A future edit that tweaks recipe-side
    // numbers will trip this check first instead of silently changing
    // what the parity wrapper integrates.
    let body0 = sim.body(0);
    let init_pos = body0.trans.position.raw_si();
    let init_vel = body0.trans.velocity.raw_si();
    let pos_drift = (init_pos - rec.position).length();
    let vel_drift = (init_vel - rec.velocity).length();
    assert!(
        pos_drift < 1e-6,
        "{label}: recipe init position drifted from CSV by {pos_drift:.6e} m"
    );
    assert!(
        vel_drift < 1e-9,
        "{label}: recipe init velocity drifted from CSV by {vel_drift:.6e} m/s"
    );

    // `step_n` advances exactly `n_steps` whole steps. (`step_until`
    // has a 1 ms slop and may stop one step short.)
    sim.step_n(n_steps).expect("step_n failed");

    let output = sim.body(0);
    let oe = output
        .orbital_elements
        .as_ref()
        .unwrap_or_else(|| panic!("{label}: orbital_elements not computed after propagation"));

    println!("Tier 3 (Simulation): {label}");
    println!(
        "  JEOD sma: {:.15e}  ours: {:.15e}",
        rec.semi_major_axis, oe.semi_major_axis
    );
    println!("  JEOD e:   {:.15e}  ours: {:.15e}", rec.e_mag, oe.e_mag);
    println!(
        "  JEOD i:   {:.15e}  ours: {:.15e}",
        rec.inclination, oe.inclination
    );

    let sma_rel_tol = 3.0e-6;
    let semiparam_rel_tol = 2.1e-6;
    let ecc_abs_tol = 2.0e-6;
    let ecc_rel_tol = 1.5e-4;
    let incl_abs_tol = 1.1e-15;
    let rmag_rel_tol = 1e-10;
    let vmag_rel_tol = 1e-10;
    let mean_motion_rel_tol = 5.6e-6;
    let energy_rel_tol = 6.0e-6;
    let ang_mom_rel_tol = 1e-10;
    let angle_tol = 1.6e-4;

    if !skip_degenerate_scalars {
        assert_close(
            &format!("{label}/semi_major_axis"),
            oe.semi_major_axis,
            rec.semi_major_axis,
            sma_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/r_mag"),
            oe.r_mag,
            rec.r_mag,
            rmag_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/vel_mag"),
            oe.vel_mag,
            rec.vel_mag,
            vmag_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/mean_motion"),
            oe.mean_motion,
            rec.mean_motion,
            mean_motion_rel_tol,
            1e-15,
        );
        assert_close(
            &format!("{label}/orb_energy"),
            oe.orb_energy,
            rec.orb_energy,
            energy_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/orb_ang_momentum"),
            oe.orb_ang_momentum,
            rec.orb_ang_momentum,
            ang_mom_rel_tol,
            1e-6,
        );
    }

    assert_close(
        &format!("{label}/semiparam"),
        oe.semiparam,
        rec.semiparam,
        semiparam_rel_tol,
        1e-6,
    );

    if rec.e_mag > 0.001 {
        assert_close(
            &format!("{label}/e_mag"),
            oe.e_mag,
            rec.e_mag,
            ecc_rel_tol,
            ecc_abs_tol,
        );
    } else {
        let ecc_diff = (oe.e_mag - rec.e_mag).abs();
        assert!(
            ecc_diff < ecc_abs_tol,
            "{label}/e_mag (near-circular): computed={:.15e}, reference={:.15e}, \
             diff={ecc_diff:.6e}, tol={ecc_abs_tol:.6e}",
            oe.e_mag,
            rec.e_mag
        );
    }

    assert_close(
        &format!("{label}/inclination"),
        oe.inclination,
        rec.inclination,
        1e-6,
        incl_abs_tol,
    );

    let skip_periapsis_angles = rec.e_mag < 0.001;
    if !skip_periapsis_angles {
        assert_angle_close(
            &format!("{label}/arg_periapsis"),
            oe.arg_periapsis,
            rec.arg_periapsis,
            angle_tol,
        );
        assert_angle_close(
            &format!("{label}/true_anom"),
            oe.true_anom,
            rec.true_anom,
            angle_tol,
        );
        assert_angle_close(
            &format!("{label}/mean_anom"),
            oe.mean_anom,
            rec.mean_anom,
            angle_tol,
        );
        assert_angle_close(
            &format!("{label}/orbital_anom"),
            oe.orbital_anom,
            rec.orbital_anom,
            angle_tol,
        );
    } else {
        println!(
            "  (skipping periapsis-referenced angles: e_mag={:.3e} < 0.001)",
            rec.e_mag
        );
    }

    if rec.inclination > 1e-10 {
        assert_angle_close(
            &format!("{label}/long_asc_node"),
            oe.long_asc_node,
            rec.long_asc_node,
            angle_tol,
        );
    } else {
        println!("  (skipping LAN: inclination={:.3e} ~ 0)", rec.inclination);
    }

    println!("  PASS: {label}");
}

#[test]
fn tier3_simulation_orbelem_t01() {
    verify_orbit_family(
        sim_orbelem_comprehensive::t01(),
        "orbelem_verif_t01_orbelem.csv",
        "T01 circular (e~0)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t10() {
    verify_orbit_family(
        sim_orbelem_comprehensive::t10(),
        "orbelem_verif_t10_orbelem.csv",
        "T10 eccentric (0<e<1)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t20() {
    verify_orbit_family(
        sim_orbelem_comprehensive::t20(),
        "orbelem_verif_t20_orbelem.csv",
        "T20 hyperbolic (e>1)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t30() {
    verify_orbit_family(
        sim_orbelem_comprehensive::t30(),
        "orbelem_verif_t30_orbelem.csv",
        "T30 near-parabolic (e~1)",
        true,
    );
}

#[test]
fn tier3_simulation_orbelem_t40() {
    verify_orbit_family(
        sim_orbelem_comprehensive::t40(),
        "orbelem_verif_t40_orbelem.csv",
        "T40 retrograde (i>90deg)",
        true,
    );
}

#[test]
fn tier3_simulation_orbelem_t50() {
    verify_orbit_family(
        sim_orbelem_comprehensive::t50(),
        "orbelem_verif_t50_orbelem.csv",
        "T50 equatorial (i~0)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t55() {
    verify_orbit_family(
        sim_orbelem_comprehensive::t55(),
        "orbelem_verif_t55_orbelem.csv",
        "T55 polar (i~90deg)",
        false,
    );
}
