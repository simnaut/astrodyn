//! Tier 3: SIM_orb_elem comprehensive -- 7 orbit families via Simulation pipeline
//!
//! Creates a `Simulation` for each orbit family, adds a body with
//! `orbital_elements_source` configured, steps once to trigger derived-state
//! computation, and compares orbital elements against JEOD reference.

use astrodyn::recipes::helpers::state_helpers::angle_diff;
use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::{DerivedStateConfig, GravitySourceEntry, VehicleConfig};
use astrodyn::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};
use astrodyn_runner::{RotationModel, Simulation};
use glam::DVec3;

fn load_mu_earth() -> f64 {
    astrodyn_gravity::fixtures::load_ggm05c().mu
}

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

/// Create a Simulation, add body with orbital_elements derived state, step once,
/// and compare orbital elements against JEOD reference.
fn verify_orbit_family(csv_name: &str, label: &str, skip_degenerate_scalars: bool) {
    let mu_earth = load_mu_earth();
    let rec = load_verif_record(csv_name);

    // Create Simulation with Earth point-mass gravity
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    // Use tiny dt so one step barely changes the state.
    // dt=1e-9 keeps position drift below 1e-5 m.
    let mut sim = Simulation::new(time, 1e-9);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: rec.position,
            velocity: rec.velocity,
        }
        .into(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        derived: DerivedStateConfig {
            orbital_elements_source: Some(earth),
            ..Default::default()
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    // Step once to trigger derived-state computation (stage 9)
    sim.step().expect("step failed");

    let output = sim.body(0);
    let oe = output
        .orbital_elements
        .as_ref()
        .unwrap_or_else(|| panic!("{label}: orbital_elements not computed after step()"));

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
    verify_orbit_family("orbelem_verif_t01_orbelem.csv", "T01 circular (e~0)", false);
}

#[test]
fn tier3_simulation_orbelem_t10() {
    verify_orbit_family(
        "orbelem_verif_t10_orbelem.csv",
        "T10 eccentric (0<e<1)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t20() {
    verify_orbit_family(
        "orbelem_verif_t20_orbelem.csv",
        "T20 hyperbolic (e>1)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t30() {
    verify_orbit_family(
        "orbelem_verif_t30_orbelem.csv",
        "T30 near-parabolic (e~1)",
        true,
    );
}

#[test]
fn tier3_simulation_orbelem_t40() {
    verify_orbit_family(
        "orbelem_verif_t40_orbelem.csv",
        "T40 retrograde (i>90deg)",
        true,
    );
}

#[test]
fn tier3_simulation_orbelem_t50() {
    verify_orbit_family(
        "orbelem_verif_t50_orbelem.csv",
        "T50 equatorial (i~0)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t55() {
    verify_orbit_family(
        "orbelem_verif_t55_orbelem.csv",
        "T55 polar (i~90deg)",
        false,
    );
}
