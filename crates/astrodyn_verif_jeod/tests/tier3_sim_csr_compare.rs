//! Tier 3: SIM_csr_compare — GGM05C 70×70 gravity-acceleration octant sweep.
//!
//! JEOD's `models/environment/gravity/verif/SIM_csr_compare/RUN_01` places a
//! **non-integrating** vehicle (translational & rotational dynamics off) and
//! teleports it through sign-permutations of the ISS-like position
//! `[x, y, z] = [7218634.798…, 18998.642…, 1938152.473…] m` at t = 1…5 s
//! (t = 0 logs the `iss_typical_state` position), logging the gravity
//! potential and acceleration each second. Because the body never integrates,
//! this is a **per-step gravity-evaluation cross-check, not a trajectory
//! propagation** — so rather than driving `Simulation::step()`, it evaluates
//! the same production gravity path the pipeline's gravity stage uses
//! (`GravityControl::evaluate_accel_only` on a GGM05C source) at each logged
//! position and compares against JEOD's `grav_accel`. (Sibling of the Tier 2
//! `grav_geospherical` reference-vector test, but at degree/order 70 and
//! against a full sim's logged output rather than a static fixture.)
//!
//! The sweep positions are JEOD *source* constants (from the RUN_01 input
//! deck / `Modified_data/iss_typical_state.py`), not JEOD output — only the
//! `grav_accel` being compared against is JEOD output, so computational
//! independence holds. The Earth orientation in this sim is the default
//! planet-generic (no RNP/rotation object in the S_define), so the
//! planet-fixed frame coincides with the integration frame and no rotation is
//! applied to the evaluation (`t_inertial_pfix = None`).
//!
//! Reference CSV regenerated via `cargo xtask regenerate-tier3` (the
//! `csr_compare_run01` entry in `trick/generate_references.sh`).

#![allow(
    clippy::float_cmp,
    reason = "position sanity check asserts exact recovery of the literal JEOD input-deck constants logged back into the CSV"
)]

use astrodyn::{gravity_fixtures, GravityControl, GravityGradient, GravityModel, GravitySource};
use astrodyn_verif_jeod::crossval::CrossvalReport;
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use glam::{DMat3, DVec3};

const DEGREE: usize = 70;
const ORDER: usize = 70;

/// One logged record from `csr_compare_run01_csr_compare.csv`.
struct Record {
    grav_accel: DVec3,
    position: DVec3,
}

/// Parse the DRAscii CSV (header row + 6 data rows). Columns:
/// time, grav_pot, grav_accel[0..2], position[0..2].
fn load_csr_compare_csv() -> Vec<Record> {
    let path = test_data_path("csr_compare_run01_csr_compare.csv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "csr_compare reference not found at {}: {e}.\n\
             Regenerate with `JEOD_HOME=… cargo xtask regenerate-tier3`.",
            path.display()
        )
    });
    text.lines()
        .skip(1) // header
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<f64> = line
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .unwrap_or_else(|e| panic!("bad CSV field {s:?}: {e}"))
                })
                .collect();
            assert_eq!(f.len(), 8, "expected 8 columns, got {}", f.len());
            Record {
                grav_accel: DVec3::new(f[2], f[3], f[4]),
                position: DVec3::new(f[5], f[6], f[7]),
            }
        })
        .collect()
}

/// The six sweep positions, reconstructed from the RUN_01 input deck
/// (JEOD source — `Modified_data/iss_typical_state.py` for t=0, then the
/// `trick.add_read` octant teleports for t=1…5).
fn expected_positions() -> [DVec3; 6] {
    // iss_typical_state.py
    let t0 = DVec3::new(-4_292_653.41, 955_168.47, 5_139_356.57);
    // input.py octant constants
    let x = 7_218_634.798_289_895;
    let y = 18_998.641_597_859_56;
    let z = 1_938_152.473_366_886;
    [
        t0,
        DVec3::new(x, y, z),
        DVec3::new(-x, y, z),
        DVec3::new(-x, -y, z),
        DVec3::new(-x, -y, -z),
        DVec3::new(-x, y, -z),
    ]
}

#[test]
fn tier3_csr_compare_gravity_octants() {
    let records = load_csr_compare_csv();
    assert_eq!(records.len(), 6, "expected 6 logged records (t=0..5)");

    // Build a GGM05C spherical-harmonics source through the `astrodyn`
    // facade (the verif crate reads physics through `astrodyn` only).
    let data = gravity_fixtures::load_ggm05c();
    assert!(
        data.degree >= DEGREE && data.order >= ORDER,
        "GGM05C fixture must cover {DEGREE}×{ORDER}"
    );
    let mu = data.mu;
    let source = GravitySource {
        mu,
        model: GravityModel::SphericalHarmonics(Box::new(data)),
    };
    // Earth orientation is the planet-generic default (no RNP object in
    // the S_define), so the planet-fixed frame coincides with the
    // integration frame: the required planet-fixed rotation is identity.
    // The source-id type is irrelevant here (evaluate takes the source
    // explicitly), so use the usize-id form the kernel tests use.
    let control =
        GravityControl::<usize>::new_nonspherical(0_usize, DEGREE, ORDER, GravityGradient::Skip);
    let t_inertial_pfix = DMat3::IDENTITY;

    let positions = expected_positions();
    let mut max_accel_err = [0.0_f64; 3];

    for (i, rec) in records.iter().enumerate() {
        // Sanity: the position we feed our kernel is the same one JEOD
        // logged (confirms row alignment + that our input constants match).
        let dp = (rec.position - positions[i]).abs();
        assert!(
            dp.max_element() < 1e-3,
            "record {i}: logged position {:?} != input-deck position {:?}",
            rec.position,
            positions[i]
        );

        let ga =
            control.evaluate_accel_only(&source, positions[i], Some(&t_inertial_pfix), 0.0, false);
        let ae = (ga.grav_accel - rec.grav_accel).abs();
        max_accel_err[0] = max_accel_err[0].max(ae.x);
        max_accel_err[1] = max_accel_err[1].max(ae.y);
        max_accel_err[2] = max_accel_err[2].max(ae.z);
    }

    let mut report = CrossvalReport::compute("tier3_csr_compare_gravity_octants", &[], &[]);
    report.add_extra("accel_err_x", max_accel_err[0], "m/s^2");
    report.add_extra("accel_err_y", max_accel_err[1], "m/s^2");
    report.add_extra("accel_err_z", max_accel_err[2], "m/s^2");
    report.write();

    // Tolerances set to 1.05× observed max per component (CLAUDE.md policy).
    const ACCEL_TOL: [f64; 3] = [ACCEL_TOL_X, ACCEL_TOL_Y, ACCEL_TOL_Z];
    for k in 0..3 {
        assert!(
            max_accel_err[k] < ACCEL_TOL[k],
            "grav_accel component {k} error {:.6e} m/s^2 exceeds tolerance {:.6e}",
            max_accel_err[k],
            ACCEL_TOL[k]
        );
    }
}

// Our GGM05C 70×70 gravity kernel reproduces JEOD's logged `grav_accel`
// to machine precision (observed per-component max: x = 0, y ≈ 3.5e-18,
// z ≈ 4.4e-16 m/s² — i.e. ~15 significant figures on a ~7 m/s² signal).
// 1.05× observed would be a sub-ULP / zero tolerance, so these use a
// uniform 1e-14 m/s² machine-epsilon floor that proves bit-level
// agreement while absorbing last-ULP variation in the 70×70 harmonic sum.
const ACCEL_TOL_X: f64 = 1.0e-14;
const ACCEL_TOL_Y: f64 = 1.0e-14;
const ACCEL_TOL_Z: f64 = 1.0e-14;
